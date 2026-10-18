"""Slide catalog, persistent inference jobs and region-level evidence."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import csv
import io
import json
import threading
import uuid
import traceback
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from ..data import SAMPLES, SlideDataset, image_path, open_image, spatial_graph
from ..features import ResNetEncoder
from ..pipeline import load_model, predict_slide, run_directory, seed
from .bank import PrototypeLibrary, TISSUES
from .messages import UserError, message, error_payload


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        seed(cfg)
        self.root = Path(cfg["paths"]["artifacts"]) / "workspace"
        self.root.mkdir(parents=True, exist_ok=True)
        self.library = PrototypeLibrary(cfg)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dragh2st-inference")
        self.lock = threading.RLock()
        self.jobs = {}
        self.models = {}
        self.slides = {}
        for dataset, names in SAMPLES.items():
            for name in names:
                identifier = f"{dataset}:{name}"
                self.slides[identifier] = {"id": identifier, "dataset": dataset, "name": name,
                                            "tissue": TISSUES[dataset], "uploaded": False,
                                            "image": str(image_path(cfg["paths"]["data"], dataset, name))}
        for path in (self.root / "uploads").glob("*/slide.json"):
            slide = json.loads(path.read_text())
            self.slides[slide["id"]] = slide

    def model_fold(self, dataset):
        if dataset == self.cfg["training"]["dataset"]:
            return self.cfg["training"]["fold"]
        return self.cfg["datasets"]["folds"][dataset]

    def model_ready(self, dataset):
        return (run_directory(self.cfg, dataset, self.model_fold(dataset)) / "model.pt").exists()

    def catalog(self):
        return {"slides": [{key: value for key, value in slide.items() if key != "image"} | {
            "predicted": self.prediction_path(slide["id"]).exists(),
            "model_ready": self.model_ready(slide["dataset"])} for slide in self.slides.values()],
                "libraries": self.library.catalog(),
                "models": [{"dataset": name, "tissue": TISSUES[name], "ready": self.model_ready(name),
                            "fold": self.model_fold(name)} for name in SAMPLES]}

    def prediction_path(self, identifier):
        # Never derive filesystem paths directly from uploaded/display names.
        import hashlib
        slide = self.slide(identifier)
        checkpoint = run_directory(self.cfg, slide["dataset"], self.model_fold(slide["dataset"])) / "model.pt"
        stamp = str(checkpoint.stat().st_mtime_ns) if checkpoint.exists() else "missing"
        key = hashlib.sha256((identifier + stamp + "reference-numerics-v1").encode()).hexdigest()
        return self.root / "predictions" / f"{key}.npz"

    def slide(self, identifier):
        if identifier not in self.slides:
            raise UserError("error.slideMissing", status=404)
        return self.slides[identifier]

    @lru_cache(maxsize=2)
    def image(self, identifier):
        with open_image(self.slide(identifier)["image"]) as image:
            result = image.convert("RGB")
            result.load()
        return result

    def info(self, identifier):
        slide = self.slide(identifier)
        image = self.image(identifier)
        pixels = self.pixels(identifier)
        return {key: value for key, value in slide.items() if key != "image"} | {
            "width": image.width, "height": image.height, "pixels": pixels,
            "spots": len(pixels), "predicted": self.prediction_path(identifier).exists(),
            "model_ready": self.model_ready(slide["dataset"]),
            "coordinate_note": message("coordinateWarning")["message"] if slide["dataset"] == "dlpfc" and not slide["uploaded"] else None,
            "coordinate_note_key": "coordinateWarning" if slide["dataset"] == "dlpfc" and not slide["uploaded"] else None}

    @lru_cache(maxsize=40)
    def pixels(self, identifier):
        slide = self.slide(identifier)
        if slide["uploaded"]:
            with np.load(Path(slide["image"]).parent / "coordinates.npz") as coordinates:
                return coordinates["pixels"].tolist()
        fold = SAMPLES[slide["dataset"]].index(slide["name"])
        dataset = SlideDataset(self.cfg["paths"]["data"], slide["dataset"], fold, "test")
        return dataset.slides[slide["name"]]["pixels"].tolist()

    @lru_cache(maxsize=8)
    def preview(self, identifier):
        image = self.image(identifier).copy()
        image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
        buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=88)
        return buffer.getvalue()

    def tile(self, identifier, level, column, row):
        if not 0 <= level <= 8 or column < 0 or row < 0:
            raise UserError("error.invalidTile")
        factor = 2 ** level
        x, y = column * 256 * factor, row * 256 * factor
        image = self.image(identifier)
        if x >= image.width or y >= image.height:
            raise UserError("error.tileOutside")
        tile = image.crop((x, y, min(x + 256 * factor, image.width), min(y + 256 * factor, image.height)))
        tile.thumbnail((256, 256), Image.Resampling.LANCZOS)
        buffer = io.BytesIO(); tile.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()

    def upload(self, filename, contents, dataset, coordinate_text=None):
        if dataset not in SAMPLES:
            raise UserError("error.selectModel")
        if len(contents) > 512 * 1024 * 1024:
            raise UserError("error.imageSize")
        try:
            with Image.open(io.BytesIO(contents)) as source:
                if source.width * source.height > 400_000_000:
                    raise UserError("error.imagePixels")
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise UserError("error.invalidImage") from error
        if min(image.size) < 112:
            raise UserError("error.imageDimensions")
        pixels, positions = [], []
        if coordinate_text:
            rows = list(csv.DictReader(io.StringIO(coordinate_text)))
            if not rows or not {"pixel_x", "pixel_y"}.issubset(rows[0]):
                raise UserError("error.csvColumns")
            pixels = [[float(row["pixel_x"]), float(row["pixel_y"])] for row in rows]
            if not np.isfinite(pixels).all():
                raise UserError("error.finiteCoordinates")
            if {"x", "y"}.issubset(rows[0]):
                positions = [[int(row["x"]), int(row["y"])] for row in rows]
            else:
                origin = np.min(pixels, axis=0)
                positions = np.floor((np.asarray(pixels) - origin) / 112).astype(int).tolist()
        else:
            # Automatically cover the image with at most 64x64 query spots.
            step = max(112, int(np.ceil(max(image.size) / 64)))
            for gy, y in enumerate(range(56, image.height - 55, step)):
                for gx, x in enumerate(range(56, image.width - 55, step)):
                    patch = np.asarray(image.crop((x-56, y-56, x+56, y+56)))
                    if (patch.min(axis=2) < 220).mean() >= 0.1:
                        pixels.append([x, y]); positions.append([gx, gy])
        pixels, positions = np.floor(np.asarray(pixels)).astype(int), np.asarray(positions, dtype=int)
        if len(pixels) < 2 or len(pixels) > 5000:
            raise UserError("error.spotCount")
        if not np.isfinite(pixels).all() or pixels.shape != (len(positions), 2):
            raise UserError("error.coordinateFormat")
        if positions.min() < 0 or positions.max() >= 128:
            raise UserError("error.gridRange")
        if len(np.unique(pixels, axis=0)) != len(pixels) or len(np.unique(positions, axis=0)) != len(positions):
            raise UserError("error.duplicateCoordinates")
        if (pixels < 56).any() or (pixels[:, 0] > image.width-56).any() or (pixels[:, 1] > image.height-56).any():
            raise UserError("error.edgeCoordinates")
        identifier = "upload:" + uuid.uuid4().hex
        directory = self.root / "uploads" / identifier.split(":")[1]
        directory.mkdir(parents=True)
        path = directory / "image.tif"; image.save(path, compression="tiff_lzw")
        np.savez_compressed(directory / "coordinates.npz", pixels=pixels, positions=positions)
        slide = {"id": identifier, "dataset": dataset, "name": Path(filename).name[:120],
                 "tissue": TISSUES[dataset], "uploaded": True, "image": str(path)}
        (directory / "slide.json").write_text(json.dumps(slide, ensure_ascii=False, indent=2))
        with self.lock:
            self.slides[identifier] = slide
        return {key: value for key, value in slide.items() if key != "image"}

    def submit(self, kind, identifier, tissue=None):
        self.slide(identifier)
        if kind not in {"predict", "index"}:
            raise UserError("error.unknownTask")
        with self.lock:
            for job in self.jobs.values():
                if job["kind"] == kind and job["slide"] == identifier and job["status"] in {"queued", "running"}:
                    return job
            job = {"id": uuid.uuid4().hex, "kind": kind, "slide": identifier, "status": "queued", **message("job.queued")}
            self.jobs[job["id"]] = job
            self.executor.submit(self.execute, job["id"], tissue)
        return dict(job)

    def execute(self, job_id, tissue):
        job = self.jobs[job_id]
        try:
            job.update(status="running", **message("job.predicting" if job["kind"] == "predict" else "job.indexing"))
            identifier = job["slide"]
            slide = self.slide(identifier)
            if job["kind"] == "predict":
                if not self.model_ready(slide["dataset"]):
                    raise UserError("error.modelNotReady")
                self.predict(identifier)
                job.update(status="complete", **message("job.predicted"))
            else:
                result = self.result(identifier)
                count = self.library.add_slide(slide["dataset"], identifier, result, tissue=tissue or slide["tissue"])
                job.update(status="complete", count=count, **message("job.indexed", count=count))
        except Exception as error:
            payload = error_payload(error)
            job.update(status="failed", **message(payload["error_key"], **payload["error_params"]))
            if not isinstance(error, UserError):
                traceback.print_exc()
        finally:
            # Keep only one model resident to accommodate large tissue graphs.
            self.models.clear()
            torch.cuda.empty_cache()

    def model(self, dataset):
        if dataset not in self.models:
            self.models.clear()
            model, _ = load_model(self.cfg, dataset, self.model_fold(dataset))
            self.models[dataset] = model
        return self.models[dataset]

    def predict(self, identifier):
        slide = self.slide(identifier)
        model = self.model(slide["dataset"])
        if not slide["uploaded"]:
            _, result = predict_slide(self.cfg, slide["dataset"], slide["name"], model)
        else:
            coordinates = np.load(Path(slide["image"]).parent / "coordinates.npz")
            pixels, positions = coordinates["pixels"], coordinates["positions"]
            image = self.image(identifier)
            patches = torch.stack([torch.Tensor(np.array(image.crop((x-56, y-56, x+56, y+56)))).permute(2, 1, 0)
                                   for x, y in pixels])
            if slide["dataset"] != "her2st":
                patches = patches / 255.0
            encoder = ResNetEncoder(self.cfg)
            features = encoder.encode(patches)
            del encoder, patches
            adjacency = spatial_graph(positions)
            with torch.inference_mode():
                prediction, _, hidden = model(features.to(model.device), torch.LongTensor(positions).to(model.device), adjacency.to(model.device))
            result = {"prediction": prediction.cpu().numpy(), "features": features.numpy(), "genes": model.genes,
                      "pixels": pixels, "barcodes": [f"spot-{i}" for i in range(len(pixels))]}
        target = self.prediction_path(identifier)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.npz")
        if not np.isfinite(result["prediction"]).all():
            raise UserError("error.nonfinitePrediction")
        np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in result.items()})
        temporary.replace(target)
        self.read_result.cache_clear()

    def result(self, identifier):
        return self.read_result(str(self.prediction_path(identifier)))

    @lru_cache(maxsize=2)
    def read_result(self, filename):
        path = Path(filename)
        if not path.exists():
            raise UserError("error.predictFirst")
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def region(self, identifier, selection):
        result = self.result(identifier)
        pixels = result["pixels"]
        if "spot" in selection:
            index = int(selection["spot"])
            if not 0 <= index < len(pixels):
                raise UserError("error.spotMissing")
            indices = np.array([index])
        elif "bounds" in selection:
            x0, y0, x1, y1 = map(float, selection["bounds"])
            if not np.isfinite([x0, y0, x1, y1]).all():
                raise UserError("error.regionCoordinates")
            mask = (pixels[:, 0] >= min(x0,x1)) & (pixels[:, 0] <= max(x0,x1)) & (pixels[:, 1] >= min(y0,y1)) & (pixels[:, 1] <= max(y0,y1))
            indices = np.flatnonzero(mask)
        else:
            indices = np.arange(len(pixels))
        if not len(indices):
            raise UserError("error.emptyRegion")
        expression = result["prediction"][indices].mean(axis=0)
        order = np.argsort(expression)[::-1]
        ranking = [{"gene": str(result["genes"][i]), "value": float(expression[i])} for i in order]
        tissue = selection.get("tissue") or self.slide(identifier)["tissue"]
        prototypes = self.library.retrieve(result["features"][indices].mean(axis=0), list(result["genes"]), tissue)
        return {"spots": len(indices), "ranking": ranking, "prototypes": self.describe_prototypes(prototypes), "tissue": tissue,
                "source": "predicted", "aggregation": "mean", "indices": indices.tolist()}

    def describe_prototypes(self, rows):
        for row in rows:
            identifier = row["slide"] if row["slide"] in self.slides else f"{row['dataset']}:{row['slide']}"
            row["slide_name"] = self.slides.get(identifier, {}).get("name", row["slide"])
        return rows

    def heatmap(self, identifier, gene):
        result = self.result(identifier)
        genes = list(result["genes"])
        if gene not in genes:
            raise UserError("error.geneNotInPanel")
        values = result["prediction"][:, genes.index(gene)]
        return {"gene": gene, "values": values.tolist(), "min": float(values.min()), "max": float(values.max())}

    def thumbnail(self, prototype_id):
        prototype = self.library.get(prototype_id)
        identifier = prototype["slide"]
        if identifier not in self.slides:
            identifier = f"{prototype['dataset']}:{identifier}"
        x, y = prototype["pixel_x"], prototype["pixel_y"]
        return self.thumbnail_image(identifier, x, y)

    @lru_cache(maxsize=2048)
    def thumbnail_image(self, identifier, x, y):
        patch = self.image(identifier).crop((x-56,y-56,x+56,y+56)).resize((224,224))
        buffer = io.BytesIO(); patch.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()
