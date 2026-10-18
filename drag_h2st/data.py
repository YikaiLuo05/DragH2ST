"""Slide-level datasets with the exact July-26 reference preprocessing."""
from pathlib import Path
import gzip
import io
import json
import os
import numpy as np
import pandas as pd
from PIL import Image
import scprep
import torch
from scipy.spatial import distance

Image.MAX_IMAGE_PIXELS = None
SAMPLES = {
    "her2st": [f"{p}{i}" for p in "AB" for i in range(1, 7)],
    "cscc": [f"{p}_ST_rep{r}" for p in ("P2", "P5", "P9", "P10") for r in range(1, 4)],
    "dlpfc": ["151507", "151508", "151509", "151510", "151669", "151670",
              "151671", "151672", "151673", "151674", "151675", "151676"],
}
GENE_FILES = {"her2st": "her_hvg_cut_1000.npy", "cscc": "cSCC_hvg_cut_1000.npy",
              "dlpfc": "dlpfc_hvg_cut_1000.npy"}


def spatial_graph(coordinates):
    """Directed four-neighbor graph, retaining distances <= 2 grid units."""
    n = len(coordinates)
    adjacency = torch.zeros((n, n))
    for i in range(n):
        distances = distance.cdist(coordinates[i:i + 1], coordinates, "euclidean")
        neighbors = distances.argsort()[0][1:5]
        for j in neighbors:
            if distances[0, j] <= 2.0:
                adjacency[i, j] = 1.0
    return adjacency


def first_match(directory, pattern):
    matches = list(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matches {directory / pattern}")
    return matches[0]


def image_path(root, dataset, slide):
    root = Path(root)
    if dataset == "her2st":
        directory = root / "her2st/data/ST-imgs" / slide[0] / slide
        return directory / next(iter(os.listdir(directory)))
    if dataset == "cscc":
        return first_match(root / "cSCC/GSE144239_RAW", f"*_{slide}.jpg*")
    # Reference convention: scaled coordinates on the full TIFF. See README.
    return root / "dlpfc" / slide / f"{slide}_full_image.tif"


def open_image(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return Image.open(io.BytesIO(handle.read()))
    return Image.open(path)


def read_metadata(root, dataset, slide):
    root = Path(root)
    if dataset in {"her2st", "cscc"}:
        if dataset == "her2st":
            counts = root / "her2st/data/ST-cnts" / f"{slide}.tsv"
            positions = root / "her2st/data/ST-spotfiles" / f"{slide}_selection.tsv"
        else:
            directory = root / "cSCC/GSE144239_RAW"
            counts = first_match(directory, f"*_{slide}_stdata.tsv*")
            positions = first_match(directory, f"*spot_data-selection-{slide}.tsv*")
        counts = pd.read_csv(counts, sep="\t", index_col=0, compression="infer")
        positions = pd.read_csv(positions, sep="\t", compression="infer")
        x, y = np.around(positions.x).astype(int), np.around(positions.y).astype(int)
        positions["id"] = [f"{a}x{b}" for a, b in zip(x, y)]
        return counts.join(positions.set_index("id"), how="left" if dataset == "her2st" else "inner")
    import scanpy as sc
    directory = root / "dlpfc" / slide
    adata = sc.read_10x_h5(directory / f"{slide}_filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    positions = pd.read_csv(directory / "tissue_positions_list.txt", header=None,
                            names=["barcode", "in_tissue", "row", "col", "pixel_y", "pixel_x"])
    positions = positions[positions.in_tissue == 1].set_index("barcode")
    scale = json.loads((directory / "scalefactors_json.json").read_text())["tissue_hires_scalef"]
    positions["pixel_x"] *= scale
    positions["pixel_y"] *= scale
    positions["x"], positions["y"] = positions.row, positions.col
    barcodes = list(set(adata.obs_names) & set(positions.index))
    adata, positions = adata[barcodes], positions.loc[barcodes]
    counts = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    return pd.DataFrame(counts, index=adata.obs_names, columns=adata.var_names).join(
        positions[["pixel_x", "pixel_y", "x", "y"]])


class SlideDataset(torch.utils.data.Dataset):
    """One section per batch. Features are supplied separately from image crops."""
    def __init__(self, root, dataset, fold, split):
        self.root, self.dataset, self.fold, self.split = str(root), dataset, fold, split
        held_out = SAMPLES[dataset][fold]
        self.names = sorted(set(SAMPLES[dataset]) - {held_out}) if split == "train" else [held_out]
        self.genes = [str(g) for g in np.load(Path(root) / GENE_FILES[dataset], allow_pickle=True)]
        self.slides = {}
        for name in self.names:
            meta = read_metadata(root, dataset, name)
            if any(g not in meta.columns for g in self.genes):
                raise ValueError(f"Target genes missing from {dataset}/{name}; use the reference gene panel")
            counts = meta[self.genes].values
            values = np.nan_to_num(counts, nan=0, posinf=0, neginf=0) if dataset != "her2st" else counts
            expression = scprep.transform.log(scprep.normalize.library_size_normalize(values))
            if dataset != "her2st":
                expression = np.nan_to_num(expression, nan=0, posinf=0, neginf=0)
            totals = counts.sum(1)
            positions = meta[["x", "y"]].values
            pixels = np.floor(meta[["pixel_x", "pixel_y"]].values).astype(int)
            self.slides[name] = {
                "expression": torch.Tensor(expression), "counts": torch.Tensor(counts),
                "size_factors": torch.Tensor(totals / np.median(totals)),
                "positions": torch.LongTensor(positions), "pixels": torch.Tensor(pixels),
                "adjacency": spatial_graph(positions), "barcodes": list(meta.index.astype(str)),
            }

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        return {"slide": name, **{k: v for k, v in self.slides[name].items() if k != "barcodes"}}

    def crops(self, name):
        """112x112 crops, preserving reference transpose and intensity scaling."""
        with open_image(image_path(self.root, self.dataset, name)) as image:
            pixels = torch.Tensor(np.array(image)).permute(1, 0, 2)
        centers = self.slides[name]["pixels"].long()
        patches = torch.zeros((len(centers), 3, 112, 112))
        for i, (x, y) in enumerate(centers.tolist()):
            patch = pixels[x - 56:x + 56, y - 56:y + 56, :]
            if patch.shape != (112, 112, 3):
                raise ValueError(f"Reference crop extends beyond image: {name}, spot {i}")
            patches[i] = patch.permute(2, 0, 1)
        return patches / 255.0 if self.dataset != "her2st" else patches
