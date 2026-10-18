"""Preparation, training and inference using config.yaml exclusively."""
import json
import random
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from .data import SlideDataset
from .features import FeatureDataset, prepare_features
from .dp_rag.prior import build_prior
from .model import DragH2ST
from .metrics import compute_metrics


def seed(cfg):
    value = cfg["training"]["seed"]
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def run_directory(cfg, dataset=None, fold=None):
    dataset = dataset or cfg["training"]["dataset"]
    fold = cfg["training"]["fold"] if fold is None else fold
    return Path(cfg["paths"]["artifacts"]) / "runs" / dataset / f"fold{fold}"


class TrainingRecord(Callback):
    def __init__(self, directory):
        self.path = directory / "epochs.jsonl"

    def on_validation_epoch_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        metrics = {key: float(value) for key, value in trainer.callback_metrics.items()}
        with self.path.open("a") as handle:
            handle.write(json.dumps({"epoch": trainer.current_epoch, **metrics}) + "\n")
        if trainer.current_epoch % 25 == 0:
            print(f"Epoch {trainer.current_epoch}: validation score {metrics.get('val_score', float('nan')):.6f}", flush=True)


def prepare(cfg):
    for name, fold in cfg["datasets"]["folds"].items():
        for split in ("train", "test"):
            seed(cfg)
            dataset = SlideDataset(cfg["paths"]["data"], name, fold, split)
            prepare_features(cfg, dataset)
        build_prior(cfg, dataset.genes, name)


def make_model(cfg, genes, name):
    similarity, embeddings, meta = build_prior(cfg, genes, name)
    model = DragH2ST(cfg, genes)
    model.cgdr.set_prior(similarity, embeddings, cfg["prior"]["similarity_threshold"], cfg["prior"]["top_k"])
    return model


def train(cfg):
    name, fold = cfg["training"]["dataset"], cfg["training"]["fold"]
    trainset = SlideDataset(cfg["paths"]["data"], name, fold, "train")
    testset = SlideDataset(cfg["paths"]["data"], name, fold, "test")
    prepare_features(cfg, trainset)
    prepare_features(cfg, testset)
    seed(cfg)  # Extraction consumes RNG; model initialization always starts here.
    model = make_model(cfg, trainset.genes, name)
    train_loader = DataLoader(FeatureDataset(cfg, trainset), batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(FeatureDataset(cfg, testset), batch_size=1, shuffle=False, num_workers=0)
    directory = run_directory(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=directory / "checkpoints", monitor="val_score", mode="max",
                                 save_top_k=1, save_weights_only=True, filename="{epoch}-{val_score:.4f}")
    trainer = pl.Trainer(accelerator="gpu", devices=[int(cfg["training"]["device"].split(":")[-1])],
                         max_epochs=cfg["training"]["max_epochs"], logger=False,
                         callbacks=[TrainingRecord(directory), checkpoint, EarlyStopping(monitor="val_score", mode="max",
                                                              patience=cfg["training"]["patience"])],
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, train_loader, test_loader)
    best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["state_dict"], strict=True)
    torch.save({"state_dict": model.state_dict(), "genes": model.genes, "config": cfg,
                "dataset": name, "fold": fold, "epoch": int(trainer.current_epoch),
                "best_epoch": best["epoch"], "best_checkpoint": checkpoint.best_model_path}, directory / "model.pt")
    evaluate(cfg)


def load_model(cfg, dataset=None, fold=None):
    directory = run_directory(cfg, dataset, fold)
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    model = DragH2ST(checkpoint["config"], checkpoint["genes"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(cfg["training"]["device"])
    return model, checkpoint


def predict_slide(cfg, dataset, slide, model=None):
    from .data import SAMPLES
    from .features import load_features
    fold = SAMPLES[dataset].index(slide)
    ds = SlideDataset(cfg["paths"]["data"], dataset, fold, "test")
    if model is None:
        model, checkpoint = load_model(cfg, dataset, cfg["datasets"]["folds"][dataset])
    if ds.genes != model.genes:
        raise ValueError("Checkpoint gene order differs from the dataset")
    sample = ds[0]
    features = load_features(cfg, dataset, slide, ds.genes, ds.slides[slide]["barcodes"])
    device = model.device
    with torch.inference_mode():
        prediction, _, hidden = model(features.to(device), sample["positions"].to(device), sample["adjacency"].to(device))
        gates = model.cgdr.context_gating(hidden)
    return ds, {"prediction": prediction.cpu().numpy(), "features": features.numpy(),
                "hidden": hidden.cpu().numpy(), "gates": gates.cpu().numpy(),
                "genes": ds.genes, "pixels": sample["pixels"].numpy(),
                "barcodes": ds.slides[slide]["barcodes"]}


def evaluate(cfg):
    model, checkpoint = load_model(cfg)
    name = cfg["training"]["dataset"]
    from .data import SAMPLES
    slide = SAMPLES[name][cfg["training"]["fold"]]
    dataset, result = predict_slide(cfg, name, slide, model)
    metrics = compute_metrics(result["prediction"], dataset.slides[slide]["expression"].numpy())
    directory = run_directory(cfg)
    np.savez_compressed(directory / "prediction.npz", **{k: np.array(v) for k, v in result.items()})
    np.save(directory / "ground_truth.npy", dataset.slides[slide]["expression"].numpy())
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"dataset": name, "fold": cfg["training"]["fold"], **metrics}, indent=2), flush=True)
    return metrics


def build_library(cfg):
    from .prototype_bank.bank import PrototypeLibrary, TISSUES, representatives
    from .features import load_features
    library = PrototypeLibrary(cfg)
    for dataset, fold in cfg["datasets"]["folds"].items():
        trainset = SlideDataset(cfg["paths"]["data"], dataset, fold, "train")
        features = []
        origins = []
        for slide in trainset.names:
            sample = trainset.slides[slide]
            values = load_features(cfg, dataset, slide, trainset.genes, sample["barcodes"]).numpy()
            features.append(values)
            origins.extend((slide, index) for index in range(len(values)))
        # Paper protocol: one K-Means over the ENTIRE training split, with a
        # fixed 256-prototype budget. Held-out measured expression is excluded.
        selected = representatives(np.concatenate(features, axis=0), 256)
        by_slide = {}
        for index in selected:
            slide, spot = origins[index]
            by_slide.setdefault(slide, []).append(spot)
        entries = []
        for slide, indices in by_slide.items():
            sample = trainset.slides[slide]
            result = {"features": load_features(cfg, dataset, slide, trainset.genes, sample["barcodes"]).numpy(),
                      "genes": trainset.genes, "pixels": sample["pixels"].numpy()}
            entries.append((slide, result, sample["expression"].numpy(), indices))
        library.replace_training(dataset, entries)
        print(f"Indexed {len(selected)} measured {dataset} training prototypes", flush=True)
