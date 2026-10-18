"""Frozen, audited ResNet50 ImageNet V2 features."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

WEIGHTS_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_hash(genes, barcodes):
    return hashlib.sha256(json.dumps([genes, barcodes], ensure_ascii=False).encode()).hexdigest()


class ResNetEncoder:
    def __init__(self, cfg):
        path = cfg["paths"]["resnet_weights"]
        if file_hash(path) != WEIGHTS_SHA256:
            raise ValueError("Expected official ResNet50 ImageNet V2 weights (resnet50-11ad3fa6.pth)")
        self.device = torch.device(cfg["training"]["device"])
        self.model = resnet50(weights=None)
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.model.fc = nn.Identity()
        self.model.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode(self, patches):
        # Resize the entire slide in the same order as the reference extractor.
        patches = patches.to(self.device)
        patches = F.interpolate(patches, size=(224, 224), mode="bilinear", align_corners=False)
        if patches.max() > 2.0:
            patches = patches / 255.0
        patches = (patches - self.mean) / self.std
        return torch.cat([self.model(patches[i:i + 64]).cpu() for i in range(0, len(patches), 64)], 0)


def feature_path(cfg, dataset, slide):
    return Path(cfg["paths"]["artifacts"]) / "features" / dataset / f"{slide}.pt"


def load_features(cfg, dataset, slide, genes, barcodes):
    path = feature_path(cfg, dataset, slide)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data["panel_hash"] != panel_hash(genes, barcodes) or data["weights_sha256"] != WEIGHTS_SHA256:
        raise ValueError(f"Feature cache does not match genes/barcode order/weights: {path}")
    if data["features"].shape != (len(barcodes), 2048):
        raise ValueError(f"Invalid feature dimensions: {path}")
    return data["features"]


def prepare_features(cfg, dataset):
    encoder = None
    for slide in dataset.names:
        path = feature_path(cfg, dataset.dataset, slide)
        barcodes = dataset.slides[slide]["barcodes"]
        if path.exists():
            load_features(cfg, dataset.dataset, slide, dataset.genes, barcodes)
            continue
        print(f"Extracting ResNet features: {dataset.dataset}/{slide}", flush=True)
        if encoder is None:
            encoder = ResNetEncoder(cfg)
        features = encoder.encode(dataset.crops(slide))
        if features.shape != (len(barcodes), 2048):
            raise ValueError(f"Invalid features for {slide}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "genes": dataset.genes, "barcodes": barcodes,
                    "panel_hash": panel_hash(dataset.genes, barcodes), "weights_sha256": WEIGHTS_SHA256,
                    "crop_size": 112}, path)
    del encoder
    torch.cuda.empty_cache()


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset):
        self.dataset = dataset
        self.features = {name: load_features(cfg, dataset.dataset, name, dataset.genes,
                                            dataset.slides[name]["barcodes"]) for name in dataset.names}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return {**sample, "features": self.features[sample["slide"]]}
