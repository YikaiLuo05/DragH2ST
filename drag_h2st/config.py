"""Single configuration entrypoint; no environment or CLI overrides."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    expected = {
        "paths": {"data", "artifacts", "resnet_weights"},
        "training": {"dataset", "fold", "seed", "device", "max_epochs", "patience", "learning_rate", "dropout"},
        "prior": {"similarity_threshold", "top_k", "weights"},
        "prediction": {"regulatory_weight"},
        "loss": {"mse", "reconstruction", "pcc", "rank", "zinb"},
        "datasets": {"folds"},
        "prototype": {"clusters_per_slide", "retrieval_k"},
        "web": {"host", "port"},
    }
    if set(cfg) != set(expected):
        raise ValueError("config.yaml contains missing or unknown sections")
    for section, keys in expected.items():
        if set(cfg[section]) != keys:
            raise ValueError(f"Invalid config.yaml keys in {section}")
    if cfg["training"]["dataset"] not in {"her2st", "cscc", "dlpfc"}:
        raise ValueError("training.dataset must be her2st, cscc or dlpfc")
    if not 0 <= cfg["training"]["fold"] < 12:
        raise ValueError("training.fold must be in 0..11")
    if set(cfg["prior"]["weights"]) != {"gene2vec", "msigdb", "string", "go"}:
        raise ValueError("Exactly four DP-RAG source weights are required")
    if any(v <= 0 for v in cfg["prior"]["weights"].values()):
        raise ValueError("DP-RAG source weights must be positive")
    if cfg["loss"]["zinb"] <= 0:
        raise ValueError("The released architecture requires a positive ZINB weight")
    if cfg["prior"]["top_k"] < 1:
        raise ValueError("prior.top_k must be positive")
    if cfg["training"]["max_epochs"] < 1 or cfg["training"]["patience"] < 1:
        raise ValueError("Training epochs and patience must be positive")
    if any(v < 1 for v in cfg["prototype"].values()):
        raise ValueError("Prototype counts and retrieval_k must be positive")
    if set(cfg["datasets"]["folds"]) != {"her2st", "cscc", "dlpfc"}:
        raise ValueError("datasets.folds must include all three benchmarks")
    if any(not 0 <= f < 12 for f in cfg["datasets"]["folds"].values()):
        raise ValueError("Dataset folds must be in 0..11")
    if not 0 <= cfg["prediction"]["regulatory_weight"] <= 1:
        raise ValueError("prediction.regulatory_weight must be in [0,1]")
    for key, value in cfg["paths"].items():
        path = Path(value).expanduser()
        cfg["paths"][key] = str(path if path.is_absolute() else ROOT / path)
    return cfg
