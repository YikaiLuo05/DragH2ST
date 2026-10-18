from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(config: Any) -> str:
    if is_dataclass(config):
        payload: Dict[str, Any] = asdict(config)
    elif isinstance(config, dict):
        payload = config
    else:
        payload = {"repr": repr(config)}

    s = stable_json_dumps(payload)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def normalize_rows_l2(mat):
    """Row-wise L2 normalize (numpy array)."""
    import numpy as np

    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return mat / denom


def cosine_sim_matrix(embeddings):
    """Compute cosine similarity matrix (numpy)."""
    import numpy as np

    x = normalize_rows_l2(embeddings)
    return np.dot(x, x.T).astype("float32")

