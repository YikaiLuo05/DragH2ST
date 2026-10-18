from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class Gene2VecLoadStats:
    loaded: int
    used: int
    missing: int


def load_gene2vec_embeddings(
    path: str,
    gene_list: List[str],
    dim: int = 200,
    normalize: bool = True,
) -> Tuple[np.ndarray, List[str], Gene2VecLoadStats]:
    """Load Gene2Vec embeddings for a gene_list.

    This is compatible with common Gene2Vec text format:
      <gene> <v1> <v2> ... <vd>

    Returns:
      - embeddings: [G, dim] float32
      - missing_genes: list
      - stats

    Offline + deterministic.
    """

    wanted = set(gene_list)
    embed: Dict[str, np.ndarray] = {}
    loaded = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < dim + 1:
                continue
            gene = parts[0]
            if gene not in wanted:
                continue
            vec = np.asarray(parts[1 : dim + 1], dtype=np.float32)
            embed[gene] = vec
            loaded += 1

    embeddings = np.zeros((len(gene_list), dim), dtype=np.float32)
    missing: List[str] = []

    for i, g in enumerate(gene_list):
        v = embed.get(g)
        if v is None:
            missing.append(g)
            continue
        embeddings[i] = v

    if normalize:
        denom = np.linalg.norm(embeddings, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        embeddings = embeddings / denom

    stats = Gene2VecLoadStats(
        loaded=loaded,
        used=len(gene_list) - len(missing),
        missing=len(missing),
    )
    return embeddings, missing, stats
