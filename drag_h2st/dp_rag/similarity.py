from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import cosine_sim_matrix


def compute_msigdb_similarity(gene_list: List[str], gene_to_sets: Dict[str, List[str]]) -> np.ndarray:
    """Gene-gene similarity based on shared MSigDB gene sets.

    We compute weighted Jaccard on set memberships:
      sim(i,j) = |Si ∩ Sj| / |Si ∪ Sj|

    If a gene has no sets, its row/col will be zeros except diagonal.
    """

    n = len(gene_list)
    sets_per_gene = [set(gene_to_sets.get(g, [])) for g in gene_list]

    sim = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(sim, 1.0)

    for i in range(n):
        si = sets_per_gene[i]
        if not si:
            continue
        for j in range(i + 1, n):
            sj = sets_per_gene[j]
            if not sj:
                continue
            inter = len(si & sj)
            if inter == 0:
                continue
            union = len(si | sj)
            val = float(inter) / float(union) if union else 0.0
            sim[i, j] = val
            sim[j, i] = val

    return sim


def compute_string_similarity(
    gene_list: List[str],
    protein_adj: Dict[str, List[Tuple[str, int]]],
    gene_to_protein: Dict[str, str],
) -> np.ndarray:
    """Gene-gene similarity from STRING interactions.

    Because offline mapping may be incomplete, we accept a gene_to_protein mapping
    (gene symbol -> STRING protein id). If mapping is missing for a gene, it contributes 0.

    Similarity is computed as:
      - direct edge score normalized to [0,1]
      - plus shared neighbor overlap (Jaccard on neighbor sets)

    This is a pragmatic offline approximation.
    """

    n = len(gene_list)
    sim = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(sim, 1.0)

    prot_ids = [gene_to_protein.get(g) for g in gene_list]
    neigh_sets: List[Optional[set[str]]] = []
    edge_score_map: Dict[Tuple[str, str], float] = {}

    for pid in prot_ids:
        if pid is None:
            neigh_sets.append(None)
            continue
        neigh = protein_adj.get(pid, [])
        neigh_sets.append(set([p for p, _ in neigh]))
        for p2, score in neigh:
            key = tuple(sorted((pid, p2)))
            edge_score_map[key] = max(edge_score_map.get(key, 0.0), score / 1000.0)

    for i in range(n):
        pid_i = prot_ids[i]
        if pid_i is None:
            continue
        ni = neigh_sets[i] or set()
        for j in range(i + 1, n):
            pid_j = prot_ids[j]
            if pid_j is None:
                continue

            # direct edge
            direct = edge_score_map.get(tuple(sorted((pid_i, pid_j))), 0.0)

            # shared neighbor jaccard
            nj = neigh_sets[j] or set()
            inter = len(ni & nj)
            union = len(ni | nj)
            shared = float(inter) / float(union) if union else 0.0

            val = 0.7 * direct + 0.3 * shared
            if val > 0:
                sim[i, j] = val
                sim[j, i] = val

    return sim


def fuse_similarity(
    sim_gene2vec: Optional[np.ndarray],
    sim_msigdb: Optional[np.ndarray],
    sim_string: Optional[np.ndarray],
    sim_go: Optional[np.ndarray],
    weight_gene2vec: float,
    weight_msigdb: float,
    weight_string: float,
    weight_go: float,
) -> np.ndarray:
    """Weighted fusion of similarity matrices.

    Each matrix is assumed in [0,1]. Fusion is:
      S = (w1*S1 + w2*S2 + w3*S3 + w4*S4) / (w1+w2+w3+w4)

    If all weights are 0 or all matrices are None, raises ValueError.
    """

    mats = []
    weights = []

    if sim_gene2vec is not None and weight_gene2vec > 0:
        mats.append(sim_gene2vec)
        weights.append(weight_gene2vec)
    if sim_msigdb is not None and weight_msigdb > 0:
        mats.append(sim_msigdb)
        weights.append(weight_msigdb)
    if sim_string is not None and weight_string > 0:
        mats.append(sim_string)
        weights.append(weight_string)
    if sim_go is not None and weight_go > 0:
        mats.append(sim_go)
        weights.append(weight_go)

    if not mats:
        raise ValueError("No similarity sources enabled. Check weights and inputs.")

    wsum = float(sum(weights))
    out = np.zeros_like(mats[0], dtype=np.float32)
    for m, w in zip(mats, weights):
        out += (w / wsum) * m.astype(np.float32)

    out = np.clip(out, 0.0, 1.0)
    np.fill_diagonal(out, 1.0)
    return out
