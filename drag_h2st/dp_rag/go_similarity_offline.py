from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .parsers.go_obo_parser import GoTerm, build_go_ancestors


@dataclass
class GoSimStats:
    genes_with_terms: int
    avg_terms_per_gene: float


def go_term_similarity(
    term1: str,
    term2: str,
    go_terms: Dict[str, GoTerm],
    ancestors: Dict[str, Set[str]],
) -> float:
    """Offline GO term similarity (M2TGLGO-style heuristic).

    Similar to the method in [`GeneSimilarityCalculator.calculate_go_term_similarity()`](go_prior/gene_similarity.py:174)
    but without goatools dependency.

    Sim = |Anc(term1) ∩ Anc(term2)| / max(depth1, depth2)

    Here we approximate depth with ancestor set size (+1 for itself) because
    minimal OBO parsing does not provide depth.
    """

    t1 = go_terms.get(term1)
    t2 = go_terms.get(term2)
    if t1 is None or t2 is None:
        return 0.0
    if t1.namespace and t2.namespace and t1.namespace != t2.namespace:
        return 0.0

    a1 = ancestors.get(term1, set())
    a2 = ancestors.get(term2, set())
    inter = len(a1 & a2)

    # depth proxy
    d1 = max(1, len(a1) + 1)
    d2 = max(1, len(a2) + 1)
    denom = float(max(d1, d2))
    return float(inter) / denom if denom > 0 else 0.0


def gene_pair_go_similarity(
    go_terms1: Set[str],
    go_terms2: Set[str],
    go_terms: Dict[str, GoTerm],
    ancestors: Dict[str, Set[str]],
) -> float:
    """Gene-gene similarity from their GO term sets.

    We average non-zero term-term similarities (same as the code-path in
    [`GeneSimilarityCalculator.calculate_gene_pair_similarity()`](go_prior/gene_similarity.py:224)).
    """

    if not go_terms1 or not go_terms2:
        return 0.0

    total = 0.0
    cnt = 0
    for t1 in go_terms1:
        for t2 in go_terms2:
            s = go_term_similarity(t1, t2, go_terms=go_terms, ancestors=ancestors)
            if s > 0:
                total += s
                cnt += 1
    return (total / cnt) if cnt else 0.0


def compute_go_gene_similarity_matrix(
    gene_list: List[str],
    gene2go: Dict[str, Set[str]],
    go_terms: Dict[str, GoTerm],
    ancestors: Optional[Dict[str, Set[str]]] = None,
    max_depth: int = 64,
) -> Tuple[np.ndarray, GoSimStats]:
    """Compute GO-based gene similarity matrix (offline).

    Requires gene2go mapping (e.g., from a GOA GAF file).

    Returns:
      sim: [G,G] float32 in [0,1]
      stats
    """

    if ancestors is None:
        ancestors = build_go_ancestors(go_terms, max_depth=max_depth)

    n = len(gene_list)
    sim = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(sim, 1.0)

    term_sets = [gene2go.get(g, set()) for g in gene_list]
    genes_with_terms = sum(1 for s in term_sets if s)
    avg_terms = float(sum(len(s) for s in term_sets)) / float(max(1, len(term_sets)))

    for i in range(n):
        si = term_sets[i]
        if not si:
            continue
        for j in range(i + 1, n):
            sj = term_sets[j]
            if not sj:
                continue
            val = gene_pair_go_similarity(si, sj, go_terms=go_terms, ancestors=ancestors)
            if val > 0:
                sim[i, j] = val
                sim[j, i] = val

    return sim, GoSimStats(genes_with_terms=genes_with_terms, avg_terms_per_gene=avg_terms)
