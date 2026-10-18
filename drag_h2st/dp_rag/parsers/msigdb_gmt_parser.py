from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class GmtParseStats:
    sets: int
    genes_total: int


def parse_gmt(path: str) -> Tuple[Dict[str, List[str]], GmtParseStats]:
    """Parse GMT file.

    GMT format:
      set_name <tab> description <tab> gene1 <tab> gene2 ...

    Returns:
      mapping: set_name -> [genes]
    """

    mapping: Dict[str, List[str]] = {}
    genes_total = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            set_name = parts[0]
            genes = [g for g in parts[2:] if g]
            mapping[set_name] = genes
            genes_total += len(genes)

    return mapping, GmtParseStats(sets=len(mapping), genes_total=genes_total)


def invert_gene_sets(
    set_to_genes: Dict[str, List[str]],
    limit_genes: Set[str] | None = None,
    max_sets_per_gene: int | None = None,
) -> Dict[str, List[str]]:
    """Invert set->genes to gene->sets.

    If limit_genes provided, only keep those genes.
    If max_sets_per_gene provided, truncate to keep determinism (sorted).
    """

    gene_to_sets: Dict[str, List[str]] = {}
    for s, genes in set_to_genes.items():
        for g in genes:
            if limit_genes is not None and g not in limit_genes:
                continue
            gene_to_sets.setdefault(g, []).append(s)

    # Deterministic order + truncate
    for g, sets in gene_to_sets.items():
        sets.sort()
        if max_sets_per_gene is not None and len(sets) > max_sets_per_gene:
            gene_to_sets[g] = sets[:max_sets_per_gene]

    return gene_to_sets
