from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class GafParseStats:
    genes: int
    associations: int


def parse_gaf_gene2go(
    path: str,
    gene_symbol_column: int = 2,
    go_id_column: int = 4,
    allowed_evidence: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Set[str]], GafParseStats]:
    """Parse GOA GAF (Gene Association File) to gene->GO terms.

    This is OFFLINE and requires a GAF file (not included by default).

    GAF 2.1 columns (0-based indices):
      2: DB Object Symbol
      4: GO ID
      6: Evidence Code

    Parameters allow for some flexibility.

    Returns:
      gene_symbol -> set(GO_ID)
    """

    gene2go: Dict[str, Set[str]] = {}
    assoc = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(gene_symbol_column, go_id_column, 6):
                continue

            evidence = parts[6]
            if allowed_evidence is not None and evidence not in allowed_evidence:
                continue

            gene = parts[gene_symbol_column]
            go_id = parts[go_id_column]
            if not gene or not go_id:
                continue

            gene2go.setdefault(gene, set()).add(go_id)
            assoc += 1

    return gene2go, GafParseStats(genes=len(gene2go), associations=assoc)
