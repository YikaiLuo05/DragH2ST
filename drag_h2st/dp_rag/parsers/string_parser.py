from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class StringParseStats:
    nodes: int
    edges: int


def parse_string_info(path: str) -> Dict[str, Dict[str, str]]:
    """Parse STRING protein info file.

    Typical columns include:
      protein_external_id, preferred_name, annotation, ...

    Returns:
      protein_id -> dict of columns

    NOTE: This does NOT provide robust mapping to gene symbols without additional
    resources. We keep it for potential future upgrades.
    """

    info: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if header is None:
                header = line.split("\t")
                continue
            parts = line.split("\t")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            pid = parts[0]
            info[pid] = row
    return info


def parse_string_links(
    path: str,
    score_threshold: int = 700,
    max_neighbors: int = 50,
) -> Tuple[Dict[str, List[Tuple[str, int]]], StringParseStats]:
    """Parse STRING links file into adjacency list.

    link file columns:
      protein1 protein2 combined_score

    Returns:
      adj: protein_id -> list of (neighbor_protein_id, score) sorted desc

    Deterministic, offline.
    """

    adj: Dict[str, List[Tuple[str, int]]] = {}
    edges = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            p1, p2, score_s = line.split()
            score = int(score_s)
            if score < score_threshold:
                continue

            adj.setdefault(p1, []).append((p2, score))
            adj.setdefault(p2, []).append((p1, score))
            edges += 1

    # sort + cap
    for p, neigh in adj.items():
        neigh.sort(key=lambda x: (-x[1], x[0]))
        if max_neighbors is not None and len(neigh) > max_neighbors:
            adj[p] = neigh[:max_neighbors]

    return adj, StringParseStats(nodes=len(adj), edges=edges)
