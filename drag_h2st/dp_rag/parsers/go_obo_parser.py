from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class GoTerm:
    go_id: str
    name: str
    namespace: str
    parents: List[str]


@dataclass
class GoParseStats:
    terms: int


def parse_go_obo_minimal(path: str) -> Tuple[Dict[str, GoTerm], GoParseStats]:
    """Parse go-basic.obo (minimal, offline).

    We avoid external deps (goatools). Only parse:
      - id
      - name
      - namespace
      - is_a

    Returns:
      go_id -> GoTerm
    """

    terms: Dict[str, GoTerm] = {}

    cur_id: Optional[str] = None
    cur_name: str = ""
    cur_ns: str = ""
    cur_parents: List[str] = []

    def flush():
        nonlocal cur_id, cur_name, cur_ns, cur_parents
        if cur_id is None:
            return
        terms[cur_id] = GoTerm(
            go_id=cur_id,
            name=cur_name,
            namespace=cur_ns,
            parents=cur_parents,
        )
        cur_id = None
        cur_name = ""
        cur_ns = ""
        cur_parents = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                flush()
                continue
            if not line or line.startswith("!"):
                continue
            if line.startswith("[Typedef]"):
                flush()
                break

            if line.startswith("id: "):
                cur_id = line.split("id: ", 1)[1].strip()
            elif line.startswith("name: "):
                cur_name = line.split("name: ", 1)[1].strip()
            elif line.startswith("namespace: "):
                cur_ns = line.split("namespace: ", 1)[1].strip()
            elif line.startswith("is_a: "):
                parent = line.split("is_a: ", 1)[1].split("!", 1)[0].strip()
                cur_parents.append(parent)

    flush()
    return terms, GoParseStats(terms=len(terms))


def build_go_ancestors(go_terms: Dict[str, GoTerm], max_depth: int = 64) -> Dict[str, Set[str]]:
    """Compute ancestors for each GO term (offline).

    This is a simple DFS with memoization. max_depth is a safety guard.
    """

    memo: Dict[str, Set[str]] = {}

    def dfs(t: str, depth: int) -> Set[str]:
        if t in memo:
            return memo[t]
        if depth > max_depth:
            memo[t] = set()
            return memo[t]
        term = go_terms.get(t)
        if term is None:
            memo[t] = set()
            return memo[t]
        out: Set[str] = set(term.parents)
        for p in term.parents:
            out |= dfs(p, depth + 1)
        memo[t] = out
        return out

    for tid in go_terms.keys():
        dfs(tid, 0)

    return memo
