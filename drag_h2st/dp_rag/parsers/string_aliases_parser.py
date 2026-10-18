from __future__ import annotations

from typing import Dict


def load_symbol_to_protein_map_tsv(path: str, uppercase: bool = True) -> Dict[str, str]:
    """Load mapping TSV written by rag_prior/scripts/build_string_gene2protein_map.py.

    Expected header: symbol\tprotein_id
    """
    mapping: Dict[str, str] = {}
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        # tolerate missing header
        if header and "\t" in header and "symbol" in header.lower() and "protein" in header.lower():
            pass
        else:
            # first line is data
            if header:
                parts = header.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    sym = parts[0].strip()
                    pid = parts[1].strip()
                    if uppercase:
                        sym = sym.upper()
                    if sym and pid:
                        mapping[sym] = pid
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            sym = parts[0].strip()
            pid = parts[1].strip()
            if uppercase:
                sym = sym.upper()
            if not sym or not pid:
                continue
            mapping[sym] = pid
    return mapping
