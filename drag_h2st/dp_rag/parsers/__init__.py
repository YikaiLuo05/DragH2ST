"""Parsers for biological databases (offline).

Each parser should be lightweight and avoid external network calls.
"""

from .string_aliases_parser import load_symbol_to_protein_map_tsv
