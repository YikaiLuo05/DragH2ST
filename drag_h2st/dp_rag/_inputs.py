"""Internal DP-RAG inputs, populated exclusively from the main config.yaml.

Parser thresholds and algorithm constants are fixed to the reference values.
"""
from dataclasses import asdict, dataclass
from typing import ClassVar


@dataclass(frozen=True)
class PriorInputs:
    db_root: str
    cache_dir: str
    weight_gene2vec: float
    weight_msigdb: float
    weight_string: float
    weight_go: float
    go_obo_path: str
    goa_gaf_path: str
    msigdb_gmt_path: str
    string_gene2protein_map_path: str

    msigdb_max_sets_per_gene: ClassVar[int] = 200
    string_links_path: ClassVar[str] = "string/9606.protein.links.v12.0.txt"
    string_info_path: ClassVar[str] = "string/9606.protein.info.v12.0.txt"
    string_score_threshold: ClassVar[int] = 700
    string_max_neighbors: ClassVar[int] = 50

    def to_dict(self):
        return asdict(self)
