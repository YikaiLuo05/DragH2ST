"""Public DP-RAG entrypoint with gene-order-aware caches."""
import hashlib
import json
from pathlib import Path
from ._inputs import PriorInputs
from .build_gene_graph import create_gene_graph_from_biological_database


def build_prior(cfg, genes, dataset):
    db = Path(cfg["paths"]["data"]) / "biological_database"
    files = [db / "gene2vec_dim_200_iter_9_w2v.txt", db / "c2.cp.v2025.1.Hs.symbols.gmt",
             db / "string/9606.protein.links.v12.0.txt", db / "string/gene2protein_symbol_to_ensp.tsv",
             db / "go-basic.obo", db / "goa_human.gaf"]
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"DP-RAG source missing: {path}")
    # Include order and file metadata to prevent the reference's length-only
    # gene-panel cache collisions.
    signature = {"genes": list(genes), "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    weights = cfg["prior"]["weights"]
    internal = PriorInputs(
        db_root=str(db), cache_dir=str(Path(cfg["paths"]["artifacts"]) / "priors" / key),
        weight_gene2vec=weights["gene2vec"], weight_msigdb=weights["msigdb"],
        weight_string=weights["string"], weight_go=weights["go"],
        go_obo_path=str(db / "go-basic.obo"), goa_gaf_path=str(db / "goa_human.gaf"),
        msigdb_gmt_path=str(db / "c2.cp.v2025.1.Hs.symbols.gmt"),
        string_gene2protein_map_path=str(db / "string/gene2protein_symbol_to_ensp.tsv"))
    similarity, embeddings, meta = create_gene_graph_from_biological_database(genes, internal, dataset)
    if similarity.shape[0] != len(genes) or embeddings.shape != (len(genes), 200):
        raise ValueError("DP-RAG prior dimensions disagree with the target panel")
    return similarity, embeddings, meta
