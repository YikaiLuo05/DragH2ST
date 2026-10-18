from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ._inputs import PriorInputs
from .utils import config_hash, cosine_sim_matrix, ensure_dir
from .parsers.gene2vec_parser import load_gene2vec_embeddings
from .parsers.msigdb_gmt_parser import invert_gene_sets, parse_gmt
from .parsers.string_parser import parse_string_links, parse_string_info
from .parsers.string_aliases_parser import load_symbol_to_protein_map_tsv
from .parsers.go_obo_parser import parse_go_obo_minimal
from .parsers.goa_gaf_parser import parse_gaf_gene2go
from .go_similarity_offline import compute_go_gene_similarity_matrix
from .similarity import compute_msigdb_similarity, compute_string_similarity, fuse_similarity


@dataclass
class RagPriorMeta:
    config: Dict[str, Any]
    config_hash: str
    sources_used: Dict[str, Any]


def create_gene_graph_from_biological_database(
    gene_list: List[str],
    config: PriorInputs,
    dataset_name: str = "unknown",
) -> Tuple[torch.Tensor, torch.Tensor, RagPriorMeta]:
    """Retrieve four biological sources and fuse their similarities.

    Preserve the reference algorithms: Gene2Vec cosine, MSigDB membership
    Jaccard, STRING direct/shared-neighbor scores and GO semantic similarity.
    No additional row scaling or temperature transform is applied here.
    """

    ensure_dir(config.cache_dir)
    h = config_hash(config)
    cache_path = os.path.join(
        config.cache_dir,
        f"gene_graph_rag_{dataset_name}_{len(gene_list)}_{h}.pkl",
    )

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        sim = torch.tensor(data["similarity"], dtype=torch.float32)
        emb = torch.tensor(data["embeddings"], dtype=torch.float32)
        meta = RagPriorMeta(
            config=data.get("config", config.to_dict()),
            config_hash=data.get("config_hash", h),
            sources_used=data.get("sources_used", {}),
        )
        return sim, emb, meta

    sources_used: Dict[str, Any] = {}

    # 1) Gene2Vec embeddings + cosine similarity
    gene2vec_path = os.path.join(config.db_root, "gene2vec_dim_200_iter_9_w2v.txt")
    gene_embeddings_np = np.zeros((len(gene_list), 200), dtype=np.float32)
    sim_gene2vec = None

    if os.path.exists(gene2vec_path):
        emb, missing, stats = load_gene2vec_embeddings(
            gene2vec_path, gene_list, dim=200, normalize=True
        )
        gene_embeddings_np = emb
        if config.weight_gene2vec > 0:
            sim_gene2vec = cosine_sim_matrix(emb)
        sources_used["gene2vec"] = {
            "path": gene2vec_path,
            "missing": len(missing),
            "stats": stats.__dict__,
            "embedding_enabled": True,
            "similarity_enabled": bool(config.weight_gene2vec > 0),
        }
    else:
        sources_used["gene2vec"] = {
            "enabled": False,
            "reason": "file missing or weight=0",
            "path": gene2vec_path,
        }

    # 2) MSigDB (GMT) shared-set similarity
    # Support both absolute and relative paths (relative to db_root).
    # Note: msigdb_gmt_path can be either:
    #   - absolute path
    #   - path relative to repo root
    #   - path relative to db_root
    # We normalize here to reduce user confusion.
    if os.path.isabs(config.msigdb_gmt_path):
        gmt_path = config.msigdb_gmt_path
    else:
        # if user already passed something like "data/biological_database/...",
        # don't prefix db_root again.
        if config.msigdb_gmt_path.startswith("data/") or config.msigdb_gmt_path.startswith("./"):
            gmt_path = config.msigdb_gmt_path.lstrip("./")
        else:
            gmt_path = os.path.join(config.db_root, config.msigdb_gmt_path)

    # Backward-compat: accept legacy config keys that may still point to entrez gmt,
    # but prefer symbols gmt for typical gene symbol lists.
    if (not os.path.exists(gmt_path)) and ("symbols.gmt" in (config.msigdb_gmt_path or "")):
        alt = os.path.join(config.db_root, "c2.cp.v2025.1.Hs.symbols.gmt")
        if os.path.exists(alt):
            gmt_path = alt
    sim_msigdb = None
    if os.path.exists(gmt_path) and config.weight_msigdb > 0:
        set_to_genes, gmt_stats = parse_gmt(gmt_path)
        gene_to_sets = invert_gene_sets(
            set_to_genes,
            limit_genes=set(gene_list),
            max_sets_per_gene=config.msigdb_max_sets_per_gene,
        )
        sim_msigdb = compute_msigdb_similarity(gene_list, gene_to_sets)
        sources_used["msigdb"] = {
            "path": gmt_path,
            "sets": gmt_stats.sets,
            "genes_total": gmt_stats.genes_total,
            "genes_hit": len(gene_to_sets),
        }
    else:
        sources_used["msigdb"] = {
            "enabled": False,
            "reason": "file missing or weight=0",
            "path": gmt_path,
        }

    # 3) STRING similarity (requires mapping from gene->protein id)
    sim_string = None

    # Same normalization for STRING paths.
    info_path = None
    if config.string_info_path:
        if os.path.isabs(config.string_info_path):
            info_path = config.string_info_path
        else:
            info_path = (
                config.string_info_path.lstrip("./")
                if config.string_info_path.startswith("data/") or config.string_info_path.startswith("./")
                else os.path.join(config.db_root, config.string_info_path)
            )

    links_path = None
    if config.string_links_path:
        if os.path.isabs(config.string_links_path):
            links_path = config.string_links_path
        else:
            links_path = (
                config.string_links_path.lstrip("./")
                if config.string_links_path.startswith("data/") or config.string_links_path.startswith("./")
                else os.path.join(config.db_root, config.string_links_path)
            )

    # Optional mapping file (TSV symbol->protein_id)
    map_path = None
    if getattr(config, "string_gene2protein_map_path", None):
        if os.path.isabs(config.string_gene2protein_map_path):
            map_path = config.string_gene2protein_map_path
        else:
            map_path = (
                config.string_gene2protein_map_path.lstrip("./")
                if config.string_gene2protein_map_path.startswith("data/")
                or config.string_gene2protein_map_path.startswith("./")
                else os.path.join(config.db_root, config.string_gene2protein_map_path)
            )

    gene_to_protein: Dict[str, str] = {}
    if map_path and os.path.exists(map_path):
        gene_to_protein = load_symbol_to_protein_map_tsv(map_path, uppercase=True)

    if links_path and os.path.exists(links_path) and config.weight_string > 0:
        protein_adj, stats = parse_string_links(
            links_path,
            score_threshold=config.string_score_threshold,
            max_neighbors=config.string_max_neighbors,
        )
        # Optional: parse info (kept for debugging/inspection)
        if info_path and os.path.exists(info_path):
            _ = parse_string_info(info_path)

        enabled_effective = bool(gene_to_protein)
        if enabled_effective:
            sim_string = compute_string_similarity(gene_list, protein_adj, gene_to_protein)

        sources_used["string"] = {
            "links_path": links_path,
            "info_path": info_path,
            "map_path": map_path,
            "nodes": stats.nodes,
            "edges": stats.edges,
            "gene_mapping_size": len(gene_to_protein),
            "enabled_effective": enabled_effective,
        }
    else:
        sources_used["string"] = {
            "enabled": False,
            "reason": "file missing or weight=0",
            "links_path": links_path,
            "info_path": info_path,
            "map_path": map_path,
        }

    # 4) GO semantic similarity (optional, requires GOA GAF)
    sim_go = None
    if config.weight_go > 0 and config.goa_gaf_path and os.path.exists(config.goa_gaf_path):
        if not os.path.exists(config.go_obo_path):
            sources_used["go"] = {
                "enabled": False,
                "reason": "go_obo_path missing",
                "go_obo_path": config.go_obo_path,
            }
        else:
            go_terms, go_stats = parse_go_obo_minimal(config.go_obo_path)
            gene2go, gaf_stats = parse_gaf_gene2go(
                config.goa_gaf_path,
                allowed_evidence=None,
            )
            sim_go, go_sim_stats = compute_go_gene_similarity_matrix(
                gene_list=gene_list,
                gene2go=gene2go,
                go_terms=go_terms,
            )
            sources_used["go"] = {
                "enabled_effective": True,
                "go_obo_path": config.go_obo_path,
                "goa_gaf_path": config.goa_gaf_path,
                "obo_terms": go_stats.terms,
                "gaf_genes": gaf_stats.genes,
                "gaf_associations": gaf_stats.associations,
                "genes_with_terms_in_list": go_sim_stats.genes_with_terms,
                "avg_terms_per_gene": go_sim_stats.avg_terms_per_gene,
            }
    else:
        reason = ""
        if config.weight_go <= 0:
            reason = "weight_go=0"
        elif not config.goa_gaf_path:
            reason = "goa_gaf_path not provided"
        elif not os.path.exists(config.goa_gaf_path):
            reason = "goa_gaf_path missing"
        sources_used["go"] = {
            "enabled": False,
            "reason": reason or "disabled",
            "weight_go": float(config.weight_go),
            "go_obo_path": config.go_obo_path,
            "goa_gaf_path": config.goa_gaf_path,
        }

    # 5) Fuse similarities
    sim_fused = fuse_similarity(
        sim_gene2vec=sim_gene2vec,
        sim_msigdb=sim_msigdb,
        sim_string=sim_string,
        sim_go=sim_go,
        weight_gene2vec=config.weight_gene2vec,
        weight_msigdb=config.weight_msigdb,
        weight_string=config.weight_string,
        weight_go=config.weight_go,
    )

    sim_tensor = torch.tensor(sim_fused, dtype=torch.float32)
    emb_tensor = torch.tensor(gene_embeddings_np, dtype=torch.float32)

    meta = RagPriorMeta(config=config.to_dict(), config_hash=h, sources_used=sources_used)

    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "similarity": sim_fused,
                "embeddings": gene_embeddings_np,
                "gene_list": gene_list,
                "config": config.to_dict(),
                "config_hash": h,
                "sources_used": sources_used,
            },
            f,
        )

    return sim_tensor, emb_tensor, meta
