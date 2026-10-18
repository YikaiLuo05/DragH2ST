"""Bounded local search around config.yaml, using the unchanged training action."""
import copy
from dataclasses import fields, replace
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from .config import ROOT, load_config
from .data import GENE_FILES
from .dp_rag._inputs import PriorInputs
from .dp_rag.prior import build_prior
from .dp_rag.utils import config_hash
from .features import file_hash
from .pipeline import run_directory


def selection_key(record):
    metrics = record["metrics"]
    return (metrics["pcc_hvg50"] + metrics["pcc_heg50"], metrics["pcc_all"], -metrics["mse_all"])


def neighbors(cfg):
    """Ten nearby settings; every model component stays active."""
    return [
        {"training.learning_rate": cfg["training"]["learning_rate"] * factor} for factor in (0.75, 1.25)
    ] + [
        {"prediction.regulatory_weight": min(0.98, max(0.02, cfg["prediction"]["regulatory_weight"] + shift))}
        for shift in (-0.05, 0.05)
    ] + [
        {"training.dropout": min(0.9, max(0.05, cfg["training"]["dropout"] + shift))}
        for shift in (-0.05, 0.05)
    ] + [
        {"loss.pcc": cfg["loss"]["pcc"] * factor} for factor in (0.8, 1.2)
    ] + [
        {"prior.similarity_threshold": min(0.98, max(0.02, cfg["prior"]["similarity_threshold"] + shift))}
        for shift in (-0.03, 0.03)
    ]


def apply_changes(cfg, changes):
    result = copy.deepcopy(cfg)
    for path, value in changes.items():
        section, key = path.split(".")
        result[section][key] = value
    return result


def reuse_prior(cfg, runtime):
    """Copy the same cached matrices under the isolated artifact path's key."""
    name = cfg["training"]["dataset"]
    genes = [str(gene) for gene in np.load(Path(cfg["paths"]["data"]) / GENE_FILES[name], allow_pickle=True)]
    expected_fields = {field.name for field in fields(PriorInputs)}
    matches = []
    for path in (Path(cfg["paths"]["artifacts"]) / "priors").glob(f"*/gene_graph_rag_{name}_*.pkl"):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        values = payload.get("config", {})
        if set(values) != expected_fields or payload.get("gene_list") != genes:
            continue
        if values["db_root"] != str(Path(cfg["paths"]["data"]) / "biological_database"):
            continue
        if any(values[f"weight_{key}"] != value for key, value in cfg["prior"]["weights"].items()):
            continue
        original = PriorInputs(**values)
        if payload["config_hash"] != config_hash(original):
            continue
        matches.append((path, payload, original))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one compatible prepared prior, found {len(matches)}. Run prepare first.")
    path, payload, original = matches[0]
    directory = runtime / "priors" / path.parent.name
    directory.mkdir(parents=True)
    translated = replace(original, cache_dir=str(directory))
    copied = dict(payload, config=translated.to_dict(), config_hash=config_hash(translated))
    target = directory / f"gene_graph_rag_{name}_{len(genes)}_{copied['config_hash']}.pkl"
    with target.open("wb") as handle:
        pickle.dump(copied, handle)
    trial_cfg = copy.deepcopy(cfg)
    trial_cfg["paths"]["artifacts"] = str(runtime)
    actual, embeddings, _ = build_prior(trial_cfg, genes, name)
    if not np.array_equal(actual.numpy(), payload["similarity"]) or not np.array_equal(embeddings.numpy(), payload["embeddings"]):
        raise AssertionError("Isolated search prior changed")
    return {"matrices_exactly_equal": True, "source_sha256": file_hash(path)}


def search(cfg):
    config_path = ROOT / "config.yaml"
    original_text = config_path.read_text()
    raw = yaml.safe_load(original_text)
    artifact_root = Path(cfg["paths"]["artifacts"])
    name, fold = cfg["training"]["dataset"], cfg["training"]["fold"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = artifact_root / "search" / f"{name}_fold{fold}_{stamp}"
    directory.mkdir(parents=True)
    runtime = directory / "runtime"
    runtime.mkdir()
    (runtime / "features").symlink_to(artifact_root / "features", target_is_directory=True)
    manifest = {"dataset": name, "fold": fold, "seed": cfg["training"]["seed"],
                "selection": "HVG50 PCC + HEG50 PCC", "max_epochs": cfg["training"]["max_epochs"],
                "patience": cfg["training"]["patience"], "baseline_config": cfg,
                "baseline_yaml_text": original_text, "trials": [], "status": "running",
                "scope": "The held-out slide is used for parameter selection; these are validation scores."}
    manifest["prior_reuse"] = reuse_prior(cfg, runtime)
    report_path = directory / "results.json"
    last_written = original_text

    def save():
        report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    def write_settings(settings):
        nonlocal last_written
        if config_path.read_text() != last_written:
            raise RuntimeError("config.yaml was edited during search; stopping without overwriting the user's edit")
        last_written = "# All user-editable settings live in this file. Commands accept actions only.\n" + yaml.safe_dump(settings, sort_keys=False, allow_unicode=True)
        temporary = config_path.with_suffix(".yaml.tmp")
        temporary.write_text(last_written)
        temporary.replace(config_path)

    def trial(changes):
        index = len(manifest["trials"])
        trial_dir = directory / f"trial_{index:02d}"
        trial_dir.mkdir()
        settings = apply_changes(raw, changes)
        settings["paths"]["artifacts"] = str(runtime)
        write_settings(settings)
        trial_cfg = load_config()
        record = {"trial": index, "changes": changes, "config": trial_cfg, "status": "running"}
        manifest["trials"].append(record)
        save()
        print(f"Search trial {index}: {changes}", flush=True)
        started = time.monotonic()
        with (trial_dir / "training.log").open("w") as handle:
            completed = subprocess.run([sys.executable, str(ROOT / "run.py"), "train"], cwd=ROOT,
                                       stdout=handle, stderr=subprocess.STDOUT)
        record["elapsed_seconds"] = time.monotonic() - started
        record["exit_code"] = completed.returncode
        output = run_directory(trial_cfg)
        if completed.returncode:
            record["status"] = "failed"
            save()
            raise RuntimeError(f"Search trial failed: {trial_dir / 'training.log'}")
        shutil.move(str(output), trial_dir / "run")
        record["run"] = str(trial_dir / "run")
        record["metrics"] = json.loads((trial_dir / "run/metrics.json").read_text())
        record["score"] = selection_key(record)[0]
        history = [json.loads(line) for line in (trial_dir / "run/epochs.jsonl").read_text().splitlines()]
        record["epochs_completed"] = len(history)
        record["best_epoch"] = max(history, key=lambda row: row["val_score"])["epoch"]
        if not all(np.isfinite(value) for value in record["metrics"].values()):
            raise RuntimeError("Search metrics contain non-finite values")
        record["status"] = "complete"
        save()
        print(f"Trial {index} complete: score={record['score']:.6f}, PCC-All={record['metrics']['pcc_all']:.6f}", flush=True)
        return record

    try:
        baseline_path = run_directory(cfg)
        if (baseline_path / "model.pt").exists() and (baseline_path / "metrics.json").exists():
            checkpoint = torch.load(baseline_path / "model.pt", map_location="cpu", weights_only=False)
            compatible = checkpoint["config"] == cfg
            best_epoch = checkpoint["best_epoch"]
            del checkpoint
        else:
            compatible = False
        if compatible:
            baseline = {"trial": 0, "changes": {}, "status": "complete", "reused": True,
                        "config": cfg, "run": str(baseline_path), "best_epoch": best_epoch,
                        "epochs_completed": len((baseline_path / "epochs.jsonl").read_text().splitlines()),
                        "metrics": json.loads((baseline_path / "metrics.json").read_text()),
                        "model_sha256": file_hash(baseline_path / "model.pt")}
            baseline["score"] = selection_key(baseline)[0]
            manifest["trials"].append(baseline)
            save()
        else:
            baseline = trial({})
        for changes in neighbors(cfg):
            trial(changes)
        winners = sorted((row for row in manifest["trials"][1:] if selection_key(row) > selection_key(baseline)),
                         key=selection_key, reverse=True)
        # Check up to three pairs of individually beneficial, distinct settings.
        combinations = 0
        for first, second in itertools.combinations(winners, 2):
            if set(first["changes"]) & set(second["changes"]):
                continue
            trial(first["changes"] | second["changes"])
            combinations += 1
            if combinations == 3:
                break
        best = max(manifest["trials"], key=selection_key)
        selected = apply_changes(raw, best["changes"])
        write_settings(selected)
        selected_cfg = load_config()
        if not best.get("reused"):
            if baseline_path.exists():
                shutil.copytree(baseline_path, directory / "baseline_run")
                if baseline.get("reused"):
                    baseline["run"] = str(directory / "baseline_run")
            candidate = directory / "selected_run"
            shutil.copytree(best["run"], candidate)
            checkpoint = torch.load(candidate / "model.pt", map_location="cpu", weights_only=False)
            checkpoint["config"] = selected_cfg
            # Preserve the immutable search checkpoint as provenance.
            checkpoint["best_checkpoint"] = str(Path(best["run"]) / "checkpoints" / Path(checkpoint["best_checkpoint"]).name)
            torch.save(checkpoint, candidate / "model.pt")
            del checkpoint
            if baseline_path.exists():
                shutil.move(str(baseline_path), directory / "previous_active_run")
            shutil.move(str(candidate), baseline_path)
        manifest.update(status="complete", best_trial=best["trial"], best_changes=best["changes"],
                        baseline_score=baseline["score"], best_score=best["score"],
                        score_gain=best["score"] - baseline["score"], selected_config=selected_cfg)
        save()
        print(json.dumps({key: manifest[key] for key in ("best_trial", "best_changes", "baseline_score", "best_score", "score_gain")}, indent=2), flush=True)
        print(f"Search evidence: {report_path}", flush=True)
    except BaseException:
        manifest["status"] = "failed"
        save()
        if config_path.read_text() == last_written:
            config_path.write_text(original_text)
        raise
