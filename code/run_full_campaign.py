"""Apply a documented seed-count amendment without changing trained models."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

import pandas as pd
from threadpoolctl import threadpool_limits

from _shared import full_data as full
from _shared import controlled as base


def validate_seed_plan(plan: dict, run_id: str) -> list[int]:
    original = list(range(10, 201, 10))
    seeds = plan["seeds"]
    if plan["run_id"] != run_id or plan["previous_seeds"] != original:
        raise ValueError("Seed amendment does not match the original run")
    if seeds != original[:10] or any(type(seed) is not int for seed in seeds):
        raise ValueError("The approved amendment requires the first ten original seeds")
    if plan["expected_dataset_seed_jobs"] != len(base.DATASETS) * len(seeds):
        raise ValueError("Amendment job count does not match the seed matrix")
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261015_full")
    parser.add_argument("--seed-plan", type=Path, required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error("Use a single directory name")
    plan_path = args.seed_plan.resolve()
    plan = json.loads(plan_path.read_text())
    seeds = validate_seed_plan(plan, args.run_id)
    central = base.ROOT / "code" / "results" / args.run_id
    metadata_path = central / "metadata.json"
    with full.run_lock(central / "run.lock"):
        metadata = json.loads(metadata_path.read_text())
        if not metadata["full_dataset"] or metadata["datasets"] != list(base.DATASETS):
            raise ValueError("The original four-dataset full-data run is required")
        if metadata["seeds"] not in (plan["previous_seeds"], seeds):
            raise ValueError("Unexpected existing seed scope")
        options = metadata["options"]
        sources = {"controlled": Path(base.__file__), "full_data": Path(full.__file__), "protocol": base.ROOT / plan["base_protocol"]}
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != options["source_sha256"][name] for name, path in sources.items()):
            raise ValueError("Training code or original protocol changed; cannot reuse checkpoints")
        if any(importlib.metadata.version(name) != version for name, version in metadata["packages"].items()):
            raise ValueError("Experiment package versions changed; cannot reuse checkpoints")
        archive = central / "metadata_before_seed_reduction_20261016.json"
        if not archive.exists():
            shutil.copyfile(metadata_path, archive)
        active_plan = central / "seed_plan.json"
        if plan_path != active_plan.resolve():
            shutil.copyfile(plan_path, active_plan)
        metadata.update(seeds=seeds, expected_seed_jobs=len(seeds) * len(base.DATASETS), seed_plan="seed_plan.json", seed_plan_sha256=hashlib.sha256(active_plan.read_bytes()).hexdigest(), seed_amendment_applied_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), campaign_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), analysis_source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {"full_analysis": base.ROOT / "code/build_full_data_results.py", "shared_analysis": base.ROOT / "code/build_controlled_results.py"}.items()}, status="running", pid=os.getpid())
        metadata.pop("error", None)
        full.write_json(metadata_path, metadata)
        try:
            with threadpool_limits(limits=options["blas_threads"]):
                for seed in seeds:
                    for dataset in base.DATASETS:
                        metadata.update(current_dataset=dataset, current_seed=seed)
                        full.write_json(metadata_path, metadata)
                        full.run_seed(dataset, seed, options)
                        gc.collect()
            paths = [base.ROOT / "code" / dataset / "results" / args.run_id / f"seed_{seed:03d}" / "metrics.csv" for dataset in base.DATASETS for seed in seeds]
            raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
            if len(raw) != len(seeds) * len(base.DATASETS) * 54:
                raise AssertionError("Incomplete amended experiment matrix")
            raw.to_csv(central / "raw_metrics.csv", index=False)
            metadata["status"] = "analyzing"
            full.write_json(metadata_path, metadata)
            from build_full_data_results import build
            build(args.run_id)
            metadata["status"] = "completed"
        except BaseException as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            full.write_json(metadata_path, metadata)


if __name__ == "__main__":
    main()
