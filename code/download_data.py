"""Acquire the four agreed benchmarks from their original distribution sites."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile, is_zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "german_numeric": (
        "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip",
        "german.data-numeric", "german-data-numeric",
    ),
    "south_german": (
        "https://archive.ics.uci.edu/static/public/522/south+german+credit.zip",
        "SouthGermanCredit.asc", "SouthGermanCredit.asc",
    ),
    "taiwan": (
        "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "default of credit card clients.xls", "default of credit card clients.xls",
    ),
    "give_me_some_credit": (
        "https://www.kaggle.com/api/v1/competitions/data/download/GiveMeSomeCredit/cs-training.csv",
        "cs-training.csv", "cs-training.csv",
    ),
    "hmeq": (
        "https://raw.githubusercontent.com/sassoftware/sas-viya-dmml-pipelines/cf9f13a0ce15c8968c879e9e6dcba11deeff200b/data/hmeq.csv",
        "hmeq.csv", "hmeq.csv",
    ),
}
ACTIVE_DATASETS = ("german_numeric", "south_german", "taiwan", "hmeq")


def fetch(url: str, token: str | None = None) -> bytes:
    headers = {"User-Agent": "quantum-kernel-credit-reproducibility/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=120) as response:
        return response.read()


def acquire(dataset: str, data_root: Path, token: str | None) -> dict:
    url, member, filename = SOURCES[dataset]
    content = fetch(url, token if dataset == "give_me_some_credit" else None)
    archive_hash = hashlib.sha256(content).hexdigest()
    if is_zipfile(BytesIO(content)):
        with ZipFile(BytesIO(content)) as archive:
            matches = [name for name in archive.namelist() if Path(name).name == member]
            if len(matches) != 1:
                raise ValueError(f"Unexpected archive member list for {dataset}")
            content = archive.read(matches[0])
            if dataset == "south_german":
                for extra in ("codetable.txt", "read_SouthGermanCredit.R"):
                    names = [name for name in archive.namelist() if Path(name).name == extra]
                    if names:
                        (data_root / extra).write_bytes(archive.read(names[0]))
    elif dataset not in ("give_me_some_credit", "hmeq"):
        raise ValueError(f"The response was not the expected dataset: {dataset}")
    if dataset == "give_me_some_credit" and b"SeriousDlqin2yrs" not in content[:2000]:
        raise ValueError("The response did not contain the original credit training CSV")
    if dataset == "hmeq":
        reader = csv.DictReader(StringIO(content.decode("utf-8-sig")))
        required = {"BAD", "LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"}
        if set(reader.fieldnames or []) != required:
            raise ValueError("Unexpected official HMEQ columns")
        rows = list(reader)
        if len(rows) != 5960 or sum(int(row["BAD"]) for row in rows) != 1189:
            raise ValueError("Unexpected official HMEQ rows or targets")
    target = data_root / filename
    target.write_bytes(content)
    return {
        "dataset": dataset, "source_url": url, "filename": filename,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "download_sha256": archive_hash, "bytes": len(content),
        "distribution": "SAS official GitHub repository" if dataset == "hmeq" else ("Kaggle competition" if dataset == "give_me_some_credit" else "UCI"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--datasets", nargs="+", choices=SOURCES, default=list(ACTIVE_DATASETS))
    parser.add_argument("--kaggle-token-stdin", action="store_true")
    args = parser.parse_args()
    token = sys.stdin.read().strip() if args.kaggle_token_stdin else None
    args.data_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_root / "acquisition_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failed = []
    for dataset in args.datasets:
        try:
            manifest[dataset] = acquire(dataset, args.data_root, token)
            print(f"downloaded {dataset}: {manifest[dataset]['bytes']} bytes", flush=True)
        except HTTPError as error:
            print(f"download failed {dataset}: HTTP {error.code}", file=sys.stderr, flush=True)
            failed.append(dataset)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit(f"Datasets requiring attention: {', '.join(failed)}")


if __name__ == "__main__":
    main()
