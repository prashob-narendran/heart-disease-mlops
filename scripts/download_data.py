"""Download the UCI Heart Disease (Cleveland) dataset.

Sources are tried in order until one succeeds:
  1. Direct download of `processed.cleveland.data` from the UCI archive.
  2. The official `ucimlrepo` client (UCI dataset id 45).
  3. A local file passed with --from-file (offline / mirror fallback).

The raw file is stored unchanged in data/raw/ and a manifest.json records
where it came from, when, and its SHA-256 checksum. The checksum acts as a
lightweight dataset version that later gets logged to MLflow with each run.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force
    python scripts/download_data.py --from-file path/to/processed.cleveland.data
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    RAW_COLUMNS,
    RAW_FILE,
    RAW_MANIFEST,
    UCI_DATASET_ID,
    UCI_RAW_URL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("download_data")

EXPECTED_ROWS = 303
EXPECTED_COLS = len(RAW_COLUMNS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw(path: Path) -> int:
    """Light structural check: 303 comma-separated rows of 14 fields."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    bad = [i for i, ln in enumerate(lines, 1)
           if len(ln.split(",")) != EXPECTED_COLS]
    if bad:
        raise ValueError(f"{len(bad)} malformed rows, e.g. line {bad[0]}")
    if len(lines) != EXPECTED_ROWS:
        log.warning("Expected %d rows, found %d", EXPECTED_ROWS, len(lines))
    return len(lines)


def from_uci_url(dest: Path, retries: int = 3) -> str:
    import requests

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(UCI_RAW_URL, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return UCI_RAW_URL
        except requests.RequestException as exc:
            log.warning("UCI download attempt %d/%d failed: %s",
                        attempt, retries, exc)
            time.sleep(2 * attempt)
    raise RuntimeError("UCI direct download failed")


def from_ucimlrepo(dest: Path) -> str:
    """Fetch via the official client and write it back in raw UCI format."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=UCI_DATASET_ID)
    df = ds.data.original if ds.data.original is not None else (
        ds.data.features.join(ds.data.targets)
    )
    df = df[RAW_COLUMNS]
    buf = io.StringIO()
    # Write missing values as "?" to match the original raw file exactly.
    df.to_csv(buf, header=False, index=False, na_rep="?")
    dest.write_text(buf.getvalue())
    return f"ucimlrepo:id={UCI_DATASET_ID}"


def from_local(src: Path, dest: Path) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copyfile(src, dest)
    return f"local:{src.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file exists")
    parser.add_argument("--from-file", type=Path,
                        help="use a local copy instead of downloading")
    args = parser.parse_args()

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists() and not args.force and not args.from_file:
        log.info("Raw data already present: %s (use --force to refresh)",
                 RAW_FILE)
        validate_raw(RAW_FILE)
        return 0

    if args.from_file:
        attempts = [("local file", lambda: from_local(args.from_file,
                                                      RAW_FILE))]
    else:
        attempts = [
            ("UCI archive", lambda: from_uci_url(RAW_FILE)),
            ("ucimlrepo", lambda: from_ucimlrepo(RAW_FILE)),
        ]

    source = None
    for name, fn in attempts:
        try:
            log.info("Trying source: %s", name)
            source = fn()
            break
        except Exception as exc:  # noqa: BLE001 - try the next source
            log.warning("Source '%s' failed: %s", name, exc)

    if source is None:
        log.error("All sources failed. Download processed.cleveland.data "
                  "manually from %s and rerun with --from-file.", UCI_RAW_URL)
        return 1

    n_rows = validate_raw(RAW_FILE)
    manifest = {
        "dataset": "UCI Heart Disease - Cleveland subset",
        "source": source,
        "file": RAW_FILE.name,
        "rows": n_rows,
        "columns": RAW_COLUMNS,
        "sha256": sha256(RAW_FILE),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }
    RAW_MANIFEST.write_text(json.dumps(manifest, indent=2))
    log.info("Saved %d rows to %s", n_rows, RAW_FILE)
    log.info("sha256=%s", manifest["sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
