from __future__ import annotations

import csv
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


# ===== Config =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_FILE = PROJECT_ROOT / "data" / "mappings" / "team_mapping_master_merged.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "elo"
MANIFEST_FILE = OUTPUT_DIR / "elo_manifest.csv"
BASE_TSV_URL_PATTERN = "https://eloratings.net/{slug}.tsv"

REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 0.5
OVERWRITE = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_unique_slugs(mapping_file: Path) -> list[str]:
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    df = pd.read_csv(mapping_file, sep=None, engine="python")

    if "elo_slug" not in df.columns:
        raise ValueError("Mapping file must contain an 'elo_slug' column.")

    slugs = (
        df["elo_slug"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )

    slugs = sorted(slugs)
    if not slugs:
        raise ValueError("No valid elo_slug values found in mapping file.")

    return slugs


def build_url(slug: str) -> str:
    return BASE_TSV_URL_PATTERN.format(slug=slug)


def download_file(url: str, target_path: Path) -> tuple[bool, int | None, str]:
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        status_code = response.status_code

        if response.status_code != 200:
            return False, status_code, f"HTTP {response.status_code}"

        content = response.content
        if not content.strip():
            return False, status_code, "Empty response body"

        target_path.write_bytes(content)
        return True, status_code, "ok"

    except requests.RequestException as e:
        return False, None, f"Request failed: {e}"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def manifest_rows(slugs: Iterable[str]) -> list[dict[str, str]]:
    timestamp = datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, str]] = []
    for slug in slugs:
        url = build_url(slug)
        file_path = OUTPUT_DIR / f"{slug}.tsv"

        if file_path.exists() and not OVERWRITE:
            rows.append(
                {
                    "elo_slug": slug,
                    "source_url": url,
                    "downloaded_at_utc": timestamp,
                    "status": "skipped_exists",
                    "http_status": "",
                    "file_name": file_path.name,
                    "file_sha256": sha256_file(file_path),
                    "notes": "File already existed and OVERWRITE=False",
                }
            )
            continue

        success, http_status, note = download_file(url, file_path)

        if success:
            file_hash = sha256_file(file_path)
            status = "downloaded"
        else:
            file_hash = ""
            status = "failed"
            if file_path.exists():
                file_path.unlink()

        rows.append(
            {
                "elo_slug": slug,
                "source_url": url,
                "downloaded_at_utc": timestamp,
                "status": status,
                "http_status": "" if http_status is None else str(http_status),
                "file_name": file_path.name,
                "file_sha256": file_hash,
                "notes": note,
            }
        )

        time.sleep(SLEEP_SECONDS)

    return rows


def write_manifest(rows: list[dict[str, str]], manifest_file: Path) -> None:
    fieldnames = [
        "elo_slug",
        "source_url",
        "downloaded_at_utc",
        "status",
        "http_status",
        "file_name",
        "file_sha256",
        "notes",
    ]

    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if "PUT_REAL_URL_PATTERN_HERE" in BASE_TSV_URL_PATTERN:
        print("Error: Set BASE_TSV_URL_PATTERN first.", file=sys.stderr)
        return 1

    ensure_output_dir(OUTPUT_DIR)

    try:
        slugs = load_unique_slugs(MAPPING_FILE)
    except Exception as e:
        print(f"Error loading slugs: {e}", file=sys.stderr)
        return 1

    print(f"Found {len(slugs)} unique elo_slug values.")
    rows = manifest_rows(slugs)
    write_manifest(rows, MANIFEST_FILE)

    downloaded = sum(r["status"] == "downloaded" for r in rows)
    skipped = sum(r["status"] == "skipped_exists" for r in rows)
    failed = sum(r["status"] == "failed" for r in rows)

    print(f"Done. Downloaded: {downloaded}, skipped: {skipped}, failed: {failed}")
    print(f"Manifest written to: {MANIFEST_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())