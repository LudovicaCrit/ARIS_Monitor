#!/usr/bin/env python3
"""
ARIS Log Monitoring — Full Dataset Pipeline Runner
=====================================================
Processes the entire Agent_e_Addestramento_LOG directory:
  1. Discovers all log files across all 16 runnables
  2. Parses them with B1 (parser)
  3. Builds feature vectors for every runnable (B6 features)
  4. Exports a single CSV with all feature vectors

This CSV is the input for training the Isolation Forest on ALL runnables.

Usage:
    python pipeline_runner.py /path/to/Agent_e_Addestramento_LOG output_features.csv

    Example:
    python pipeline_runner.py ~/Azienda/Use_Case_2/Agent_e_Addestramento_LOG features_all.csv

Output:
    A CSV file with one row per 5-minute window per runnable.
    Columns: runnable, window_end, window_end_epoch, + all feature columns.

Expected runtime: 5-15 minutes on 3.3 GB of logs (depends on disk speed).
"""

import sys
import os
import csv
import time
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from parser_b1 import parse_file, LogFamily, detect_log_family, is_noise
from features_b6 import FeatureBuilder, FeatureVector


# ─── File discovery ───────────────────────────────────────────────────────────

# Files to skip entirely (too large, binary, or pure noise)
SKIP_EXTENSIONS = {".jfr", ".csv", ".xml", ".properties", ".conf", ".yml",
                   ".cfg", ".crt", ".pid", ".xsd", ".policy", ".includes",
                   ".template", ".info", ".json", ".txt"}

SKIP_FILENAMES = {"agent.login.log", "agent_login.log"}  # 100% Clous noise

# Map of known file → source_path patterns for runnable detection
def discover_log_files(base_dir: str) -> list:
    """
    Walk the Agent_e_Addestramento_LOG directory and find all parseable log files.

    Returns:
        list of dicts: [{filepath, source_path, runnable_hint, log_family}, ...]
    """
    base = Path(base_dir)
    files = []

    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue

        # Skip by extension
        if path.suffix.lower() in SKIP_EXTENSIONS:
            continue

        # Skip known noise files
        if path.name.lower() in SKIP_FILENAMES:
            continue

        # Skip files without extension that aren't logs
        if not path.suffix and path.name not in ("instanceId",):
            # Check if it looks like a log (has newlines and text)
            continue

        # Only process .log files and rotated logs (.log.1, .1, .2, etc.)
        name = path.name.lower()
        is_log = name.endswith(".log")
        is_rotated = any(name.endswith(f".{i}") for i in range(1, 20))
        is_rotated_log = any(name.endswith(f".log{i}") for i in range(1, 20))

        if not (is_log or is_rotated or is_rotated_log):
            continue

        # Build source_path (relative to base)
        rel_path = str(path.relative_to(base))
        source_path = rel_path.replace("\\", "/")

        # Detect runnable from path
        runnable_hint = ""
        for part in path.parts:
            if part.startswith("runnable_") and part.endswith("_s"):
                runnable_hint = part.replace("runnable_", "")
                break
        if not runnable_hint and "agent" in str(path).lower():
            runnable_hint = "agent"

        # Detect log family
        log_family = detect_log_family(path.name, source_path)

        files.append({
            "filepath": str(path),
            "source_path": source_path,
            "runnable_hint": runnable_hint,
            "log_family": log_family.value,
            "size_mb": path.stat().st_size / (1024 * 1024),
        })

    return files


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(base_dir: str, output_csv: str, interval_seconds: int = 300):
    """
    Run the full pipeline: discover → parse → features → export.

    Args:
        base_dir: Path to Agent_e_Addestramento_LOG
        output_csv: Path for output CSV file
        interval_seconds: Feature vector interval (default 300 = 5 min)
    """
    start_time = time.time()

    # ── Step 1: Discover files ──
    print("="*70)
    print("STEP 1: Discovering log files...")
    print("="*70)

    files = discover_log_files(base_dir)
    total_size = sum(f["size_mb"] for f in files)

    print(f"  Found {len(files)} log files ({total_size:.1f} MB)")

    # Show breakdown by runnable
    by_runnable = {}
    for f in files:
        r = f["runnable_hint"] or "unknown"
        if r not in by_runnable:
            by_runnable[r] = {"count": 0, "size_mb": 0}
        by_runnable[r]["count"] += 1
        by_runnable[r]["size_mb"] += f["size_mb"]

    for r in sorted(by_runnable.keys()):
        info = by_runnable[r]
        print(f"    {r:25s}: {info['count']:4d} files, {info['size_mb']:8.1f} MB")

    # ── Step 2: Parse all files ──
    print(f"\n{'='*70}")
    print("STEP 2: Parsing log files with B1...")
    print("="*70)

    builder = FeatureBuilder()
    total_records = 0
    files_parsed = 0
    files_failed = 0

    for i, f in enumerate(files):
        filepath = f["filepath"]
        filename = Path(filepath).name
        size_mb = f["size_mb"]

        # Progress
        if (i + 1) % 10 == 0 or size_mb > 5:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(files)}] ({elapsed:.0f}s) Parsing {filename} "
                  f"({size_mb:.1f} MB, {f['runnable_hint']})...")

        try:
            count = 0
            for record in parse_file(
                filepath,
                source_path=f["source_path"],
                runnable_hint=f["runnable_hint"],
                apply_noise_filter=True,
            ):
                builder.add_log_record(record)
                count += 1

            total_records += count
            files_parsed += 1

        except Exception as e:
            files_failed += 1
            print(f"  WARNING: Failed to parse {filename}: {e}")

    elapsed = time.time() - start_time
    print(f"\n  Parsing complete in {elapsed:.0f}s")
    print(f"  Files parsed: {files_parsed}, failed: {files_failed}")
    print(f"  Total records: {total_records}")

    # ── Step 2b: Build temporal index for fast lookups ──
    print(f"\n  Building temporal index (bisect)...")
    idx_start = time.time()
    builder.log_accumulator.prepare_index()
    print(f"  Index built in {time.time() - idx_start:.1f}s")

    # ── Step 3: Build feature vectors ──
    print(f"\n{'='*70}")
    print("STEP 3: Building feature vectors for all runnables...")
    print("="*70)

    runnables = builder.log_accumulator.get_runnables()
    print(f"  Runnables with data: {sorted(runnables)}")

    # Show record counts per runnable so we know who's the big one
    for rn in sorted(runnables):
        rn_recs = len(builder.log_accumulator._records.get(rn, []))
        print(f"    {rn:25s}: {rn_recs:>8d} records")

    all_vectors = []
    for runnable in sorted(runnables):
        r_start = time.time()

        # Get time range for this runnable to show progress
        rn_epochs = builder.log_accumulator._epochs.get(runnable, [])
        if rn_epochs:
            rn_start = (rn_epochs[0] // interval_seconds) * interval_seconds
            rn_end = rn_epochs[-1]
            total_windows = (rn_end - rn_start) // interval_seconds
        else:
            total_windows = 0

        print(f"    {runnable:25s}: ~{total_windows:>7d} windows to build...", end="", flush=True)

        vectors = builder.build_vectors(runnable, interval_seconds=interval_seconds)
        build_time = time.time() - r_start

        if vectors:
            # Filter out vectors where everything is zero (saves space)
            # Use a set of indices for O(1) lookup instead of O(n) list search
            meaningful_idx = set()
            for i, v in enumerate(vectors):
                if (v.error_count_5m > 0 or v.error_count_1h > 0 or
                    v.fatal_count_1h > 0 or v.state_transitions_1h > 0 or
                    v.restarts_24h > 0 or v.pool_exhaustion_count_5m > 0 or
                    v.current_state not in ("STARTED", "UNKNOWN") or
                    v.heap_committed_mb > 0):
                    meaningful_idx.add(i)

            # Also keep one "normal" vector per hour for baseline
            last_normal_epoch = 0
            for i, v in enumerate(vectors):
                if i not in meaningful_idx:
                    if v.window_end_epoch - last_normal_epoch >= 3600:
                        meaningful_idx.add(i)
                        last_normal_epoch = v.window_end_epoch

            meaningful = [vectors[i] for i in sorted(meaningful_idx)]
            all_vectors.extend(meaningful)
            r_elapsed = time.time() - r_start
            print(f"\r    {runnable:25s}: {len(vectors):7d} total → {len(meaningful):7d} meaningful ({r_elapsed:.1f}s, build {build_time:.1f}s)")
        else:
            print(f"\r    {runnable:25s}: no vectors")

    # ── Step 4: Export CSV ──
    print(f"\n{'='*70}")
    print(f"STEP 4: Exporting to {output_csv}...")
    print("="*70)

    if not all_vectors:
        print("  ERROR: No vectors to export!")
        return

    # Sort all vectors by time
    all_vectors.sort(key=lambda v: (v.window_end_epoch, v.runnable))

    # Write CSV
    fieldnames = list(all_vectors[0].to_dict().keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in all_vectors:
            writer.writerow(v.to_dict())

    file_size = os.path.getsize(output_csv) / (1024 * 1024)
    elapsed = time.time() - start_time

    print(f"\n  Exported {len(all_vectors)} vectors to {output_csv} ({file_size:.1f} MB)")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"  Total time:       {elapsed:.0f} seconds")
    print(f"  Files processed:  {files_parsed}")
    print(f"  Records parsed:   {total_records}")
    print(f"  Feature vectors:  {len(all_vectors)}")
    print(f"  Output file:      {output_csv} ({file_size:.1f} MB)")
    print(f"  Runnables:        {len(runnables)}")

    # Per-runnable summary
    runnable_counts = {}
    for v in all_vectors:
        runnable_counts[v.runnable] = runnable_counts.get(v.runnable, 0) + 1
    print(f"\n  Vectors per runnable:")
    for r in sorted(runnable_counts.keys()):
        print(f"    {r:25s}: {runnable_counts[r]:7d}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pipeline_runner.py <Agent_e_Addestramento_LOG_dir> <output.csv>")
        print()
        print("Example:")
        print("  python pipeline_runner.py ~/Azienda/Use_Case_2/Agent_e_Addestramento_LOG features_all.csv")
        sys.exit(1)

    base_dir = sys.argv[1]
    output_csv = sys.argv[2]

    if not os.path.isdir(base_dir):
        print(f"ERROR: Directory not found: {base_dir}")
        sys.exit(1)

    run_pipeline(base_dir, output_csv)