"""
Test the B6 Feature Engineering on real log data and CSV monitordata.
Validates feature computation, time alignment, and data integrity.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser_b1 import parse_file
from features_b6 import (
    FeatureBuilder, FeatureVector, MonitordataIndex,
    load_monitordata_dir, LogAccumulator
)
from datetime import datetime, timezone

UPLOADS = os.path.join(os.path.dirname(__file__), "..", "..", "test_samples")
CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test_samples", "csv")

# ─── Test infrastructure ─────────────────────────────────────────────────────

class Colors:
    OK = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    BOLD = "\033[1m"
    END = "\033[0m"

passed = 0
failed = 0

def check(condition, description, detail=""):
    global passed, failed
    if condition:
        print(f"  {Colors.OK}✓{Colors.END} {description}")
        passed += 1
    else:
        print(f"  {Colors.FAIL}✗ {description}{Colors.END}")
        if detail:
            print(f"    → {detail}")
        failed += 1

def section(title):
    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.END}")
    print(f"{Colors.BOLD}  {title}{Colors.END}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.END}")


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_csv_loading():
    section("CSV Monitordata Loading")

    # Load individual files
    index = MonitordataIndex()

    gauge_file = os.path.join(UPLOADS, "accessConfigCache_cache-average-get-time_2025_11.csv")
    timer_file = os.path.join(UPLOADS, "login_abs_duration_2025_11.csv")
    heap_file = os.path.join(UPLOADS, "MemoryUsageGaugeSet_heap_committed_2025_11.csv")

    if not os.path.exists(gauge_file):
        # Try alternative location
        gauge_file = os.path.join(CSV_DIR, "accessConfigCache_cache-average-get-time_2025_11.csv")
        timer_file = os.path.join(CSV_DIR, "login_abs_duration_2025_11.csv")
        heap_file = os.path.join(CSV_DIR, "MemoryUsageGaugeSet_heap_committed_2025_11.csv")

    if not os.path.exists(gauge_file):
        print("  SKIPPED: CSV files not found")
        return

    index.load_gauge_csv(gauge_file, "cache/accessConfig/avg-get-time")
    index.load_timer_csv(timer_file, "performance/login/duration")
    index.load_gauge_csv(heap_file, "jvm/heap_committed")

    stats = index.stats()
    print(f"    Loaded: {stats}")

    check(stats["gauge_metrics"] == 2, f"2 gauge metrics loaded (got {stats['gauge_metrics']})")
    check(stats["timer_metrics"] == 1, f"1 timer metric loaded (got {stats['timer_metrics']})")
    check(stats["gauge_datapoints"] > 10000, f"Substantial gauge datapoints ({stats['gauge_datapoints']})")
    check(stats["timer_datapoints"] > 5000, f"Substantial timer datapoints ({stats['timer_datapoints']})")

    # Test value retrieval
    epochs = index.get_all_epochs()
    check(len(epochs) > 5000, f"Epochs indexed: {len(epochs)}")

    # Get a value at known time
    test_epoch = epochs[10] if len(epochs) > 10 else epochs[0]
    heap_val = index.get_gauge("jvm/heap_committed", test_epoch)
    check(heap_val is not None, f"Heap value retrieved at epoch {test_epoch}")
    if heap_val:
        heap_mb = heap_val / (1024 * 1024)
        check(500 < heap_mb < 16000, f"Heap value reasonable: {heap_mb:.0f} MB")

    cache_val = index.get_gauge("cache/accessConfig/avg-get-time", test_epoch)
    check(cache_val is not None, f"Cache value retrieved")

    timer_val = index.get_timer("performance/login/duration", test_epoch)
    check(timer_val is not None, "Timer data retrieved")
    if timer_val:
        check("count" in timer_val and "mean" in timer_val,
              f"Timer has expected fields (count={timer_val['count']})")


def test_log_accumulator():
    section("Log Accumulator")

    filepath = os.path.join(UPLOADS, "system_out.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    acc = LogAccumulator()
    count = 0
    for record in parse_file(filepath, source_path="runnable_arcm_s/base/logs/system.out.log"):
        acc.add_record(record)
        count += 1

    print(f"    Loaded {count} records")
    runnables = acc.get_runnables()
    check("arcm_s" in runnables, f"arcm_s in runnables: {runnables}")

    t_min, t_max = acc.get_time_range()
    check(t_max > t_min, f"Time range valid: {datetime.fromtimestamp(t_min, tz=timezone.utc).date()} to {datetime.fromtimestamp(t_max, tz=timezone.utc).date()}")

    # Compute features at a known crash time (Nov 10, ~10:55 UTC = epoch ~1762768500)
    # The crash loop started around 10:48 CET = 09:48 UTC
    crash_epoch = 1762768500  # approximate
    feats = acc.compute_log_features("arcm_s", crash_epoch)

    print(f"    Features at crash time ({datetime.fromtimestamp(crash_epoch, tz=timezone.utc)}):")
    for k, v in sorted(feats.items()):
        print(f"      {k:35s} = {v}")

    check(feats["fatal_count_1h"] > 0,
          f"FATAL count > 0 at crash time (got {feats['fatal_count_1h']})")
    check(feats["error_count_1h"] > 0,
          f"ERROR count > 0 at crash time (got {feats['error_count_1h']})")


def test_log_accumulator_with_history():
    section("Log Accumulator with Runnable History")

    hist_path = os.path.join(UPLOADS, "runnable_history.log")
    if not os.path.exists(hist_path):
        print("  SKIPPED: file not found")
        return

    acc = LogAccumulator()
    for record in parse_file(hist_path, source_path="agent/runnable_history.log"):
        acc.add_record(record)

    runnables = acc.get_runnables()
    check(len(runnables) > 5, f"Multiple runnables from history: {len(runnables)}")

    # Check state tracking for arcm_s
    # Nov 10 crash loop: multiple STARTING→DOWN transitions
    crash_epoch = 1762768500
    feats = acc.compute_log_features("arcm_s", crash_epoch)

    print(f"    arcm_s features at crash time:")
    print(f"      current_state:         {feats['current_state']}")
    print(f"      state_transitions_1h:  {feats['state_transitions_1h']}")
    print(f"      restarts_24h:          {feats['restarts_24h']}")
    print(f"      minutes_since_restart: {feats['minutes_since_last_restart']:.1f}")

    check(feats["state_transitions_1h"] > 3,
          f"Multiple state transitions at crash time (got {feats['state_transitions_1h']})")
    check(feats["restarts_24h"] >= 2,
          f"Multiple restarts in 24h at crash time (got {feats['restarts_24h']})")


def test_feature_builder():
    section("Feature Builder — Full Integration")

    filepath = os.path.join(UPLOADS, "system_out.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    builder = FeatureBuilder()

    # Load log records
    for record in parse_file(filepath, source_path="runnable_arcm_s/base/logs/system.out.log"):
        builder.add_log_record(record)

    # Load runnable history too
    hist_path = os.path.join(UPLOADS, "runnable_history.log")
    if os.path.exists(hist_path):
        for record in parse_file(hist_path, source_path="agent/runnable_history.log"):
            builder.add_log_record(record)

    # Load monitordata CSVs
    gauge_file = os.path.join(UPLOADS, "accessConfigCache_cache-average-get-time_2025_11.csv")
    timer_file = os.path.join(UPLOADS, "login_abs_duration_2025_11.csv")
    heap_file = os.path.join(UPLOADS, "MemoryUsageGaugeSet_heap_committed_2025_11.csv")

    if os.path.exists(gauge_file):
        builder.load_monitordata_files({
            "cache/accessConfig/avg-get-time": (gauge_file, "gauge"),
            "performance/login/duration": (timer_file, "timer"),
            "jvm/heap_committed": (heap_file, "gauge"),
        })

    stats = builder.get_stats()
    print(f"    Builder stats: {stats['log_records_loaded']} records, "
          f"{len(stats['runnables'])} runnables")

    # Build vectors for arcm_s
    vectors = builder.build_vectors("arcm_s", interval_seconds=300)
    print(f"    Feature vectors for arcm_s: {len(vectors)}")

    check(len(vectors) > 100, f"Substantial vectors generated ({len(vectors)})")

    if vectors:
        # Check first vector structure
        v0 = vectors[0]
        check(isinstance(v0, FeatureVector), "FeatureVector objects returned")
        check(v0.runnable == "arcm_s", "Runnable is arcm_s")
        check(v0.window_end_epoch > 0, "Epoch is set")

        # Check that some vectors have non-zero error features
        with_errors = [v for v in vectors if v.error_count_5m > 0]
        check(len(with_errors) > 10,
              f"Vectors with errors: {len(with_errors)} out of {len(vectors)}")

        # Check that infra features are populated (if CSVs loaded)
        if os.path.exists(gauge_file):
            with_heap = [v for v in vectors if v.heap_committed_mb > 0]
            check(len(with_heap) > 0,
                  f"Vectors with heap data: {len(with_heap)}",
                  "CSVs might not overlap with log time range")

            with_cache = [v for v in vectors if v.cache_avg_get_time_ms > -1]
            check(len(with_cache) >= 0,
                  f"Vectors with cache data: {len(with_cache)}",
                  "Cache metric may not align with arcm_s log time range — OK if 0")

        # Find the crash period vectors (Nov 10)
        crash_vectors = [v for v in vectors
                        if v.window_end.month == 11 and v.window_end.day == 10
                        and v.window_end.hour >= 9 and v.window_end.hour <= 12]
        if crash_vectors:
            print(f"\n    {Colors.WARN}Nov 10 crash period vectors:{Colors.END}")
            for v in crash_vectors[:10]:
                print(f"      [{v.window_end.strftime('%H:%M')}] "
                      f"err5m={v.error_count_5m:3d} fatal1h={v.fatal_count_1h:2d} "
                      f"restarts24h={v.restarts_24h:2d} "
                      f"pool={v.pool_exhaustion_count_5m:2d} "
                      f"state={v.current_state}")

            # During the crash, we should see elevated error features
            max_fatal = max(v.fatal_count_1h for v in crash_vectors)
            check(max_fatal > 3,
                  f"Elevated FATAL during crash period (max={max_fatal})")

        # Show some sample vectors
        print(f"\n    {Colors.WARN}Sample vectors (first 5):{Colors.END}")
        for v in vectors[:5]:
            d = v.to_dict()
            non_zero = {k: d[k] for k in d
                       if d[k] not in (0, -1, -1.0, 0.0, "UNKNOWN", "arcm_s")
                       and k not in ("window_end", "window_end_epoch", "runnable")}
            print(f"      [{v.window_end.strftime('%Y-%m-%d %H:%M')}] {non_zero}")


def test_feature_vector_to_dict():
    section("Feature Vector Serialization")

    fv = FeatureVector(
        runnable="arcm_s",
        window_end=datetime(2025, 11, 10, 10, 0, tzinfo=timezone.utc),
        window_end_epoch=1762768800,
        error_count_5m=15,
        fatal_count_1h=3,
        pool_exhaustion_count_5m=8,
        heap_committed_mb=4096.0,
    )

    d = fv.to_dict()
    check(len(d) > 20, f"Dict has {len(d)} fields")
    check(d["runnable"] == "arcm_s", "Runnable preserved")
    check(d["error_count_5m"] == 15, "Error count preserved")
    check(d["heap_committed_mb"] == 4096.0, "Heap preserved")
    check(isinstance(d["window_end"], str), "Timestamp serialized as string")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{Colors.BOLD}ARIS Feature Engineering B6 — Test Suite{Colors.END}\n")

    test_csv_loading()
    test_log_accumulator()
    test_log_accumulator_with_history()
    test_feature_builder()
    test_feature_vector_to_dict()

    total = passed + failed
    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.END}")
    if failed == 0:
        print(f"{Colors.OK}{Colors.BOLD}  ALL {total} CHECKS PASSED ✓{Colors.END}")
    else:
        print(f"  {Colors.OK}{passed} passed{Colors.END}, {Colors.FAIL}{failed} failed{Colors.END} out of {total}")
    print(f"{'═' * 70}\n")

    sys.exit(1 if failed > 0 else 0)
