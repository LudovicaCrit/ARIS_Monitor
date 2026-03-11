"""
ARIS Log Monitoring System — Feature Engineering for Predictor B6
===================================================================
Computes feature vectors from two data sources:

  1. LOG FEATURES (from B1 parser output):
     - Error/fatal counts in sliding windows (5min, 1h)
     - Error rate changes (delta between windows)
     - Distinct error types
     - State transitions and restarts
     - Pool exhaustion counts (arcm_s specific)

  2. INFRA FEATURES (from CSV monitordata):
     - JVM heap usage and pressure
     - Cache performance (hit times)
     - Login/request performance (latency, throughput)
     - HTTP connection pool usage

Feature vectors are computed every 5 minutes per runnable,
aligned to the CSV sampling interval.
"""

import csv
import re
import os
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Iterator
from pathlib import Path
from collections import defaultdict

from parser_b1 import LogRecord, LogLevel, FormatType


# ─── Feature vector ──────────────────────────────────────────────────────────

@dataclass
class FeatureVector:
    """A single feature vector for one runnable at one point in time."""
    runnable: str
    window_end: datetime
    window_end_epoch: int = 0

    # ── Log-derived features ──
    error_count_5m: int = 0
    error_count_1h: int = 0
    fatal_count_1h: int = 0
    warn_count_1h: int = 0
    distinct_errors_5m: int = 0
    error_rate_delta: float = 0.0     # (errors_last_5m - errors_prev_5m) / max(errors_prev_5m, 1)
    has_stack_trace_5m: int = 0       # count of records with stack traces in 5min

    # State features (from runnable_history)
    current_state: str = "UNKNOWN"
    state_transitions_1h: int = 0
    restarts_24h: int = 0
    minutes_since_last_restart: float = -1.0  # -1 = no restart seen

    # Specific pattern features
    pool_exhaustion_count_5m: int = 0   # "No free database connection" (arcm_s)
    bus_error_count_5m: int = 0         # BUS-XXXXX errors
    zkc_error_count_5m: int = 0         # ZKC registration errors

    # ── Infrastructure features (from CSV monitordata) ──
    heap_committed_mb: float = -1.0
    heap_used_pct: float = -1.0         # requires both committed and used
    cache_avg_get_time_ms: float = -1.0
    login_count_5m: int = -1
    login_mean_duration_ms: float = -1.0
    login_p95_duration_ms: float = -1.0
    login_rate_per_sec: float = -1.0
    http_pool_available: int = -1
    http_pool_leased: int = -1
    http_pool_pending: int = -1

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "runnable": self.runnable,
            "window_end": self.window_end.isoformat(),
            "window_end_epoch": self.window_end_epoch,
            "error_count_5m": self.error_count_5m,
            "error_count_1h": self.error_count_1h,
            "fatal_count_1h": self.fatal_count_1h,
            "warn_count_1h": self.warn_count_1h,
            "distinct_errors_5m": self.distinct_errors_5m,
            "error_rate_delta": self.error_rate_delta,
            "has_stack_trace_5m": self.has_stack_trace_5m,
            "current_state": self.current_state,
            "state_transitions_1h": self.state_transitions_1h,
            "restarts_24h": self.restarts_24h,
            "minutes_since_last_restart": self.minutes_since_last_restart,
            "pool_exhaustion_count_5m": self.pool_exhaustion_count_5m,
            "bus_error_count_5m": self.bus_error_count_5m,
            "zkc_error_count_5m": self.zkc_error_count_5m,
            "heap_committed_mb": self.heap_committed_mb,
            "heap_used_pct": self.heap_used_pct,
            "cache_avg_get_time_ms": self.cache_avg_get_time_ms,
            "login_count_5m": self.login_count_5m,
            "login_mean_duration_ms": self.login_mean_duration_ms,
            "login_p95_duration_ms": self.login_p95_duration_ms,
            "login_rate_per_sec": self.login_rate_per_sec,
            "http_pool_available": self.http_pool_available,
            "http_pool_leased": self.http_pool_leased,
            "http_pool_pending": self.http_pool_pending,
        }


# ─── CSV monitordata loader ──────────────────────────────────────────────────

@dataclass
class MonitordataIndex:
    """
    Index of CSV monitordata files, organized by metric type.
    Maps epoch timestamps to metric values for fast lookup.
    """
    # metric_name → {epoch_t: value_or_dict}
    _gauges: dict = field(default_factory=lambda: defaultdict(dict))
    _timers: dict = field(default_factory=lambda: defaultdict(dict))

    def load_gauge_csv(self, filepath: str, metric_name: str):
        """Load a gauge CSV (t, value) into the index."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = int(row["t"])
                    v = float(row["value"])
                    self._gauges[metric_name][t] = v
                except (ValueError, KeyError):
                    continue

    def load_timer_csv(self, filepath: str, metric_name: str):
        """Load a timer/histogram CSV into the index."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = int(row["t"])
                    self._timers[metric_name][t] = {
                        "count": int(row.get("count", 0)),
                        "max": float(row.get("max", 0)),
                        "mean": float(row.get("mean", 0)),
                        "min": float(row.get("min", 0)),
                        "stddev": float(row.get("stddev", 0)),
                        "p50": float(row.get("p50", 0)),
                        "p75": float(row.get("p75", 0)),
                        "p95": float(row.get("p95", 0)),
                        "p99": float(row.get("p99", 0)),
                        "mean_rate": float(row.get("mean_rate", 0)),
                        "m1_rate": float(row.get("m1_rate", 0)),
                        "m5_rate": float(row.get("m5_rate", 0)),
                    }
                except (ValueError, KeyError):
                    continue

    def get_gauge(self, metric_name: str, epoch_t: int, tolerance: int = 30) -> Optional[float]:
        """Get a gauge value at a given time, with tolerance for alignment."""
        data = self._gauges.get(metric_name, {})
        if epoch_t in data:
            return data[epoch_t]
        # Search within tolerance
        for dt in range(-tolerance, tolerance + 1):
            if (epoch_t + dt) in data:
                return data[epoch_t + dt]
        return None

    def get_timer(self, metric_name: str, epoch_t: int, tolerance: int = 30) -> Optional[dict]:
        """Get timer data at a given time."""
        data = self._timers.get(metric_name, {})
        if epoch_t in data:
            return data[epoch_t]
        for dt in range(-tolerance, tolerance + 1):
            if (epoch_t + dt) in data:
                return data[epoch_t + dt]
        return None

    def get_all_epochs(self) -> list:
        """Get all unique epoch timestamps across all metrics, sorted."""
        epochs = set()
        for data in self._gauges.values():
            epochs.update(data.keys())
        for data in self._timers.values():
            epochs.update(data.keys())
        return sorted(epochs)

    def stats(self) -> dict:
        """Return loading statistics."""
        return {
            "gauge_metrics": len(self._gauges),
            "timer_metrics": len(self._timers),
            "gauge_datapoints": sum(len(v) for v in self._gauges.values()),
            "timer_datapoints": sum(len(v) for v in self._timers.values()),
        }


def load_monitordata_dir(base_dir: str) -> MonitordataIndex:
    """
    Auto-discover and load all CSV files from a monitordata directory.

    Expected structure:
      base_dir/
        caches/<cacheName>/<metricName>_YYYY_MM.csv     → gauge
        performance/<endpoint>/<metricName>_YYYY_MM.csv  → timer
        jvm/<metricSet>/<metricName>_YYYY_MM.csv         → gauge
        httpclient/<metricName>_YYYY_MM.csv              → gauge or timer
    """
    index = MonitordataIndex()
    base = Path(base_dir)

    if not base.exists():
        return index

    for csv_path in base.rglob("*.csv"):
        rel = csv_path.relative_to(base)
        parts = rel.parts
        filename = csv_path.stem  # without .csv

        # Determine metric type from directory structure
        if len(parts) >= 2:
            category = parts[0]  # caches, performance, jvm, httpclient, etc.
        else:
            category = "unknown"

        # Build metric name from path
        metric_name = "/".join(str(p) for p in parts[:-1]) + "/" + filename

        # Detect CSV type by peeking at header
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                header = f.readline().strip()
        except:
            continue

        if header == "t,value":
            index.load_gauge_csv(str(csv_path), metric_name)
        elif "count" in header and "mean" in header:
            index.load_timer_csv(str(csv_path), metric_name)
        # else: skip unknown formats

    return index


# ─── Log record accumulator ──────────────────────────────────────────────────

class LogAccumulator:
    """
    Accumulates B1 log records and computes log-derived features
    for any given time window.

    After all records are added, call prepare_index() to sort by epoch
    and build a bisect-compatible index for O(log n) window lookups.
    Without the index, compute_log_features falls back to O(n) linear scan.
    """

    def __init__(self):
        # runnable → list of (timestamp_epoch, LogRecord)
        self._records: dict = defaultdict(list)
        # runnable → list of (timestamp_epoch, new_state)
        self._state_history: dict = defaultdict(list)
        # runnable → list of restart timestamps (epoch)
        self._restarts: dict = defaultdict(list)
        # ── bisect index (built by prepare_index) ──
        self._sorted: bool = False
        # runnable → sorted list of epoch ints (parallel to _records)
        self._epochs: dict = {}
        # runnable → sorted list of epoch ints for state_history
        self._state_epochs: dict = {}
        # runnable → sorted list of restart epochs
        self._restart_epochs: dict = {}

    def add_record(self, record: LogRecord):
        """Add a parsed log record."""
        epoch = int(record.timestamp.timestamp())
        self._records[record.runnable].append((epoch, record))
        self._sorted = False  # mark index as stale

        # Track state changes
        if record.format_type == FormatType.RUNNABLE_HIST and "State change:" in record.message:
            m = re.search(r'State change:\s*\w+\s*->\s*(\w+)', record.message)
            if m:
                new_state = m.group(1).rstrip(".")
                self._state_history[record.runnable].append((epoch, new_state))
                if new_state == "STARTING":
                    self._restarts[record.runnable].append(epoch)

        # Track restarts from PLAIN_TS (Getting RuntimeMXBean)
        if "Getting RuntimeMXBean" in record.message:
            self._restarts[record.runnable].append(epoch)

    def add_records(self, records):
        """Add multiple records."""
        for r in records:
            self.add_record(r)

    def prepare_index(self):
        """
        Sort all records by epoch and build parallel epoch arrays for bisect.
        Call this ONCE after all records have been added, before build_vectors.

        This turns compute_log_features from O(n) per call to O(log n + k),
        where k is the number of records in the window — typically very small.

        For the full ARIS dataset (~945k records, ~153k windows per runnable),
        this reduces build_vectors from hours to seconds.
        """
        for runnable, records in self._records.items():
            records.sort(key=lambda x: x[0])
            self._epochs[runnable] = [t for t, _ in records]

        for runnable, history in self._state_history.items():
            history.sort(key=lambda x: x[0])
            self._state_epochs[runnable] = [t for t, _ in history]

        for runnable, restarts in self._restarts.items():
            restarts.sort()
            self._restart_epochs[runnable] = restarts  # already sorted in place

        self._sorted = True

    def _get_records_in_range(self, runnable: str, start_epoch: int, end_epoch: int) -> list:
        """
        Return LogRecord objects in [start_epoch, end_epoch] using bisect.
        Falls back to linear scan if index is not built.
        """
        records = self._records.get(runnable, [])
        if not records:
            return []

        if self._sorted and runnable in self._epochs:
            epochs = self._epochs[runnable]
            lo = bisect_left(epochs, start_epoch)
            hi = bisect_right(epochs, end_epoch)
            return [records[i][1] for i in range(lo, hi)]
        else:
            # Fallback: linear scan (used if prepare_index not called)
            return [r for t, r in records if start_epoch <= t <= end_epoch]

    def _count_states_in_range(self, runnable: str, start_epoch: int, end_epoch: int) -> int:
        """Count state transitions in [start_epoch, end_epoch] using bisect."""
        history = self._state_history.get(runnable, [])
        if not history:
            return 0

        if self._sorted and runnable in self._state_epochs:
            epochs = self._state_epochs[runnable]
            lo = bisect_left(epochs, start_epoch)
            hi = bisect_right(epochs, end_epoch)
            return hi - lo
        else:
            return sum(1 for t, _ in history if start_epoch <= t <= end_epoch)

    def _get_current_state(self, runnable: str, at_epoch: int) -> str:
        """Get most recent state at or before at_epoch using bisect."""
        history = self._state_history.get(runnable, [])
        if not history:
            return "UNKNOWN"

        if self._sorted and runnable in self._state_epochs:
            epochs = self._state_epochs[runnable]
            idx = bisect_right(epochs, at_epoch) - 1
            if idx >= 0:
                return history[idx][1]
            return "UNKNOWN"
        else:
            for t, s in reversed(history):
                if t <= at_epoch:
                    return s
            return "UNKNOWN"

    def _count_restarts_in_range(self, runnable: str, start_epoch: int, end_epoch: int) -> int:
        """Count restarts in [start_epoch, end_epoch] using bisect."""
        restarts = self._restarts.get(runnable, [])
        if not restarts:
            return 0

        if self._sorted and runnable in self._restart_epochs:
            epochs = self._restart_epochs[runnable]
            lo = bisect_left(epochs, start_epoch)
            hi = bisect_right(epochs, end_epoch)
            return hi - lo
        else:
            return sum(1 for t in restarts if start_epoch <= t <= end_epoch)

    def _get_last_restart_before(self, runnable: str, at_epoch: int) -> Optional[int]:
        """Get most recent restart epoch at or before at_epoch."""
        restarts = self._restarts.get(runnable, [])
        if not restarts:
            return None

        if self._sorted and runnable in self._restart_epochs:
            epochs = self._restart_epochs[runnable]
            idx = bisect_right(epochs, at_epoch) - 1
            if idx >= 0:
                return epochs[idx]
            return None
        else:
            recent = [t for t in restarts if t <= at_epoch]
            return max(recent) if recent else None

    def compute_log_features(self, runnable: str, window_end_epoch: int) -> dict:
        """
        Compute log-derived features for a runnable at a given time.

        If prepare_index() has been called, uses bisect for O(log n + k) lookups.
        Otherwise falls back to O(n) linear scan (backward compatible).

        Returns a dict with feature values.
        """
        w5m = window_end_epoch - 300      # 5 minutes ago
        w1h = window_end_epoch - 3600     # 1 hour ago
        w24h = window_end_epoch - 86400   # 24 hours ago
        prev_5m_start = w5m - 300         # previous 5-min window

        # ── Get records in each window using bisect ──
        in_5m = self._get_records_in_range(runnable, w5m, window_end_epoch)
        in_1h = self._get_records_in_range(runnable, w1h, window_end_epoch)
        in_prev_5m = self._get_records_in_range(runnable, prev_5m_start, w5m - 1)

        # Error counts
        errors_5m = sum(1 for r in in_5m if r.level in (LogLevel.ERROR, LogLevel.FATAL))
        errors_1h = sum(1 for r in in_1h if r.level in (LogLevel.ERROR, LogLevel.FATAL))
        fatals_1h = sum(1 for r in in_1h if r.level == LogLevel.FATAL)
        warns_1h = sum(1 for r in in_1h if r.level == LogLevel.WARN)

        # Distinct error messages in 5m
        error_hashes = set(r.message_hash for r in in_5m if r.level in (LogLevel.ERROR, LogLevel.FATAL))
        distinct_5m = len(error_hashes)

        # Error rate delta
        errors_prev_5m = sum(1 for r in in_prev_5m if r.level in (LogLevel.ERROR, LogLevel.FATAL))
        denominator = max(errors_prev_5m, 1)
        error_rate_delta = (errors_5m - errors_prev_5m) / denominator

        # Stack traces
        stack_5m = sum(1 for r in in_5m if r.has_stack_trace)

        # Specific patterns
        pool_5m = sum(1 for r in in_5m if "No free database connection" in r.message)
        bus_5m = sum(1 for r in in_5m if "BUS-" in r.message)
        zkc_5m = sum(1 for r in in_5m if "No registered Instance" in r.message)

        # State features (using bisect helpers)
        transitions_1h = self._count_states_in_range(runnable, w1h, window_end_epoch)
        current_state = self._get_current_state(runnable, window_end_epoch)

        # Restarts in 24h
        restarts_24h = self._count_restarts_in_range(runnable, w24h, window_end_epoch)

        # Minutes since last restart
        last_restart = self._get_last_restart_before(runnable, window_end_epoch)
        if last_restart is not None:
            minutes_since = (window_end_epoch - last_restart) / 60
        else:
            minutes_since = -1.0

        return {
            "error_count_5m": errors_5m,
            "error_count_1h": errors_1h,
            "fatal_count_1h": fatals_1h,
            "warn_count_1h": warns_1h,
            "distinct_errors_5m": distinct_5m,
            "error_rate_delta": error_rate_delta,
            "has_stack_trace_5m": stack_5m,
            "current_state": current_state,
            "state_transitions_1h": transitions_1h,
            "restarts_24h": restarts_24h,
            "minutes_since_last_restart": minutes_since,
            "pool_exhaustion_count_5m": pool_5m,
            "bus_error_count_5m": bus_5m,
            "zkc_error_count_5m": zkc_5m,
        }

    def get_runnables(self) -> list:
        """Return all runnables that have records."""
        return list(self._records.keys())

    def get_runnable_time_range(self, runnable: str) -> tuple:
        """Return (min_epoch, max_epoch) for a specific runnable."""
        if self._sorted and runnable in self._epochs:
            epochs = self._epochs[runnable]
            if epochs:
                return (epochs[0], epochs[-1])
            return (0, 0)
        else:
            records = self._records.get(runnable, [])
            if not records:
                return (0, 0)
            times = [t for t, _ in records]
            return (min(times), max(times))

    def get_time_range(self) -> tuple:
        """Return (min_epoch, max_epoch) across all records."""
        if self._sorted:
            # Fast path: just check first/last of each sorted epoch array
            min_t = None
            max_t = None
            for epochs in self._epochs.values():
                if epochs:
                    if min_t is None or epochs[0] < min_t:
                        min_t = epochs[0]
                    if max_t is None or epochs[-1] > max_t:
                        max_t = epochs[-1]
            return (min_t or 0, max_t or 0)
        else:
            all_times = []
            for records in self._records.values():
                for t, _ in records:
                    all_times.append(t)
            if not all_times:
                return (0, 0)
            return (min(all_times), max(all_times))


# ─── Feature vector builder ──────────────────────────────────────────────────

class FeatureBuilder:
    """
    Builds feature vectors by combining log features and infra metrics.

    Usage:
        builder = FeatureBuilder()

        # Load log records
        for record in parse_file(...):
            builder.add_log_record(record)

        # Load monitordata (optional)
        builder.load_monitordata("/path/to/monitordata/abs0000000000")

        # Generate feature vectors
        vectors = builder.build_vectors(runnable="arcm_s", interval_seconds=300)
    """

    def __init__(self):
        self.log_accumulator = LogAccumulator()
        self.monitordata: Optional[MonitordataIndex] = None
        self._records_loaded = 0

    def add_log_record(self, record: LogRecord):
        """Add a B1 log record."""
        self.log_accumulator.add_record(record)
        self._records_loaded += 1

    def add_log_records(self, records):
        """Add multiple log records."""
        for r in records:
            self.add_log_record(r)

    def load_monitordata(self, monitordata_dir: str):
        """Load CSV monitordata from a directory."""
        self.monitordata = load_monitordata_dir(monitordata_dir)

    def load_monitordata_files(self, files: dict):
        """
        Load specific monitordata files.

        Args:
            files: dict mapping metric_name → (filepath, type)
                   type is "gauge" or "timer"
        """
        if self.monitordata is None:
            self.monitordata = MonitordataIndex()

        for metric_name, (filepath, metric_type) in files.items():
            if metric_type == "gauge":
                self.monitordata.load_gauge_csv(filepath, metric_name)
            elif metric_type == "timer":
                self.monitordata.load_timer_csv(filepath, metric_name)

    def build_vectors(
        self,
        runnable: str,
        interval_seconds: int = 300,
        start_epoch: Optional[int] = None,
        end_epoch: Optional[int] = None,
        progress_every: int = 10000,
    ) -> list:
        """
        Build feature vectors for a runnable at regular intervals.

        Args:
            runnable: The runnable to compute features for.
            interval_seconds: Time between vectors (default 300 = 5 minutes).
            start_epoch: Start time (default: first record FOR THIS RUNNABLE).
            end_epoch: End time (default: last record FOR THIS RUNNABLE).
            progress_every: Print progress every N vectors (0 to disable).

        Returns:
            List of FeatureVector objects.
        """
        # Determine time range — PER RUNNABLE, not global
        rn_start, rn_end = self.log_accumulator.get_runnable_time_range(runnable)
        if rn_start == 0 and rn_end == 0:
            return []

        if start_epoch is None:
            start_epoch = rn_start
        if end_epoch is None:
            end_epoch = rn_end

        if start_epoch >= end_epoch:
            return []

        # Align to interval boundaries
        start_epoch = (start_epoch // interval_seconds) * interval_seconds
        total_windows = (end_epoch - start_epoch) // interval_seconds
        vectors = []

        t = start_epoch + interval_seconds  # first window_end
        count = 0
        while t <= end_epoch:
            fv = FeatureVector(
                runnable=runnable,
                window_end=datetime.fromtimestamp(t, tz=timezone.utc),
                window_end_epoch=t,
            )

            # ── Log features ──
            log_feats = self.log_accumulator.compute_log_features(runnable, t)
            fv.error_count_5m = log_feats["error_count_5m"]
            fv.error_count_1h = log_feats["error_count_1h"]
            fv.fatal_count_1h = log_feats["fatal_count_1h"]
            fv.warn_count_1h = log_feats["warn_count_1h"]
            fv.distinct_errors_5m = log_feats["distinct_errors_5m"]
            fv.error_rate_delta = log_feats["error_rate_delta"]
            fv.has_stack_trace_5m = log_feats["has_stack_trace_5m"]
            fv.current_state = log_feats["current_state"]
            fv.state_transitions_1h = log_feats["state_transitions_1h"]
            fv.restarts_24h = log_feats["restarts_24h"]
            fv.minutes_since_last_restart = log_feats["minutes_since_last_restart"]
            fv.pool_exhaustion_count_5m = log_feats["pool_exhaustion_count_5m"]
            fv.bus_error_count_5m = log_feats["bus_error_count_5m"]
            fv.zkc_error_count_5m = log_feats["zkc_error_count_5m"]

            # ── Infra features (from monitordata) ──
            if self.monitordata:
                self._fill_infra_features(fv, t)

            vectors.append(fv)
            t += interval_seconds
            count += 1

            # Progress indicator
            if progress_every > 0 and count % progress_every == 0:
                pct = count * 100 // max(total_windows, 1)
                print(f"\r      ... {count:>7d}/{total_windows} windows ({pct}%)", end="", flush=True)

        if progress_every > 0 and count > progress_every:
            print(f"\r      ... {count:>7d}/{total_windows} windows (100%)   ")

        return vectors

    def _fill_infra_features(self, fv: FeatureVector, epoch_t: int):
        """Fill infrastructure features from monitordata."""
        md = self.monitordata

        # Heap committed (search for any heap_committed metric)
        for metric_name in md._gauges:
            if "heap_committed" in metric_name:
                val = md.get_gauge(metric_name, epoch_t)
                if val is not None:
                    fv.heap_committed_mb = val / (1024 * 1024)
                break

        # Heap used percentage (if both committed and used available)
        for metric_name in md._gauges:
            if "heap_used" in metric_name and "pct" not in metric_name:
                used = md.get_gauge(metric_name, epoch_t)
                if used is not None and fv.heap_committed_mb > 0:
                    fv.heap_used_pct = (used / (1024 * 1024)) / fv.heap_committed_mb * 100
                break

        # Cache average get time
        for metric_name in md._gauges:
            if "cache-average-get-time" in metric_name:
                val = md.get_gauge(metric_name, epoch_t)
                if val is not None:
                    fv.cache_avg_get_time_ms = val * 1000  # convert to ms
                break

        # Login performance
        for metric_name in md._timers:
            if "login" in metric_name and "duration" in metric_name:
                timer = md.get_timer(metric_name, epoch_t)
                if timer:
                    fv.login_count_5m = timer["count"]
                    fv.login_mean_duration_ms = timer["mean"]
                    fv.login_p95_duration_ms = timer["p95"]
                    fv.login_rate_per_sec = timer["m5_rate"]
                break

        # HTTP connection pool
        for metric_name in md._gauges:
            if "available-connections" in metric_name:
                val = md.get_gauge(metric_name, epoch_t)
                if val is not None:
                    fv.http_pool_available = int(val)
                break

        for metric_name in md._gauges:
            if "leased-connections" in metric_name:
                val = md.get_gauge(metric_name, epoch_t)
                if val is not None:
                    fv.http_pool_leased = int(val)
                break

        for metric_name in md._gauges:
            if "pending-connections" in metric_name:
                val = md.get_gauge(metric_name, epoch_t)
                if val is not None:
                    fv.http_pool_pending = int(val)
                break

    def build_all_runnables(self, interval_seconds: int = 300) -> dict:
        """
        Build feature vectors for all runnables.
        Returns: dict mapping runnable → list[FeatureVector]
        """
        result = {}
        for runnable in self.log_accumulator.get_runnables():
            vectors = self.build_vectors(runnable, interval_seconds)
            if vectors:
                result[runnable] = vectors
        return result

    def get_stats(self) -> dict:
        """Return builder statistics."""
        md_stats = self.monitordata.stats() if self.monitordata else {}
        return {
            "log_records_loaded": self._records_loaded,
            "runnables": self.log_accumulator.get_runnables(),
            "time_range": self.log_accumulator.get_time_range(),
            "monitordata": md_stats,
        }