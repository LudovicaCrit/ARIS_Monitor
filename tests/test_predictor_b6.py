"""
Test the B6 Predictor — Isolation Forest on arcm_s.
RETROSPECTIVE VALIDATION: does the model flag anomalies BEFORE known crashes?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser_b1 import parse_file
from features_b6 import FeatureBuilder
from predictor_b6 import Predictor, RiskScore
from datetime import datetime, timezone, timedelta

UPLOADS = os.path.join(os.path.dirname(__file__), "..", "..", "test_samples")
RULES = os.path.join(os.path.dirname(__file__), "..", "config", "rules.yaml")

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


# ─── Build feature vectors ────────────────────────────────────────────────────

def build_arcm_vectors():
    """Build feature vectors for arcm_s from all available data."""
    builder = FeatureBuilder()

    sysout = os.path.join(UPLOADS, "system_out.log")
    hist = os.path.join(UPLOADS, "runnable_history.log")
    heap = os.path.join(UPLOADS, "MemoryUsageGaugeSet_heap_committed_2025_11.csv")

    if not os.path.exists(sysout):
        return None, None

    for record in parse_file(sysout, source_path="runnable_arcm_s/base/logs/system.out.log"):
        builder.add_log_record(record)

    if os.path.exists(hist):
        for record in parse_file(hist, source_path="agent/runnable_history.log"):
            builder.add_log_record(record)

    if os.path.exists(heap):
        builder.load_monitordata_files({
            "jvm/heap_committed": (heap, "gauge"),
        })

    vectors = builder.build_vectors("arcm_s", interval_seconds=300)
    return vectors, builder


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_model_training():
    section("Model Training")

    vectors, _ = build_arcm_vectors()
    if not vectors:
        print("  SKIPPED: data not found")
        return None

    predictor = Predictor(contamination=0.02, n_estimators=200)

    print(f"    Training on {len(vectors)} vectors...")
    predictor.fit(vectors, model_version="arcm_s_v1")

    check(predictor._is_fitted, "Model fitted successfully")
    check(predictor.model is not None, "Isolation Forest model created")
    check(predictor.scaler is not None, "Scaler created")
    check(len(predictor.feature_names) == 12, f"12 features used ({len(predictor.feature_names)})")

    return predictor, vectors


def test_prediction(predictor, vectors):
    section("Prediction + Risk Scores")

    if predictor is None:
        print("  SKIPPED: no model")
        return None

    print(f"    Scoring {len(vectors)} vectors...")
    risk_scores = predictor.predict(vectors)

    stats = predictor.get_stats(risk_scores)
    print(f"    Stats: {stats}")

    check(len(risk_scores) == len(vectors), "One score per vector")
    check(stats["red_count"] > 0, f"RED alerts generated: {stats['red_count']}")
    check(stats["green_count"] > stats["red_count"],
          "Most vectors are GREEN (normal)",
          f"GREEN={stats['green_count']}, RED={stats['red_count']}")
    check(stats["red_pct"] < 10, f"RED < 10% of total ({stats['red_pct']}%)")

    return risk_scores


def test_retrospective_validation(risk_scores, vectors):
    section("RETROSPECTIVE VALIDATION — Would the model have warned?")

    if risk_scores is None:
        print("  SKIPPED: no scores")
        return

    # Build lookup: epoch → risk_score
    score_map = {rs.predicted_at_epoch: rs for rs in risk_scores}

    # Known crash events with their onset times
    crashes = [
        {
            "name": "Migration failure restart loop",
            "crash_time": datetime(2025, 11, 10, 9, 50, tzinfo=timezone.utc),
            "description": "arcm_s entered STARTING→DOWN→FAILED loop, 11 restarts in 30min",
        },
        {
            "name": "Pool exhaustion onset (Dec 3)",
            "crash_time": datetime(2025, 12, 3, 3, 15, tzinfo=timezone.utc),
            "description": "First 'No free database connection' errors",
        },
        {
            "name": "Pool exhaustion explosion (Dec 18)",
            "crash_time": datetime(2025, 12, 18, 0, 0, tzinfo=timezone.utc),
            "description": "Massive pool saturation, 289 errors/day for 6 days",
        },
        {
            "name": "Pool exhaustion recurrence (Jan 15)",
            "crash_time": datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc),
            "description": "Pool exhaustion returns after 3 weeks quiet",
        },
    ]

    for crash in crashes:
        crash_epoch = int(crash["crash_time"].timestamp())
        # Align to 5-min grid
        crash_epoch = (crash_epoch // 300) * 300

        print(f"\n  {Colors.BOLD}{crash['name']}{Colors.END}")
        print(f"  Crash at: {crash['crash_time'].strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Description: {crash['description']}")

        # Get scores in the window: 2h before to 1h after crash
        window_start = crash_epoch - 7200   # 2h before
        window_end = crash_epoch + 3600     # 1h after

        window_scores = []
        t = window_start
        while t <= window_end:
            if t in score_map:
                window_scores.append(score_map[t])
            t += 300

        if not window_scores:
            print(f"  {Colors.WARN}  No scores in window (data gap){Colors.END}")
            continue

        # Show timeline
        print(f"  {'Time':>7} | {'Risk':>5} | {'Level':>6} | Top features")
        print(f"  {'-'*65}")

        pre_crash_max = 0
        first_red_before_crash = None
        first_yellow_before_crash = None

        for rs in window_scores:
            time_str = rs.predicted_at.strftime("%H:%M")
            is_crash_time = abs(rs.predicted_at_epoch - crash_epoch) < 300

            marker = " <<<< CRASH" if is_crash_time else ""
            top_feat_str = ", ".join(f"{f['feature']}={f['value']}" for f in rs.top_features[:3])

            color = Colors.FAIL if rs.risk_level == "RED" else \
                    Colors.WARN if rs.risk_level == "YELLOW" else ""
            end = Colors.END if color else ""

            print(f"  {time_str:>7} | {color}{rs.risk_score:5.1f}{end} | {color}{rs.risk_level:>6}{end} | "
                  f"{top_feat_str}{marker}")

            # Track pre-crash signals
            if rs.predicted_at_epoch < crash_epoch:
                pre_crash_max = max(pre_crash_max, rs.risk_score)
                if rs.risk_level == "RED" and first_red_before_crash is None:
                    first_red_before_crash = rs
                if rs.risk_level == "YELLOW" and first_yellow_before_crash is None:
                    first_yellow_before_crash = rs

        # Validate
        during_crash = [rs for rs in window_scores
                       if abs(rs.predicted_at_epoch - crash_epoch) < 600]
        if during_crash:
            crash_risk = max(rs.risk_score for rs in during_crash)
            check(crash_risk > 50,
                  f"Risk score elevated AT crash time (score={crash_risk:.1f})")

        if first_red_before_crash:
            lead_minutes = (crash_epoch - first_red_before_crash.predicted_at_epoch) / 60
            check(lead_minutes > 0,
                  f"RED warning {lead_minutes:.0f} min BEFORE crash")
        elif first_yellow_before_crash:
            lead_minutes = (crash_epoch - first_yellow_before_crash.predicted_at_epoch) / 60
            check(lead_minutes > 0,
                  f"YELLOW warning {lead_minutes:.0f} min BEFORE crash")
        else:
            check(pre_crash_max > 20,
                  f"Some pre-crash signal detected (max={pre_crash_max:.1f})")


def test_model_save_load(predictor, vectors):
    section("Model Save/Load")

    if predictor is None:
        print("  SKIPPED: no model")
        return

    model_path = "/tmp/test_model_b6.pkl"
    predictor.save(model_path)
    check(os.path.exists(model_path), "Model saved to disk")

    # Load into new predictor
    predictor2 = Predictor()
    predictor2.load(model_path)
    check(predictor2._is_fitted, "Model loaded successfully")
    check(predictor2.model_version == "arcm_s_v1", "Version preserved")

    # Predict with loaded model — should give same results
    scores_original = predictor.predict(vectors[:10])
    scores_loaded = predictor2.predict(vectors[:10])

    scores_match = all(
        abs(a.risk_score - b.risk_score) < 0.01
        for a, b in zip(scores_original, scores_loaded)
    )
    check(scores_match, "Loaded model produces identical scores")

    os.remove(model_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{Colors.BOLD}ARIS Predictor B6 — Isolation Forest Test + Retrospective Validation{Colors.END}\n")

    result = test_model_training()
    if result:
        predictor, vectors = result
        risk_scores = test_prediction(predictor, vectors)
        test_retrospective_validation(risk_scores, vectors)
        test_model_save_load(predictor, vectors)
    else:
        print("  SKIPPED: could not build vectors")

    total = passed + failed
    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.END}")
    if failed == 0:
        print(f"{Colors.OK}{Colors.BOLD}  ALL {total} CHECKS PASSED ✓{Colors.END}")
    else:
        print(f"  {Colors.OK}{passed} passed{Colors.END}, {Colors.FAIL}{failed} failed{Colors.END} out of {total}")
    print(f"{'═' * 70}\n")

    sys.exit(1 if failed > 0 else 0)
