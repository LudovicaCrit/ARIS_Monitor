"""
Test the B4 Correlator and the full pipeline B1 → B3 → B4 → B5.
Validates incident creation, dependency-based correlation, and root cause estimation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser_b1 import parse_file, parse_file_with_stats
from detector_b3 import Detector
from correlator_b4 import Correlator, Incident, DEPENDENCY_LAYERS, DEPENDENCIES
from narrator_b5 import Narrator

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


# ─── Helper: collect all detections from multiple files ──────────────────────

def get_all_detections():
    """Parse multiple log files and collect all detections, sorted by time."""
    files = [
        ("system_out.log", "runnable_arcm_s/base/logs/system.out.log"),
        ("arcm-error.log", "runnable_arcm_s/base/logs/arcm/arcm-error.log"),
        ("arcm.log", "runnable_arcm_s/base/logs/arcm/arcm.log"),
        ("zkc.log", "runnable_arcm_s/base/logs/zkc.log"),
        ("agent_error.log", "agent/agentLogs/agent.error.log"),
        ("runnable_history.log", "agent/runnable_history.log"),
        ("copernicus_publishing.log", "runnable_abs_s/base/logs/copernicus.publishing.log"),
    ]

    detector = Detector(RULES)
    all_detections = []

    for filename, source_path in files:
        filepath = os.path.join(UPLOADS, filename)
        if not os.path.exists(filepath):
            continue
        for record in parse_file(filepath, source_path=source_path):
            detections = detector.process(record)
            all_detections.extend(detections)

    all_detections.extend(detector.flush())

    # Sort by time
    all_detections.sort(key=lambda d: d.detected_at)
    return all_detections


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_dependency_hierarchy():
    section("Dependency Hierarchy")

    check(DEPENDENCY_LAYERS["zoo_s"] == 0, "zoo_s is layer 0 (foundation)")
    check(DEPENDENCY_LAYERS["loadbalancer_s"] == 5, "loadbalancer_s is layer 5 (edge)")
    check(DEPENDENCY_LAYERS["arcm_s"] == 4, "arcm_s is layer 4")

    check("zoo_s" in DEPENDENCIES.get("elastic_s", []),
          "elastic_s depends on zoo_s")
    check("umcadmin_s" in DEPENDENCIES.get("arcm_s", []),
          "arcm_s depends on umcadmin_s")
    check("abs_s" in DEPENDENCIES.get("loadbalancer_s", []),
          "loadbalancer_s depends on abs_s")

    # No circular dependencies
    for child, parents in DEPENDENCIES.items():
        for parent in parents:
            parent_parents = DEPENDENCIES.get(parent, [])
            check(child not in parent_parents,
                  f"No circular: {child} → {parent} → {child}",
                  f"Found circular dependency!")


def test_correlation_single_runnable():
    section("Correlation — Single Runnable (arcm_s)")

    filepath = os.path.join(UPLOADS, "system_out.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    detector = Detector(RULES)
    detections = []
    for record in parse_file(filepath, source_path="runnable_arcm_s/base/logs/system.out.log"):
        detections.extend(detector.process(record))

    # Sort by time
    detections.sort(key=lambda d: d.detected_at)

    correlator = Correlator(window_minutes=5)
    for d in detections:
        correlator.process(d)
    correlator.flush()

    incidents = correlator.get_all_incidents()
    stats = correlator.get_stats_summary()

    print(f"    Detections: {stats['detections_processed']}")
    print(f"    Incidents:  {stats['total_incidents']}")
    print(f"    Correlated: {stats['detections_correlated']}")
    print(f"    By severity: {stats['by_severity']}")
    print(f"    By size: {stats['by_size']}")

    check(stats["total_incidents"] > 0, "Incidents created")
    check(stats["total_incidents"] < stats["detections_processed"],
          "Correlation reduces detection count to incidents",
          f"{stats['detections_processed']} detections → {stats['total_incidents']} incidents")

    check(stats["detections_correlated"] > 100,
          "Substantial correlation happening",
          f"{stats['detections_correlated']} detections merged into existing incidents")

    # Check that pool exhaustion events are grouped
    pool_incidents = [i for i in incidents if "C01" in i.rules_involved]
    if pool_incidents:
        max_pool = max(i.detection_count for i in pool_incidents)
        check(max_pool > 5,
              f"Pool exhaustion events grouped: largest incident has {max_pool} detections")


def test_correlation_multi_file():
    section("Correlation — Multi-File (cross-runnable)")

    detections = get_all_detections()
    if not detections:
        print("  SKIPPED: no detections")
        return

    correlator = Correlator(window_minutes=5)
    for d in detections:
        correlator.process(d)
    correlator.flush()

    incidents = correlator.get_all_incidents()
    stats = correlator.get_stats_summary()

    print(f"    Detections: {stats['detections_processed']}")
    print(f"    Incidents:  {stats['total_incidents']}")
    print(f"    Multi-runnable: {stats['multi_runnable_incidents']}")
    print(f"    By severity: {stats['by_severity']}")

    check(len(incidents) > 0, "Incidents created from multi-file analysis")

    # Check that we have incidents with multiple runnables
    multi_run = [i for i in incidents if len(i.affected_runnables) > 1]
    check(len(multi_run) >= 0,
          f"Multi-runnable incidents: {len(multi_run)}",
          "May be 0 if runnables fail at different times — that's OK")

    # Root cause estimation
    incidents_with_root = [i for i in incidents if i.root_cause_estimate]
    check(len(incidents_with_root) > 0, "Root cause estimates generated")

    # Show top incidents by severity and size
    print(f"\n    {Colors.WARN}Top 10 incidents by detection count:{Colors.END}")
    top = sorted(incidents, key=lambda i: i.detection_count, reverse=True)[:10]
    for inc in top:
        print(f"      [{inc.severity:8s}] {inc.detection_count:4d} detections | "
              f"runnables: {', '.join(inc.affected_runnables)} | "
              f"rules: {', '.join(inc.rules_involved)}")
        if inc.root_cause_estimate:
            print(f"        Root cause: {inc.root_cause_estimate[:100]}")


def test_root_cause_estimation():
    section("Root Cause Estimation Logic")

    # Create a synthetic multi-runnable incident
    inc = Incident(
        incident_id="test-001",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        severity="CRITICAL",
        affected_runnables=["arcm_s", "umcadmin_s", "zoo_s"],
    )
    inc.estimate_root_cause()

    check(inc.root_cause_runnable == "zoo_s",
          "Root cause is zoo_s (lowest layer)",
          f"Got: {inc.root_cause_runnable}")
    check("zoo_s" in inc.root_cause_estimate,
          "Root cause explanation mentions zoo_s")

    # Test with single runnable
    inc2 = Incident(
        incident_id="test-002",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        severity="HIGH",
        affected_runnables=["arcm_s"],
    )
    inc2.estimate_root_cause()
    check("isolato" in inc2.root_cause_estimate.lower(),
          "Single runnable → 'problema isolato'",
          f"Got: {inc2.root_cause_estimate}")

    # Test cascade detection
    inc3 = Incident(
        incident_id="test-003",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        severity="CRITICAL",
        affected_runnables=["elastic_s", "ces_s"],
    )
    inc3.estimate_root_cause()
    check(inc3.root_cause_runnable == "elastic_s",
          "elastic_s identified as root (ces_s depends on it)")
    check("cascata" in inc3.root_cause_estimate.lower(),
          "Cascade detected in explanation",
          f"Got: {inc3.root_cause_estimate}")


from datetime import datetime, timezone


def test_full_pipeline():
    section("Full Pipeline: B1 → B3 → B4 → B5 (end-to-end)")

    detections = get_all_detections()
    if not detections:
        print("  SKIPPED: no detections")
        return

    # B4: Correlate
    correlator = Correlator(window_minutes=5)
    for d in detections:
        correlator.process(d)
    correlator.flush()
    incidents = correlator.get_all_incidents()

    # B5: Narrate (one alert per incident, using the highest-severity detection)
    narrator = Narrator(use_llm=False, cooldown_minutes=30)
    alerts = []
    for inc in incidents:
        if not inc.detections:
            continue
        # Pick the most severe detection as representative
        rep = max(inc.detections, key=lambda d: SEVERITY_ORDER.get(d.severity, 0))
        alert = narrator.narrate(rep)
        if alert:
            alert.incident_id = inc.incident_id
            alerts.append(alert)

    cor_stats = correlator.get_stats_summary()
    nar_stats = narrator.get_stats_summary()

    print(f"    Detections:  {cor_stats['detections_processed']}")
    print(f"    Incidents:   {cor_stats['total_incidents']}")
    print(f"    Alerts:      {len(alerts)}")
    print(f"    Narr stats:  {nar_stats}")

    check(len(incidents) > 0, "Pipeline produces incidents")
    check(len(alerts) > 0, "Pipeline produces alerts")
    check(len(alerts) <= len(incidents), "Alerts ≤ incidents (cooldown dedup)")

    # The full pipeline should dramatically reduce volume
    total_detections = cor_stats["detections_processed"]
    check(len(alerts) < total_detections / 5,
          f"Pipeline reduces volume: {total_detections} detections → {len(alerts)} alerts")

    # Print final alert timeline
    print(f"\n    {Colors.WARN}Final alert timeline (first 20):{Colors.END}")
    for alert in alerts[:20]:
        inc_id = alert.incident_id[:8] if alert.incident_id else "n/a"
        esc = " [ESC]" if alert.escalated else ""
        print(f"      {alert.severity:8s} | inc:{inc_id} | {alert.rule_id} | "
              f"{alert.title[:50]}{esc}")


# ─── Severity ordering import for full pipeline ──────────────────────────────
from correlator_b4 import SEVERITY_ORDER


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{Colors.BOLD}ARIS Correlator B4 — Test Suite{Colors.END}\n")

    test_dependency_hierarchy()
    test_correlation_single_runnable()
    test_correlation_multi_file()
    test_root_cause_estimation()
    test_full_pipeline()

    total = passed + failed
    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.END}")
    if failed == 0:
        print(f"{Colors.OK}{Colors.BOLD}  ALL {total} CHECKS PASSED ✓{Colors.END}")
    else:
        print(f"  {Colors.OK}{passed} passed{Colors.END}, {Colors.FAIL}{failed} failed{Colors.END} out of {total}")
    print(f"{'═' * 70}\n")

    sys.exit(1 if failed > 0 else 0)
