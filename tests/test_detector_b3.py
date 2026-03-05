"""
Test the B3 Detector on real log data parsed by B1.
Validates rule matching, frequency escalation, and state sequence detection.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser_b1 import parse_file, parse_file_with_stats, LogLevel, FormatType
from detector_b3 import Detector, load_rules
from pathlib import Path

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


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_rule_loading():
    section("Rule Loading")
    rules = load_rules(RULES)

    check(len(rules) > 15, f"Loaded {len(rules)} rules (expected 15+)")

    # Check all severity levels present
    severities = set(r.severity for r in rules)
    check("CRITICAL" in severities, "CRITICAL rules present")
    check("HIGH" in severities, "HIGH rules present")
    check("MEDIUM" in severities, "MEDIUM rules present")
    check("LOW" in severities, "LOW rules present")

    # Check all pattern types present
    types = set(r.pattern_type for r in rules)
    check("KEYWORD" in types, "KEYWORD rules present")
    check("REGEX" in types, "REGEX rules present")
    check("STATE_SEQUENCE" in types, "STATE_SEQUENCE rules present")

    # Check regex compilation
    regex_rules = [r for r in rules if r.pattern_type == "REGEX"]
    check(all(r._compiled_regex is not None for r in regex_rules),
          "All REGEX rules compiled successfully")

    # Check specific rules exist
    rule_ids = set(r.rule_id for r in rules)
    for rid in ["C01", "C02", "C03", "C04", "C05", "H01", "H02", "S01"]:
        check(rid in rule_ids, f"Rule {rid} present")


def test_detector_on_system_out():
    section("Detector on system_out.log (arcm_s)")
    filepath = os.path.join(UPLOADS, "system_out.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    detector = Detector(RULES)
    all_detections = []

    for record in parse_file(filepath, source_path="runnable_arcm_s/base/logs/system.out.log"):
        detections = detector.process(record)
        all_detections.extend(detections)

    stats = detector.get_stats_summary()
    print(f"    Total detections: {stats['total_detections']}")
    print(f"    By rule: {stats['by_rule']}")

    check(stats["total_detections"] > 100, "Substantial detections generated",
          f"Got {stats['total_detections']}")

    # ── C01: Pool exhaustion ──
    c01 = [d for d in all_detections if d.rule_id == "C01"]
    check(len(c01) > 500, f"C01 pool exhaustion: {len(c01)} detections (expected 500+)")

    # Check escalation on pool exhaustion
    escalated_c01 = [d for d in c01 if d.escalated]
    check(len(escalated_c01) > 0, "C01 frequency escalation triggered",
          f"Got {len(escalated_c01)} escalated detections")

    # ── C03: Migration failure ──
    c03 = [d for d in all_detections if d.rule_id == "C03"]
    check(len(c03) > 5, f"C03 migration failure: {len(c03)} detections")

    # ── C04: INOPERATIVE ──
    c04 = [d for d in all_detections if d.rule_id == "C04"]
    check(len(c04) > 5, f"C04 INOPERATIVE: {len(c04)} detections")

    # ── C05: BeanCreation ──
    c05 = [d for d in all_detections if d.rule_id == "C05"]
    check(len(c05) > 3, f"C05 BeanCreation: {len(c05)} detections")

    # ── H02: BUS-10507 ──
    h02 = [d for d in all_detections if d.rule_id == "H02"]
    check(len(h02) > 100, f"H02 BUS-10507: {len(h02)} detections")

    # ── L01: ZKC ──
    l01 = [d for d in all_detections if d.rule_id == "L01"]
    # ZKC errors are in zkc.log, not in system_out, so might be 0
    # That's OK — we test it separately

    # ── Severity distribution ──
    by_severity = {}
    for d in all_detections:
        by_severity[d.severity] = by_severity.get(d.severity, 0) + 1
    print(f"    By severity: {by_severity}")
    check("CRITICAL" in by_severity, "CRITICAL detections generated")

    # ── Sample detections ──
    print(f"\n    {Colors.WARN}Sample CRITICAL detections:{Colors.END}")
    criticals = [d for d in all_detections if d.severity == "CRITICAL"]
    for d in criticals[:5]:
        print(f"      [{d.detected_at.isoformat()[:19]}] {d.rule_id} | {d.trigger_summary[:80]}")


def test_detector_on_zkc():
    section("Detector on zkc.log")
    filepath = os.path.join(UPLOADS, "zkc.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    detector = Detector(RULES)
    all_detections = []

    for record in parse_file(filepath, source_path="runnable_arcm_s/base/logs/zkc.log"):
        detections = detector.process(record)
        all_detections.extend(detections)

    l01 = [d for d in all_detections if d.rule_id == "L01"]
    check(len(l01) > 50, f"L01 ZKC Instance Not Found: {len(l01)} detections")
    check(all(d.severity == "LOW" for d in l01), "All ZKC detections are LOW severity")


def test_detector_on_arcm_error():
    section("Detector on arcm-error.log")
    filepath = os.path.join(UPLOADS, "arcm-error.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    detector = Detector(RULES)
    all_detections = []

    for record in parse_file(filepath, source_path="runnable_arcm_s/base/logs/arcm/arcm-error.log"):
        detections = detector.process(record)
        all_detections.extend(detections)

    # Should catch NullPointerException in JobMonitoringTag (H01)
    h01 = [d for d in all_detections if d.rule_id == "H01"]
    check(len(h01) > 0, f"H01 NullPointerException: {len(h01)} detections")

    print(f"    Total detections: {len(all_detections)}")
    for d in all_detections:
        print(f"      {d.rule_id} ({d.severity}) | {d.trigger_summary[:70]}")


def test_detector_on_agent_error():
    section("Detector on agent_error.log")
    filepath = os.path.join(UPLOADS, "agent_error.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    detector = Detector(RULES)
    all_detections = []

    for record in parse_file(filepath, source_path="agent/agentLogs/agent.error.log"):
        detections = detector.process(record)
        all_detections.extend(detections)

    # MBean errors (M02)
    m02 = [d for d in all_detections if d.rule_id == "M02"]
    check(len(m02) > 0, f"M02 MBean Not Found: {len(m02)} detections")

    # APG errors (L02)
    l02 = [d for d in all_detections if d.rule_id == "L02"]
    check(len(l02) > 0, f"L02 APG Invalid AppType: {len(l02)} detections")


def test_detector_on_runnable_history():
    section("Detector on runnable_history.log (state sequences)")
    filepath = os.path.join(UPLOADS, "runnable_history.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    detector = Detector(RULES)
    all_detections = []

    for record in parse_file(filepath, source_path="agent/runnable_history.log"):
        detections = detector.process(record)
        all_detections.extend(detections)

    # Flush pending state rules
    final = detector.flush()
    all_detections.extend(final)

    stats = detector.get_stats_summary()
    print(f"    Total detections: {stats['total_detections']}")
    print(f"    By rule: {stats['by_rule']}")

    # S01: Restart loop (arcm_s on Nov 10 had 11 restarts)
    s01 = [d for d in all_detections if d.rule_id == "S01"]
    check(len(s01) > 0, f"S01 Restart Loop: {len(s01)} detections (expected for arcm_s Nov 10)")

    if s01:
        arcm_loops = [d for d in s01 if d.runnable == "arcm_s"]
        check(len(arcm_loops) > 0, "S01 detected specifically for arcm_s",
              f"Found: {set(d.runnable for d in s01)}")

    # Print all state detections
    print(f"\n    {Colors.WARN}State sequence detections:{Colors.END}")
    for d in all_detections:
        print(f"      [{d.detected_at.isoformat()[:19]}] {d.rule_id} ({d.severity}) | {d.runnable} | {d.trigger_summary[:70]}")


def test_detector_on_copernicus():
    section("Detector on copernicus_publishing.log")
    filepath = os.path.join(UPLOADS, "copernicus_publishing.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    detector = Detector(RULES)
    all_detections = []

    for record in parse_file(filepath, source_path="runnable_abs_s/base/logs/copernicus.publishing.log"):
        detections = detector.process(record)
        all_detections.extend(detections)

    stats = detector.get_stats_summary()
    print(f"    Total detections: {stats['total_detections']}")

    # M04: Plugin localization
    m04 = [d for d in all_detections if d.rule_id == "M04"]
    check(len(m04) > 0, f"M04 Plugin Localization: {len(m04)} detections")


def test_escalation_logic():
    section("Frequency Escalation Logic")
    filepath = os.path.join(UPLOADS, "system_out.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    detector = Detector(RULES)
    all_detections = []

    for record in parse_file(filepath, source_path="runnable_arcm_s/base/logs/system.out.log"):
        detections = detector.process(record)
        all_detections.extend(detections)

    # C01 with escalation: 10 pool errors in 5 min → P1
    c01_escalated = [d for d in all_detections if d.rule_id == "C01" and d.escalated]
    check(len(c01_escalated) > 0, "C01 pool exhaustion escalated to P1",
          f"Got {len(c01_escalated)} escalated")

    if c01_escalated:
        check(c01_escalated[0].severity == "P1",
              f"Escalated severity is P1 (got {c01_escalated[0].severity})")
        check(c01_escalated[0].match_count >= 10,
              f"Match count >= 10 at escalation (got {c01_escalated[0].match_count})")

    # H02 with escalation: 50 BUS errors in 10 min → CRITICAL
    h02_escalated = [d for d in all_detections if d.rule_id == "H02" and d.escalated]
    if h02_escalated:
        check(h02_escalated[0].severity == "CRITICAL",
              f"H02 escalated to CRITICAL (got {h02_escalated[0].severity})")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{Colors.BOLD}ARIS Detector B3 — Test Suite{Colors.END}")
    print(f"Rules: {RULES}\n")

    test_rule_loading()
    test_detector_on_system_out()
    test_detector_on_zkc()
    test_detector_on_arcm_error()
    test_detector_on_agent_error()
    test_detector_on_runnable_history()
    test_detector_on_copernicus()
    test_escalation_logic()

    total = passed + failed
    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.END}")
    if failed == 0:
        print(f"{Colors.OK}{Colors.BOLD}  ALL {total} CHECKS PASSED ✓{Colors.END}")
    else:
        print(f"  {Colors.OK}{passed} passed{Colors.END}, {Colors.FAIL}{failed} failed{Colors.END} out of {total}")
    print(f"{'═' * 70}\n")

    sys.exit(1 if failed > 0 else 0)
