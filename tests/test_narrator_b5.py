"""
Test the B5 Narrator on real detections from B3.
Validates template generation, cooldown deduplication, and alert quality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser_b1 import parse_file
from detector_b3 import Detector
from narrator_b5 import Narrator, narrate_with_template, Alert

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


# ─── Helper: get detections from a file ──────────────────────────────────────

def get_detections(filename, source_path):
    filepath = os.path.join(UPLOADS, filename)
    if not os.path.exists(filepath):
        return []
    detector = Detector(RULES)
    detections = []
    for record in parse_file(filepath, source_path=source_path):
        detections.extend(detector.process(record))
    detections.extend(detector.flush())
    return detections


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_template_narration():
    section("Template Narration (all rule templates)")

    detections = get_detections(
        "system_out.log",
        "runnable_arcm_s/base/logs/system.out.log"
    )
    if not detections:
        print("  SKIPPED: no detections")
        return

    # Get one detection per rule_id
    by_rule = {}
    for d in detections:
        if d.rule_id not in by_rule:
            by_rule[d.rule_id] = d

    print(f"    Testing templates for {len(by_rule)} distinct rules")

    for rule_id, detection in sorted(by_rule.items()):
        alert = narrate_with_template(detection)
        check(isinstance(alert, Alert), f"{rule_id}: Alert object created")
        check(len(alert.title) > 10, f"{rule_id}: Title has content ({len(alert.title)} chars)")
        check(len(alert.narrative) > 50, f"{rule_id}: Narrative has substance ({len(alert.narrative)} chars)")
        check(alert.severity == detection.severity, f"{rule_id}: Severity preserved")
        check(alert.runnable == detection.runnable, f"{rule_id}: Runnable preserved")
        check(len(alert.notification_channels) > 0, f"{rule_id}: Channels assigned")

        print(f"    {Colors.WARN}[{rule_id}] {alert.title}{Colors.END}")
        print(f"      {alert.narrative[:120]}...")
        print()


def test_cooldown_dedup():
    section("Cooldown Deduplication")

    detections = get_detections(
        "system_out.log",
        "runnable_arcm_s/base/logs/system.out.log"
    )
    if not detections:
        print("  SKIPPED: no detections")
        return

    # Narrate all with cooldown
    narrator = Narrator(use_llm=False, cooldown_minutes=15)
    alerts = narrator.narrate_batch(detections)
    stats = narrator.get_stats_summary()

    print(f"    Detections in:   {len(detections)}")
    print(f"    Alerts out:      {len(alerts)}")
    print(f"    Suppressed:      {stats['suppressed_by_cooldown']}")
    print(f"    By severity:     {stats['by_severity']}")

    check(len(alerts) < len(detections),
          "Cooldown reduced alert count",
          f"{len(detections)} → {len(alerts)}")

    check(stats["suppressed_by_cooldown"] > 100,
          "Substantial suppression by cooldown",
          f"Suppressed {stats['suppressed_by_cooldown']}")

    # The reduction should be dramatic for pool exhaustion (C01: 998 → should be ~20-30)
    c01_alerts = [a for a in alerts if a.rule_id == "C01"]
    check(len(c01_alerts) < 200,
          f"C01 pool exhaustion deduplicated: {len(c01_alerts)} alerts from 998 detections")


def test_severity_channels():
    section("Severity → Notification Channels")

    detections = get_detections(
        "system_out.log",
        "runnable_arcm_s/base/logs/system.out.log"
    )
    if not detections:
        print("  SKIPPED: no detections")
        return

    # Find detections of each severity
    by_sev = {}
    for d in detections:
        if d.severity not in by_sev:
            by_sev[d.severity] = d

    for sev, det in by_sev.items():
        alert = narrate_with_template(det)
        if sev in ("CRITICAL", "P1"):
            check("sms" in alert.notification_channels,
                  f"{sev}: SMS channel included",
                  f"Channels: {alert.notification_channels}")
        elif sev == "HIGH":
            check("email" in alert.notification_channels,
                  f"{sev}: email channel included")
            check("sms" not in alert.notification_channels,
                  f"{sev}: SMS NOT included (correct)")
        elif sev == "LOW":
            check("log" in alert.notification_channels,
                  f"{sev}: log-only channel")


def test_state_sequence_narration():
    section("State Sequence Alert Narration")

    detections = get_detections(
        "runnable_history.log",
        "agent/runnable_history.log"
    )
    if not detections:
        print("  SKIPPED: no detections")
        return

    # S01: Restart loop
    s01 = [d for d in detections if d.rule_id == "S01"]
    if s01:
        alert = narrate_with_template(s01[0])
        check("restart" in alert.title.lower() or "loop" in alert.title.lower() or "crash" in alert.title.lower(),
              "S01 title mentions restart/loop/crash",
              f"Title: {alert.title}")
        check("NON riavviare" in alert.narrative,
              "S01 narrative warns against manual restart")
        check(alert.severity == "CRITICAL",
              "S01 alert is CRITICAL")

        print(f"    {Colors.WARN}S01 Alert:{Colors.END}")
        print(f"      Title: {alert.title}")
        print(f"      {alert.narrative}")
    else:
        print("  No S01 detections found")

    # S02: Down without recovery
    s02 = [d for d in detections if d.rule_id == "S02"]
    if s02:
        alert = narrate_with_template(s02[0])
        check("crash" in alert.title.lower() or "down" in alert.title.lower() or "recovery" in alert.title.lower(),
              "S02 title mentions crash/down",
              f"Title: {alert.title}")

        print(f"\n    {Colors.WARN}S02 Alert:{Colors.END}")
        print(f"      Title: {alert.title}")
        print(f"      {alert.narrative}")


def test_escalated_alert():
    section("Escalated Alert Narration")

    detections = get_detections(
        "system_out.log",
        "runnable_arcm_s/base/logs/system.out.log"
    )
    if not detections:
        print("  SKIPPED: no detections")
        return

    escalated = [d for d in detections if d.escalated]
    if escalated:
        alert = narrate_with_template(escalated[0])
        check(alert.escalated, "Alert preserves escalation flag")
        check(alert.severity == escalated[0].severity,
              f"Alert has escalated severity: {alert.severity}")

        print(f"    {Colors.WARN}Escalated alert:{Colors.END}")
        print(f"      [{alert.severity}] {alert.title}")
        print(f"      Matches: {alert.match_count}")
    else:
        print("  No escalated detections found")


def test_full_pipeline_b1_b3_b5():
    section("Full Pipeline: B1 → B3 → B5 (end-to-end)")

    filepath = os.path.join(UPLOADS, "system_out.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    # B1: Parse
    from parser_b1 import parse_file_with_stats
    records, parse_stats = parse_file_with_stats(
        filepath, source_path="runnable_arcm_s/base/logs/system.out.log"
    )

    # B3: Detect
    detector = Detector(RULES)
    detections = []
    for r in records:
        detections.extend(detector.process(r))

    # B5: Narrate (with cooldown)
    narrator = Narrator(use_llm=False, cooldown_minutes=30)
    alerts = narrator.narrate_batch(detections)

    print(f"    B1: {parse_stats.total_lines} lines → {parse_stats.records_emitted} records")
    print(f"    B3: {len(detections)} detections")
    print(f"    B5: {len(alerts)} alerts (after cooldown dedup)")
    print(f"    Reduction: {parse_stats.total_lines} → {len(alerts)} "
          f"({len(alerts)/parse_stats.total_lines*100:.2f}%)")

    check(len(alerts) > 5, "Pipeline produces meaningful alerts", f"Got {len(alerts)}")
    check(len(alerts) < 500, "Cooldown keeps alert volume manageable", f"Got {len(alerts)}")

    # Show the alert timeline
    print(f"\n    {Colors.WARN}Alert timeline (first 15):{Colors.END}")
    for alert in alerts[:15]:
        esc = " [ESCALATED]" if alert.escalated else ""
        print(f"      [{alert.generated_at.isoformat()[:16]}] "
              f"{alert.severity:8s} | {alert.rule_id} | {alert.title[:55]}{esc}")

    # Verify the most important alerts are present
    alert_rules = set(a.rule_id for a in alerts)
    check("C01" in alert_rules, "Pool exhaustion alert (C01) in output")
    check("C03" in alert_rules, "Migration failure alert (C03) in output")
    check("C04" in alert_rules, "INOPERATIVE alert (C04) in output")
    check("C05" in alert_rules, "BeanCreation alert (C05) in output")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{Colors.BOLD}ARIS Narrator B5 — Test Suite{Colors.END}\n")

    test_template_narration()
    test_cooldown_dedup()
    test_severity_channels()
    test_state_sequence_narration()
    test_escalated_alert()
    test_full_pipeline_b1_b3_b5()

    total = passed + failed
    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.END}")
    if failed == 0:
        print(f"{Colors.OK}{Colors.BOLD}  ALL {total} CHECKS PASSED ✓{Colors.END}")
    else:
        print(f"  {Colors.OK}{passed} passed{Colors.END}, {Colors.FAIL}{failed} failed{Colors.END} out of {total}")
    print(f"{'═' * 70}\n")

    sys.exit(1 if failed > 0 else 0)
