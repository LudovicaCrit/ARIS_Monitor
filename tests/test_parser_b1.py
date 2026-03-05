"""
Test the B1 parser on all uploaded sample log files.
Validates parsing, format detection, multiline aggregation, and noise filtering.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser_b1 import (
    parse_file_with_stats, LogLevel, LogFamily, FormatType,
    detect_log_family, detect_runnable
)
from pathlib import Path
from datetime import datetime, timezone

UPLOADS = os.path.join(os.path.dirname(__file__), "..", "..", "test_samples")

# ─── Test infrastructure ─────────────────────────────────────────────────────

class Colors:
    OK = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    BOLD = "\033[1m"
    END = "\033[0m"

passed = 0
failed = 0

def check(condition: bool, description: str, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  {Colors.OK}✓{Colors.END} {description}")
        passed += 1
    else:
        print(f"  {Colors.FAIL}✗ {description}{Colors.END}")
        if detail:
            print(f"    → {detail}")
        failed += 1


def section(title: str):
    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.END}")
    print(f"{Colors.BOLD}  {title}{Colors.END}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.END}")


def print_stats(stats):
    print(f"    Lines: {stats.total_lines} total, {stats.noise_filtered} noise filtered")
    print(f"    Records: {stats.records_emitted} emitted, {stats.errors_and_above} ERROR+, {stats.has_stack_trace_count} with stack trace")
    print(f"    Formats: {stats.by_format}")
    print(f"    Levels:  {stats.by_level}")


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_system_out_arcm():
    section("system_out.log (arcm_s — 21 MB)")
    filepath = os.path.join(UPLOADS, "system_out.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, stats = parse_file_with_stats(
        filepath,
        source_path="runnable_arcm_s/base/logs/system.out.log",
    )
    print_stats(stats)

    check(stats.records_emitted > 0, "Emitted records > 0", f"Got {stats.records_emitted}")
    check(stats.by_format.get("PIPE_8FIELD", 0) > 1000, "Majority of structured lines are PIPE_8FIELD",
          f"Got {stats.by_format.get('PIPE_8FIELD', 0)}")
    check(stats.by_level.get("ERROR", 0) > 4000, "ERROR count matches expected ~4655",
          f"Got {stats.by_level.get('ERROR', 0)}")
    check(stats.by_level.get("FATAL", 0) > 40, "FATAL count matches expected ~57",
          f"Got {stats.by_level.get('FATAL', 0)}")
    check(stats.has_stack_trace_count > 100, "Stack traces aggregated",
          f"Got {stats.has_stack_trace_count} records with stack trace")

    # Check runnable detection
    runnables = set(r.runnable for r in records)
    check("arcm_s" in runnables, "Runnable correctly detected as arcm_s", f"Found: {runnables}")

    # Check pool exhaustion is captured
    pool_errors = [r for r in records if "No free database connection" in r.message]
    check(len(pool_errors) > 500, "Pool exhaustion errors captured",
          f"Got {len(pool_errors)} records (some are multi-line aggregated)")

    # Check FATAL/INOPERATIVE
    fatals = [r for r in records if r.level == LogLevel.FATAL]
    inop = [r for r in fatals if "INOPERATIVE" in r.message]
    check(len(inop) > 5, "INOPERATIVE fatals captured", f"Got {len(inop)}")

    # Check migration failure
    migration = [r for r in records if "Cannot calculate migration" in r.message]
    check(len(migration) > 5, "Migration failure errors captured", f"Got {len(migration)}")

    # Check BeanCreation
    bean = [r for r in records if "BeanCreationNotAllowedException" in r.message or
            (r.stack_trace and "BeanCreationNotAllowedException" in r.stack_trace)]
    check(len(bean) > 5, "BeanCreationNotAllowedException captured", f"Got {len(bean)}")

    # Check restart detection (PLAIN_TS with "Getting RuntimeMXBean")
    restarts = [r for r in records if "Getting RuntimeMXBean" in r.message]
    check(len(restarts) >= 25, "JVM restarts detected via PLAIN_TS",
          f"Got {len(restarts)} (expected ~30)")

    # Spot check: first record should be a startup line
    if records:
        first = records[0]
        check(first.timestamp.year == 2025, "First record timestamp is 2025",
              f"Got {first.timestamp}")
        check(first.log_family == LogFamily.SYSTEM_OUT, "Log family is SYSTEM_OUT")

    # BUS message aggregation
    bus = [r for r in records if "BUS-" in r.message and "\n" in r.message]
    check(len(bus) > 0, "BUS messages aggregated into multi-line messages",
          f"Got {len(bus)} multi-line BUS records")

    # Print some interesting records
    print(f"\n    {Colors.WARN}Sample records:{Colors.END}")
    for r in fatals[:3]:
        print(f"      [{r.timestamp.isoformat()}] {r.level.value} | {r.logger} | {r.message[:100]}")


def test_arcm_error():
    section("arcm-error.log")
    filepath = os.path.join(UPLOADS, "arcm-error.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, stats = parse_file_with_stats(
        filepath,
        source_path="runnable_arcm_s/base/logs/arcm/arcm-error.log",
    )
    print_stats(stats)

    check(stats.records_emitted > 0, "Records emitted")
    check(stats.by_format.get("PIPE_8FIELD", 0) == stats.records_emitted,
          "All records are PIPE_8FIELD",
          f"{stats.by_format}")
    check(all(r.level in (LogLevel.ERROR, LogLevel.FATAL) for r in records),
          "All records are ERROR or FATAL level")

    # Check NullPointerException in JobMonitoringTag
    npe = [r for r in records if "NullPointerException" in (r.stack_trace or "")]
    check(len(npe) > 0, "NullPointerException stack traces captured", f"Got {len(npe)}")

    if records:
        check(records[0].instance_id == "arcm0000000000", "Instance ID parsed correctly",
              f"Got: {records[0].instance_id}")
        check(records[0].runnable == "arcm_s", "Runnable from instance_id")


def test_arcm_log():
    section("arcm.log")
    filepath = os.path.join(UPLOADS, "arcm.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, stats = parse_file_with_stats(
        filepath,
        source_path="runnable_arcm_s/base/logs/arcm/arcm.log",
    )
    print_stats(stats)

    check(stats.records_emitted > 0, "Records emitted")
    check(stats.by_format.get("PIPE_8FIELD", 0) > 0, "PIPE_8FIELD format detected")

    # BootLog messages
    bootlog = [r for r in records if r.logger == "BootLog"]
    check(len(bootlog) > 10, "BootLog messages captured", f"Got {len(bootlog)}")

    # Check ACCEPTING_TRAFFIC
    accepting = [r for r in records if "ACCEPTING_TRAFFIC" in r.message]
    check(len(accepting) > 0, "ACCEPTING_TRAFFIC events captured", f"Got {len(accepting)}")


def test_zkc():
    section("zkc.log")
    filepath = os.path.join(UPLOADS, "zkc.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, stats = parse_file_with_stats(
        filepath,
        source_path="runnable_arcm_s/base/logs/zkc.log",
    )
    print_stats(stats)

    check(stats.records_emitted > 0, "Records emitted")
    check(all(r.level == LogLevel.ERROR for r in records), "All records are ERROR (ZKC startup race)")

    zkc_msgs = [r for r in records if "No registered Instance" in r.message]
    check(len(zkc_msgs) > 10, "ZKC 'No registered Instance' messages captured", f"Got {len(zkc_msgs)}")


def test_agent():
    section("agent.log (14 MB, 99.9% noise)")
    filepath = os.path.join(UPLOADS, "agent.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, stats = parse_file_with_stats(filepath, source_path="agent/agentLogs/agent.log")
    print_stats(stats)

    check(stats.noise_filtered > 60000, "Noise filter removed ZKC heartbeat",
          f"Filtered {stats.noise_filtered} of {stats.total_lines} lines")
    check(stats.records_emitted < 100, "Only meaningful records remain",
          f"Got {stats.records_emitted}")
    check(stats.records_emitted > 0, "Some records survived", f"Got {stats.records_emitted}")

    if records:
        check(records[0].format_type == FormatType.AGENT, "Format detected as AGENT")
        check(records[0].runnable == "agent", "Runnable is 'agent'")


def test_agent_error():
    section("agent_error.log")
    filepath = os.path.join(UPLOADS, "agent_error.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, stats = parse_file_with_stats(
        filepath, source_path="agent/agentLogs/agent.error.log"
    )
    print_stats(stats)

    check(stats.records_emitted > 0, "Records emitted")
    check(stats.by_level.get("ERROR", 0) > 0, "ERROR level present")

    # MBean errors
    mbean = [r for r in records if "MBean" in r.message or "MBean" in (r.stack_trace or "")]
    check(len(mbean) > 0, "MBean Not Found errors captured", f"Got {len(mbean)}")

    # APG invalid apptype
    apg = [r for r in records if "APG" in r.message or "apptype" in r.message]
    check(len(apg) > 0, "APG apptype-specific errors captured", f"Got {len(apg)}")


def test_wrapper():
    section("ARISCloudAgent.log (wrapper)")
    filepath = os.path.join(UPLOADS, "ARISCloudAgent.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, stats = parse_file_with_stats(
        filepath, source_path="agent/wrapperLogs/ARISCloudAgent.log"
    )
    print_stats(stats)

    check(stats.noise_filtered > 1000, "Noise filter works through wrapper prefix",
          f"Filtered {stats.noise_filtered}")
    check(stats.records_emitted >= 0, "Parsing completes without errors")

    # Wrapper should unwrap to AGENT format or WRAPPER format
    wrapper_count = stats.by_format.get("WRAPPER", 0)
    agent_count = stats.by_format.get("AGENT", 0)
    check(wrapper_count > 0 or agent_count > 0 or stats.records_emitted == 0,
          "Records detected as WRAPPER or AGENT after unwrapping",
          f"WRAPPER={wrapper_count}, AGENT={agent_count}")


def test_runnable_history():
    section("runnable_history.log")
    filepath = os.path.join(UPLOADS, "runnable_history.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, stats = parse_file_with_stats(filepath, source_path="agent/runnable_history.log")
    print_stats(stats)

    check(stats.records_emitted > 100, "Substantial records", f"Got {stats.records_emitted}")
    check(stats.by_format.get("RUNNABLE_HIST", 0) == stats.records_emitted,
          "All records are RUNNABLE_HIST format")

    # State changes
    state_changes = [r for r in records if "State change:" in r.message]
    check(len(state_changes) > 200, "State change events captured", f"Got {len(state_changes)}")

    # FAILED states should be ERROR level
    failed_records = [r for r in records if "FAILED" in r.message]
    check(all(r.level == LogLevel.ERROR for r in failed_records),
          "FAILED states mapped to ERROR level", f"Got {len(failed_records)} FAILED records")

    # Runnable names
    runnables = set(r.runnable for r in records)
    check("arcm_s" in runnables, "arcm_s in runnables")
    check("zoo_s" in runnables, "zoo_s in runnables")
    check("abs_s" in runnables, "abs_s in runnables")
    check(len(runnables) > 10, "Multiple runnables present", f"Found {len(runnables)}: {sorted(runnables)}")


def test_copernicus_publishing():
    section("copernicus_publishing.log")
    filepath = os.path.join(UPLOADS, "copernicus_publishing.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, stats = parse_file_with_stats(
        filepath,
        source_path="runnable_abs_s/base/logs/copernicus.publishing.log",
    )
    print_stats(stats)

    check(stats.records_emitted > 100, "Substantial records", f"Got {stats.records_emitted}")
    check(stats.by_format.get("PIPE_8FIELD", 0) > 0, "PIPE_8FIELD detected")

    # Check publishing errors
    errors = [r for r in records if r.level == LogLevel.ERROR]
    check(len(errors) > 0, "ERROR records captured", f"Got {len(errors)}")


def test_rest_operations():
    section("agent_rest_operations.log")
    filepath = os.path.join(UPLOADS, "agent_rest_operations.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, stats = parse_file_with_stats(filepath, source_path="agent/agent_rest_operations.log")
    print_stats(stats)

    check(stats.records_emitted > 0, "Records emitted", f"Got {stats.records_emitted}")
    check(stats.by_format.get("REST_OPS", 0) > 0, "REST_OPS format detected")

    # Check duration parsing
    if records:
        check("[" in records[0].message and "ms]" in records[0].message,
              "Duration included in message", f"First: {records[0].message[:80]}")

    # Configure operations
    configures = [r for r in records if "configure" in r.message.lower()]
    check(len(configures) > 0, "Configure operations captured", f"Got {len(configures)}")


def test_message_hash():
    section("Message Hash (deduplication)")
    filepath = os.path.join(UPLOADS, "system_out.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, _ = parse_file_with_stats(
        filepath,
        source_path="runnable_arcm_s/base/logs/system.out.log",
    )

    # Pool exhaustion messages should share the same hash
    pool = [r for r in records if "No free database connection" in r.message]
    if len(pool) > 2:
        hashes = set(r.message_hash for r in pool)
        check(len(hashes) < len(pool) / 2,
              "Pool exhaustion messages share similar hashes (dedup works)",
              f"{len(hashes)} unique hashes for {len(pool)} records")

    # Same ZKC error should hash identically
    zkc_file = os.path.join(UPLOADS, "zkc.log")
    if os.path.exists(zkc_file):
        zkc_records, _ = parse_file_with_stats(zkc_file)
        if len(zkc_records) > 5:
            zkc_hashes = set(r.message_hash for r in zkc_records)
            check(len(zkc_hashes) <= 5,
                  "ZKC repeated messages deduplicate well",
                  f"{len(zkc_hashes)} unique hashes for {len(zkc_records)} records")


def test_timestamp_utc():
    section("Timestamp UTC Normalization")
    filepath = os.path.join(UPLOADS, "system_out.log")
    if not os.path.exists(filepath):
        print("  SKIPPED: file not found")
        return

    records, _ = parse_file_with_stats(
        filepath,
        source_path="runnable_arcm_s/base/logs/system.out.log",
    )

    if records:
        # All timestamps should be UTC
        for r in records[:10]:
            check(r.timestamp.tzinfo is not None, f"Timestamp has tzinfo: {r.timestamp.isoformat()}")
            break  # just check first

        # Find a PLAIN_Z record (UTC) and a PIPE_8FIELD record from same time
        plain_z = [r for r in records if r.format_type == FormatType.PLAIN_TS and "Getting RuntimeMXBean" in r.message]
        pipe8 = [r for r in records if r.format_type == FormatType.PIPE_8FIELD]

        if plain_z and pipe8:
            # The first plain_z and first pipe_8 in November should show correct UTC conversion
            pz = plain_z[0]
            check(pz.timestamp.tzinfo == timezone.utc,
                  f"PLAIN_Z timestamp is UTC: {pz.timestamp.isoformat()}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{Colors.BOLD}ARIS Log Parser B1 — Test Suite{Colors.END}")
    print(f"Testing on sample files in {UPLOADS}\n")

    test_system_out_arcm()
    test_arcm_error()
    test_arcm_log()
    test_zkc()
    test_agent()
    test_agent_error()
    test_wrapper()
    test_runnable_history()
    test_copernicus_publishing()
    test_rest_operations()
    test_message_hash()
    test_timestamp_utc()

    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.END}")
    total = passed + failed
    if failed == 0:
        print(f"{Colors.OK}{Colors.BOLD}  ALL {total} CHECKS PASSED ✓{Colors.END}")
    else:
        print(f"  {Colors.OK}{passed} passed{Colors.END}, {Colors.FAIL}{failed} failed{Colors.END} out of {total}")
    print(f"{'═' * 70}\n")

    sys.exit(1 if failed > 0 else 0)
