"""
ARIS Log Monitoring System — Detector B3
==========================================
Matches normalized log records (B1 output) against a YAML rule catalog
and emits detection events.

Supports 4 pattern types:
  - KEYWORD:        Fast substring match on message
  - REGEX:          Regex match on message (and optionally stack_trace)
  - FREQUENCY:      Counts matches in a sliding window, triggers on threshold
  - STATE_SEQUENCE: Tracks runnable state transitions from runnable_history

Detection events are the input for B4 (Correlator) and B5 (Narrator).
"""

import re
import yaml
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path
from collections import defaultdict

# Import B1 types
from parser_b1 import LogRecord, LogLevel, FormatType


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Rule:
    """A single detection rule loaded from YAML."""
    rule_id: str
    name: str
    description: str
    pattern_type: str             # KEYWORD, REGEX, FREQUENCY, STATE_SEQUENCE
    severity: str                 # CRITICAL, HIGH, MEDIUM, LOW
    enabled: bool = True

    # Pattern matching
    pattern: str = ""             # substring (KEYWORD) or regex string (REGEX)
    _compiled_regex: object = field(default=None, repr=False)

    # Filters
    runnable_filter: list = field(default_factory=list)
    level_filter: list = field(default_factory=list)
    match_stack_trace: bool = False

    # State sequence (for STATE_SEQUENCE type)
    state_sequence: list = field(default_factory=list)
    window_minutes: int = 10
    no_followup_state: Optional[str] = None

    # Frequency escalation (optional, for KEYWORD/REGEX rules)
    frequency_escalation: Optional[dict] = None

    # Metadata
    tags: list = field(default_factory=list)
    suggested_action: str = ""

    def __post_init__(self):
        if self.pattern_type == "REGEX" and self.pattern:
            self._compiled_regex = re.compile(self.pattern, re.IGNORECASE | re.DOTALL)


@dataclass
class Detection:
    """A detection event — output of B3."""
    detection_id: str
    detected_at: datetime
    rule_id: str
    rule_name: str
    severity: str
    runnable: str
    trigger_log_ids: list         # line numbers of triggering records
    trigger_summary: str          # human-readable summary of what matched
    context_window: dict = field(default_factory=dict)
    suggested_action: str = ""
    tags: list = field(default_factory=list)

    # For frequency-based escalations
    match_count: int = 1
    escalated: bool = False
    original_severity: str = ""


# ─── Rule loader ──────────────────────────────────────────────────────────────

def load_rules(yaml_path: str) -> list:
    """Load rules from a YAML file and return a list of Rule objects."""
    path = Path(yaml_path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules = []
    for r in data.get("rules", []):
        rule = Rule(
            rule_id=r["rule_id"],
            name=r["name"],
            description=r.get("description", ""),
            pattern_type=r["pattern_type"],
            severity=r["severity"],
            enabled=r.get("enabled", True),
            pattern=r.get("pattern", ""),
            runnable_filter=r.get("runnable_filter", []),
            level_filter=r.get("level_filter", []),
            match_stack_trace=r.get("match_stack_trace", False),
            state_sequence=r.get("state_sequence", []),
            window_minutes=r.get("window_minutes", 10),
            no_followup_state=r.get("no_followup_state"),
            frequency_escalation=r.get("frequency_escalation"),
            tags=r.get("tags", []),
            suggested_action=r.get("suggested_action", ""),
        )
        rules.append(rule)

    return [r for r in rules if r.enabled]


# ─── Matchers ─────────────────────────────────────────────────────────────────

def _passes_filters(record: LogRecord, rule: Rule) -> bool:
    """Check if a record passes the runnable and level filters of a rule."""
    # Runnable filter (empty = match all)
    if rule.runnable_filter and record.runnable not in rule.runnable_filter:
        return False

    # Level filter (empty = match all)
    if rule.level_filter:
        level_names = [lf if isinstance(lf, str) else lf.value for lf in rule.level_filter]
        if record.level.value not in level_names:
            return False

    return True


def _match_keyword(record: LogRecord, rule: Rule) -> bool:
    """Match a KEYWORD rule against a record's message."""
    if rule.pattern in record.message:
        return True
    if rule.match_stack_trace and record.stack_trace and rule.pattern in record.stack_trace:
        return True
    return False


def _match_regex(record: LogRecord, rule: Rule) -> bool:
    """Match a REGEX rule against a record's message."""
    if rule._compiled_regex is None:
        return False
    if rule._compiled_regex.search(record.message):
        return True
    if rule.match_stack_trace and record.stack_trace:
        if rule._compiled_regex.search(record.stack_trace):
            return True
    return False


# ─── Frequency tracker ────────────────────────────────────────────────────────

class FrequencyTracker:
    """
    Tracks match counts per (rule_id, runnable) in sliding time windows.
    Used for frequency_escalation on KEYWORD/REGEX rules.
    """

    def __init__(self):
        # Key: (rule_id, runnable) → list of timestamps
        self._events: dict = defaultdict(list)

    def record_match(self, rule_id: str, runnable: str, timestamp: datetime):
        """Record a match event."""
        key = (rule_id, runnable)
        self._events[key].append(timestamp)

    def count_in_window(self, rule_id: str, runnable: str, timestamp: datetime,
                        window_minutes: int) -> int:
        """Count matches in the time window ending at timestamp."""
        key = (rule_id, runnable)
        cutoff = timestamp - timedelta(minutes=window_minutes)
        events = self._events[key]

        # Prune old events (keep memory bounded)
        self._events[key] = [t for t in events if t >= cutoff]

        return len(self._events[key])


# ─── State sequence tracker ──────────────────────────────────────────────────

class StateTracker:
    """
    Tracks runnable state transitions for STATE_SEQUENCE rules.
    Expects records from runnable_history.log (format RUNNABLE_HIST).
    """

    def __init__(self):
        # Key: runnable → list of (timestamp, state)
        self._history: dict = defaultdict(list)

    def record_state(self, runnable: str, timestamp: datetime, message: str):
        """Extract and record state transitions from runnable_history messages."""
        m = re.search(r'State change:\s*\w+\s*->\s*(\w+)', message)
        if m:
            new_state = m.group(1).rstrip(".")
            self._history[runnable].append((timestamp, new_state))

    def check_sequence(self, rule: Rule, runnable: str, current_time: datetime) -> Optional[list]:
        """
        Check if the required state sequence occurred within the time window.
        Returns the matching timestamps if found, None otherwise.
        """
        if not rule.state_sequence:
            return None

        history = self._history.get(runnable, [])
        if not history:
            return None

        window_start = current_time - timedelta(minutes=rule.window_minutes)

        # Filter to window
        windowed = [(t, s) for t, s in history if t >= window_start]
        states = [s for _, s in windowed]
        times = [t for t, _ in windowed]

        # Check if the sequence appears in order
        seq = rule.state_sequence
        seq_idx = 0
        match_times = []

        for i, state in enumerate(states):
            if state == seq[seq_idx]:
                match_times.append(times[i])
                seq_idx += 1
                if seq_idx >= len(seq):
                    return match_times  # Full sequence matched

        return None

    def check_no_followup(self, rule: Rule, runnable: str, current_time: datetime) -> Optional[list]:
        """
        For rules with no_followup_state: check if a state sequence occurred
        WITHOUT the expected followup state within the window.
        """
        if not rule.no_followup_state or len(rule.state_sequence) < 2:
            return None

        history = self._history.get(runnable, [])
        if not history:
            return None

        # Find the last occurrence of the final state in the sequence
        target_state = rule.state_sequence[-1]
        window_start = current_time - timedelta(minutes=rule.window_minutes)

        for t, s in reversed(history):
            if t < window_start:
                break
            if s == target_state:
                # Check if the expected followup appeared after this point
                followup_found = False
                for t2, s2 in history:
                    if t2 > t and s2 == rule.no_followup_state:
                        followup_found = True
                        break
                if not followup_found:
                    # Check enough time has passed
                    if (current_time - t).total_seconds() >= rule.window_minutes * 60:
                        return [t]
        return None


# ─── Detector B3 ──────────────────────────────────────────────────────────────

class Detector:
    """
    The B3 Detector engine. Processes normalized log records against a rule
    catalog and emits Detection events.

    Usage:
        detector = Detector("config/rules.yaml")
        for record in parse_file(...):
            detections = detector.process(record)
            for d in detections:
                print(d)
        # At end, check pending state-based detections
        final = detector.flush()
    """

    def __init__(self, rules_path: str):
        self.rules = load_rules(rules_path)
        self.frequency_tracker = FrequencyTracker()
        self.state_tracker = StateTracker()

        # Separate rules by type for efficient dispatch
        self._keyword_rules = [r for r in self.rules if r.pattern_type == "KEYWORD"]
        self._regex_rules = [r for r in self.rules if r.pattern_type == "REGEX"]
        self._state_rules = [r for r in self.rules if r.pattern_type == "STATE_SEQUENCE"]

        # Track last process time for flush
        self._last_time: Optional[datetime] = None

        # Stats
        self.stats = defaultdict(int)

    def process(self, record: LogRecord) -> list:
        """
        Process a single log record and return any Detection events triggered.
        """
        self._last_time = record.timestamp
        detections = []

        # ── State tracking (always, even if no state rules) ──
        if record.format_type == FormatType.RUNNABLE_HIST and "State change:" in record.message:
            self.state_tracker.record_state(
                record.runnable, record.timestamp, record.message
            )
            # Check state sequence rules
            for rule in self._state_rules:
                if rule.runnable_filter and record.runnable not in rule.runnable_filter:
                    continue

                match_times = self.state_tracker.check_sequence(
                    rule, record.runnable, record.timestamp
                )
                if match_times:
                    d = self._create_detection(
                        rule=rule,
                        record=record,
                        summary=f"State sequence {' → '.join(rule.state_sequence)} "
                                f"detected for {record.runnable} within {rule.window_minutes}min",
                    )
                    detections.append(d)
                    self.stats[rule.rule_id] += 1

        # ── KEYWORD rules ──
        for rule in self._keyword_rules:
            if not _passes_filters(record, rule):
                continue
            if _match_keyword(record, rule):
                d = self._create_detection(
                    rule=rule,
                    record=record,
                    summary=f"Keyword match: '{rule.pattern[:60]}' in {record.runnable}",
                )
                detections.append(d)
                self.stats[rule.rule_id] += 1

                # Frequency tracking
                self._check_frequency_escalation(d, rule, record)

        # ── REGEX rules ──
        for rule in self._regex_rules:
            if not _passes_filters(record, rule):
                continue
            if _match_regex(record, rule):
                d = self._create_detection(
                    rule=rule,
                    record=record,
                    summary=f"Regex match: rule {rule.rule_id} ({rule.name}) in {record.runnable}",
                )
                detections.append(d)
                self.stats[rule.rule_id] += 1

                # Frequency tracking
                self._check_frequency_escalation(d, rule, record)

        return detections

    def _check_frequency_escalation(self, detection: Detection, rule: Rule, record: LogRecord):
        """Check if frequency escalation threshold is exceeded."""
        if not rule.frequency_escalation:
            return

        esc = rule.frequency_escalation
        self.frequency_tracker.record_match(rule.rule_id, record.runnable, record.timestamp)
        count = self.frequency_tracker.count_in_window(
            rule.rule_id, record.runnable, record.timestamp, esc["window_minutes"]
        )
        detection.match_count = count

        if count >= esc["threshold"]:
            detection.original_severity = detection.severity
            detection.severity = esc["escalate_to"]
            detection.escalated = True
            detection.trigger_summary += (
                f" [ESCALATED: {count} matches in {esc['window_minutes']}min "
                f"→ {esc['escalate_to']}]"
            )

    def flush(self) -> list:
        """
        Check pending state-based rules (no_followup_state) at end of processing.
        Call this after all records have been processed.
        """
        detections = []
        if self._last_time is None:
            return detections

        for rule in self._state_rules:
            if not rule.no_followup_state:
                continue

            # Check all known runnables
            for runnable in list(self.state_tracker._history.keys()):
                if rule.runnable_filter and runnable not in rule.runnable_filter:
                    continue

                match_times = self.state_tracker.check_no_followup(
                    rule, runnable, self._last_time
                )
                if match_times:
                    d = Detection(
                        detection_id=str(uuid.uuid4())[:12],
                        detected_at=match_times[0],
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        runnable=runnable,
                        trigger_log_ids=[],
                        trigger_summary=(
                            f"State '{rule.state_sequence[-1]}' without followup "
                            f"'{rule.no_followup_state}' for {runnable} "
                            f"within {rule.window_minutes}min"
                        ),
                        suggested_action=rule.suggested_action,
                        tags=rule.tags,
                    )
                    detections.append(d)
                    self.stats[rule.rule_id] += 1

        return detections

    def _create_detection(self, rule: Rule, record: LogRecord, summary: str) -> Detection:
        """Create a Detection from a rule match."""
        return Detection(
            detection_id=str(uuid.uuid4())[:12],
            detected_at=record.timestamp,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            runnable=record.runnable,
            trigger_log_ids=[record.line_number],
            trigger_summary=summary,
            suggested_action=rule.suggested_action,
            tags=rule.tags,
        )

    def get_stats_summary(self) -> dict:
        """Return detection statistics."""
        return {
            "total_detections": sum(self.stats.values()),
            "by_rule": dict(self.stats),
            "rules_loaded": len(self.rules),
        }
