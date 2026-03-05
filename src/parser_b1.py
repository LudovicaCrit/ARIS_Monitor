"""
ARIS Log Monitoring System — Parser B1
=======================================
Normalizes all ARIS log formats into a unified schema.

Supports 7 format families:
  1. WRAPPER       — Tanuki Java Service Wrapper (ARISCloudAgent.log)
  2. PIPE_8FIELD   — ARIS application logs (arcm.log, zkc.log, system.out of arcm_s, copernicus_publishing*.log)
  3. SPRING        — Spring Boot structured (system.out of abs_s, ces_s, dashboarding_s, ecp_s)
  4. RUNNABLE_HIST — Runnable state machine (runnable_history.log)
  5. AGENT         — ARIS Agent logs (agent.log, agent_error.log)
  6. REST_OPS      — Agent REST API operations (agent_rest_operations.log)
  7. PLAIN_TS      — Timestamped but unstructured (JVM bootstrap lines)

Lines without timestamps (stack traces, BUS- messages, banners, SMTP debug)
are aggregated to the preceding structured record.
"""

import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Iterator
from pathlib import Path


# ─── Enums ───────────────────────────────────────────────────────────────────

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"
    UNKNOWN = "UNKNOWN"


class LogFamily(Enum):
    SYSTEM_OUT = "SYSTEM_OUT"
    AGENT_LOG = "AGENT_LOG"
    AGENT_ERROR = "AGENT_ERROR"
    AGENT_LOGIN = "AGENT_LOGIN"
    WRAPPER_LOG = "WRAPPER_LOG"
    RUNNABLE_HISTORY = "RUNNABLE_HISTORY"
    REST_OPS = "REST_OPS"
    ARCM_APP = "ARCM_APP"          # arcm.log, arcm-error.log
    ARCM_ZKC = "ARCM_ZKC"          # zkc.log
    COPERNICUS_PUB = "COPERNICUS_PUB"  # copernicus_publishing*.log
    UNKNOWN = "UNKNOWN"


class FormatType(Enum):
    WRAPPER = "WRAPPER"
    PIPE_8FIELD = "PIPE_8FIELD"
    SPRING = "SPRING"
    RUNNABLE_HIST = "RUNNABLE_HIST"
    AGENT = "AGENT"
    REST_OPS = "REST_OPS"
    PLAIN_TS = "PLAIN_TS"
    CONTINUATION = "CONTINUATION"
    UNSTRUCTURED = "UNSTRUCTURED"


# ─── Output record ───────────────────────────────────────────────────────────

@dataclass
class LogRecord:
    """Unified normalized log record — output of B1."""
    timestamp: datetime
    level: LogLevel
    instance_id: Optional[str] = None
    tenant: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    thread: Optional[str] = None
    logger: Optional[str] = None
    message: str = ""
    stack_trace: Optional[str] = None
    runnable: str = "unknown"
    log_family: LogFamily = LogFamily.UNKNOWN
    source_file: str = ""
    source_path: str = ""
    collected_at: Optional[datetime] = None
    line_number: int = 0
    format_type: FormatType = FormatType.UNSTRUCTURED
    has_stack_trace: bool = False
    message_hash: str = ""
    _raw_lines: list = field(default_factory=list, repr=False)

    def finalize(self):
        """Compute derived fields after aggregation is complete."""
        self.has_stack_trace = bool(self.stack_trace)
        self.message_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        normalized = self._normalize_message(self.message)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _normalize_message(msg: str) -> str:
        """Remove variable parts for semantic grouping."""
        s = msg.strip()
        # UUIDs
        s = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<UUID>', s, flags=re.I)
        # Hex IDs (8+ chars)
        s = re.sub(r'\b[0-9a-f]{8,}\b', '<HEX>', s, flags=re.I)
        # IP addresses
        s = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', s)
        # Numbers (standalone, not in identifiers)
        s = re.sub(r'(?<![A-Za-z_])\d+(?![A-Za-z_])', '<N>', s)
        # Timestamps in messages
        s = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.,]?\d*Z?', '<TS>', s)
        return s


# ─── Timezone handling ───────────────────────────────────────────────────────

# ARIS server is in Europe/Paris (CET = UTC+1, CEST = UTC+2).
# We use a simple rule: CET for Nov-Mar, CEST for Apr-Oct.
# For production, use pytz or zoneinfo.
CET = timedelta(hours=1)
CEST = timedelta(hours=2)


def _is_summer_time(dt: datetime) -> bool:
    """Approximate CET/CEST boundary (last Sunday of March/October)."""
    m = dt.month
    if m < 3 or m > 10:
        return False
    if 4 <= m <= 9:
        return True
    # March or October: approximate
    if m == 3:
        last_sun = 31 - ((datetime(dt.year, 3, 31).weekday() + 1) % 7)
        return dt.day >= last_sun
    else:  # October
        last_sun = 31 - ((datetime(dt.year, 10, 31).weekday() + 1) % 7)
        return dt.day < last_sun


def _to_utc(dt: datetime, has_z_suffix: bool = False) -> datetime:
    """Convert a naive datetime to UTC."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    if has_z_suffix:
        return dt.replace(tzinfo=timezone.utc)
    offset = CEST if _is_summer_time(dt) else CET
    return (dt - offset).replace(tzinfo=timezone.utc)


# ─── Regex patterns ─────────────────────────────────────────────────────────

# Priority 1: WRAPPER (Tanuki prefix wrapping agent log)
RE_WRAPPER = re.compile(
    r'^(INFO|ERROR|WARN|STATUS|DEBUG)\s+\|\s*jvm\s+\d+\s+\|\s*\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+\|\s*(.*)',
    re.DOTALL
)

# Priority 2: PIPE_8FIELD (ARIS application logs)
RE_PIPE8 = re.compile(
    r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2},\d{3})'  # timestamp
    r'\|(\w+)\s*'                                        # level
    r'\|([^|]*)'                                         # instance_id
    r'\|([^|]*)'                                         # tenant
    r'\|([^|]*)'                                         # session_id
    r'\|([^|]*)'                                         # request_id
    r'\|([^|]*)'                                         # thread
    r'\|(.*)',                                            # logger + message
    re.DOTALL
)

# Priority 3: SPRING (Spring Boot structured)
RE_SPRING = re.compile(
    r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2},\d{3})'   # timestamp
    r'\s+-\s+'
    r'(INFO|WARN|ERROR|FATAL|DEBUG|TRACE)\s+'            # level
    r'\[([^\]]+)\]'                                       # [thread:logger@line]
    r'\s+-\s+'
    r'(.*)',                                              # message
    re.DOTALL
)

# Priority 4: RUNNABLE_HISTORY
RE_HIST = re.compile(
    r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})'  # timestamp
    r'\|(\w+)\s+-\s+'                                     # runnable
    r'(.*)',                                               # message
    re.DOTALL
)

# Priority 5: AGENT (4 pipe-delimited fields)
RE_AGENT = re.compile(
    r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})'  # timestamp
    r'\|(\w+)\s*'                                         # level
    r'\|([^|]*)'                                          # thread
    r'\|\s*(.*)',                                          # logger + message
    re.DOTALL
)

# Priority 6: REST_OPS
RE_REST = re.compile(
    r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})'  # start_ts
    r'\s+--\s+'
    r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})'   # end_ts
    r'\s+\((\d+)ms\)'                                      # duration
    r'\s+-\s+([^\s]+)'                                     # version
    r'\s+-\s+([^\s]+)'                                     # ip
    r'\s+-\s+(\w+)'                                        # user
    r'\s+-\s+(.*)',                                         # command
    re.DOTALL
)

# Priority 7: PLAIN_TS (timestamp only, no structure)
RE_PLAIN_Z = re.compile(
    r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{2,3}Z)\s+(.*)',
    re.DOTALL
)
RE_PLAIN = re.compile(
    r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2},\d{3})\s+(.*)',
    re.DOTALL
)

# Continuation detection
RE_STACKTRACE = re.compile(r'^\s+(at\s|\.\.\.)')
RE_CAUSED_BY = re.compile(r'^\s*Caused by:\s')
RE_EXCEPTION_HEADER = re.compile(r'^[A-Za-z][\w.]*(?:Exception|Error|Throwable)[\s:]')


# ─── Timestamp parsing ──────────────────────────────────────────────────────

def _parse_ts_iso_comma(s: str) -> datetime:
    """Parse '2025-11-05T09:14:50,641'"""
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S,%f")

def _parse_ts_iso_dot_z(s: str) -> datetime:
    """Parse '2025-11-05T09:14:55.813Z' or '2025-11-10T10:56:33.96Z'"""
    s = s.rstrip("Z")
    # Pad fractional seconds to 3 digits
    parts = s.split(".")
    if len(parts) == 2:
        parts[1] = parts[1].ljust(3, "0")
        s = ".".join(parts)
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")

def _parse_ts_space_comma(s: str) -> datetime:
    """Parse '2026-02-25 15:34:44,164'"""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")

def _parse_ts_space_dot(s: str) -> datetime:
    """Parse '2024-09-06 00:56:48.955'"""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


# ─── Noise filters (B0) ─────────────────────────────────────────────────────

NOISE_PATTERNS = [
    re.compile(r'Reusing existing ZKC instance'),
    re.compile(r'Accepted login for user Clous'),
]

def is_noise(line: str) -> bool:
    """Return True if the line should be dropped pre-parsing."""
    for p in NOISE_PATTERNS:
        if p.search(line):
            return True
    return False


# ─── File → log_family / runnable mapping ────────────────────────────────────

# Map instance_id prefix → runnable name
INSTANCE_TO_RUNNABLE = {
    "arcm": "arcm_s",
    "abs": "abs_s",
    "copernicus": "copernicus_s",
    "umcadmin": "umcadmin_s",
    "adsadmin": "adsadmin_s",
    "ces": "ces_s",
    "ecp": "ecp_s",
    "dashboarding": "dashboarding_s",
    "octopus": "octopus_s",
    "simulation": "simulation_s",
    "cloudsearch": "cloudsearch_s",
    "elastic": "elastic_s",
    "zoo": "zoo_s",
    "cdf": "cdf_s",
    "apg": "apg_s",
    "loadbalancer": "loadbalancer_s",
}


def detect_log_family(filename: str, source_path: str = "") -> LogFamily:
    """Determine the log family from filename and path."""
    fn = filename.lower()
    if fn == "ariscloudagent.log" or "wrapperlog" in fn.lower():
        return LogFamily.WRAPPER_LOG
    if fn == "runnable_history.log":
        return LogFamily.RUNNABLE_HISTORY
    if fn == "agent_rest_operations.log":
        return LogFamily.REST_OPS
    if fn == "agent_login.log":
        return LogFamily.AGENT_LOGIN
    if fn == "agent_error.log" or fn == "agent.error.log":
        return LogFamily.AGENT_ERROR
    if fn.startswith("agent"):
        return LogFamily.AGENT_LOG
    if fn == "zkc.log":
        return LogFamily.ARCM_ZKC
    if fn in ("arcm.log", "arcm-error.log"):
        return LogFamily.ARCM_APP
    if fn.startswith("copernicus_publishing") or fn.startswith("copernicus.publishing"):
        return LogFamily.COPERNICUS_PUB
    if fn == "system.out.log" or fn.startswith("system_out"):
        return LogFamily.SYSTEM_OUT
    return LogFamily.UNKNOWN


def detect_runnable(source_path: str, instance_id: str = "", thread: str = "") -> str:
    """Derive runnable from path, instance_id, or thread name."""
    # 1. From path: runnable_X_s/... or work_X_s/...
    m = re.search(r'(?:runnable_|work_)(\w+_s)', source_path)
    if m:
        return m.group(1)

    # 2. From instance_id: arcm0000000000 → arcm_s
    if instance_id:
        prefix = re.match(r'^([a-z]+)', instance_id)
        if prefix and prefix.group(1) in INSTANCE_TO_RUNNABLE:
            return INSTANCE_TO_RUNNABLE[prefix.group(1)]

    # 3. From thread name: "Loadbalancer loadbalancer_s Application Scan Thread"
    m = re.search(r'\b(\w+_s)\b', thread)
    if m:
        return m.group(1)

    # 4. If in agent/ path
    if "agent" in source_path.lower():
        return "agent"

    return "unknown"


# ─── Core parser ─────────────────────────────────────────────────────────────

def _split_logger_message(text: str) -> tuple:
    """Split 'LoggerName - actual message' into (logger, message)."""
    parts = text.split(" - ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", text.strip()


def _parse_line(line: str) -> Optional[LogRecord]:
    """
    Try to parse a single line into a LogRecord.
    Returns None if the line is a continuation/unstructured.
    """
    line = line.rstrip("\r\n")
    if not line.strip():
        return None

    # P1: WRAPPER — unwrap and re-parse inner content
    m = RE_WRAPPER.match(line)
    if m:
        inner = m.group(2)
        record = _parse_line(inner)
        if record:
            record.format_type = FormatType.WRAPPER
            return record
        # If inner doesn't parse, treat whole thing as plain
        return None

    # P2: PIPE_8FIELD
    m = RE_PIPE8.match(line)
    if m:
        ts = _to_utc(_parse_ts_iso_comma(m.group(1)))
        level = LogLevel[m.group(2).strip()] if m.group(2).strip() in LogLevel.__members__ else LogLevel.UNKNOWN
        logger, message = _split_logger_message(m.group(8))
        return LogRecord(
            timestamp=ts,
            level=level,
            instance_id=m.group(3).strip() or None,
            tenant=m.group(4).strip() or None,
            session_id=m.group(5).strip() or None,
            request_id=m.group(6).strip() or None,
            thread=m.group(7).strip() or None,
            logger=logger or None,
            message=message,
            format_type=FormatType.PIPE_8FIELD,
        )

    # P3: SPRING
    m = RE_SPRING.match(line)
    if m:
        ts = _to_utc(_parse_ts_iso_comma(m.group(1)))
        level_str = m.group(2).strip()
        level = LogLevel[level_str] if level_str in LogLevel.__members__ else LogLevel.UNKNOWN
        thread_logger = m.group(3)
        thread, logger = thread_logger, ""
        if ":" in thread_logger:
            thread, logger = thread_logger.split(":", 1)
            # Remove @linenum
            logger = re.sub(r'@\d+$', '', logger)
        return LogRecord(
            timestamp=ts,
            level=level,
            thread=thread.strip() or None,
            logger=logger.strip() or None,
            message=m.group(4).strip(),
            format_type=FormatType.SPRING,
        )

    # P4: RUNNABLE_HISTORY
    m = RE_HIST.match(line)
    if m:
        ts = _to_utc(_parse_ts_space_comma(m.group(1)))
        runnable = m.group(2).strip()
        message = m.group(3).strip()
        # Extract level from state transitions
        level = LogLevel.INFO
        if "FAILED" in message or "INOPERATIVE" in message:
            level = LogLevel.ERROR
        elif "DOWN" in message:
            level = LogLevel.WARN
        record = LogRecord(
            timestamp=ts,
            level=level,
            runnable=runnable,  # parsed directly from line, not from path
            message=message,
            format_type=FormatType.RUNNABLE_HIST,
        )
        record._runnable_from_content = True  # flag: don't override in enrichment
        return record

    # P5: AGENT
    m = RE_AGENT.match(line)
    if m:
        ts = _to_utc(_parse_ts_space_comma(m.group(1)))
        level_str = m.group(2).strip()
        level = LogLevel[level_str] if level_str in LogLevel.__members__ else LogLevel.UNKNOWN
        thread = m.group(3).strip()
        logger, message = _split_logger_message(m.group(4))
        return LogRecord(
            timestamp=ts,
            level=level,
            thread=thread or None,
            logger=logger or None,
            message=message,
            format_type=FormatType.AGENT,
        )

    # P6: REST_OPS
    m = RE_REST.match(line)
    if m:
        ts = _to_utc(_parse_ts_space_dot(m.group(1)))
        end_ts = _to_utc(_parse_ts_space_dot(m.group(2)))
        duration_ms = int(m.group(3))
        command = m.group(7).strip()
        level = LogLevel.INFO
        if "FAILED" in command.upper():
            level = LogLevel.ERROR
        return LogRecord(
            timestamp=ts,
            level=level,
            thread=f"{m.group(5).strip()}@{m.group(6).strip()}",  # ip@user
            logger="REST_OPS",
            message=f"[{duration_ms}ms] {command}",
            format_type=FormatType.REST_OPS,
        )

    # P7: PLAIN_TS (with Z suffix = UTC)
    m = RE_PLAIN_Z.match(line)
    if m:
        ts = _to_utc(_parse_ts_iso_dot_z(m.group(1)), has_z_suffix=True)
        return LogRecord(
            timestamp=ts,
            level=LogLevel.UNKNOWN,
            message=m.group(2).strip(),
            format_type=FormatType.PLAIN_TS,
        )

    # P7b: PLAIN_TS (without Z = CET)
    m = RE_PLAIN.match(line)
    if m:
        ts = _to_utc(_parse_ts_iso_comma(m.group(1)))
        return LogRecord(
            timestamp=ts,
            level=LogLevel.UNKNOWN,
            message=m.group(2).strip(),
            format_type=FormatType.PLAIN_TS,
        )

    # Not a structured line
    return None


def _is_continuation(line: str) -> bool:
    """Check if a line is a stack trace or continuation."""
    stripped = line.rstrip("\r\n")
    if not stripped.strip():
        return True  # blank lines between stack frames
    if RE_STACKTRACE.match(stripped):
        return True
    if RE_CAUSED_BY.match(stripped):
        return True
    if RE_EXCEPTION_HEADER.match(stripped):
        return True
    return False


def _classify_unstructured(line: str) -> str:
    """Tag unstructured lines for aggregation."""
    s = line.strip()
    if s.startswith("BUS-"):
        return "BUS"
    if any(s.startswith(p) for p in ("DEBUG", "EHLO", "250", "220", "STARTTLS", "RCPT", "AUTH", "MAIL FROM", "354", "DATA")):
        return "SMTP"
    if s.startswith(("Date:", "From:", "Reply-To:", "To:", "Message-ID:", "Subject:", "MIME-", "Content-", "---")):
        return "EMAIL"
    if re.match(r'^[\s_/\\|()U\"\'>*=]+$', s) or s.startswith("ARIS version") or s.startswith("Spring boot"):
        return "BANNER"
    return "OTHER"


# ─── File parser (with multiline aggregation) ───────────────────────────────

def parse_file(
    filepath: str,
    source_path: str = "",
    runnable_hint: str = "",
    apply_noise_filter: bool = True,
    collected_at: Optional[datetime] = None,
) -> Iterator[LogRecord]:
    """
    Parse a log file and yield normalized LogRecords.

    Args:
        filepath: Local path to the log file.
        source_path: Original path on the ARIS server (for runnable detection).
        runnable_hint: Override runnable if known.
        apply_noise_filter: Whether to drop noise lines (B0 filter).
        collected_at: Timestamp of collection.
    """
    path = Path(filepath)
    filename = path.name
    log_family = detect_log_family(filename, source_path)

    if collected_at is None:
        collected_at = datetime.now(timezone.utc)

    current: Optional[LogRecord] = None
    line_number = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line_number += 1
            line = raw_line.rstrip("\r\n")

            # B0: noise filter
            if apply_noise_filter and is_noise(line):
                continue

            # Try to parse as structured
            record = _parse_line(line)

            if record is not None:
                # Yield previous record
                if current is not None:
                    current.finalize()
                    yield current

                # Enrich new record
                record.source_file = filename
                record.source_path = source_path or filepath
                record.log_family = log_family
                record.line_number = line_number
                record.collected_at = collected_at
                record._raw_lines = [line]

                # Runnable detection
                if runnable_hint:
                    record.runnable = runnable_hint
                elif getattr(record, '_runnable_from_content', False):
                    pass  # keep runnable parsed from log content (e.g. RUNNABLE_HIST)
                else:
                    record.runnable = detect_runnable(
                        source_path or filepath,
                        record.instance_id or "",
                        record.thread or "",
                    )

                current = record
            else:
                # Continuation / unstructured → aggregate to current
                if current is not None:
                    current._raw_lines.append(line)

                    if _is_continuation(line):
                        # Stack trace line
                        if current.stack_trace is None:
                            current.stack_trace = line
                        else:
                            current.stack_trace += "\n" + line
                    else:
                        tag = _classify_unstructured(line)
                        if tag == "BANNER":
                            pass  # drop
                        elif tag in ("SMTP", "EMAIL"):
                            # Don't pollute main message, just tag
                            pass
                        else:
                            # BUS messages and others: append to message
                            if line.strip():
                                current.message += "\n" + line.strip()

    # Yield last record
    if current is not None:
        current.finalize()
        yield current


# ─── Convenience: parse with stats ───────────────────────────────────────────

@dataclass
class ParseStats:
    """Statistics from parsing a file."""
    total_lines: int = 0
    noise_filtered: int = 0
    records_emitted: int = 0
    by_format: dict = field(default_factory=dict)
    by_level: dict = field(default_factory=dict)
    errors_and_above: int = 0
    has_stack_trace_count: int = 0


def parse_file_with_stats(filepath: str, **kwargs) -> tuple:
    """Parse a file and return (list[LogRecord], ParseStats)."""
    records = []
    stats = ParseStats()

    # Count total and noise lines
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            stats.total_lines += 1
            if kwargs.get("apply_noise_filter", True) and is_noise(raw_line):
                stats.noise_filtered += 1

    for record in parse_file(filepath, **kwargs):
        records.append(record)
        stats.records_emitted += 1
        fmt = record.format_type.value
        stats.by_format[fmt] = stats.by_format.get(fmt, 0) + 1
        lvl = record.level.value
        stats.by_level[lvl] = stats.by_level.get(lvl, 0) + 1
        if record.level in (LogLevel.ERROR, LogLevel.FATAL):
            stats.errors_and_above += 1
        if record.has_stack_trace:
            stats.has_stack_trace_count += 1

    return records, stats
