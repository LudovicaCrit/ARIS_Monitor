"""
ARIS Log Monitoring System — Correlator B4
=============================================
Aggregates B3 Detection events into correlated Incidents.

When multiple runnables fail in a short time window, the failures are often
causally related (e.g., zoo_s goes down → elastic_s fails → ces_s fails →
everything downstream crashes). B4 groups these into a single Incident with
a root cause estimate based on the ARIS dependency hierarchy.

Correlation logic:
  1. TIME WINDOW:  Detections within 5 minutes are candidates for grouping
  2. DEPENDENCY:   If runnable A depends on B, and both fail, B is likely root cause
  3. SAME RULE:    Repeated detections of the same rule on the same runnable merge
  4. ESCALATION:   Incident severity = max severity of its detections
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from collections import defaultdict

from detector_b3 import Detection


# ─── ARIS Dependency Hierarchy ────────────────────────────────────────────────
# Lower layer = more foundational. If a lower layer fails, higher layers cascade.
# Layer 0 = infrastructure, Layer 5 = edge services

DEPENDENCY_LAYERS = {
    "zoo_s":           0,  # ZooKeeper — everything depends on this
    "elastic_s":       1,  # Elasticsearch
    "cdf_s":           1,  # CDF
    "cloudsearch_s":   1,  # CloudSearch
    "ces_s":           2,  # Central Execution Service
    "umcadmin_s":      2,  # User Management
    "adsadmin_s":      3,  # ADS Admin
    "abs_s":           4,  # ABS (main app server)
    "arcm_s":          4,  # ARCM (Risk & Compliance)
    "copernicus_s":    4,  # Copernicus
    "ecp_s":           4,  # ECP
    "dashboarding_s":  4,  # Dashboarding
    "apg_s":           4,  # APG (deactivated)
    "octopus_s":       4,  # Octopus
    "simulation_s":    4,  # Simulation
    "loadbalancer_s":  5,  # Load Balancer (edge)
    "agent":           -1, # Agent itself — outside the hierarchy
}

# Direct dependencies: child → [parents]
# If a parent fails, the child is likely affected
DEPENDENCIES = {
    "elastic_s":       ["zoo_s"],
    "cdf_s":           ["zoo_s"],
    "cloudsearch_s":   ["zoo_s"],
    "ces_s":           ["elastic_s", "zoo_s"],
    "umcadmin_s":      ["ces_s", "zoo_s"],
    "adsadmin_s":      ["umcadmin_s"],
    "abs_s":           ["adsadmin_s", "umcadmin_s", "ces_s", "cloudsearch_s"],
    "arcm_s":          ["adsadmin_s", "umcadmin_s", "ces_s"],
    "copernicus_s":    ["abs_s", "ces_s"],
    "ecp_s":           ["abs_s"],
    "dashboarding_s":  ["abs_s", "adsadmin_s"],
    "apg_s":           ["abs_s"],
    "octopus_s":       ["abs_s"],
    "simulation_s":    ["abs_s"],
    "loadbalancer_s":  ["abs_s", "arcm_s", "copernicus_s"],
}


# ─── Severity ordering ───────────────────────────────────────────────────────

SEVERITY_ORDER = {
    "P1": 5,
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}


def _max_severity(*severities) -> str:
    """Return the highest severity from a list."""
    return max(severities, key=lambda s: SEVERITY_ORDER.get(s, 0))


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Incident:
    """A correlated incident — output of B4."""
    incident_id: str
    created_at: datetime
    updated_at: datetime
    severity: str
    status: str = "OPEN"  # OPEN, ACKNOWLEDGED, RESOLVED, FALSE_POSITIVE

    # Constituent detections
    detections: list = field(default_factory=list)
    detection_count: int = 0

    # Affected runnables
    affected_runnables: list = field(default_factory=list)

    # Root cause analysis
    root_cause_runnable: Optional[str] = None
    root_cause_rule: Optional[str] = None
    root_cause_estimate: str = ""

    # Timeline
    timeline: list = field(default_factory=list)  # [{timestamp, runnable, event}]

    # Rules involved
    rules_involved: list = field(default_factory=list)

    def add_detection(self, detection: Detection):
        """Add a detection to this incident."""
        self.detections.append(detection)
        self.detection_count = len(self.detections)
        self.updated_at = max(self.updated_at, detection.detected_at)
        self.severity = _max_severity(self.severity, detection.severity)

        if detection.runnable not in self.affected_runnables:
            self.affected_runnables.append(detection.runnable)

        if detection.rule_id not in self.rules_involved:
            self.rules_involved.append(detection.rule_id)

        self.timeline.append({
            "timestamp": detection.detected_at.isoformat(),
            "runnable": detection.runnable,
            "event": f"[{detection.rule_id}] {detection.trigger_summary[:80]}",
        })
        self.timeline.sort(key=lambda x: x["timestamp"])

    def estimate_root_cause(self):
        """
        Estimate root cause based on dependency hierarchy.
        The affected runnable at the lowest layer is most likely the root cause.
        """
        if not self.affected_runnables:
            return

        # Find the runnable at the lowest dependency layer
        runnables_with_layers = [
            (r, DEPENDENCY_LAYERS.get(r, 99))
            for r in self.affected_runnables
        ]
        runnables_with_layers.sort(key=lambda x: x[1])

        root_runnable = runnables_with_layers[0][0]
        root_layer = runnables_with_layers[0][1]
        self.root_cause_runnable = root_runnable

        # Find the first detection on the root runnable
        root_detections = [
            d for d in self.detections if d.runnable == root_runnable
        ]
        if root_detections:
            root_detections.sort(key=lambda d: d.detected_at)
            self.root_cause_rule = root_detections[0].rule_id

        # Generate explanation
        if len(self.affected_runnables) == 1:
            self.root_cause_estimate = (
                f"Problema isolato su {root_runnable}."
            )
        else:
            dependents = [
                r for r in self.affected_runnables
                if r != root_runnable
            ]
            # Check if dependents actually depend on root
            cascade_runnables = []
            for dep in dependents:
                parents = DEPENDENCIES.get(dep, [])
                if root_runnable in parents:
                    cascade_runnables.append(dep)

            if cascade_runnables:
                self.root_cause_estimate = (
                    f"Probabile causa root: {root_runnable} (layer {root_layer}). "
                    f"Effetto a cascata su: {', '.join(cascade_runnables)}. "
                    f"Risolvere prima {root_runnable}."
                )
            else:
                self.root_cause_estimate = (
                    f"Runnable coinvolti: {', '.join(self.affected_runnables)}. "
                    f"Runnable più basso nella gerarchia: {root_runnable} (layer {root_layer}). "
                    f"Verificare se i problemi sono correlati o indipendenti."
                )


# ─── Correlator engine ────────────────────────────────────────────────────────

class Correlator:
    """
    The B4 Correlator engine. Groups B3 detections into Incidents.

    Correlation strategy:
    - Time window: detections within `window_minutes` are candidates
    - Same runnable + same rule = merge into existing incident
    - Multiple runnables in same window = correlate if dependency exists
    - Each incident gets a root cause estimate

    Usage:
        correlator = Correlator(window_minutes=5)
        for detection in all_detections:
            incident = correlator.process(detection)
            if incident:  # new or updated incident
                print(incident)
        incidents = correlator.get_all_incidents()
    """

    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self._incidents: list = []
        self._active_window: dict = {}  # runnable → incident_id
        self._window_start: Optional[datetime] = None

        # Stats
        self.stats = {
            "detections_processed": 0,
            "incidents_created": 0,
            "detections_correlated": 0,
            "multi_runnable_incidents": 0,
        }

    def process(self, detection: Detection) -> Optional[Incident]:
        """
        Process a detection and return the Incident it belongs to
        (new or existing).
        """
        self.stats["detections_processed"] += 1

        # Check if we need to close the current window
        if self._window_start is not None:
            elapsed = (detection.detected_at - self._window_start).total_seconds() / 60
            if elapsed > self.window_minutes:
                self._close_window()

        # Start new window if needed
        if self._window_start is None:
            self._window_start = detection.detected_at

        # Try to find an existing incident to add to
        incident = self._find_matching_incident(detection)

        if incident is None:
            # Create new incident
            incident = Incident(
                incident_id=str(uuid.uuid4())[:12],
                created_at=detection.detected_at,
                updated_at=detection.detected_at,
                severity=detection.severity,
            )
            self._incidents.append(incident)
            self.stats["incidents_created"] += 1
        else:
            self.stats["detections_correlated"] += 1

        incident.add_detection(detection)
        self._active_window[detection.runnable] = incident.incident_id

        return incident

    def _find_matching_incident(self, detection: Detection) -> Optional[Incident]:
        """Find an existing incident that this detection should be added to."""
        # Strategy 1: Same runnable, same rule, within window
        for inc in reversed(self._incidents):
            if not self._in_window(inc, detection):
                continue

            # Same runnable + same rule = definitely merge
            if (detection.runnable in inc.affected_runnables and
                    detection.rule_id in inc.rules_involved):
                return inc

            # Same runnable + different rule but related severity = merge
            if detection.runnable in inc.affected_runnables:
                if SEVERITY_ORDER.get(detection.severity, 0) >= 3:  # HIGH+
                    return inc

        # Strategy 2: Different runnable but dependency-related within window
        for inc in reversed(self._incidents):
            if not self._in_window(inc, detection):
                continue

            if self._has_dependency(detection.runnable, inc.affected_runnables):
                # Only correlate if both are significant (HIGH+)
                if (SEVERITY_ORDER.get(detection.severity, 0) >= 3 and
                        SEVERITY_ORDER.get(inc.severity, 0) >= 3):
                    return inc

        return None

    def _in_window(self, incident: Incident, detection: Detection) -> bool:
        """Check if a detection falls within an incident's time window."""
        delta = abs((detection.detected_at - incident.updated_at).total_seconds())
        return delta <= self.window_minutes * 60

    def _has_dependency(self, runnable: str, other_runnables: list) -> bool:
        """Check if runnable has a dependency relationship with any of the others."""
        # Check if runnable depends on any of the others
        parents = DEPENDENCIES.get(runnable, [])
        for other in other_runnables:
            if other in parents:
                return True

        # Check if any of the others depend on runnable
        for other in other_runnables:
            other_parents = DEPENDENCIES.get(other, [])
            if runnable in other_parents:
                return True

        return False

    def _close_window(self):
        """Close the current window and finalize incidents."""
        self._active_window.clear()
        self._window_start = None

        # Estimate root causes for multi-runnable incidents
        for inc in self._incidents:
            if len(inc.affected_runnables) > 1:
                inc.estimate_root_cause()
                self.stats["multi_runnable_incidents"] += 1

    def flush(self):
        """Finalize all pending incidents. Call after processing all detections."""
        self._close_window()

        # Final root cause estimation for all incidents
        for inc in self._incidents:
            if not inc.root_cause_estimate:
                inc.estimate_root_cause()

    def get_all_incidents(self) -> list:
        """Return all incidents."""
        return self._incidents

    def get_open_incidents(self) -> list:
        """Return only OPEN incidents."""
        return [i for i in self._incidents if i.status == "OPEN"]

    def get_stats_summary(self) -> dict:
        """Return correlation statistics."""
        severity_dist = defaultdict(int)
        size_dist = defaultdict(int)
        for inc in self._incidents:
            severity_dist[inc.severity] += 1
            bucket = "1" if inc.detection_count == 1 else \
                     "2-5" if inc.detection_count <= 5 else \
                     "6-20" if inc.detection_count <= 20 else "20+"
            size_dist[bucket] += 1

        return {
            **self.stats,
            "total_incidents": len(self._incidents),
            "by_severity": dict(severity_dist),
            "by_size": dict(size_dist),
        }
