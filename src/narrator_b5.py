"""
ARIS Log Monitoring System — Narrator B5
==========================================
Generates human-readable alert narratives from B3 Detection events.

Two modes:
  1. LLM mode:      Uses Claude API to generate contextual narratives in Italian
  2. Template mode:  Falls back to static templates when API is unavailable

The Narrator enriches detections with:
  - A human-readable title
  - A narrative (3-5 sentences explaining what's happening, the risk, and the action)
  - Suggested actions from the rule catalog
  - Historical context (optional, from B2 if available)
  - Confidence score
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

from detector_b3 import Detection


# ─── Output structure ─────────────────────────────────────────────────────────

@dataclass
class Alert:
    """A human-readable alert — output of B5."""
    alert_id: str
    incident_id: Optional[str] = None  # Set by B4 if correlated
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: str = ""
    title: str = ""
    narrative: str = ""
    suggested_action: str = ""
    historical_context: str = ""
    confidence: float = 0.8
    notification_channels: list = field(default_factory=list)

    # Source detection(s)
    detection_ids: list = field(default_factory=list)
    rule_id: str = ""
    runnable: str = ""
    match_count: int = 1
    escalated: bool = False


# ─── Static templates (fallback) ─────────────────────────────────────────────

TEMPLATES = {
    "C01": {
        "title": "Pool di connessioni DB esaurito — {runnable}",
        "narrative": (
            "Il pool di connessioni al database del runnable {runnable} è saturo. "
            "Sono state rilevate {match_count} occorrenze di 'No free database connection' "
            "a partire dalle {time}. "
            "Questo indica che le 50 connessioni disponibili sono tutte occupate, "
            "probabilmente a causa di query lente o bloccate. "
            "Se non risolto, il servizio diventerà non disponibile."
        ),
    },
    "C02": {
        "title": "Login SQL Server fallito — {runnable}",
        "narrative": (
            "Il runnable {runnable} non riesce a connettersi al database ARIS10DB. "
            "L'errore 'Cannot open database' indica che il database è offline, "
            "le credenziali non sono valide, o il server SQL non è raggiungibile. "
            "Il servizio è completamente bloccato finché la connessione non viene ripristinata."
        ),
    },
    "C03": {
        "title": "Migrazione schema fallita — {runnable} in restart loop",
        "narrative": (
            "Il runnable {runnable} non riesce ad avviarsi perché lo schema del database "
            "non corrisponde alla versione attesa dal software. "
            "L'errore 'Cannot calculate migration' causerà un ciclo infinito di restart. "
            "NON riavviare manualmente — il problema persisterà. "
            "È necessario un intervento manuale sullo schema del database."
        ),
    },
    "C04": {
        "title": "Runnable INOPERATIVE — {runnable}",
        "narrative": (
            "Il runnable {runnable} è terminato in stato INOPERATIVE alle {time}. "
            "Questo è lo stato terminale: il runnable ha tentato di avviarsi "
            "ma ha riscontrato un errore fatale che impedisce il funzionamento. "
            "Verificare i log di boot (system.out.log) per identificare la causa root."
        ),
    },
    "C05": {
        "title": "Crash durante shutdown — {runnable}",
        "narrative": (
            "Il runnable {runnable} ha subito un errore BeanCreationNotAllowedException "
            "durante la fase di shutdown alle {time}. "
            "Questo è un bug noto di Spring Boot che si verifica quando lo shutdown "
            "tenta di creare nuovi bean. Il runnable si riavvierà automaticamente. "
            "Se il pattern si ripete frequentemente, investigare la causa dello shutdown iniziale."
        ),
    },
    "H01": {
        "title": "NullPointerException in JobMonitoring — {runnable}",
        "narrative": (
            "Il runnable {runnable} ha generato {match_count} NullPointerException "
            "nel componente JobMonitoringTag. "
            "Questo errore è un noto precursore di crash completi di arcm_s. "
            "Se si ripete più di 3 volte in un'ora, il rischio di crash è elevato. "
            "Monitorare attentamente nelle prossime ore."
        ),
    },
    "H02": {
        "title": "Errori massivi di caricamento dati — {runnable}",
        "narrative": (
            "Il data layer di {runnable} ha generato {match_count} errori BUS-10507 "
            "(cannot load object). "
            "In volumi elevati, questo indica problemi di integrità dati o "
            "inconsistenza dello schema. "
            "Verificare i log SQL per query fallite correlate."
        ),
    },
    "H03": {
        "title": "Errori di salvataggio dati — {runnable}",
        "narrative": (
            "Il data layer di {runnable} non riesce a salvare oggetti. "
            "Rilevate {match_count} occorrenze di BUS-10201. "
            "Possibili cause: lock sul database, pool connessioni saturo, "
            "o vincoli di integrità violati. "
            "Verificare correlazione con il pool di connessioni."
        ),
    },
    "H04": {
        "title": "Autenticazione BulkExport fallita — {runnable}",
        "narrative": (
            "L'utente di servizio BulkExport non riesce ad autenticarsi su {runnable}. "
            "Verificare le credenziali nella configurazione del runnable "
            "e la validità dell'account su SQL Server."
        ),
    },
    "S01": {
        "title": "RESTART LOOP — {runnable} in ciclo di crash",
        "narrative": (
            "Il runnable {runnable} è intrappolato in un ciclo di restart. "
            "La sequenza STARTING → DOWN si è ripetuta almeno 3 volte in {window}. "
            "Ogni tentativo di avvio fallisce con lo stesso errore. "
            "NON riavviare manualmente — il problema si ripeterà. "
            "Verificare i FATAL nel system.out.log per la causa root "
            "(tipicamente: migrazione schema fallita o database non raggiungibile)."
        ),
    },
    "S02": {
        "title": "Runnable crashato senza recovery — {runnable}",
        "narrative": (
            "Il runnable {runnable} è passato in stato DOWN e non si è riavviato "
            "automaticamente entro 5 minuti. "
            "Verificare che l'agent sia attivo e che il runnable sia configurato "
            "per l'auto-restart."
        ),
    },
    "S03": {
        "title": "Shutdown lento — {runnable}",
        "narrative": (
            "Il runnable {runnable} è in fase di STOPPING da più di 3 minuti. "
            "Possibili cause: thread bloccati, connessioni non chiuse, o job ancora in esecuzione. "
            "Se supera i 5 minuti, considerare un kill forzato."
        ),
    },
}

# Default template for rules without a specific one
DEFAULT_TEMPLATE = {
    "title": "Alert {severity} — {rule_name} su {runnable}",
    "narrative": (
        "Rilevata anomalia su {runnable}: {summary}. "
        "Severità: {severity}. Regola: {rule_id} ({rule_name}). "
        "{suggested_action}"
    ),
}


# ─── Template narrator ───────────────────────────────────────────────────────

def _format_time(dt: datetime) -> str:
    """Format datetime for Italian narratives."""
    return dt.strftime("%H:%M del %d/%m/%Y")


def narrate_with_template(detection: Detection) -> Alert:
    """Generate an alert using static templates."""
    template = TEMPLATES.get(detection.rule_id, DEFAULT_TEMPLATE)

    # Build context for formatting
    ctx = {
        "runnable": detection.runnable,
        "time": _format_time(detection.detected_at),
        "match_count": detection.match_count,
        "severity": detection.severity,
        "rule_id": detection.rule_id,
        "rule_name": detection.rule_name,
        "summary": detection.trigger_summary,
        "suggested_action": detection.suggested_action,
        "window": f"{detection.context_window.get('window_minutes', '?')} minuti"
                  if detection.context_window else "finestra temporale",
    }

    title = template["title"].format(**ctx)
    narrative = template["narrative"].format(**ctx)

    # Determine notification channels based on severity
    channels = []
    if detection.severity in ("CRITICAL", "P1"):
        channels = ["email", "sms", "teams"]
    elif detection.severity == "HIGH":
        channels = ["email", "teams"]
    elif detection.severity == "MEDIUM":
        channels = ["email"]
    else:
        channels = ["log"]

    return Alert(
        alert_id=detection.detection_id,
        generated_at=datetime.now(timezone.utc),
        severity=detection.severity,
        title=title,
        narrative=narrative,
        suggested_action=detection.suggested_action,
        confidence=0.9,  # templates are deterministic
        notification_channels=channels,
        detection_ids=[detection.detection_id],
        rule_id=detection.rule_id,
        runnable=detection.runnable,
        match_count=detection.match_count,
        escalated=detection.escalated,
    )


# ─── LLM narrator ────────────────────────────────────────────────────────────

LLM_SYSTEM_PROMPT = """Sei un operatore senior di ARIS (Software AG) con 10 anni di esperienza.
Il tuo compito è generare alert chiari e azionabili per il team operativo.

Regole:
- Scrivi in italiano
- Massimo 5 frasi
- Struttura: (1) cosa sta succedendo, (2) qual è il rischio, (3) cosa fare
- Sii specifico: includi nomi di runnable, timestamp, conteggi
- Non usare gergo tecnico inutile ma non semplificare troppo
- Se l'errore è noto e ha una soluzione, indicala
- Se l'errore è un falso positivo noto, dillo chiaramente

Rispondi SOLO con un oggetto JSON con questa struttura (niente altro testo):
{
  "title": "titolo breve dell'alert",
  "narrative": "narrativa di 3-5 frasi",
  "confidence": 0.0-1.0
}"""


def _build_llm_prompt(detection: Detection, historical_context: str = "") -> str:
    """Build the user prompt for the LLM."""
    prompt = f"""Genera un alert per questa detection:

Regola: {detection.rule_id} — {detection.rule_name}
Severità: {detection.severity}
Runnable: {detection.runnable}
Timestamp: {_format_time(detection.detected_at)}
Descrizione: {detection.trigger_summary}
Occorrenze: {detection.match_count}
Escalato: {"Sì" if detection.escalated else "No"}
Tags: {', '.join(detection.tags)}

Azione suggerita dal catalogo:
{detection.suggested_action}
"""
    if historical_context:
        prompt += f"\nContesto storico:\n{historical_context}\n"

    return prompt


def narrate_with_llm(
    detection: Detection,
    historical_context: str = "",
    api_key: Optional[str] = None,
) -> Optional[Alert]:
    """
    Generate an alert using Claude API.
    Returns None if API call fails (caller should fall back to template).
    """
    try:
        import httpx
    except ImportError:
        return None

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None

    prompt = _build_llm_prompt(detection, historical_context)

    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "system": LLM_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15.0,
        )

        if response.status_code != 200:
            return None

        data = response.json()
        text = data["content"][0]["text"]

        # Parse JSON response
        result = json.loads(text)

        channels = []
        if detection.severity in ("CRITICAL", "P1"):
            channels = ["email", "sms", "teams"]
        elif detection.severity == "HIGH":
            channels = ["email", "teams"]
        elif detection.severity == "MEDIUM":
            channels = ["email"]
        else:
            channels = ["log"]

        return Alert(
            alert_id=detection.detection_id,
            generated_at=datetime.now(timezone.utc),
            severity=detection.severity,
            title=result.get("title", f"Alert {detection.rule_id}"),
            narrative=result.get("narrative", ""),
            suggested_action=detection.suggested_action,
            historical_context=historical_context,
            confidence=result.get("confidence", 0.7),
            notification_channels=channels,
            detection_ids=[detection.detection_id],
            rule_id=detection.rule_id,
            runnable=detection.runnable,
            match_count=detection.match_count,
            escalated=detection.escalated,
        )

    except Exception:
        return None


# ─── Narrator engine ─────────────────────────────────────────────────────────

class Narrator:
    """
    The B5 Narrator engine. Converts B3 detections into human-readable alerts.

    Usage:
        narrator = Narrator(use_llm=True)  # or False for template-only
        alert = narrator.narrate(detection)

    Deduplication: the narrator suppresses repeated alerts for the same rule+runnable
    within a configurable cooldown window.
    """

    def __init__(
        self,
        use_llm: bool = False,
        api_key: Optional[str] = None,
        cooldown_minutes: int = 15,
    ):
        self.use_llm = use_llm
        self.api_key = api_key
        self.cooldown_minutes = cooldown_minutes

        # Dedup: (rule_id, runnable) → last alert timestamp
        self._last_alert: dict = {}

        # Stats
        self.stats = {
            "total_narrated": 0,
            "llm_generated": 0,
            "template_generated": 0,
            "suppressed_by_cooldown": 0,
            "by_severity": defaultdict(int),
        }

    def narrate(
        self,
        detection: Detection,
        historical_context: str = "",
        force: bool = False,
    ) -> Optional[Alert]:
        """
        Generate an alert from a detection.
        Returns None if suppressed by cooldown.
        """
        # Cooldown deduplication
        if not force:
            key = (detection.rule_id, detection.runnable)
            last = self._last_alert.get(key)
            if last is not None:
                elapsed = (detection.detected_at - last).total_seconds() / 60
                if elapsed < self.cooldown_minutes:
                    self.stats["suppressed_by_cooldown"] += 1
                    return None

        # Try LLM first, fall back to template
        alert = None
        if self.use_llm:
            alert = narrate_with_llm(detection, historical_context, self.api_key)
            if alert:
                self.stats["llm_generated"] += 1

        if alert is None:
            alert = narrate_with_template(detection)
            self.stats["template_generated"] += 1

        # Update cooldown tracker
        self._last_alert[(detection.rule_id, detection.runnable)] = detection.detected_at

        # Update stats
        self.stats["total_narrated"] += 1
        self.stats["by_severity"][alert.severity] += 1

        return alert

    def narrate_batch(
        self,
        detections: list,
        historical_context: str = "",
    ) -> list:
        """Narrate a list of detections, applying cooldown dedup."""
        alerts = []
        for d in detections:
            alert = self.narrate(d, historical_context)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def get_stats_summary(self) -> dict:
        return {
            **self.stats,
            "by_severity": dict(self.stats["by_severity"]),
        }
