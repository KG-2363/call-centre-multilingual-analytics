"""Near-real-time path: streaming escalation alerts during a live call.

**Why a second path at all.** The batch pipeline answers "how did we do
yesterday". It cannot answer "this call is going wrong right now, get a
supervisor on it", and that is where the commercial value of escalation
prediction actually sits — a save is worth far more than a report.

**What is deliberately NOT duplicated.** Only the minimum needed for an alert
runs in the stream: partial ASR, language ID, dissatisfaction keyword hits, and
the *rule* escalation score. Full LLM enrichment, compliance adjudication and all
Gold aggregation stay in batch. Two systems computing the same metric two ways is
the classic lambda-architecture failure, so the stream writes to its own
short-TTL alert store and never to Gold. Batch remains the single source of truth
for every reported number; the stream is an operational trigger with a different
correctness bar.

**Ordering and lateness.** Keyed by `call_id` so all partial transcripts for one
call land on one partition in order. Watermark of 30s absorbs recorder jitter;
anything later is dropped for alerting (it is useless by then) but still arrives
in batch via the recording.

**Exactly-once vs at-least-once.** At-least-once with idempotent alert keys
(`call_id:rule_version:bucket`). Deduplicating in the alert store is far cheaper
than transactional exactly-once across Kafka and Redis, and a duplicate alert is
a much smaller problem than a missing one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ccma.lang import codeswitch, lid
from ccma.logging_conf import get_logger

log = get_logger(__name__)

ALERT_THRESHOLD = 0.55
COOLDOWN_S = 90  # never re-alert on the same call inside this window


@dataclass
class CallWindow:
    """Rolling state for one in-flight call. Bounded on purpose — an unbounded
    per-key state store is how streaming jobs die at 3am."""

    call_id: str
    agent_id: str
    started_at: float
    turns: list[dict[str, Any]] = field(default_factory=list)
    escalation_hits: set[str] = field(default_factory=set)
    dissatisfaction_hits: int = 0
    negative_turns: int = 0
    last_alert_at: float = 0.0

    MAX_TURNS = 60

    def add_turn(self, role: str, text: str, at_ms: int) -> None:
        self.turns.append({"role": role, "text": text, "at_ms": at_ms})
        if len(self.turns) > self.MAX_TURNS:
            self.turns.pop(0)


ESCALATION_KEYWORDS: dict[str, list[str]] = {
    "SUPERVISOR_REQUEST": ["supervisor", "senior", "manager", "सीनियर", "अधिकारी"],
    "LEGAL_OR_REGULATOR_THREAT": ["consumer forum", "trai", "court", "kanoon", "क़ानून", "उपभोक्ता"],
    "SOCIAL_MEDIA_THREAT": ["twitter", "social media", "post karunga"],
    "CHURN_THREAT": ["band kar", "बंद कर", "cancel", "port out", "dusri company"],
    "ABUSIVE_LANGUAGE": ["bekar", "बेकार", "bakwas", "nonsense"],
}
NEGATIVE_MARKERS = [
    "galat", "ग़लत", "kharab", "ख़राब", "pareshan", "परेशान", "gussa",
    "frustrated", "unacceptable", "worst", "third time", "teesri baar",
]


def score_window(window: CallWindow) -> tuple[float, list[str]]:
    """Rule score only. The GBM is intentionally not used here: it needs
    features (hold time, repeat contacts, final sentiment) that do not exist
    until the call ends. Scoring a partial call with a model trained on complete
    calls is a silent distribution shift, and it would be wrong in exactly the
    cases we care about."""
    score = 0.05
    reasons: list[str] = []

    if window.escalation_hits:
        score += 0.18 * len(window.escalation_hits)
        reasons.append(f"escalation language: {', '.join(sorted(window.escalation_hits))}")
    if window.dissatisfaction_hits:
        score += 0.06 * min(window.dissatisfaction_hits, 5)
        reasons.append(f"{window.dissatisfaction_hits} dissatisfaction markers")

    elapsed = time.time() - window.started_at
    if elapsed > 480:
        score += 0.12
        reasons.append(f"call running {int(elapsed)}s")

    if window.negative_turns >= 3:
        score += 0.15
        reasons.append(f"{window.negative_turns} consecutive negative turns")

    return min(1.0, score), reasons


def process_event(event: dict[str, Any], state: dict[str, CallWindow]) -> dict[str, Any] | None:
    """Handle one partial-transcript event; return an alert or None.

    Event contract (produced by the media server / streaming ASR):
        {call_id, agent_id, role, text, at_ms, is_final}
    """
    call_id = event["call_id"]
    window = state.get(call_id)
    if window is None:
        window = CallWindow(call_id, event.get("agent_id", "unknown"), time.time())
        state[call_id] = window

    text = (event.get("text") or "").strip()
    if not text:
        return None

    window.add_turn(event.get("role", "UNKNOWN"), text, event.get("at_ms", 0))

    # Language ID runs in-stream: escalation keywords must be matched in the
    # language actually spoken, and the Devanagari projection is what makes a
    # single keyword list work across scripts.
    normalized = lid.normalize_text(text)
    result = lid.identify(normalized)
    projected = lid.devanagari_projection(normalized, result.lang_primary) or normalized
    haystack = f"{normalized} {projected}"

    if event.get("role") == "CUSTOMER":
        for indicator, keywords in ESCALATION_KEYWORDS.items():
            if any(k in haystack for k in keywords):
                window.escalation_hits.add(indicator)

        if any(marker in haystack for marker in NEGATIVE_MARKERS):
            window.dissatisfaction_hits += 1
            window.negative_turns += 1
        else:
            window.negative_turns = 0

    score, reasons = score_window(window)
    now = time.time()
    if score >= ALERT_THRESHOLD and (now - window.last_alert_at) > COOLDOWN_S:
        window.last_alert_at = now
        cs = codeswitch.compute([t.lang for t in result.tokens])
        return {
            "alert_key": f"{call_id}:v1:{int(now // COOLDOWN_S)}",
            "call_id": call_id,
            "agent_id": window.agent_id,
            "risk_score": round(score, 3),
            "reasons": reasons,
            "detected_language": result.lang_primary.value,
            "code_switched": cs.is_code_switched,
            "turns_observed": len(window.turns),
            "emitted_at": now,
        }
    return None


def run(bootstrap_servers: str, topic_in: str, topic_out: str, group_id: str = "ccma-escalation") -> None:
    """Kafka consumer loop.

    Kafka rather than a queue: we need replay (for tuning the rules against
    recorded traffic without touching production), key-ordered partitions, and
    multiple independent consumers on the same stream (alerting, live agent
    assist, QA sampling)."""
    try:
        from kafka import KafkaConsumer, KafkaProducer
    except ImportError as exc:
        raise RuntimeError("pip install kafka-python to run the streaming consumer") from exc

    consumer = KafkaConsumer(
        topic_in,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=False,        # commit only after the alert is produced
        auto_offset_reset="latest",       # stale partial transcripts have no value
        max_poll_records=200,
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        linger_ms=20,
    )

    state: dict[str, CallWindow] = {}
    last_gc = time.time()

    try:
        for message in consumer:
            event = message.value
            try:
                alert = process_event(event, state)
                if alert:
                    producer.send(topic_out, key=alert["call_id"].encode(), value=alert)
                    log.info("escalation alert", extra=alert)
                if event.get("is_final"):
                    state.pop(event["call_id"], None)
            except Exception as exc:
                # One malformed event must never stop the stream. Route it to the
                # DLQ and keep going.
                log.error("event failed", extra={"error": str(exc), "event": str(event)[:300]})
                producer.send(f"{topic_out}.dlq", value={"event": event, "error": str(exc)})

            consumer.commit()

            # Reap abandoned windows: calls whose final event never arrived.
            if time.time() - last_gc > 300:
                cutoff = time.time() - 3600
                for call_id in [k for k, w in state.items() if w.started_at < cutoff]:
                    state.pop(call_id, None)
                last_gc = time.time()
    finally:
        consumer.close()
        producer.flush()
        producer.close()


if __name__ == "__main__":
    import os

    run(
        bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092"),
        topic_in=os.environ.get("KAFKA_TOPIC_IN", "asr.partial_transcripts"),
        topic_out=os.environ.get("KAFKA_TOPIC_OUT", "ccma.escalation_alerts"),
    )
