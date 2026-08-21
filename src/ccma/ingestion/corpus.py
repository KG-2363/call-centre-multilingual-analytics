"""Deterministic corpus generation: 15 templates -> N calls with ground truth.

Determinism is a hard requirement, not a convenience. The same seed must produce
byte-identical Bronze so that (a) tests can assert on concrete values, (b) a
model change can be A/B'd against a fixed input, and (c) the ground-truth labels
stay aligned with the audio across regenerations.

Variation injected per call: agent, timestamp, hold/transfer behaviour, queue
wait, speaking-rate jitter, and an acoustic-difficulty rating that the mock ASR
backend later converts into realistic recognition errors.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..models.bronze import CallMetadata
from ..models.enums import CallDirection, ComplianceRule, ResolutionStatus, SpeakerRole
from .templates import TEMPLATES, CallTemplate, TurnTemplate

AGENTS: list[tuple[str, str, str]] = [
    ("AGT-1001", "Rohit Sharma", "billing_hi"),
    ("AGT-1002", "Priya Nair", "tech_hi"),
    ("AGT-1003", "Anjali Menon", "billing_en"),
    ("AGT-1004", "Imran Qureshi", "fraud"),
    ("AGT-1005", "Deepak Rao", "sales"),
    ("AGT-1006", "Sneha Patil", "complaint"),
    ("AGT-1007", "Vikram Singh", "retention"),
    ("AGT-1008", "Meera Iyer", "account"),
    ("AGT-1009", "Arjun Das", "logistics"),
    ("AGT-1010", "Fatima Sheikh", "info_hi"),
]

CITIES: list[tuple[str, str]] = [
    ("Bengaluru", "Karnataka"),
    ("Hyderabad", "Telangana"),
    ("Delhi", "Delhi"),
    ("Lucknow", "Uttar Pradesh"),
    ("Jaipur", "Rajasthan"),
    ("Patna", "Bihar"),
    ("Pune", "Maharashtra"),
    ("Indore", "Madhya Pradesh"),
]

# Acoustic conditions we simulate. `difficulty` drives the mock ASR error rate
# and mirrors what actually degrades Whisper on Indian call audio: 8 kHz codec
# bandwidth, background noise, and heavy regional accent.
ACOUSTIC_PROFILES: list[tuple[str, float]] = [
    ("clean_studio", 0.05),
    ("mobile_indoor", 0.12),
    ("mobile_street_noise", 0.28),
    ("landline_8k_codec", 0.22),
    ("heavy_accent_regional", 0.30),
]


@dataclass
class GroundTruthTurn:
    turn_index: int
    role: SpeakerRole
    text: str
    lang: str
    start_ms: int
    end_ms: int
    overlaps_next: bool = False


@dataclass
class GeneratedCall:
    metadata: CallMetadata
    turns: list[GroundTruthTurn]
    template: CallTemplate
    acoustic_profile: str
    difficulty: float
    tts_voices: dict[str, str] = field(default_factory=dict)

    def ground_truth(self) -> dict:
        """Labels used by the evaluation harness. Never read by the pipeline
        itself — keeping this separate is what stops label leakage."""
        return {
            "call_id": self.metadata.call_id,
            "template_id": self.template.template_id,
            "language_style": self.template.language_style,
            "acoustic_profile": self.acoustic_profile,
            "difficulty": self.difficulty,
            "intent_l1": self.template.intent_l1,
            "intent_l2": self.template.intent_l2,
            "intent_l3": self.template.intent_l3,
            "resolution_status": self.template.resolution.value,
            "escalated": self.template.escalated,
            "fcr": self.template.fcr,
            "compliance_present": [r.value for r in self.template.compliance_present],
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "role": t.role.value,
                    "text": t.text,
                    "lang": t.lang,
                    "start_ms": t.start_ms,
                    "end_ms": t.end_ms,
                }
                for t in self.turns
            ],
        }


def _hash_customer(seed: str) -> str:
    return hashlib.sha256(f"ccma-demo-salt::{seed}".encode()).hexdigest()


def _lay_out_turns(
    turn_templates: list[TurnTemplate], rng: random.Random
) -> tuple[list[GroundTruthTurn], int]:
    """Assign wall-clock spans, injecting pauses and overlaps.

    Overlaps are generated deliberately: `overlaps_next` turns start *before*
    the previous turn ends. This is the single most common cause of diarization
    error on real call audio, and a pipeline that never sees it in test will
    fail silently in production."""
    turns: list[GroundTruthTurn] = []
    cursor = 0
    for idx, tt in enumerate(turn_templates):
        # Speaking-rate jitter: +/-20% around the template's nominal duration.
        duration = max(600, int(tt.ms * rng.uniform(0.8, 1.2)))
        start = cursor
        end = start + duration
        turns.append(
            GroundTruthTurn(idx, tt.role, tt.text, tt.lang, start, end, tt.overlaps_next)
        )
        if tt.overlaps_next:
            cursor = end - rng.randint(300, 900)  # next speaker cuts in
        else:
            cursor = end + rng.randint(150, 1200)  # natural inter-turn gap
    total = turns[-1].end_ms if turns else 0
    return turns, total


def generate_corpus(
    n_calls: int = 30, seed: int = 20260731, tenant_id: str = "demo-tenant"
) -> list[GeneratedCall]:
    rng = random.Random(seed)
    batch_id = f"batch-{seed}"
    base_time = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)
    ingest_time = datetime(2026, 7, 31, 2, 0, tzinfo=timezone.utc)

    calls: list[GeneratedCall] = []
    for i in range(n_calls):
        template = TEMPLATES[i % len(TEMPLATES)]
        call_id = f"CALL-{seed}-{i:04d}"

        turns, duration_ms = _lay_out_turns(list(template.turns), rng)

        agent_id, agent_name, _ = next(
            (a for a in AGENTS if a[2] == template.queue),
            AGENTS[i % len(AGENTS)],
        )
        # Every third call is handled by a rotating agent so scorecards have
        # more than one agent per queue to compare.
        if i % 3 == 0:
            agent_id, agent_name, _ = AGENTS[(i // 3) % len(AGENTS)]

        city, state = CITIES[i % len(CITIES)]
        profile_name, difficulty = ACOUSTIC_PROFILES[
            rng.randrange(len(ACOUSTIC_PROFILES))
        ]

        escalating = template.resolution in (
            ResolutionStatus.ESCALATED,
            ResolutionStatus.UNRESOLVED,
        )
        hold_count = rng.randint(1, 3) if escalating else rng.randint(0, 1)
        hold_total = hold_count * rng.randint(20_000, 95_000)

        recorded_at = base_time + timedelta(
            days=i % 10, hours=rng.randint(0, 9), minutes=rng.randint(0, 59)
        )

        metadata = CallMetadata(
            call_id=call_id,
            tenant_id=tenant_id,
            audio_uri=f"audio/{call_id}.wav",
            audio_sha256=None,
            duration_ms=duration_ms,
            recorded_at=recorded_at,
            ingested_at=ingest_time,
            ingest_batch_id=batch_id,
            agent_id=agent_id,
            agent_name=agent_name,
            queue=template.queue,
            campaign="inbound_care" if i % 4 else "outbound_retention",
            direction=CallDirection.OUTBOUND
            if template.intent_l1 in ("SALES", "RETENTION") and i % 4 == 0
            else CallDirection.INBOUND,
            customer_id_hash=_hash_customer(f"cust-{i % 22}"),
            customer_city=city,
            customer_state=state,
            hold_count=hold_count,
            hold_total_ms=hold_total,
            transfer_count=1 if template.escalated else 0,
            queue_wait_ms=rng.randint(2_000, 180_000),
            disconnect_party="CUSTOMER" if escalating else "AGENT",
            source_system="synthetic-template-bank",
        )

        calls.append(
            GeneratedCall(
                metadata=metadata,
                turns=turns,
                template=template,
                acoustic_profile=profile_name,
                difficulty=difficulty,
            )
        )
    return calls


def compliance_expected(template: CallTemplate) -> set[ComplianceRule]:
    return set(template.compliance_present)
