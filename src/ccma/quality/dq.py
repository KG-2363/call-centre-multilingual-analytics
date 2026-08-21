"""Data-quality checks between layers.

Modelled on Great Expectations semantics but dependency-free so checks can run
inside the same process as the transform. Every check declares a severity:

* **BLOCKING** — fails the task; the partition is not published. Reserved for
  violations that would corrupt downstream aggregates (duplicate keys, negative
  durations, referential breaks).
* **WARNING** — publishes, but raises an alert. Distribution drift, elevated
  quarantine rates, unusual language mix.
* **INFO** — recorded for trend analysis only.

The distinction matters operationally: a pipeline that blocks on every anomaly
gets its checks disabled within a month. Block on corruption, alert on drift.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..models.bronze import AsrSegment, CallMetadata
from ..models.enrichment import CallEnrichment
from ..models.enums import ValidationStatus
from ..models.silver import Utterance


class Severity(str, Enum):
    BLOCKING = "BLOCKING"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: Severity
    observed: Any = None
    expected: str = ""
    detail: str = ""


@dataclass
class DqReport:
    layer: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def blocking_failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == Severity.BLOCKING]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.blocking_failures

    def summary(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "blocking_failures": len(self.blocking_failures),
            "warnings": len(self.warnings),
            "failed_checks": [
                {"name": r.name, "severity": r.severity.value, "detail": r.detail}
                for r in self.results
                if not r.passed
            ],
        }


def check_bronze(calls: list[CallMetadata], segments: list[AsrSegment]) -> DqReport:
    report = DqReport("bronze")
    add = report.results.append

    call_ids = [c.call_id for c in calls]
    duplicates = [cid for cid, n in Counter(call_ids).items() if n > 1]
    add(
        CheckResult(
            "bronze.call_id_unique", not duplicates, Severity.BLOCKING,
            len(duplicates), "0 duplicates",
            f"duplicate call_ids: {duplicates[:5]}" if duplicates else "",
        )
    )

    bad_duration = [c.call_id for c in calls if c.duration_ms <= 0]
    add(
        CheckResult(
            "bronze.duration_positive", not bad_duration, Severity.BLOCKING,
            len(bad_duration), "all > 0",
            f"non-positive duration: {bad_duration[:5]}" if bad_duration else "",
        )
    )

    # A raw phone number in a hash column means the PII redaction upstream broke.
    unhashed = [c.call_id for c in calls if len(c.customer_id_hash) != 64]
    add(
        CheckResult(
            "bronze.customer_id_hashed", not unhashed, Severity.BLOCKING,
            len(unhashed), "sha256 hex (64 chars)",
            f"unhashed customer ids: {unhashed[:5]}" if unhashed else "",
        )
    )

    known = set(call_ids)
    orphans = {s.call_id for s in segments} - known
    add(
        CheckResult(
            "bronze.segments_reference_calls", not orphans, Severity.BLOCKING,
            len(orphans), "0 orphans",
            f"segments for unknown calls: {sorted(orphans)[:5]}" if orphans else "",
        )
    )

    inverted = [s.segment_id for s in segments if s.end_ms < s.start_ms]
    add(
        CheckResult(
            "bronze.segment_timestamps_ordered", not inverted, Severity.BLOCKING,
            len(inverted), "end_ms >= start_ms",
            f"inverted segments: {inverted[:5]}" if inverted else "",
        )
    )

    silent = [c.call_id for c in calls if c.call_id not in {s.call_id for s in segments}]
    silent_ratio = len(silent) / max(1, len(calls))
    add(
        CheckResult(
            "bronze.calls_have_transcript", silent_ratio < 0.05, Severity.WARNING,
            round(silent_ratio, 4), "< 5% of calls with no segments",
            f"{len(silent)} calls produced no ASR output",
        )
    )
    return report


def check_silver(
    utterances: list[Utterance], calls: list[CallMetadata], min_lang_confidence: float = 0.5
) -> DqReport:
    report = DqReport("silver")
    add = report.results.append

    by_call: dict[str, list[Utterance]] = {}
    for utt in utterances:
        by_call.setdefault(utt.call_id, []).append(utt)

    bad_turn_index = [
        cid
        for cid, utts in by_call.items()
        if sorted(u.turn_index for u in utts) != list(range(len(utts)))
    ]
    add(
        CheckResult(
            "silver.turn_index_contiguous", not bad_turn_index, Severity.BLOCKING,
            len(bad_turn_index), "0..n-1 per call",
            f"non-contiguous turn_index: {bad_turn_index[:5]}" if bad_turn_index else "",
        )
    )

    out_of_range = [
        u.call_id for u in utterances if not (0.0 <= u.asr_confidence <= 1.0)
    ]
    add(
        CheckResult(
            "silver.confidence_in_range", not out_of_range, Severity.BLOCKING,
            len(out_of_range), "0.0 <= c <= 1.0", "",
        )
    )

    unknown_ratio = (
        sum(1 for u in utterances if u.lang_primary.value == "und") / max(1, len(utterances))
    )
    add(
        CheckResult(
            "silver.language_identified", unknown_ratio < 0.10, Severity.WARNING,
            round(unknown_ratio, 4), "< 10% unknown language",
            f"{unknown_ratio:.1%} of utterances have no language label",
        )
    )

    low_lang_conf = (
        sum(1 for u in utterances if u.lang_confidence < min_lang_confidence)
        / max(1, len(utterances))
    )
    add(
        CheckResult(
            "silver.language_confidence", low_lang_conf < 0.25, Severity.WARNING,
            round(low_lang_conf, 4), "< 25% below threshold",
            f"{low_lang_conf:.1%} of utterances have low-confidence LID",
        )
    )

    # Both roles must be present — a call transcribed as one speaker means
    # diarization or channel mapping failed, and every agent metric derived from
    # it will be wrong.
    single_role = [
        cid for cid, utts in by_call.items() if len({u.speaker_role for u in utts}) < 2
    ]
    single_ratio = len(single_role) / max(1, len(by_call))
    add(
        CheckResult(
            "silver.both_speakers_present", single_ratio < 0.10, Severity.WARNING,
            round(single_ratio, 4), "< 10% single-speaker calls",
            f"single-speaker calls: {single_role[:5]}" if single_role else "",
        )
    )

    covered = len(by_call) / max(1, len(calls))
    add(
        CheckResult(
            "silver.call_coverage", covered > 0.90, Severity.WARNING,
            round(covered, 4), "> 90% of bronze calls present in silver",
            f"only {covered:.1%} of calls have utterances",
        )
    )

    hallucination_ratio = sum(
        1 for u in utterances if "hallucination_suspect" in u.quality_flags
    ) / max(1, len(utterances))
    add(
        CheckResult(
            "silver.hallucination_rate", hallucination_ratio < 0.08, Severity.WARNING,
            round(hallucination_ratio, 4), "< 8%",
            f"{hallucination_ratio:.1%} of utterances flagged as possible hallucinations",
        )
    )
    return report


def check_enrichment(
    enrichments: list[CallEnrichment], expected_calls: int, max_quarantine_rate: float = 0.05
) -> DqReport:
    report = DqReport("enrichment")
    add = report.results.append

    if not enrichments:
        add(CheckResult("enrichment.non_empty", False, Severity.BLOCKING, 0, "> 0 rows"))
        return report

    quarantined = [
        e for e in enrichments if e.validation_status == ValidationStatus.QUARANTINED
    ]
    rate = len(quarantined) / len(enrichments)
    add(
        CheckResult(
            "enrichment.quarantine_rate", rate <= max_quarantine_rate, Severity.WARNING,
            round(rate, 4), f"<= {max_quarantine_rate:.0%}",
            f"{len(quarantined)} of {len(enrichments)} calls quarantined",
        )
    )

    coverage = len(enrichments) / max(1, expected_calls)
    add(
        CheckResult(
            "enrichment.coverage", coverage > 0.95, Severity.BLOCKING,
            round(coverage, 4), "> 95% of silver calls enriched",
            f"only {coverage:.1%} of calls enriched",
        )
    )

    ids = [e.call_id for e in enrichments]
    duplicates = [cid for cid, n in Counter(ids).items() if n > 1]
    add(
        CheckResult(
            "enrichment.one_row_per_call", not duplicates, Severity.BLOCKING,
            len(duplicates), "0 duplicates",
            f"duplicate enrichment rows: {duplicates[:5]}" if duplicates else "",
        )
    )

    # A model that classifies everything as OTHER is broken, not cautious.
    other_rate = sum(1 for e in enrichments if e.primary_intent.l1 == "OTHER") / len(enrichments)
    add(
        CheckResult(
            "enrichment.intent_specificity", other_rate < 0.20, Severity.WARNING,
            round(other_rate, 4), "< 20% OTHER",
            f"{other_rate:.1%} of calls classified as OTHER/UNCLASSIFIED",
        )
    )

    # Confidence collapsed to a single value means calibration is gone.
    confidences = [e.primary_intent.confidence for e in enrichments]
    spread = max(confidences) - min(confidences)
    add(
        CheckResult(
            "enrichment.confidence_spread", spread > 0.05, Severity.WARNING,
            round(spread, 4), "> 0.05",
            "intent confidence has collapsed to a near-constant value",
        )
    )

    disputed = sum(
        1 for e in enrichments for c in e.compliance if c.detector_disagreement
    )
    add(
        CheckResult(
            "enrichment.compliance_agreement",
            disputed / max(1, len(enrichments)) < 1.0,
            Severity.INFO,
            disputed,
            "tracked for review queue sizing",
            f"{disputed} compliance assessments need human review",
        )
    )
    return report
