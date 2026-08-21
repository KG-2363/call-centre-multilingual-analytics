"""Stage functions. The CLI and the Airflow DAG both call these — there is no
orchestration logic inside the transforms themselves, which is what lets the same
code run under Airflow, in a test, or from a laptop.

Each stage is:
  * **idempotent** — re-running overwrites its own partition and nothing else,
  * **partition-scoped** — the unit of work is a `dt=YYYY-MM-DD` partition,
  * **independently replayable** — Silver rebuilds from Bronze without touching
    audio; Gold rebuilds from Silver without touching the LLM.

That last property is the one that matters in practice. Prompt changes are
frequent, ASR model changes are rare, and re-running a 6-hour ASR job because
someone edited a prompt is how teams stop iterating.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .asr.base import TranscriptionRequest, get_backend
from .config import Settings
from .gold.build import build_gold
from .ingestion.bronze import load_bronze_calls, partition_for, write_bronze, write_ground_truth
from .ingestion.corpus import generate_corpus
from .ingestion.synthesize import synthesize_call
from .logging_conf import get_logger
from .models.bronze import AsrSegment
from .models.enrichment import CallEnrichment
from .models.silver import ConversationLanguageProfile, Utterance
from .quality import asr_quality, dq
from .silver.build import build_utterances
from .storage import read_models, write_json, write_jsonl

log = get_logger(__name__)


@dataclass
class StageResult:
    stage: str
    rows: int = 0
    partitions: dict[str, int] = field(default_factory=dict)
    dq_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def stage_seed(settings: Settings, tts: str | None = None) -> StageResult:
    """Generate the synthetic corpus, optionally render audio, land Bronze."""
    settings.ensure_dirs()
    calls = generate_corpus(n_calls=settings.corpus_size, seed=settings.random_seed)

    backend = tts or settings.tts_backend
    audio_paths: dict[str, Path] = {}
    if backend != "none":
        log.info("rendering audio", extra={"backend": backend, "calls": len(calls)})
        for call in calls:
            path = synthesize_call(call, settings.audio_dir, backend)
            if path:
                audio_paths[call.metadata.call_id] = path

    counts = write_bronze(calls, settings, audio_paths)
    write_ground_truth(calls, settings)

    return StageResult(
        "seed",
        rows=len(calls),
        partitions=counts,
        notes=[f"audio_backend={backend}", f"audio_files={len(audio_paths)}"],
    )


def stage_transcribe(settings: Settings, partition: str | None = None) -> StageResult:
    """Bronze audio -> Bronze ASR segments -> Silver utterances + quality."""
    settings.ensure_dirs()
    calls = load_bronze_calls(settings, partition)
    if not calls:
        raise FileNotFoundError("no bronze calls found; run `ccma seed` first")

    backend = get_backend(settings)
    backend.warmup()
    calibration = asr_quality.CalibrationModel.load(
        settings.data_root / "models" / "asr_calibration.json"
    )

    segments_by_partition: dict[str, list[AsrSegment]] = defaultdict(list)
    utterances_by_partition: dict[str, list[Utterance]] = defaultdict(list)
    profiles_by_partition: dict[str, list[ConversationLanguageProfile]] = defaultdict(list)
    quality_by_partition: dict[str, list[Any]] = defaultdict(list)

    all_segments: list[AsrSegment] = []
    all_utterances: list[Utterance] = []

    for call in calls:
        part = partition_for(call)
        audio_path = (
            settings.data_root / call.audio_uri
            if call.audio_uri and (settings.data_root / call.audio_uri).exists()
            else None
        )
        segments = backend.transcribe(
            TranscriptionRequest(call=call, audio_path=audio_path)
        )
        utterances, profile = build_utterances(call, segments)
        quality = asr_quality.assess_call(call.call_id, segments, utterances, calibration)

        segments_by_partition[part].extend(segments)
        utterances_by_partition[part].extend(utterances)
        profiles_by_partition[part].append(profile)
        quality_by_partition[part].append(quality)
        all_segments.extend(segments)
        all_utterances.extend(utterances)

    counts: dict[str, int] = {}
    for part in sorted(utterances_by_partition):
        write_jsonl(settings.bronze_dir / part / "asr_segments.jsonl", segments_by_partition[part])
        counts[part] = write_jsonl(
            settings.silver_dir / part / "utterances.jsonl", utterances_by_partition[part]
        )
        write_jsonl(
            settings.silver_dir / part / "language_profile.jsonl", profiles_by_partition[part]
        )
        write_jsonl(settings.silver_dir / part / "call_quality.jsonl", quality_by_partition[part])

    bronze_report = dq.check_bronze(calls, all_segments)
    silver_report = dq.check_silver(all_utterances, calls)
    _persist_dq(settings, [bronze_report, silver_report])

    if not bronze_report.ok or not silver_report.ok:
        failures = bronze_report.blocking_failures + silver_report.blocking_failures
        raise RuntimeError(
            "blocking data-quality failures: "
            + "; ".join(f"{f.name} ({f.detail})" for f in failures)
        )

    return StageResult(
        "transcribe",
        rows=len(all_utterances),
        partitions=counts,
        dq_summary={"bronze": bronze_report.summary(), "silver": silver_report.summary()},
        notes=[f"asr_backend={backend.name}", f"segments={len(all_segments)}"],
    )


def stage_enrich(settings: Settings, partition: str | None = None) -> StageResult:
    """Silver utterances -> Silver enrichment via the LLM."""
    from .enrichment.base import get_llm_backend
    from .enrichment.runner import enrich_batch

    calls = {c.call_id: c for c in load_bronze_calls(settings, partition)}
    if not calls:
        raise FileNotFoundError("no bronze calls found; run `ccma seed` first")

    parts = (
        [partition]
        if partition
        else sorted(p.parent.name for p in settings.silver_dir.glob("dt=*/utterances.jsonl"))
    )
    if not parts:
        raise FileNotFoundError("no silver utterances found; run `ccma transcribe` first")

    backend = get_llm_backend(settings)
    counts: dict[str, int] = {}
    all_enrichments: list[CallEnrichment] = []

    for part in parts:
        utterances = read_models(settings.silver_dir / part / "utterances.jsonl", Utterance)
        profiles = {
            p.call_id: p
            for p in read_models(
                settings.silver_dir / part / "language_profile.jsonl",
                ConversationLanguageProfile,
            )
        }
        by_call: dict[str, list[Utterance]] = defaultdict(list)
        for utt in utterances:
            by_call[utt.call_id].append(utt)

        items = [
            (calls[cid], sorted(utts, key=lambda u: u.turn_index), profiles[cid])
            for cid, utts in by_call.items()
            if cid in calls and cid in profiles
        ]
        outcomes = enrich_batch(items, backend, settings)
        enrichments = [o.enrichment for o in outcomes]

        counts[part] = write_jsonl(
            settings.silver_dir / part / "enrichment.jsonl", enrichments
        )
        all_enrichments.extend(enrichments)
        log.info(
            "enriched partition",
            extra={
                "partition": part,
                "calls": len(enrichments),
                "heavy_model_used": sum(1 for o in outcomes if o.used_heavy_model),
                "repairs": sum(o.attempts - 1 for o in outcomes),
            },
        )

    report = dq.check_enrichment(all_enrichments, expected_calls=len(calls))
    _persist_dq(settings, [report])
    if not report.ok:
        raise RuntimeError(
            "blocking data-quality failures: "
            + "; ".join(f"{f.name} ({f.detail})" for f in report.blocking_failures)
        )

    return StageResult(
        "enrich",
        rows=len(all_enrichments),
        partitions=counts,
        dq_summary={"enrichment": report.summary()},
        notes=[f"llm_backend={backend.name}"],
    )


def stage_gold(settings: Settings) -> StageResult:
    """Silver -> Gold star schema, then score escalation risk."""
    counts = build_gold(settings)
    notes = _score_escalation(settings)
    return StageResult("gold", rows=sum(counts.values()), partitions=counts, notes=notes)


def _score_escalation(settings: Settings) -> list[str]:
    import duckdb

    from .gold import escalation

    model_path = settings.data_root / "models" / "escalation_model.json"
    con = duckdb.connect(str(settings.warehouse_path))
    try:
        df = con.execute(
            """
            SELECT f.*, r.prior_contacts_7d, r.subsequent_contacts_7d
            FROM fact_call f
            LEFT JOIN fact_repeat_contact r ON r.call_id = f.call_id
            """
        ).fetchdf()

        if df.empty:
            return ["escalation: no rows to score"]

        model = escalation.train(df, model_path)
        scores = escalation.predict(df, model)

        con.execute("DROP TABLE IF EXISTS fact_escalation_risk")
        con.register("escalation_scores", scores)
        con.execute("CREATE TABLE fact_escalation_risk AS SELECT * FROM escalation_scores")
        con.unregister("escalation_scores")

        flagged = int(scores["is_flagged"].sum())
        return [
            f"escalation_model={'gbm' if model.trained else 'rule_baseline'}",
            f"flagged_calls={flagged}",
            f"metrics={model.metrics}",
        ]
    finally:
        con.close()


def _persist_dq(settings: Settings, reports: list[dq.DqReport]) -> None:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json(
        settings.data_root / "dq" / f"{stamp}.json",
        {"generated_at": stamp, "reports": [r.summary() for r in reports]},
    )
    for report in reports:
        for warning in report.warnings:
            log.warning(
                "dq warning",
                extra={"layer": report.layer, "check": warning.name, "detail": warning.detail},
            )


def run_all(settings: Settings, tts: str | None = None) -> list[StageResult]:
    return [
        stage_seed(settings, tts),
        stage_transcribe(settings),
        stage_enrich(settings),
        stage_gold(settings),
    ]
