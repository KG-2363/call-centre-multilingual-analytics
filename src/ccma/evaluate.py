"""Evaluation harness: pipeline output vs. the held-out ground truth.

Runs against `data/eval/ground_truth.jsonl`, which no pipeline stage can read.
It reports the metric set defined in docs/04-evaluation-framework.md, and it also
**fits the ASR calibration model** — the labelled corpus is precisely the "small
human-labelled seed set" the reference-free quality estimator needs.

Metrics are reported *sliced by language mode and acoustic profile*, never as a
single aggregate. A headline WER of 0.22 hides the only fact that matters: it is
probably 0.09 on English and 0.40 on Hinglish, and those two numbers demand
completely different responses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .logging_conf import get_logger
from .models.bronze import AsrSegment
from .models.enrichment import CallEnrichment
from .models.silver import Utterance
from .quality import asr_quality
from .quality.text_metrics import error_rates
from .storage import read_jsonl, read_models, write_json

log = get_logger(__name__)


@dataclass
class Slice:
    n: int = 0
    values: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, metric: str, value: float) -> None:
        self.values[metric].append(value)

    def mean(self, metric: str) -> float | None:
        vals = self.values.get(metric, [])
        return round(sum(vals) / len(vals), 4) if vals else None

    def summary(self) -> dict[str, Any]:
        return {"n": self.n, **{k: self.mean(k) for k in sorted(self.values)}}


def _load_ground_truth(settings: Settings) -> dict[str, dict]:
    path = settings.data_root / "eval" / "ground_truth.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `ccma seed` first")
    return {row["call_id"]: row for row in read_jsonl(path)}


def _load_silver(settings: Settings) -> tuple[dict[str, list[Utterance]], dict[str, list[AsrSegment]], dict[str, CallEnrichment]]:
    utterances: dict[str, list[Utterance]] = defaultdict(list)
    for path in sorted(settings.silver_dir.glob("dt=*/utterances.jsonl")):
        for utt in read_models(path, Utterance):
            utterances[utt.call_id].append(utt)
    for utts in utterances.values():
        utts.sort(key=lambda u: u.turn_index)

    segments: dict[str, list[AsrSegment]] = defaultdict(list)
    for path in sorted(settings.bronze_dir.glob("dt=*/asr_segments.jsonl")):
        for seg in read_models(path, AsrSegment):
            segments[seg.call_id].append(seg)

    enrichment: dict[str, CallEnrichment] = {}
    for path in sorted(settings.silver_dir.glob("dt=*/enrichment.jsonl")):
        for row in read_models(path, CallEnrichment):
            enrichment[row.call_id] = row

    return utterances, segments, enrichment


def evaluate_asr(
    ground_truth: dict[str, dict], utterances: dict[str, list[Utterance]]
) -> dict[str, Any]:
    """Call-level WER by concatenation.

    Concatenated rather than turn-aligned because ASR segment boundaries do not
    match true turn boundaries — forcing an alignment would measure segmentation
    error as if it were recognition error. Concatenating both sides and scoring
    once measures what we actually care about: did we capture what was said."""
    by_language: dict[str, Slice] = defaultdict(Slice)
    by_acoustic: dict[str, Slice] = defaultdict(Slice)
    overall = Slice()
    per_call: list[dict[str, Any]] = []

    for call_id, truth in ground_truth.items():
        hyp_turns = utterances.get(call_id, [])
        if not hyp_turns:
            continue

        reference = " ".join(t["text"] for t in truth["turns"])
        hypothesis = " ".join(u.text_raw for u in hyp_turns)
        rates = error_rates(reference, hypothesis)

        lang_mode = truth["language_style"]
        acoustic = truth["acoustic_profile"]

        for bucket in (overall, by_language[lang_mode], by_acoustic[acoustic]):
            bucket.n += 1
            bucket.add("wer", rates.wer)
            bucket.add("cer", rates.cer)
            bucket.add("translit_invariant_wer", rates.translit_invariant_wer)

        per_call.append(
            {
                "call_id": call_id,
                "language_style": lang_mode,
                "acoustic_profile": acoustic,
                "wer": rates.wer,
                "cer": rates.cer,
                "translit_invariant_wer": rates.translit_invariant_wer,
            }
        )

    return {
        "overall": overall.summary(),
        "by_language_mode": {k: v.summary() for k, v in sorted(by_language.items())},
        "by_acoustic_profile": {k: v.summary() for k, v in sorted(by_acoustic.items())},
        "per_call": per_call,
    }


def evaluate_lid(
    ground_truth: dict[str, dict], utterances: dict[str, list[Utterance]]
) -> dict[str, Any]:
    """Utterance-level LID, aligned by time overlap with the true turns.

    `hi_latn` and `mixed` are both scored as correct for a true `hinglish` turn:
    the practically important distinction is Hindi-content vs English-only, and
    penalising the model for choosing `mixed` over `hi_latn` measures a labelling
    convention rather than a capability."""
    acceptable: dict[str, set[str]] = {
        "en": {"en"},
        "hi": {"hi"},
        "hi_latn": {"hi_latn", "mixed", "hi"},
        "hinglish": {"mixed", "hi_latn", "hi"},
    }

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    correct = total = 0

    for call_id, truth in ground_truth.items():
        hyp_turns = utterances.get(call_id, [])
        if not hyp_turns:
            continue

        for true_turn in truth["turns"]:
            overlapping = [
                u
                for u in hyp_turns
                if u.start_ms < true_turn["end_ms"] and u.end_ms > true_turn["start_ms"]
            ]
            if not overlapping:
                continue
            predicted = max(
                overlapping,
                key=lambda u: min(u.end_ms, true_turn["end_ms"]) - max(u.start_ms, true_turn["start_ms"]),
            ).lang_primary.value

            expected = true_turn["lang"]
            confusion[expected][predicted] += 1
            total += 1
            if predicted in acceptable.get(expected, {expected}):
                correct += 1

    per_language: dict[str, dict[str, Any]] = {}
    for expected, predictions in confusion.items():
        n = sum(predictions.values())
        hits = sum(v for k, v in predictions.items() if k in acceptable.get(expected, {expected}))
        per_language[expected] = {
            "n": n,
            "recall": round(hits / n, 4) if n else None,
            "predictions": dict(sorted(predictions.items(), key=lambda kv: -kv[1])),
        }

    return {
        "accuracy": round(correct / total, 4) if total else None,
        "n": total,
        "per_true_language": per_language,
    }


def evaluate_enrichment(
    ground_truth: dict[str, dict], enrichment: dict[str, CallEnrichment]
) -> dict[str, Any]:
    intent_l1 = intent_l2 = resolution = 0
    escalation_tp = escalation_fp = escalation_fn = escalation_tn = 0
    fcr_correct = 0
    n = 0
    per_intent: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    confidence_bins: dict[int, list[int]] = defaultdict(list)

    for call_id, truth in ground_truth.items():
        row = enrichment.get(call_id)
        if row is None or row.validation_status.value == "QUARANTINED":
            continue
        n += 1

        l1_hit = row.primary_intent.l1 == truth["intent_l1"]
        l2_hit = l1_hit and row.primary_intent.l2 == truth["intent_l2"]
        intent_l1 += int(l1_hit)
        intent_l2 += int(l2_hit)

        per_intent[truth["intent_l1"]]["n"] += 1
        per_intent[truth["intent_l1"]]["correct"] += int(l1_hit)

        bucket = min(9, int(row.primary_intent.confidence * 10))
        confidence_bins[bucket].append(int(l1_hit))

        resolution += int(row.resolution_status.value == truth["resolution_status"])
        fcr_correct += int(row.fcr_candidate == truth["fcr"])

        predicted_escalation = row.escalation.escalated_in_call
        actual_escalation = truth["escalated"]
        if predicted_escalation and actual_escalation:
            escalation_tp += 1
        elif predicted_escalation and not actual_escalation:
            escalation_fp += 1
        elif not predicted_escalation and actual_escalation:
            escalation_fn += 1
        else:
            escalation_tn += 1

    precision = escalation_tp / (escalation_tp + escalation_fp) if (escalation_tp + escalation_fp) else None
    recall = escalation_tp / (escalation_tp + escalation_fn) if (escalation_tp + escalation_fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )

    # Expected Calibration Error: does a stated confidence of 0.8 correspond to
    # 80% accuracy? A model that is accurate but miscalibrated cannot be
    # thresholded, which makes selective automation impossible.
    ece = 0.0
    for bucket, hits in confidence_bins.items():
        if not hits:
            continue
        weight = len(hits) / max(1, n)
        ece += weight * abs((sum(hits) / len(hits)) - ((bucket + 0.5) / 10))

    return {
        "n": n,
        "intent_l1_accuracy": round(intent_l1 / n, 4) if n else None,
        "intent_l2_accuracy": round(intent_l2 / n, 4) if n else None,
        "resolution_accuracy": round(resolution / n, 4) if n else None,
        "fcr_accuracy": round(fcr_correct / n, 4) if n else None,
        "escalation": {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "tp": escalation_tp, "fp": escalation_fp,
            "fn": escalation_fn, "tn": escalation_tn,
        },
        "intent_calibration_ece": round(ece, 4),
        "per_intent_l1": {
            k: {**v, "accuracy": round(v["correct"] / v["n"], 4) if v["n"] else None}
            for k, v in sorted(per_intent.items())
        },
    }


def evaluate_compliance(
    ground_truth: dict[str, dict], enrichment: dict[str, CallEnrichment]
) -> dict[str, Any]:
    per_rule: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "disputed": 0}
    )

    for call_id, truth in ground_truth.items():
        row = enrichment.get(call_id)
        if row is None:
            continue
        expected = set(truth["compliance_present"])

        for check in row.compliance:
            rule = check.rule.value
            predicted_present = check.outcome.value == "PRESENT"
            actually_present = rule in expected

            # NOT_APPLICABLE is scored only when the ground truth also says the
            # rule was not exercised; otherwise it is a miss.
            if check.outcome.value == "NOT_APPLICABLE" and not actually_present:
                per_rule[rule]["tn"] += 1
                continue

            if predicted_present and actually_present:
                per_rule[rule]["tp"] += 1
            elif predicted_present and not actually_present:
                per_rule[rule]["fp"] += 1
            elif not predicted_present and actually_present:
                per_rule[rule]["fn"] += 1
            else:
                per_rule[rule]["tn"] += 1

            if check.detector_disagreement:
                per_rule[rule]["disputed"] += 1

    out: dict[str, Any] = {}
    macro_precision: list[float] = []
    macro_recall: list[float] = []
    for rule, counts in sorted(per_rule.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        if precision is not None:
            macro_precision.append(precision)
        if recall is not None:
            macro_recall.append(recall)
        out[rule] = {
            **counts,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
        }

    return {
        "per_rule": out,
        "macro_precision": round(sum(macro_precision) / len(macro_precision), 4) if macro_precision else None,
        "macro_recall": round(sum(macro_recall) / len(macro_recall), 4) if macro_recall else None,
        "total_disputed": sum(c["disputed"] for c in per_rule.values()),
    }


def calibrate_asr_quality(
    settings: Settings,
    ground_truth: dict[str, dict],
    utterances: dict[str, list[Utterance]],
    segments: dict[str, list[AsrSegment]],
) -> dict[str, Any]:
    """Fit the reference-free WER estimator on the labelled corpus, then report
    how well it predicts held-out WER."""
    samples: list[tuple[dict[str, float], float, str | None]] = []

    for call_id, truth in ground_truth.items():
        hyp_turns = utterances.get(call_id, [])
        call_segments = segments.get(call_id, [])
        if not hyp_turns or not call_segments:
            continue
        reference = " ".join(t["text"] for t in truth["turns"])
        hypothesis = " ".join(u.text_raw for u in hyp_turns)
        true_wer = error_rates(reference, hypothesis).wer
        signals = asr_quality.collect_signals(call_segments)
        samples.append((signals, true_wer, truth["language_style"]))

    if len(samples) < 10:
        return {"fitted": False, "n": len(samples), "reason": "insufficient samples"}

    model = asr_quality.fit_calibration(samples)
    model_path = settings.data_root / "models" / "asr_calibration.json"
    model.save(model_path)

    errors = [abs(model.predict(s, lang) - true) for s, true, lang in samples]
    mae = sum(errors) / len(errors)

    # Rank correlation matters more than absolute error: the estimator's job is
    # to order calls by quality so we can gate and route, not to hit WER exactly.
    predicted = [model.predict(s, lang) for s, _, lang in samples]
    actual = [true for _, true, _ in samples]
    rho = _spearman(predicted, actual)

    return {
        "fitted": True,
        "n": len(samples),
        "mae": round(mae, 4),
        "spearman_rho": round(rho, 4),
        "residual_std": model.residual_std,
        "per_language_bias": model.per_language_bias,
        "model_path": str(model_path),
    }


def _spearman(a: list[float], b: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for position, index in enumerate(order):
            ranks[index] = float(position)
        return ranks

    ra, rb = rank(a), rank(b)
    n = len(ra)
    if n < 2:
        return 0.0
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den_a = sum((x - mean_a) ** 2 for x in ra) ** 0.5
    den_b = sum((y - mean_b) ** 2 for y in rb) ** 0.5
    return num / (den_a * den_b) if den_a and den_b else 0.0


def evaluate_diarization(
    ground_truth: dict[str, dict], utterances: dict[str, list[Utterance]]
) -> dict[str, Any]:
    """Speaker-attribution accuracy, weighted by duration.

    Duration-weighted because misattributing a 30-second explanation matters far
    more than misattributing a one-word backchannel."""
    correct_ms = total_ms = 0

    for call_id, truth in ground_truth.items():
        for utt in utterances.get(call_id, []):
            overlapping = [
                t
                for t in truth["turns"]
                if utt.start_ms < t["end_ms"] and utt.end_ms > t["start_ms"]
            ]
            if not overlapping:
                continue
            best = max(
                overlapping,
                key=lambda t: min(utt.end_ms, t["end_ms"]) - max(utt.start_ms, t["start_ms"]),
            )
            duration = utt.end_ms - utt.start_ms
            total_ms += duration
            if best["role"] == utt.speaker_role.value:
                correct_ms += duration

    return {
        "speaker_attribution_accuracy": round(correct_ms / total_ms, 4) if total_ms else None,
        "scored_ms": total_ms,
    }


def run_evaluation(settings: Settings, output_path: Path | None = None) -> dict[str, Any]:
    ground_truth = _load_ground_truth(settings)
    utterances, segments, enrichment = _load_silver(settings)

    report = {
        "corpus_size": len(ground_truth),
        "asr": evaluate_asr(ground_truth, utterances),
        "language_id": evaluate_lid(ground_truth, utterances),
        "diarization": evaluate_diarization(ground_truth, utterances),
        "enrichment": evaluate_enrichment(ground_truth, enrichment),
        "compliance": evaluate_compliance(ground_truth, enrichment),
        "asr_quality_calibration": calibrate_asr_quality(
            settings, ground_truth, utterances, segments
        ),
    }

    path = output_path or (settings.data_root / "eval" / "evaluation_report.json")
    write_json(path, report)
    log.info("evaluation written", extra={"path": str(path)})
    return report


def print_report(report: dict[str, Any]) -> str:
    """Compact text rendering for the CLI."""
    lines: list[str] = []
    asr = report["asr"]["overall"]
    lines.append(f"ASR      n={asr.get('n')}  WER={asr.get('wer')}  CER={asr.get('cer')}  "
                 f"translit-invariant WER={asr.get('translit_invariant_wer')}")
    for mode, stats in report["asr"]["by_language_mode"].items():
        lines.append(f"           {mode:<10} n={stats['n']:<3} WER={stats['wer']}  "
                     f"TI-WER={stats['translit_invariant_wer']}")

    lid = report["language_id"]
    lines.append(f"LID      accuracy={lid.get('accuracy')} over n={lid.get('n')}")
    for lang, stats in lid["per_true_language"].items():
        lines.append(f"           {lang:<10} recall={stats['recall']}  n={stats['n']}")

    dia = report["diarization"]
    lines.append(f"DIARIZE  speaker accuracy={dia.get('speaker_attribution_accuracy')}")

    enr = report["enrichment"]
    lines.append(
        f"INTENT   L1={enr.get('intent_l1_accuracy')}  L2={enr.get('intent_l2_accuracy')}  "
        f"ECE={enr.get('intent_calibration_ece')}"
    )
    lines.append(
        f"RESOLVE  accuracy={enr.get('resolution_accuracy')}  FCR accuracy={enr.get('fcr_accuracy')}"
    )
    esc = enr["escalation"]
    lines.append(
        f"ESCALATE precision={esc.get('precision')}  recall={esc.get('recall')}  f1={esc.get('f1')}"
    )

    comp = report["compliance"]
    lines.append(
        f"COMPLY   macro precision={comp.get('macro_precision')}  "
        f"macro recall={comp.get('macro_recall')}  disputed={comp.get('total_disputed')}"
    )

    cal = report["asr_quality_calibration"]
    if cal.get("fitted"):
        lines.append(
            f"QUALITY  reference-free WER estimator MAE={cal['mae']}  "
            f"spearman={cal['spearman_rho']}  n={cal['n']}"
        )
    else:
        lines.append(f"QUALITY  calibration skipped ({cal.get('reason')})")

    return "\n".join(lines)


__all__ = [
    "calibrate_asr_quality",
    "evaluate_asr",
    "evaluate_compliance",
    "evaluate_diarization",
    "evaluate_enrichment",
    "evaluate_lid",
    "print_report",
    "run_evaluation",
]
