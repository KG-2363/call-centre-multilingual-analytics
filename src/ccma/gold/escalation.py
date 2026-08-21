"""Escalation-risk prediction.

**What we predict.** Not "did the customer shout" — the probability that this
call generates a *downstream cost event* within 7 days: a repeat contact on the
same intent, a supervisor escalation, or a regulator/consumer-forum complaint.
That framing makes the target observable in the warehouse without any annotation,
which is the whole reason this model is trainable at all.

**Weak supervision.** True escalation labels are scarce and lag by a week, so the
training label is assembled from three noisy sources whose errors are largely
independent:

  1. an in-call escalation actually occurred (high precision, low recall),
  2. a repeat contact on the same L1 intent within 7 days (the outcome we care
     about, but only observable after the fact),
  3. explicit threat language — regulator, legal, social media, churn (high
     precision, moderate recall).

Any source firing produces a positive. This over-generates positives, which we
accept: in this application a false positive costs a supervisor two minutes and a
false negative costs a customer. The threshold is tuned for recall accordingly.

**Two models, always both.** A transparent rule baseline runs alongside the
learned model. The baseline is what ships on day one, what we fall back to when
the model degrades, and what we benchmark against — a gradient-boosted model that
cannot beat seven weighted rules is not worth operating.

**Calibration.** Raw GBM scores are not probabilities. We fit isotonic
calibration on a held-out split so a "0.8" genuinely means roughly 80%, because
supervisors will threshold on this number and an uncalibrated score makes that
threshold meaningless.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..logging_conf import get_logger

log = get_logger(__name__)

FEATURE_COLUMNS: list[str] = [
    "sentiment_delta",
    "sentiment_min",
    "dissatisfaction_signal_count",
    "escalation_indicator_count",
    "llm_escalation_risk",
    "hold_count",
    "hold_time_s",
    "transfer_count",
    "queue_wait_s",
    "handling_time_s",
    "turn_count",
    "agent_talk_ratio",
    "silence_ratio",
    "overlap_count",
    "compliance_score",
    "prior_contacts_7d",
    "mean_asr_confidence",
    "call_cmi",
    "agent_customer_language_match",
]

# Rule baseline. Weights are odds-ratio-ish contributions chosen from domain
# reasoning, then sanity-checked against the corpus; they are not fitted.
RULE_WEIGHTS: dict[str, float] = {
    "escalation_indicator_count": 0.16,
    "dissatisfaction_signal_count": 0.07,
    "negative_sentiment_close": 0.20,
    "sentiment_worsened": 0.14,
    "repeat_contact": 0.22,
    "long_hold": 0.10,
    "transfer": 0.09,
    "compliance_gap": 0.06,
    "language_mismatch": 0.05,
}


@dataclass
class EscalationModel:
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    estimator: Any = None
    calibrator: Any = None
    threshold: float = 0.35
    metrics: dict[str, float] = field(default_factory=dict)
    trained: bool = False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "trained": self.trained,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self.trained and self.estimator is not None:
            try:
                import pickle

                with open(path.with_suffix(".pkl"), "wb") as fh:
                    pickle.dump({"estimator": self.estimator, "calibrator": self.calibrator}, fh)
            except Exception as exc:
                log.warning("could not persist estimator", extra={"error": str(exc)})

    @classmethod
    def load(cls, path: Path) -> EscalationModel:
        if not path.exists():
            return cls()
        meta = json.loads(path.read_text(encoding="utf-8"))
        model = cls(
            feature_names=meta.get("feature_names", list(FEATURE_COLUMNS)),
            threshold=meta.get("threshold", 0.35),
            metrics=meta.get("metrics", {}),
            trained=meta.get("trained", False),
        )
        pkl = path.with_suffix(".pkl")
        if model.trained and pkl.exists():
            try:
                import pickle

                with open(pkl, "rb") as fh:
                    blob = pickle.load(fh)
                model.estimator = blob.get("estimator")
                model.calibrator = blob.get("calibrator")
            except Exception as exc:
                log.warning("could not load estimator; falling back to rules", extra={"error": str(exc)})
                model.trained = False
        return model


def build_weak_labels(df: pd.DataFrame) -> pd.Series:
    """Assemble the noisy target from independent signals."""
    in_call = df.get("escalated_in_call", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    repeat = df.get("subsequent_contacts_7d", pd.Series(0, index=df.index)).fillna(0) > 0
    threats = df.get("escalation_indicator_count", pd.Series(0, index=df.index)).fillna(0) >= 2
    bad_close = (
        df.get("customer_sentiment_end", pd.Series("", index=df.index)).isin(
            ["NEGATIVE", "VERY_NEGATIVE"]
        )
        & (df.get("dissatisfaction_signal_count", pd.Series(0, index=df.index)).fillna(0) >= 2)
    )
    return (in_call | repeat | threats | bad_close).astype(int)


def rule_score(df: pd.DataFrame) -> pd.Series:
    """Transparent baseline. Every term is inspectable and explainable to a
    supervisor, which matters when the output drives an intervention."""
    def col(name: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(df.get(name, pd.Series(default, index=df.index)), errors="coerce").fillna(default)

    score = pd.Series(0.05, index=df.index)
    score += RULE_WEIGHTS["escalation_indicator_count"] * col("escalation_indicator_count").clip(0, 4)
    score += RULE_WEIGHTS["dissatisfaction_signal_count"] * col("dissatisfaction_signal_count").clip(0, 6)
    score += RULE_WEIGHTS["negative_sentiment_close"] * df.get(
        "customer_sentiment_end", pd.Series("", index=df.index)
    ).isin(["NEGATIVE", "VERY_NEGATIVE"]).astype(float)
    score += RULE_WEIGHTS["sentiment_worsened"] * (col("sentiment_delta") < -0.3).astype(float)
    score += RULE_WEIGHTS["repeat_contact"] * (col("prior_contacts_7d") > 0).astype(float)
    score += RULE_WEIGHTS["long_hold"] * (col("hold_time_s") > 90).astype(float)
    score += RULE_WEIGHTS["transfer"] * (col("transfer_count") > 0).astype(float)
    score += RULE_WEIGHTS["compliance_gap"] * (col("compliance_score", 1.0) < 0.7).astype(float)
    score += RULE_WEIGHTS["language_mismatch"] * (
        col("agent_customer_language_match", 1.0) < 0.6
    ).astype(float)
    return score.clip(0.0, 1.0)


def train(df: pd.DataFrame, model_path: Path | None = None) -> EscalationModel:
    """Fit the learned model, but only adopt it if it beats the rule baseline."""
    model = EscalationModel()
    labels = build_weak_labels(df)

    positives = int(labels.sum())
    if positives < 15 or (len(labels) - positives) < 15:
        log.warning(
            "insufficient class balance for training; shipping rule baseline",
            extra={"n": len(labels), "positives": positives},
        )
        model.metrics = {"n": float(len(labels)), "positives": float(positives)}
        if model_path:
            model.save(model_path)
        return model

    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        log.warning("scikit-learn unavailable; shipping rule baseline")
        if model_path:
            model.save(model_path)
        return model

    features = _feature_frame(df, model.feature_names)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )

    estimator = HistGradientBoostingClassifier(
        max_iter=180,
        max_depth=4,          # shallow: the dataset is small and wide trees overfit
        learning_rate=0.06,
        l2_regularization=1.0,
        min_samples_leaf=8,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv=3)
    calibrated.fit(x_train, y_train)

    probabilities = calibrated.predict_proba(x_test)[:, 1]
    baseline = rule_score(df.loc[x_test.index])

    try:
        model_auc = float(roc_auc_score(y_test, probabilities))
        model_ap = float(average_precision_score(y_test, probabilities))
        rule_auc = float(roc_auc_score(y_test, baseline))
        rule_ap = float(average_precision_score(y_test, baseline))
    except ValueError:
        log.warning("test split is single-class; shipping rule baseline")
        if model_path:
            model.save(model_path)
        return model

    model.metrics = {
        "model_roc_auc": round(model_auc, 4),
        "model_pr_auc": round(model_ap, 4),
        "rule_roc_auc": round(rule_auc, 4),
        "rule_pr_auc": round(rule_ap, 4),
        "n_train": float(len(x_train)),
        "n_test": float(len(x_test)),
        "positive_rate": round(float(labels.mean()), 4),
        "brier": round(float(((probabilities - y_test) ** 2).mean()), 4),
    }

    if model_ap > rule_ap + 0.02:
        model.estimator, model.calibrator, model.trained = estimator, calibrated, True
        model.threshold = _threshold_for_recall(y_test.tolist(), probabilities.tolist(), 0.85)
        log.info("learned model adopted", extra=model.metrics)
    else:
        log.info("learned model did not beat the rule baseline; keeping rules", extra=model.metrics)

    if model_path:
        model.save(model_path)
    return model


def _feature_frame(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name in names:
        out[name] = pd.to_numeric(df.get(name, 0.0), errors="coerce").fillna(0.0)
    return out


def _threshold_for_recall(y_true: list[int], scores: list[float], target_recall: float) -> float:
    """Pick the highest threshold that still achieves the target recall.

    Recall-first, because the asymmetry is stark: missing an escalation costs a
    customer relationship, flagging one unnecessarily costs a supervisor a
    two-minute review."""
    positives = sum(y_true)
    if positives == 0:
        return 0.35
    ordered = sorted(zip(scores, y_true), reverse=True)
    found = 0
    for score, label in ordered:
        found += label
        if found / positives >= target_recall:
            return round(max(0.05, min(0.95, score)), 4)
    return 0.05


def predict(df: pd.DataFrame, model: EscalationModel) -> pd.DataFrame:
    """Score calls. Always emits both scores plus the top contributing drivers."""
    baseline = rule_score(df)

    if model.trained and model.calibrator is not None:
        try:
            learned = pd.Series(
                model.calibrator.predict_proba(_feature_frame(df, model.feature_names))[:, 1],
                index=df.index,
            )
            source = "gbm_isotonic"
        except Exception as exc:
            log.warning("scoring failed; using rule baseline", extra={"error": str(exc)})
            learned, source = baseline, "rule_fallback"
    else:
        learned, source = baseline, "rule_baseline"

    out = pd.DataFrame(
        {
            "call_id": df.get("call_id"),
            "date_key": df.get("date_key"),
            "agent_id": df.get("agent_id"),
            "intent_l1": df.get("intent_l1"),
            "rule_score": baseline.round(4),
            "risk_score": learned.round(4),
            "risk_band": pd.cut(
                learned,
                bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
                labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            ).astype(str),
            "is_flagged": (learned >= model.threshold),
            "model_source": source,
            "threshold": model.threshold,
        }
    )
    out["top_drivers"] = [_drivers(row) for _, row in df.iterrows()]
    return out


def _drivers(row: pd.Series) -> str:
    """Human-readable reasons, ordered by contribution. A risk score with no
    explanation does not get acted on."""
    reasons: list[tuple[float, str]] = []

    def num(name: str, default: float = 0.0) -> float:
        value = row.get(name, default)
        try:
            return float(value) if value is not None and not (isinstance(value, float) and math.isnan(value)) else default
        except (TypeError, ValueError):
            return default

    if num("escalation_indicator_count") > 0:
        reasons.append((num("escalation_indicator_count") * 0.16, f"{int(num('escalation_indicator_count'))} escalation indicator(s)"))
    if row.get("customer_sentiment_end") in ("NEGATIVE", "VERY_NEGATIVE"):
        reasons.append((0.20, "call closed on negative sentiment"))
    if num("sentiment_delta") < -0.3:
        reasons.append((0.14, "sentiment worsened during the call"))
    if num("prior_contacts_7d") > 0:
        reasons.append((0.22, f"{int(num('prior_contacts_7d'))} prior contact(s) in 7 days"))
    if num("hold_time_s") > 90:
        reasons.append((0.10, f"{int(num('hold_time_s'))}s on hold"))
    if num("transfer_count") > 0:
        reasons.append((0.09, "call was transferred"))
    if num("dissatisfaction_signal_count") >= 2:
        reasons.append((0.07, f"{int(num('dissatisfaction_signal_count'))} dissatisfaction signals"))
    if num("agent_customer_language_match", 1.0) < 0.6:
        reasons.append((0.05, "agent did not match the customer's language"))

    reasons.sort(reverse=True)
    return "; ".join(text for _, text in reasons[:3]) or "no strong risk drivers"
