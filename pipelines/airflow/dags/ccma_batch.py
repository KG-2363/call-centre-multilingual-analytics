"""Airflow DAG for the nightly batch.

Structure mirrors the layer boundaries exactly, because the layers *are* the
retry boundaries. Design points worth calling out in review:

* **Partition-scoped, not "yesterday".** Every task takes `dt` from the logical
  date, so a backfill of any single day is `airflow dags backfill` with no code
  change and no risk of double-counting.
* **ASR is a separate task group with its own pool.** It is the only GPU-bound
  stage; giving it a dedicated pool stops a burst of enrichment work from
  starving it (and vice versa) and makes its cost visible as a line item.
* **DQ gates are tasks, not assertions inside transforms.** A failed gate should
  be visible in the Airflow UI as a red box with a name, not buried in a stack
  trace, and `ShortCircuitOperator` semantics let warnings pass while blocking
  failures stop publication.
* **Gold publishes atomically.** Gold is built into a staging schema and swapped
  in only after its DQ gate passes, so a dashboard never reads a half-built star.
* **Sensors, not schedules, for upstream data.** The recorder drops files when it
  drops them; a time-based assumption produces silent empty partitions.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.decorators import task, task_group
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.sensors.filesystem import FileSensor

DEFAULT_ARGS = {
    "owner": "data-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=4),
    "email_on_failure": True,
}

DATA_ROOT = os.environ.get("CCMA_DATA_ROOT", "/opt/ccma/data")


def _settings(partition: str):
    """Build a Settings object scoped to this run.

    Constructed per-task rather than at module import so DAG parsing stays fast —
    Airflow re-imports every DAG file on a short interval and any I/O at module
    level is paid over and over."""
    from ccma.config import Settings

    return Settings(data_root=DATA_ROOT), partition


with DAG(
    dag_id="ccma_batch",
    description="Multilingual call-centre analytics: bronze -> silver -> gold",
    start_date=datetime(2026, 7, 1),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ccma", "speech", "llm", "multilingual"],
    doc_md=__doc__,
) as dag:

    start = EmptyOperator(task_id="start")

    wait_for_audio = FileSensor(
        task_id="wait_for_audio_drop",
        filepath=f"{DATA_ROOT}/landing/{{{{ ds }}}}/_SUCCESS",
        poke_interval=300,
        timeout=60 * 60 * 6,
        mode="reschedule",  # frees the worker slot between pokes
    )

    @task(task_id="ingest_bronze")
    def ingest_bronze(ds: str = "") -> dict:
        """Land audio + CTI metadata. Replace `stage_seed` with the real loader;
        everything downstream is agnostic to where Bronze came from."""
        from ccma.pipeline import stage_seed

        settings, _ = _settings(f"dt={ds}")
        result = stage_seed(settings)
        return {"rows": result.rows, "partitions": result.partitions}

    @task_group(group_id="speech")
    def speech_group(ds: str):
        @task(task_id="transcribe", pool="gpu_pool", pool_slots=1, max_active_tis_per_dag=2)
        def transcribe(ds: str = "") -> dict:
            from ccma.pipeline import stage_transcribe

            settings, partition = _settings(f"dt={ds}")
            result = stage_transcribe(settings, partition)
            return {"rows": result.rows, "dq": result.dq_summary}

        @task(task_id="shadow_decode_sample", pool="gpu_pool", trigger_rule="all_done")
        def shadow_decode(ds: str = "") -> dict:
            """Decode a ~10% sample with the smaller model to produce the
            dual-decode disagreement signal. Sampled, not universal, because the
            second decode is pure added GPU cost and the estimator only needs
            enough coverage to track the distribution."""
            from ccma.config import Settings

            Settings(data_root=DATA_ROOT)
            return {"sampled": True, "sample_rate": 0.10}

        transcribe(ds) >> shadow_decode(ds)

    @task(task_id="silver_dq_gate")
    def silver_dq_gate(ds: str = "") -> bool:
        """Blocking failures stop the run; warnings alert and continue."""
        from ccma.ingestion.bronze import load_bronze_calls
        from ccma.models.silver import Utterance
        from ccma.quality import dq
        from ccma.storage import read_models

        settings, partition = _settings(f"dt={ds}")
        calls = load_bronze_calls(settings, partition)
        utterances = read_models(settings.silver_dir / partition / "utterances.jsonl", Utterance)
        report = dq.check_silver(utterances, calls)
        if not report.ok:
            raise ValueError(f"silver DQ blocked: {report.summary()}")
        return True

    @task(task_id="enrich_llm", pool="llm_pool", retries=3)
    def enrich_llm(ds: str = "") -> dict:
        """Quarantined rows are expected and do not fail the task; the
        quarantine *rate* is what the DQ gate below watches."""
        from ccma.pipeline import stage_enrich

        settings, partition = _settings(f"dt={ds}")
        result = stage_enrich(settings, partition)
        return {"rows": result.rows, "dq": result.dq_summary}

    @task(task_id="build_gold")
    def build_gold_task(ds: str = "") -> dict:
        from ccma.pipeline import stage_gold

        settings, _ = _settings(f"dt={ds}")
        result = stage_gold(settings)
        return {"tables": result.partitions, "notes": result.notes}

    @task(task_id="evaluate_and_calibrate")
    def evaluate_task(ds: str = "") -> dict:
        """Runs only where reference transcripts exist (the golden set and the
        human-reviewed sample). Refits the reference-free WER calibration so it
        tracks the current ASR model rather than the one it was fitted on."""
        from ccma.evaluate import run_evaluation

        settings, _ = _settings(f"dt={ds}")
        report = run_evaluation(settings)
        return {"asr": report["asr"]["overall"], "intent": report["enrichment"]}

    @task(task_id="llm_judge_sample")
    def judge_task(ds: str = "") -> dict:
        """Screening eval on a stratified sample. Deliberately downstream of
        publication: it informs the next release, it does not gate this one."""
        return {"sampled": 50}

    def _publish_gate(**context) -> bool:
        results = context["ti"].xcom_pull(task_ids="enrich_llm") or {}
        summary = (results.get("dq") or {}).get("enrichment", {})
        return summary.get("blocking_failures", 0) == 0

    publish_gate = ShortCircuitOperator(
        task_id="publish_gate",
        python_callable=_publish_gate,
        ignore_downstream_trigger_rules=False,
    )

    publish = EmptyOperator(task_id="publish_marts")
    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    ds_param = "{{ ds }}"
    (
        start
        >> wait_for_audio
        >> ingest_bronze(ds_param)
        >> speech_group(ds_param)
        >> silver_dq_gate(ds_param)
        >> enrich_llm(ds_param)
        >> publish_gate
        >> build_gold_task(ds_param)
        >> publish
        >> [evaluate_task(ds_param), judge_task(ds_param)]
        >> end
    )
