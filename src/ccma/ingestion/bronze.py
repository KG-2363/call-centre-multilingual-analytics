"""Bronze writer: land call metadata, audio and (for the synthetic corpus) the
ground-truth reference in an immutable, replayable layout.

Layout (identical locally and on S3):

    bronze/
      dt=YYYY-MM-DD/
        calls.jsonl              CallMetadata, one per line
        asr_segments.jsonl       written later by the ASR stage
      _manifest/<batch_id>.json  what this batch contained + checksums
    eval/
      ground_truth.jsonl         reference labels (never read by the pipeline)

Partitioning by `dt` (call date, not ingest date) is deliberate: reprocessing a
day is then a single-partition overwrite, and the warehouse can prune on the same
column that every dashboard filters by.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings
from ..logging_conf import get_logger
from ..models.bronze import CallMetadata
from ..storage import write_json, write_jsonl
from .corpus import GeneratedCall

log = get_logger(__name__)


def partition_for(call: CallMetadata) -> str:
    return call.recorded_at.strftime("dt=%Y-%m-%d")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bronze(
    calls: list[GeneratedCall], settings: Settings, audio_paths: dict[str, Path] | None = None
) -> dict[str, int]:
    """Write one batch. Returns per-partition row counts for the DQ layer."""
    audio_paths = audio_paths or {}
    by_partition: dict[str, list[CallMetadata]] = defaultdict(list)

    for call in calls:
        meta = call.metadata
        audio_path = audio_paths.get(meta.call_id)
        if audio_path and audio_path.exists():
            meta = meta.model_copy(
                update={
                    "audio_sha256": _sha256_file(audio_path),
                    "audio_uri": str(audio_path.relative_to(settings.data_root)).replace(
                        "\\", "/"
                    ),
                }
            )
            call.metadata = meta
        by_partition[partition_for(meta)].append(meta)

    counts: dict[str, int] = {}
    for partition, metas in sorted(by_partition.items()):
        path = settings.bronze_dir / partition / "calls.jsonl"
        counts[partition] = write_jsonl(path, metas)
        log.info("wrote bronze partition", extra={"partition": partition, "rows": counts[partition]})

    batch_id = calls[0].metadata.ingest_batch_id if calls else "empty"
    write_json(
        settings.bronze_dir / "_manifest" / f"{batch_id}.json",
        {
            "batch_id": batch_id,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "call_count": len(calls),
            "partitions": counts,
            "call_ids": [c.metadata.call_id for c in calls],
            "source_system": calls[0].metadata.source_system if calls else None,
        },
    )
    return counts


def write_ground_truth(calls: list[GeneratedCall], settings: Settings) -> int:
    """Reference labels live outside bronze/silver/gold on purpose — no pipeline
    stage has a path to them, so they cannot leak into a feature or a prompt."""
    path = settings.data_root / "eval" / "ground_truth.jsonl"
    n = write_jsonl(path, [c.ground_truth() for c in calls])
    log.info("wrote ground truth", extra={"rows": n, "path": str(path)})
    return n


def load_bronze_calls(settings: Settings, partition: str | None = None) -> list[CallMetadata]:
    from ..storage import read_models

    if partition:
        paths = [settings.bronze_dir / partition / "calls.jsonl"]
    else:
        paths = sorted(settings.bronze_dir.glob("dt=*/calls.jsonl"))

    calls: list[CallMetadata] = []
    for path in paths:
        calls.extend(read_models(path, CallMetadata))
    return calls
