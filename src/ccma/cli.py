"""`ccma` command-line interface.

Every pipeline stage is a subcommand so an operator can replay any single layer
without re-running the ones before it. That is the whole point of the layered
design and it should be one command away, not a notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import pipeline
from .config import get_settings
from .logging_conf import configure_logging

app = typer.Typer(
    name="ccma",
    help="Multilingual (en/hi/hinglish) call-centre analytics pipeline.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _setup():
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    return settings


def _render(results: list[pipeline.StageResult]) -> None:
    table = Table(title="Pipeline", show_lines=False)
    table.add_column("Stage", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Partitions", justify="right")
    table.add_column("Notes", style="dim")
    for result in results:
        table.add_row(
            result.stage,
            str(result.rows),
            str(len(result.partitions)),
            "; ".join(result.notes)[:80],
        )
    console.print(table)


@app.command()
def seed(
    tts: Optional[str] = typer.Option(None, help="Override TTS backend: none | edge_tts"),
    size: Optional[int] = typer.Option(None, help="Number of calls to generate"),
) -> None:
    """Generate the synthetic corpus and land the Bronze layer."""
    settings = _setup()
    if size:
        settings.corpus_size = size
    result = pipeline.stage_seed(settings, tts)
    _render([result])
    console.print(f"[green]Bronze written to[/green] {settings.bronze_dir}")


@app.command()
def transcribe(
    partition: Optional[str] = typer.Option(None, help="Only this partition, e.g. dt=2026-07-20"),
) -> None:
    """Run ASR, diarization and language ID -> Silver utterances."""
    settings = _setup()
    result = pipeline.stage_transcribe(settings, partition)
    _render([result])


@app.command()
def enrich(
    partition: Optional[str] = typer.Option(None, help="Only this partition"),
) -> None:
    """Run LLM conversation intelligence -> Silver enrichment."""
    settings = _setup()
    result = pipeline.stage_enrich(settings, partition)
    _render([result])


@app.command()
def gold() -> None:
    """Build the Gold star schema and score escalation risk."""
    settings = _setup()
    result = pipeline.stage_gold(settings)
    _render([result])

    table = Table(title="Gold tables")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right")
    for name, count in sorted(result.partitions.items()):
        table.add_row(name, str(count))
    console.print(table)


@app.command("run-all")
def run_all(
    tts: Optional[str] = typer.Option(None, help="Override TTS backend"),
) -> None:
    """Seed -> transcribe -> enrich -> gold, end to end."""
    settings = _setup()
    _render(pipeline.run_all(settings, tts))
    console.print(f"[green]Warehouse ready:[/green] {settings.warehouse_path}")
    console.print("Next: [cyan]ccma evaluate[/cyan] then [cyan]streamlit run dashboard/app.py[/cyan]")


@app.command()
def evaluate(
    output: Optional[Path] = typer.Option(None, help="Write the JSON report here"),
    json_out: bool = typer.Option(False, "--json", help="Print the full JSON report"),
) -> None:
    """Score the pipeline against ground truth and fit ASR quality calibration."""
    from .evaluate import print_report, run_evaluation

    settings = _setup()
    report = run_evaluation(settings, output)
    if json_out:
        console.print_json(json.dumps(report))
    else:
        console.print(print_report(report))


@app.command()
def judge(
    sample: int = typer.Option(10, help="Number of calls to sample"),
) -> None:
    """Run LLM-as-judge over a sample of enriched calls."""
    from collections import defaultdict

    from .enrichment.base import get_llm_backend
    from .models.enrichment import CallEnrichment
    from .models.silver import Utterance
    from .quality.judge import aggregate_verdicts, judge_enrichment
    from .storage import read_models

    settings = _setup()
    backend = get_llm_backend(settings)

    utterances: dict[str, list[Utterance]] = defaultdict(list)
    for path in sorted(settings.silver_dir.glob("dt=*/utterances.jsonl")):
        for utt in read_models(path, Utterance):
            utterances[utt.call_id].append(utt)
    for utts in utterances.values():
        utts.sort(key=lambda u: u.turn_index)

    enrichments: list[CallEnrichment] = []
    for path in sorted(settings.silver_dir.glob("dt=*/enrichment.jsonl")):
        enrichments.extend(read_models(path, CallEnrichment))

    if not enrichments:
        console.print("[red]No enrichment found. Run `ccma enrich` first.[/red]")
        raise typer.Exit(1)

    selected = enrichments[:sample]
    verdicts = [
        judge_enrichment(e, utterances.get(e.call_id, []), backend, settings.ollama_model_judge)
        for e in selected
    ]
    summary = aggregate_verdicts(verdicts)
    console.print_json(json.dumps(summary, indent=2))
    if summary.get("judge_degenerate"):
        console.print(
            "[yellow]Judge scores show no spread — treat this evaluation as invalid.[/yellow]"
        )


@app.command()
def query(sql: str = typer.Argument(..., help="SQL to run against the warehouse")) -> None:
    """Run ad-hoc SQL against the Gold warehouse."""
    from .gold.build import query as run_query

    settings = _setup()
    df = run_query(settings, sql)
    console.print(df.to_string(index=False, max_rows=60))


@app.command()
def kpis() -> None:
    """Print the executive KPI summary."""
    from .gold.build import query as run_query
    from .gold.kpis import EXEC_SUMMARY_SQL

    settings = _setup()
    df = run_query(settings, EXEC_SUMMARY_SQL)
    if df.empty:
        console.print("[red]No data. Run `ccma run-all` first.[/red]")
        raise typer.Exit(1)

    table = Table(title="Executive KPI summary")
    table.add_column("KPI", style="cyan")
    table.add_column("Value", justify="right")
    for column in df.columns:
        value = df.iloc[0][column]
        formatted = f"{value:,.3f}" if isinstance(value, float) else f"{value:,}"
        table.add_row(column.replace("_", " ").title(), formatted)
    console.print(table)


@app.command()
def doctor() -> None:
    """Check the environment: paths, backends, optional dependencies."""
    settings = _setup()
    table = Table(title="Environment")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    table.add_row("data_root", "OK" if settings.data_root.exists() else "MISSING", str(settings.data_root))
    table.add_row("asr_backend", "OK", settings.asr_backend)
    table.add_row("llm_backend", "OK", settings.llm_backend)

    if settings.llm_backend == "ollama":
        from .enrichment.base import OllamaBackend

        backend = OllamaBackend(settings)
        healthy = backend.health()
        models = backend.available_models() if healthy else []
        table.add_row(
            "ollama",
            "OK" if healthy else "UNREACHABLE",
            f"{settings.ollama_host} models={len(models)}",
        )
        for required in (settings.ollama_model, settings.ollama_model_judge, settings.ollama_embed_model):
            present = any(m.split(":")[0] == required.split(":")[0] for m in models)
            table.add_row(f"  model {required}", "OK" if present else "MISSING",
                          "" if present else f"run: ollama pull {required}")

    for module, purpose in (
        ("faster_whisper", "real ASR"),
        ("streamlit", "dashboard"),
        ("sklearn", "escalation model"),
        ("duckdb", "warehouse"),
    ):
        try:
            __import__(module)
            table.add_row(module, "OK", purpose)
        except ImportError:
            table.add_row(module, "MISSING", f"{purpose} unavailable")

    table.add_row(
        "warehouse",
        "OK" if settings.warehouse_path.exists() else "NOT BUILT",
        str(settings.warehouse_path),
    )
    console.print(table)


@app.command()
def schemas(output: Path = typer.Option(Path("schemas"), help="Directory to write into")) -> None:
    """Export JSON Schemas for every persisted model (contract documentation)."""
    from .enrichment.schema import enrichment_json_schema
    from .models.bronze import AsrSegment, CallMetadata
    from .models.enrichment import CallEnrichment
    from .models.silver import CallQuality, ConversationLanguageProfile, Utterance
    from .storage import write_json

    output.mkdir(parents=True, exist_ok=True)
    exports = {
        "bronze_call_metadata": CallMetadata.model_json_schema(),
        "bronze_asr_segment": AsrSegment.model_json_schema(),
        "silver_utterance": Utterance.model_json_schema(),
        "silver_call_quality": CallQuality.model_json_schema(),
        "silver_language_profile": ConversationLanguageProfile.model_json_schema(),
        "silver_call_enrichment": CallEnrichment.model_json_schema(),
        "llm_output_contract": enrichment_json_schema(),
    }
    for name, schema in exports.items():
        write_json(output / f"{name}.schema.json", schema)
    console.print(f"[green]Wrote {len(exports)} schemas to[/green] {output}")


if __name__ == "__main__":
    app()
