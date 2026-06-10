import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from book_mash.config import load_config
from book_mash.corpus.loader import load_chapters
from book_mash.init import run_init
from book_mash.output.annotations import write_annotations
from book_mash.output.ledger import write_ledger
from book_mash.output.report import render_report
from book_mash.runners.measurement import run_measurement
from book_mash.runners.planner import MeasurementPlan, plan_measurement


app = typer.Typer(help="MASH-style multi-judge measurement engine for book manuscripts.")
console = Console()

_DIM_ORDER = ["humanness", "voice", "usefulness", "evidence_density", "claim_defensibility", "redundancy"]


@app.command()
def init(
    directory: str = typer.Option(
        ".",
        "--directory",
        "-d",
        help="Target directory in which to write book-mash.toml.",
    ),
) -> None:
    """Generate a book-mash.toml interactively."""
    run_init(Path(directory).resolve())


@app.command()
def measure(
    config: str = typer.Option(..., "--config", help="Path to book-mash.toml"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show planned LLM calls + cost estimate without executing any judge calls.",
    ),
):
    """Run a measurement pass over the manuscript."""
    cfg = load_config(config)
    if dry_run:
        plan = plan_measurement(cfg)
        _print_dry_run(plan)
        return
    run = asyncio.run(run_measurement(cfg))
    run_dir = Path(cfg.runs_dir) / run.id
    write_ledger(run, run_dir)
    render_report(run, run_dir)
    write_annotations(
        load_chapters(cfg.chapters_glob, cfg.skip_sections),
        run.scores,
        run_dir / "annotations",
        run_id=run.id,
    )
    _print_summary(run, run_dir)


def _print_dry_run(plan: MeasurementPlan) -> None:
    console.rule("book-mash measure — dry run")
    console.print(
        f"Corpus: [bold]{plan.n_chapters}[/bold] chapter(s), "
        f"[bold]{plan.n_sections}[/bold] section(s), "
        f"[bold]{plan.n_paragraphs}[/bold] paragraph(s)."
    )
    console.print()

    table = Table(title="Planned judge calls (no LLM will be invoked)")
    table.add_column("Dimension")
    table.add_column("Granularity")
    table.add_column("Model")
    table.add_column("Calls", justify="right")
    table.add_column("Est. cost", justify="right")
    for d in plan.dims:
        table.add_row(
            d.dim_name,
            d.granularity,
            d.model_id,
            f"{d.call_count:,}",
            f"${d.estimated_cost_usd:.2f}",
        )
    table.add_row(
        "[bold]Total[/bold]",
        "",
        "",
        f"[bold]{plan.total_calls:,}[/bold]",
        f"[bold]${plan.total_estimated_cost_usd:.2f}[/bold]",
    )
    console.print(table)
    console.print()

    over_budget = plan.total_estimated_cost_usd > plan.budget_cap_usd
    budget_line = f"Budget cap (from \\[budget] max_cost_usd): ${plan.budget_cap_usd:.2f}"
    if over_budget:
        console.print(f"[red]{budget_line} — plan exceeds budget by "
                      f"${plan.total_estimated_cost_usd - plan.budget_cap_usd:.2f}.[/red]")
    else:
        console.print(f"[green]{budget_line} — plan fits within budget.[/green]")

    if plan.embedder_available:
        console.print(
            f"[dim]Embedding prefilter: ~{plan.estimated_embedding_calls} call(s); "
            f"cost not estimated here (Voyage AI / OpenAI tier-dependent).[/dim]"
        )
    else:
        console.print(
            "[yellow]No embedding client available — redundancy will skip the prefilter. "
            "Consider setting VOYAGE_API_KEY or OPENAI_API_KEY for cheaper redundancy.[/yellow]"
        )

    console.print()
    console.print(
        "[dim]Estimates assume a cold cache. Warm-cache reruns typically reuse ~90% of calls. "
        "Per-call token counts are typical, not exact.[/dim]"
    )
    console.print()
    console.print("Run [cyan]book-mash measure --config <toml>[/cyan] (without --dry-run) to execute.")


def _print_summary(run, run_dir: Path) -> None:
    console.rule(f"book-mash run {run.id}")
    console.print(f"Cost: ${run.total_cost_usd:.2f} - Status: {run.status.value}")
    console.print(f"Outputs: {run_dir}")
    book = run.rollups.get("book", {})
    if book:
        table = Table(title="Book-level heatmap")
        for dim in _DIM_ORDER:
            table.add_column(dim)
        row = []
        for dim in _DIM_ORDER:
            val = book.get(dim)
            row.append(f"{val:.0f}" if isinstance(val, (int, float)) else "-")
        table.add_row(*row)
        console.print(table)
