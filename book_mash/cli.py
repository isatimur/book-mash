import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from book_mash.config import load_config
from book_mash.corpus.loader import load_chapters
from book_mash.output.annotations import write_annotations
from book_mash.output.ledger import write_ledger
from book_mash.output.report import render_report
from book_mash.runners.measurement import run_measurement


app = typer.Typer(help="MASH-style multi-judge measurement engine for book manuscripts.")
console = Console()

_DIM_ORDER = ["humanness", "voice", "usefulness", "evidence_density", "claim_defensibility", "redundancy"]


@app.command()
def measure(config: str = typer.Option(..., "--config", help="Path to book-mash.toml")):
    """Run a measurement pass over the manuscript."""
    cfg = load_config(config)
    run = asyncio.run(run_measurement(cfg))
    run_dir = Path(cfg.runs_dir) / run.id
    write_ledger(run, run_dir)
    render_report(run, run_dir)
    write_annotations(
        load_chapters(cfg.chapters_glob),
        run.scores,
        run_dir / "annotations",
        run_id=run.id,
    )
    _print_summary(run, run_dir)


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
