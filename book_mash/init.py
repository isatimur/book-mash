"""Interactive wizard that generates a book-mash.toml for a consumer repo."""

from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, Prompt


console = Console()


def render_toml(
    chapters_glob: str,
    claims_dir: str,
    evidence_dir: str,
    voice_baseline_chapters: list[str],
    runs_dir: str,
    max_cost_usd: float,
) -> str:
    """Render a book-mash.toml string from the wizard's collected answers.

    Pure function — easy to test without driving the interactive prompts.
    """
    lines = ["[corpus]"]
    lines.append(f'chapters_glob = "{chapters_glob}"')
    lines.append(f'claims_dir = "{claims_dir}"')
    lines.append(f'evidence_dir = "{evidence_dir}"')
    if voice_baseline_chapters:
        baseline_repr = ", ".join(f'"{c}"' for c in voice_baseline_chapters)
        lines.append(f"voice_baseline_chapters = [{baseline_repr}]")
    lines.append("")
    lines.append("[output]")
    lines.append(f'runs_dir = "{runs_dir}"')
    lines.append("")
    lines.append("[budget]")
    lines.append(f"max_cost_usd = {max_cost_usd}")
    lines.append("")
    return "\n".join(lines)


def run_init(target_dir: Path) -> Path | None:
    """Drive the interactive wizard and write book-mash.toml to target_dir.

    Returns the path of the written file, or None if the user aborted.
    """
    target = target_dir / "book-mash.toml"
    if target.exists():
        if not Confirm.ask(
            f"[yellow]{target} already exists. Overwrite?[/yellow]",
            default=False,
        ):
            console.print("[red]Aborted.[/red]")
            return None

    console.rule("book-mash init")
    console.print(
        "Generate a book-mash.toml for your project. Answer 5 questions; "
        "defaults shown in brackets are usually fine."
    )
    console.print()

    chapters_glob = Prompt.ask(
        "Chapter files glob (relative to this directory)",
        default="public/drafting/*.md",
    )
    claims_dir = Prompt.ask("Claims directory", default="claims/")
    evidence_dir = Prompt.ask("Evidence directory", default="evidence/")

    # Suggest a voice baseline by listing the matching files
    matches = sorted(target_dir.glob(chapters_glob))
    if matches:
        console.print(f"[dim]Found {len(matches)} matching chapter file(s).[/dim]")
        baseline_default = matches[0].name
    else:
        console.print(
            f"[yellow]No files match {chapters_glob!r} in {target_dir} — "
            f"voice baseline will be hard to fill in.[/yellow]"
        )
        baseline_default = ""

    baseline_raw = Prompt.ask(
        "Voice baseline chapters (comma-separated filenames; leave blank to skip voice judging)",
        default=baseline_default,
    )
    voice_baseline_chapters = [s.strip() for s in baseline_raw.split(",") if s.strip()]

    runs_dir = Prompt.ask(
        "Run output directory (add this to .gitignore)",
        default=".book-mash-runs/",
    )
    max_cost_usd = FloatPrompt.ask("Cost budget per run, USD", default=10.0)

    content = render_toml(
        chapters_glob=chapters_glob,
        claims_dir=claims_dir,
        evidence_dir=evidence_dir,
        voice_baseline_chapters=voice_baseline_chapters,
        runs_dir=runs_dir,
        max_cost_usd=max_cost_usd,
    )
    target.write_text(content)

    console.print()
    console.print(f"[green]Wrote {target}[/green]")
    console.print()
    console.print("Next steps:")
    console.print(f"  1. Add [cyan]{runs_dir}[/cyan] to your repo's .gitignore")
    console.print("  2. Set [cyan]ANTHROPIC_API_KEY[/cyan] in your environment")
    console.print(
        f"  3. Run [cyan]book-mash measure --config {target.name}[/cyan]"
    )

    return target
