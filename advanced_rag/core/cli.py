"""Command-line interface: arag ingest | ask | chat | eval | info."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings
from .evaluate import Evaluator, load_dataset, report_to_dict
from .pipeline import RAGPipeline

app = typer.Typer(add_completion=False, help="Advanced RAG on Google Gemini.")
console = Console()


def _check_key() -> None:
    try:
        get_settings().require_key()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    semantic: bool = typer.Option(False, help="Use embedding-based semantic chunking"),
    graph: bool = typer.Option(True, help="Build the GraphRAG knowledge graph"),
):
    """Load -> chunk -> embed -> build graph -> persist."""
    _check_key()
    pipe = RAGPipeline()
    with console.status("[bold]Ingesting (chunk → embed → graph)…"):
        n = pipe.ingest(path, build_graph=graph, semantic=semantic)
    g = pipe.graph
    console.print(Panel.fit(
        f"Indexed [bold]{n}[/bold] chunks from [cyan]{path}[/cyan]\n"
        + (f"Graph: {g.graph.number_of_nodes()} entities, "
           f"{g.graph.number_of_edges()} relations, {len(g.communities)} communities"
           if g else "Graph: skipped"),
        title="Ingestion complete",
    ))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question"),
    mode: str = typer.Option("agentic", help="agentic | simple"),
    source: Optional[list[str]] = typer.Option(None, help="Restrict to sources (substring)"),
    trace: bool = typer.Option(False, help="Show retrieval/agent trace"),
):
    """Ask a single question against the persisted index."""
    _check_key()
    pipe = RAGPipeline.load()
    with console.status(f"[bold]Thinking ({mode})…"):
        res = pipe.ask(question, mode=mode, sources=source)
    console.print(Panel(res.answer.render(), title=f"Answer ({res.mode})", border_style="green"))
    if trace:
        _print_trace(res)


@app.command()
def chat(mode: str = typer.Option("agentic", help="agentic | simple")):
    """Interactive multi-turn chat with conversation memory."""
    _check_key()
    pipe = RAGPipeline.load()
    console.print("[dim]Type your questions. Ctrl-C or 'exit' to quit.[/dim]")
    while True:
        try:
            q = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        with console.status(f"[bold]Thinking ({mode})…"):
            res = pipe.chat(q, mode=mode)
        console.print(Panel(res.answer.render(), border_style="green"))


@app.command()
def evaluate(
    dataset: str = typer.Argument(..., help="Path to eval JSON dataset"),
    mode: str = typer.Option("agentic", help="agentic | simple"),
    out: Optional[str] = typer.Option(None, help="Write full report JSON here"),
):
    """Run the evaluation harness and print a metrics summary."""
    _check_key()
    pipe = RAGPipeline.load()
    data = load_dataset(dataset)
    with console.status(f"[bold]Evaluating {len(data)} cases ({mode})…"):
        report = Evaluator(pipe).run(data, mode=mode)
    table = Table(title="Evaluation summary")
    table.add_column("metric"); table.add_column("score", justify="right")
    for k, v in report.summary().items():
        table.add_row(k, f"{v:.3f}")
    console.print(table)
    if out:
        Path(out).write_text(json.dumps(report_to_dict(report), indent=2))
        console.print(f"[dim]Full report → {out}[/dim]")


@app.command()
def info():
    """Show stats about the persisted index."""
    s = get_settings()
    if not (s.index_dir / "chunks.jsonl").exists():
        console.print(f"[yellow]No index at {s.index_dir}. Run `arag ingest` first.[/yellow]")
        raise typer.Exit()
    pipe = RAGPipeline.load()
    table = Table(title="Index info")
    table.add_column("field"); table.add_column("value", justify="right")
    table.add_row("index dir", str(s.index_dir))
    table.add_row("chunks", str(len(pipe.store.chunks)))
    table.add_row("embed model", s.embed_model)
    table.add_row("embed dim", str(s.embed_dim))
    table.add_row("gen model", s.gen_model)
    if pipe.graph:
        table.add_row("graph entities", str(pipe.graph.graph.number_of_nodes()))
        table.add_row("graph relations", str(pipe.graph.graph.number_of_edges()))
        table.add_row("communities", str(len(pipe.graph.communities)))
    console.print(table)


def _print_trace(res) -> None:
    lines: list[str] = []
    if res.agent:
        lines += [f"plan: {p}" for p in res.agent.plan]
        lines += res.agent.trace
    elif res.retrieval:
        lines += res.retrieval.trace
    if lines:
        console.print(Panel("\n".join(lines), title="Trace", border_style="blue"))


if __name__ == "__main__":
    app()
