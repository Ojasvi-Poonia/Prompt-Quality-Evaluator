import os
import math
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def generate_radar_chart(
    rubric_dimensions: List[dict],
    output_path: str = "output/radar_chart.png",
) -> str:
    labels = [d["name"] for d in rubric_dimensions]
    scores = [d["score"] for d in rubric_dimensions]
    n = len(labels)

    angles = [i / n * 2 * math.pi for i in range(n)]
    angles += angles[:1]
    scores_closed = scores + scores[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", fontsize=8)
    ax.set_ylim(0, 1)

    ax.plot(angles, scores_closed, "o-", linewidth=2, color="#e74c3c")
    ax.fill(angles, scores_closed, alpha=0.25, color="#e74c3c")

    for angle, score in zip(angles[:-1], scores):
        ax.annotate(
            f"{score:.2f}",
            xy=(angle, score),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#2c3e50",
        )

    ax.set_title("Prompt Quality Radar (LLM-judged)", fontsize=14, fontweight="bold", pad=20)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return os.path.abspath(output_path)


def _score_color(score: float) -> str:
    if score >= 0.75:
        return "green"
    if score >= 0.50:
        return "yellow"
    if score >= 0.30:
        return "dark_orange"
    return "red"


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def print_report(result: dict, console: Optional[Console] = None) -> None:
    if console is None:
        console = Console()

    final_score = result["final_score"]
    label = result["label"]
    pillar_scores = result.get("pillar_scores", {})
    rubric_dims = result["rubric_dimensions"]
    suggestions = result.get("suggestions", [])
    domain = result.get("detected_domain", "general")
    engine = result.get("engine", "llm")
    cached = result.get("from_cache", False)

    color = _score_color(final_score / 100)

    header = Text()
    header.append(f"  PROMPT QUALITY SCORE: {final_score:.1f}/100  ", style=f"bold {color}")
    header.append(f"  [{label}]", style=f"bold {color}")
    if domain != "general":
        header.append(f"   Domain: {domain}", style="bold cyan")
    header.append(f"   Engine: {engine}", style="bold magenta")
    if cached:
        header.append("   (cached)", style="dim")
    console.print(Panel(header, border_style=color, padding=(1, 2)))

    if pillar_scores:
        console.print("\n[bold]Pillar Breakdown[/bold]")
        pillar_table = Table(show_header=True, header_style="bold cyan")
        pillar_table.add_column("Pillar", style="bold")
        pillar_table.add_column("Score", justify="center")
        pillar_table.add_column("Bar", justify="left")
        for name, score in pillar_scores.items():
            display_name = name.replace("_", " ").title()
            c = _score_color(score)
            pillar_table.add_row(
                display_name,
                f"[{c}]{score:.2f}[/{c}]",
                f"[{c}]{_bar(score)}[/{c}]",
            )
        console.print(pillar_table)

    console.print("\n[bold]Rubric Dimensions[/bold]")
    dim_table = Table(show_header=True, header_style="bold cyan")
    dim_table.add_column("Dimension", style="bold", min_width=22)
    dim_table.add_column("Score", justify="center", min_width=6)
    dim_table.add_column("Bar", min_width=22)
    dim_table.add_column("Label", justify="center", min_width=10)
    dim_table.add_column("Key Finding", max_width=50)

    for d in rubric_dims:
        c = _score_color(d["score"])
        finding = d["findings"][0] if d["findings"] else "-"
        dim_table.add_row(
            d["name"],
            f"[{c}]{d['score']:.2f}[/{c}]",
            f"[{c}]{_bar(d['score'])}[/{c}]",
            f"[{c}]{d['label']}[/{c}]",
            finding[:50],
        )
    console.print(dim_table)

    if suggestions:
        console.print("\n[bold]Top Improvement Suggestions[/bold]")
        for i, s in enumerate(suggestions, 1):
            console.print(f"  {i}. {s}")

    console.print()
