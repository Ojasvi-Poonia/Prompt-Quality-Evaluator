#!/usr/bin/env python3

import argparse
import json
import sys
import os

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from evaluator.scorer import evaluate, evaluate_with_confidence, ablation_analysis
from evaluator.improvements import build_rewrite_template
from evaluator.visualizer import generate_radar_chart, print_report


def _ensure_nlp_data():
    import nltk

    for resource, path in [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("brown", "corpora/brown"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)

    try:
        import spacy
        for model_name in ("en_core_web_md", "en_core_web_sm"):
            try:
                spacy.load(model_name)
                break
            except OSError:
                continue
    except OSError:
        from spacy.cli import download
        download("en_core_web_md")


def _load_prompt(args) -> str:
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as fh:
            text = fh.read()
    elif args.prompt:
        text = args.prompt
    else:
        return ""
    return text


def _print_compare(result_a: dict, result_b: dict, console: Console) -> None:
    a_score = result_a["final_score"]
    b_score = result_b["final_score"]
    diff = a_score - b_score

    console.print()
    if abs(diff) < 1.0:
        verdict = "[bold yellow]Tie[/bold yellow]"
    elif diff > 0:
        verdict = f"[bold green]Prompt A wins by {abs(diff):.1f} points[/bold green]"
    else:
        verdict = f"[bold green]Prompt B wins by {abs(diff):.1f} points[/bold green]"

    header = f"  COMPARISON: A={a_score:.1f} vs B={b_score:.1f}    {verdict}"
    console.print(Panel(header, padding=(1, 2)))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Dimension", style="bold")
    table.add_column("Prompt A", justify="center")
    table.add_column("Prompt B", justify="center")
    table.add_column("Δ (A-B)", justify="center")
    table.add_column("Winner", justify="center")

    a_dims = {d["name"]: d["score"] for d in result_a["rubric_dimensions"]}
    b_dims = {d["name"]: d["score"] for d in result_b["rubric_dimensions"]}

    for name in a_dims:
        a_val = a_dims[name]
        b_val = b_dims.get(name, 0.0)
        d = a_val - b_val
        if abs(d) < 0.05:
            winner = "[yellow]≈[/yellow]"
        elif d > 0:
            winner = "[green]A[/green]"
        else:
            winner = "[green]B[/green]"
        table.add_row(name, f"{a_val:.2f}", f"{b_val:.2f}", f"{d:+.2f}", winner)

    console.print(table)
    console.print()


def _print_ablation(text: str, console: Console) -> None:
    result = ablation_analysis(text)

    console.print()
    console.print(Panel(
        f"  ABLATION ANALYSIS    Final Score: [bold]{result['final_score']:.1f}[/bold]",
        padding=(1, 2),
    ))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Dimension", style="bold")
    table.add_column("Score", justify="center")
    table.add_column("Contribution to Final", justify="center")
    table.add_column("Bar", justify="left")

    contributions = result["dimension_contributions"]
    sorted_contrib = sorted(
        contributions.items(),
        key=lambda x: -x[1]["weighted_contribution_to_final"],
    )

    for name, info in sorted_contrib:
        score = info["score"]
        contrib = info["weighted_contribution_to_final"]
        bar_width = max(0, int(contrib / 12 * 20))
        bar = "█" * bar_width + "░" * (20 - bar_width)
        color = "green" if score >= 0.7 else ("yellow" if score >= 0.5 else "red")
        table.add_row(
            name,
            f"[{color}]{score:.2f}[/{color}]",
            f"[{color}]{contrib:.1f} pts[/{color}]",
            f"[{color}]{bar}[/{color}]",
        )

    console.print(table)

    console.print("\n[bold]Top 3 Strongest Dimensions:[/bold]")
    for d in result["highest_dimensions"]:
        console.print(f"  - {d['name']} ({d['score']:.2f})")
    console.print("\n[bold]Top 3 Weakest Dimensions (focus here):[/bold]")
    for d in result["lowest_dimensions"]:
        console.print(f"  - {d['name']} ({d['score']:.2f})")
    console.print()


def _run_batch(prompts_file: str, output_csv: str, console: Console) -> None:
    if not os.path.isfile(prompts_file):
        console.print(f"[red]Error: prompts file not found: {prompts_file}[/red]")
        sys.exit(1)

    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    console.print(f"[dim]Evaluating {len(data)} prompts...[/dim]\n")

    for i, entry in enumerate(data, 1):
        if isinstance(entry, str):
            prompt = entry
            label = f"#{i}"
        else:
            prompt = entry.get("prompt", "")
            label = entry.get("tier") or entry.get("label") or f"#{i}"

        if not prompt.strip():
            continue

        r = evaluate(prompt)
        rows.append({
            "id": i,
            "label": label,
            "preview": prompt[:80] + ("..." if len(prompt) > 80 else ""),
            "score": r["final_score"],
            "verdict": r["label"],
            "domain": r.get("detected_domain", "general"),
        })

    rows.sort(key=lambda x: -x["score"])

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Rank", justify="right")
    table.add_column("Label", style="bold")
    table.add_column("Score", justify="center")
    table.add_column("Verdict", justify="center")
    table.add_column("Domain", justify="center")
    table.add_column("Prompt")

    for rank, row in enumerate(rows, 1):
        color = "green" if row["score"] >= 60 else ("yellow" if row["score"] >= 40 else "red")
        table.add_row(
            str(rank),
            row["label"],
            f"[{color}]{row['score']:.1f}[/{color}]",
            f"[{color}]{row['verdict']}[/{color}]",
            row["domain"],
            row["preview"],
        )

    console.print(table)

    if output_csv:
        import csv
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "label", "preview", "score", "verdict", "domain"])
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"\n[dim]CSV saved to: {os.path.abspath(output_csv)}[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM prompt quality using rule-based, semantic, and information-theoretic metrics.",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt text to evaluate (inline).")
    parser.add_argument("--file", "-f", help="Read prompt from a text file.")
    parser.add_argument("--json", "-j", action="store_true", dest="json_output", help="Output raw JSON.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all individual metrics.")
    parser.add_argument("--no-chart", action="store_true", help="Skip radar chart generation.")
    parser.add_argument("--output", "-o", default="output", help="Output directory for chart (default: output/).")

    parser.add_argument("--rewrite", action="store_true",
                        help="Print a template-based improved version of the prompt.")
    parser.add_argument("--confidence", action="store_true",
                        help="Compute bootstrap 95%% confidence interval (slower, ~30 evaluations).")
    parser.add_argument("--ablation", action="store_true",
                        help="Show per-dimension contribution to the final score.")

    parser.add_argument("--compare", help="Path to a second prompt file. Will compare prompt vs --compare.")

    parser.add_argument("--batch", help="Path to JSON file with list of prompts (or {tier, prompt} entries).")
    parser.add_argument("--export-csv", help="When using --batch, write results to this CSV path.")

    args = parser.parse_args()

    _ensure_nlp_data()

    console = Console(stderr=True) if args.json_output else Console()

    if args.batch:
        _run_batch(args.batch, args.export_csv, console)
        return

    text = _load_prompt(args)
    if not text:
        parser.print_help()
        sys.exit(1)
    if not text.strip():
        print("Error: prompt text is empty.", file=sys.stderr)
        sys.exit(1)

    if args.compare:
        if not os.path.isfile(args.compare):
            console.print(f"[red]Compare file not found: {args.compare}[/red]")
            sys.exit(1)
        with open(args.compare, "r", encoding="utf-8") as f:
            text_b = f.read()
        if not args.json_output:
            console.print("[dim]Evaluating both prompts...[/dim]")
        result_a = evaluate(text)
        result_b = evaluate(text_b)
        if args.json_output:
            print(json.dumps({"prompt_a": result_a, "prompt_b": result_b}, indent=2, default=str))
        else:
            _print_compare(result_a, result_b, console)
        return

    if args.ablation:
        _print_ablation(text, console)
        return

    if not args.json_output:
        if args.confidence:
            console.print("[dim]Evaluating prompt with bootstrap confidence intervals...[/dim]\n")
        else:
            console.print("[dim]Evaluating prompt...[/dim]\n")

    if args.confidence:
        result = evaluate_with_confidence(text)
    else:
        result = evaluate(text)

    if args.json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result, console=console)

        if args.confidence and "confidence_interval_95" in result:
            ci = result["confidence_interval_95"]
            console.print(
                f"[dim]95% confidence interval: [{ci[0]:.1f}, {ci[1]:.1f}]  "
                f"(std={result['score_std']:.2f}, n={result['bootstrap_samples']})[/dim]"
            )

        if args.verbose:
            console.print("\n[bold]All Raw Metrics[/bold]")
            for key, val in sorted(result["raw_metrics"].items()):
                console.print(f"  {key}: {val['score']:.3f}")

        if args.rewrite:
            console.print()
            template = build_rewrite_template(
                result["raw_metrics"],
                result["rubric_dimensions"],
                result.get("detected_domain", "general"),
                text,
            )
            console.print(template)

    if not args.no_chart and not args.json_output:
        chart_path = os.path.join(args.output, "radar_chart.png")
        saved = generate_radar_chart(result["rubric_dimensions"], output_path=chart_path)
        console.print(f"[dim]Radar chart saved to: {saved}[/dim]")


if __name__ == "__main__":
    main()
