#!/usr/bin/env python3
"""CLI entry point for the Prompt Quality Evaluator.

Usage examples::

    python main.py "Your prompt text here"
    python main.py --file prompt.txt
    python main.py "Your prompt" --json
    python main.py "Your prompt" --verbose
    python main.py "Your prompt" --no-chart
    python main.py "Your prompt" --output report/
"""

import argparse
import json
import sys
import os

from rich.console import Console

from evaluator.scorer import evaluate
from evaluator.visualizer import generate_radar_chart, print_report


def _ensure_nlp_data():
    """Download required NLTK data and spaCy model if missing."""
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
        spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")


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

    args = parser.parse_args()

    # --- Resolve prompt text ---
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as fh:
            text = fh.read()
    elif args.prompt:
        text = args.prompt
    else:
        parser.print_help()
        sys.exit(1)

    if not text.strip():
        print("Error: prompt text is empty.", file=sys.stderr)
        sys.exit(1)

    # --- Ensure NLP resources ---
    _ensure_nlp_data()

    # --- Evaluate ---
    console = Console(stderr=True) if args.json_output else Console()

    if not args.json_output:
        console.print("[dim]Evaluating prompt...[/dim]\n")

    result = evaluate(text)

    # --- Output ---
    if args.json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result, console=console)

        if args.verbose:
            console.print("\n[bold]All Raw Metrics[/bold]")
            for key, val in sorted(result["raw_metrics"].items()):
                console.print(f"  {key}: {val['score']:.3f}")

    # --- Radar chart ---
    if not args.no_chart:
        chart_path = os.path.join(args.output, "radar_chart.png")
        saved = generate_radar_chart(result["rubric_dimensions"], output_path=chart_path)
        if not args.json_output:
            console.print(f"[dim]Radar chart saved to: {saved}[/dim]")


if __name__ == "__main__":
    main()
