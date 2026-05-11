"""
Main entry point for the LLM translation quality judge.

Usage:
    # Run all models sequentially (each uses concurrent workers)
    python -m llm_judge.run

    # Run specific models
    python -m llm_judge.run --models "Qwen/Qwen3-30B-A3B,meta-llama/Llama-3.3-70B-Instruct"

    # Run on specific languages with a row limit (pilot)
    python -m llm_judge.run --languages "Arabic,French" --limit 20

    # Only run analysis on existing results
    python -m llm_judge.run --analyze-only

    # List available models
    python -m llm_judge.run --list-models

    # Export results to Excel
    python -m llm_judge.run --export-xlsx
"""

import argparse
import os
import sys

from .config import JUDGE_MODELS, INPUT_XLSX, RESULTS_DIR, HF_TOKEN, CONCURRENCY
from .data_loader import load_xlsx
from .judge import run_judge
from .analysis import generate_report, export_results_xlsx


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM Translation Quality Judge - evaluate translations with multiple LLMs"
    )
    p.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model IDs to run (default: all in config)"
    )
    p.add_argument(
        "--languages", type=str, default=None,
        help="Comma-separated language names (default: all 10)"
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Max items per language (for pilot runs)"
    )
    p.add_argument(
        "--input", type=str, default=INPUT_XLSX,
        help="Path to the input xlsx file"
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Do not resume from existing results (start fresh)"
    )
    p.add_argument(
        "--concurrency", type=int, default=None,
        help=f"Parallel API requests per model (default: {CONCURRENCY})"
    )
    p.add_argument(
        "--analyze-only", action="store_true",
        help="Skip judging, only run analysis on existing results"
    )
    p.add_argument(
        "--export-xlsx", action="store_true",
        help="Export analysis to Excel workbook"
    )
    p.add_argument(
        "--list-models", action="store_true",
        help="List configured judge models and exit"
    )
    p.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace API token (overrides HF_TOKEN env var)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("Configured judge models:")
        print("-" * 60)
        for m in JUDGE_MODELS:
            print(f"  {m['id']}")
            print(f"    Short: {m['short']}")
            print(f"    Notes: {m.get('notes', '')}")
            print()
        return

    token = args.token or HF_TOKEN
    if not token and not args.analyze_only:
        print("ERROR: HF_TOKEN not set.")
        print("Set it via: $env:HF_TOKEN='hf_your_token_here' (PowerShell)")
        print("       or: export HF_TOKEN=hf_your_token_here (bash)")
        sys.exit(1)

    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
    else:
        model_ids = [m["id"] for m in JUDGE_MODELS]

    languages = None
    if args.languages:
        languages = [l.strip() for l in args.languages.split(",")]

    concurrency = args.concurrency or CONCURRENCY

    # Run judging
    if not args.analyze_only:
        print(f"Loading data from {args.input}")
        items = load_xlsx(args.input, languages=languages, limit=args.limit)
        print(f"Loaded {len(items)} items")

        if languages:
            print(f"Languages: {languages}")
        print(f"Limit per language: {args.limit or 'all'}")
        print(f"Models to run: {len(model_ids)} (sequential)")
        print(f"Workers per model: {concurrency}")
        print()

        for i, model_id in enumerate(model_ids):
            print(f"\n{'='*60}")
            print(f"Model {i+1}/{len(model_ids)}: {model_id}")
            print(f"{'='*60}")
            try:
                run_judge(
                    model_id, items,
                    token=token,
                    resume=not args.no_resume,
                    concurrency=concurrency,
                )
            except KeyboardInterrupt:
                print("\n\nCtrl+C — stopping. Progress is saved.")
                sys.exit(1)
            except Exception as e:
                print(f"FAILED: {e}")
                print("Continuing with next model...")

    # Analysis
    if os.path.exists(RESULTS_DIR) and os.listdir(RESULTS_DIR):
        print(f"\n{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")
        report = generate_report(output_path=os.path.join(RESULTS_DIR, "report.txt"))
        print(report)

        if args.export_xlsx or args.analyze_only:
            export_results_xlsx()
    else:
        print("\nNo results found for analysis.")


if __name__ == "__main__":
    main()
