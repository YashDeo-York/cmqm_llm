"""
Entry point for MQM scoring of translation errors.

Only scores items that were flagged edit_required=yes by each model's
CMQM results, filtered to selected languages.

Usage:
    # Run all models on Chinese Mandarin + Urdu (default)
    python -m llm_judge.run_mqm

    # Run specific models
    python -m llm_judge.run_mqm --models "Qwen/Qwen3-30B-A3B,Qwen/Qwen3-8B"

    # Run on specific languages
    python -m llm_judge.run_mqm --languages "Chinese Mandarin,Urdu,Arabic"

    # Dry run: show item counts without calling the API
    python -m llm_judge.run_mqm --dry-run
"""

import argparse
import sys

from .config import JUDGE_MODELS, INPUT_XLSX, HF_TOKEN, CONCURRENCY
from .data_loader import load_xlsx
from .mqm_judge import run_mqm_judge, load_edit_required_items

DEFAULT_LANGUAGES = ["Chinese Mandarin", "Urdu"]


def parse_args():
    p = argparse.ArgumentParser(
        description="MQM scoring for translation errors (edit_required=yes items only)"
    )
    p.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model IDs (default: all in config)",
    )
    p.add_argument(
        "--languages", type=str, default=None,
        help=f"Comma-separated languages (default: {', '.join(DEFAULT_LANGUAGES)})",
    )
    p.add_argument(
        "--concurrency", type=int, default=None,
        help=f"Parallel API requests per model (default: {CONCURRENCY})",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, ignoring existing MQM results",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show item counts per model/language without calling the API",
    )
    p.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace API token (overrides HF_TOKEN env var)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    token = args.token or HF_TOKEN
    if not token and not args.dry_run:
        print("ERROR: HF_TOKEN not set.")
        print("Set it via: $env:HF_TOKEN='hf_your_token_here' (PowerShell)")
        sys.exit(1)

    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
    else:
        model_ids = [m["id"] for m in JUDGE_MODELS]

    languages = DEFAULT_LANGUAGES
    if args.languages:
        languages = [l.strip() for l in args.languages.split(",")]

    concurrency = args.concurrency or CONCURRENCY

    print(f"Loading data from {INPUT_XLSX}")
    all_items = load_xlsx(INPUT_XLSX)
    print(f"Loaded {len(all_items)} total items")
    print(f"Languages for MQM: {languages}")
    print(f"Models: {len(model_ids)}")
    print()

    for i, model_id in enumerate(model_ids):
        print(f"{'='*60}")
        print(f"Model {i+1}/{len(model_ids)}: {model_id}")
        print(f"{'='*60}")

        try:
            items = load_edit_required_items(model_id, languages, all_items)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        per_lang = {}
        for it in items:
            per_lang[it.language] = per_lang.get(it.language, 0) + 1
        for lang, count in sorted(per_lang.items()):
            print(f"  {lang}: {count} items needing MQM scoring")
        print(f"  Total: {len(items)} items")

        if args.dry_run:
            print("  (dry run — skipping API calls)")
            continue

        if not items:
            print("  No edit_required=yes items for this model/language combo")
            continue

        try:
            run_mqm_judge(
                model_id, items,
                token=token,
                resume=not args.no_resume,
                concurrency=concurrency,
            )
        except KeyboardInterrupt:
            print("\n\nCtrl+C — stopping. Progress is saved.")
            sys.exit(1)
        except Exception as e:
            print(f"  FAILED: {e}")
            print("  Continuing with next model...")

    print("\nMQM scoring complete.")


if __name__ == "__main__":
    main()
