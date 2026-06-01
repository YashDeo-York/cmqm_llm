"""
Clinical harm re-scoring runner.

Re-scores edit_required=yes items using the human-aligned harm scale
(none | minor | major) instead of the original LLM scale (low | moderate | high).

Usage:
    python -m llm_judge.harm_rescore
    python -m llm_judge.harm_rescore --models "Qwen/Qwen3-30B-A3B"
    python -m llm_judge.harm_rescore --dry-run
"""

import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.n = kwargs.get("initial", 0)
            self.total = kwargs.get("total", 0)
            self.postfix = kwargs.get("postfix", [0])
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
                    self.n += 1
        def update(self, n=1):
            self.n += n
        def close(self):
            pass
        @staticmethod
        def write(message):
            print(message)

import sys
from .config import JUDGE_MODELS, INPUT_XLSX, RESULTS_DIR, HF_TOKEN, CONCURRENCY
from .data_loader import load_xlsx, EvalItem
from .prompts import build_harm_rescore_prompt
from .hf_client import call_hf_chat, SpendingLimitError

HARM_DIR = os.path.join(RESULTS_DIR, "harm_rescore")
VALID_HARMS = {"none", "minor", "major"}

_shutdown = threading.Event()


def _sanitize(model_id: str) -> str:
    return model_id.replace("/", "__").replace(".", "_")


def _result_path(model_id: str) -> str:
    os.makedirs(HARM_DIR, exist_ok=True)
    return os.path.join(HARM_DIR, f"{_sanitize(model_id)}.jsonl")


def _load_existing_keys(path: str) -> set[str]:
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                keys.add(f"{rec['language']}|{rec['identifier']}")
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def load_edit_yes_items(model_id: str, all_items: list[EvalItem]) -> list[EvalItem]:
    """Get items where this model said edit_required=yes in CMQM results."""
    safe = _sanitize(model_id)
    cmqm_path = os.path.join(RESULTS_DIR, f"{safe}.jsonl")
    if not os.path.exists(cmqm_path):
        raise FileNotFoundError(f"No CMQM results at {cmqm_path}")

    edit_keys = set()
    with open(cmqm_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("edit_required") == "yes":
                edit_keys.add(f"{rec['language']}|{rec['identifier']}")

    return [it for it in all_items
            if f"{it.language}|{it.identifier}" in edit_keys]


def _normalize_harm(result: dict) -> dict:
    harm = str(result.get("harm", "")).strip().lower()
    if harm not in VALID_HARMS:
        # Try alternate field names
        harm = str(result.get("harm_potential", result.get("clinical_harm", ""))).strip().lower()
    if harm not in VALID_HARMS:
        return {"harm": "error", "rationale": str(result), "_parse_error": True}
    return {
        "harm": harm,
        "rationale": str(result.get("rationale", result.get("brief_rationale", ""))),
        "_parse_error": False,
    }


def _worker(work_q, result_q, model_id, token):
    while not _shutdown.is_set():
        try:
            item = work_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            break

        messages = build_harm_rescore_prompt(item)
        try:
            raw = call_hf_chat(model_id, messages, token=token)
            result = _normalize_harm(raw)
            if not _shutdown.is_set() and result.get("_parse_error"):
                time.sleep(1)
                raw = call_hf_chat(model_id, messages, token=token)
                result = _normalize_harm(raw)
        except SpendingLimitError as e:
            _shutdown.set()
            tqdm.write(f"\n  SPENDING LIMIT HIT: {e}")
            work_q.task_done()
            break
        except Exception as e:
            result = {"harm": "error", "rationale": f"API_ERROR: {str(e)[:200]}",
                      "_parse_error": True}

        result_q.put((item, result))
        work_q.task_done()


def run_harm_rescore(model_id, items, token=None, resume=True, concurrency=None):
    workers = concurrency or CONCURRENCY
    out_path = _result_path(model_id)
    done_keys = _load_existing_keys(out_path) if resume else set()

    pending = [it for it in items
               if f"{it.language}|{it.identifier}" not in done_keys]

    total = len(items)
    skipped = total - len(pending)
    if skipped:
        print(f"  Resuming: {skipped}/{total} already done, {len(pending)} remaining")
    if not pending:
        print(f"  All {total} items already scored for {model_id}")
        return out_path

    short_name = model_id.split("/")[-1][:25]
    print(f"  Workers: {workers} | Items: {len(pending)}")

    _shutdown.clear()
    work_q = queue.Queue(maxsize=workers * 2)
    result_q = queue.Queue()

    threads = []
    for _ in range(workers):
        t = threading.Thread(target=_worker,
                             args=(work_q, result_q, model_id, token),
                             daemon=True)
        t.start()
        threads.append(t)

    errors = 0
    processed = 0

    pbar = tqdm(total=total, desc=f"  Harm {short_name}", unit="item",
                initial=skipped,
                bar_format="{desc} |{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] errs={postfix[0]}",
                postfix=[errors])

    def feeder():
        for item in pending:
            if _shutdown.is_set():
                break
            work_q.put(item)
        for _ in range(workers):
            work_q.put(None)

    feed_thread = threading.Thread(target=feeder, daemon=True)
    feed_thread.start()

    try:
        with open(out_path, "a", encoding="utf-8") as f:
            while processed < len(pending) and not _shutdown.is_set():
                try:
                    item, result = result_q.get(timeout=2.0)
                except queue.Empty:
                    continue
                processed += 1
                if result.get("_parse_error"):
                    errors += 1
                record = {
                    "model": model_id,
                    "language": item.language,
                    "identifier": item.identifier,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **result,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                pbar.postfix[0] = errors
                pbar.update(1)
    except KeyboardInterrupt:
        print(f"\n  Ctrl+C — stopping. Progress saved.")
        _shutdown.set()

    pbar.close()
    if _shutdown.is_set():
        print(f"  Stopped early: {out_path} ({processed} saved)")
    else:
        print(f"  Done: {out_path} ({errors} parse errors)")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Re-score clinical harm on edit=yes items")
    p.add_argument("--models", type=str, default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--token", type=str, default=None)
    args = p.parse_args()

    token = args.token or HF_TOKEN
    if not token and not args.dry_run:
        print("ERROR: HF_TOKEN not set.")
        sys.exit(1)

    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
    else:
        model_ids = [m["id"] for m in JUDGE_MODELS]

    print(f"Loading data from {INPUT_XLSX}")
    all_items = load_xlsx(INPUT_XLSX)
    print(f"Loaded {len(all_items)} total items")
    print(f"Scale: none | minor | major (matching human annotators)\n")

    for i, model_id in enumerate(model_ids):
        print(f"{'='*60}")
        print(f"Model {i+1}/{len(model_ids)}: {model_id}")
        print(f"{'='*60}")

        try:
            items = load_edit_yes_items(model_id, all_items)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  {len(items)} edit_required=yes items")

        if args.dry_run:
            print("  (dry run)")
            continue

        if not items:
            continue

        try:
            run_harm_rescore(model_id, items, token=token,
                             resume=not args.no_resume,
                             concurrency=args.concurrency)
        except KeyboardInterrupt:
            print("\n\nCtrl+C — stopping. Progress saved.")
            sys.exit(1)
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\nHarm re-scoring complete.")


if __name__ == "__main__":
    main()
