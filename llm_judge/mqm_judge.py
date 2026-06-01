"""
MQM judge runner: scores translation errors using the GEMBA-MQM taxonomy.

Only processes items where edit_required == 'yes' in existing CMQM results,
filtered to selected languages. Reuses the same concurrent worker architecture
as judge.py.
"""

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

from .config import (
    RESULTS_DIR, CONCURRENCY, ALL_MQM_TYPES, MQM_SEVERITY_WEIGHTS,
)
from .data_loader import EvalItem
from .prompts import build_mqm_prompt
from .hf_client import call_hf_chat, SpendingLimitError

_shutdown = threading.Event()

MQM_RESULTS_DIR = os.path.join(RESULTS_DIR, "mqm")


def _sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(".", "_")


def _result_path(model_id: str) -> str:
    os.makedirs(MQM_RESULTS_DIR, exist_ok=True)
    safe = _sanitize_model_name(model_id)
    return os.path.join(MQM_RESULTS_DIR, f"{safe}.jsonl")


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


def _normalize_mqm_result(result: dict) -> dict:
    """Validate and normalize raw MQM JSON from the LLM."""
    errors = result.get("errors", [])
    if not isinstance(errors, list):
        return _error_result("errors field is not a list")

    normalized_errors = []
    for err in errors:
        if not isinstance(err, dict):
            continue
        cat = str(err.get("category", "other")).strip().lower()
        sev = str(err.get("severity", "minor")).strip().lower()
        explanation = str(err.get("explanation", ""))

        if sev not in MQM_SEVERITY_WEIGHTS:
            sev = "minor"

        normalized_errors.append({
            "category": cat,
            "severity": sev,
            "explanation": explanation,
        })

    # Compute score
    n_crit = sum(1 for e in normalized_errors if e["severity"] == "critical")
    n_maj = sum(1 for e in normalized_errors if e["severity"] == "major")
    n_min = sum(1 for e in normalized_errors if e["severity"] == "minor")
    score = max(-25, -(25 * n_crit) - (5 * n_maj) - (1 * n_min))

    return {
        "errors": normalized_errors,
        "mqm_score": score,
        "n_critical": n_crit,
        "n_major": n_maj,
        "n_minor": n_min,
        "_parse_error": False,
    }


def _error_result(reason: str) -> dict:
    return {
        "errors": [],
        "mqm_score": -99,
        "n_critical": 0,
        "n_major": 0,
        "n_minor": 0,
        "_parse_error": True,
        "error_reason": reason,
    }


def _worker(work_q: queue.Queue, result_q: queue.Queue,
            model_id: str, token: str | None):
    while not _shutdown.is_set():
        try:
            item = work_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            break

        messages = build_mqm_prompt(item)
        try:
            raw_result = call_hf_chat(model_id, messages, token=token)
            result = _normalize_mqm_result(raw_result)
            if not _shutdown.is_set() and result.get("_parse_error"):
                time.sleep(1)
                raw_result = call_hf_chat(model_id, messages, token=token)
                result = _normalize_mqm_result(raw_result)
        except SpendingLimitError as e:
            _shutdown.set()
            tqdm.write(f"\n  SPENDING LIMIT HIT: {e}")
            work_q.task_done()
            break
        except Exception as e:
            result = _error_result(f"API_ERROR: {str(e)[:200]}")

        result_q.put((item, result))
        work_q.task_done()


def load_edit_required_items(model_id: str, languages: list[str],
                             all_items: list[EvalItem]) -> list[EvalItem]:
    """
    Load identifiers where a CMQM model said edit_required=yes,
    then return the matching EvalItems filtered to the given languages.
    """
    safe = _sanitize_model_name(model_id)
    cmqm_path = os.path.join(RESULTS_DIR, f"{safe}.jsonl")
    if not os.path.exists(cmqm_path):
        raise FileNotFoundError(f"No CMQM results for {model_id} at {cmqm_path}")

    lang_set = set(languages)
    edit_keys = set()
    with open(cmqm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if (rec.get("edit_required") == "yes"
                        and rec.get("language") in lang_set):
                    edit_keys.add(f"{rec['language']}|{rec['identifier']}")
            except (json.JSONDecodeError, KeyError):
                continue

    items = [
        it for it in all_items
        if it.language in lang_set
        and f"{it.language}|{it.identifier}" in edit_keys
    ]
    return items


def run_mqm_judge(model_id: str, items: list[EvalItem],
                  token: str | None = None,
                  resume: bool = True,
                  concurrency: int | None = None) -> str:
    """Run MQM annotation for a single model over filtered items."""
    workers = concurrency or CONCURRENCY
    out_path = _result_path(model_id)
    done_keys = _load_existing_keys(out_path) if resume else set()

    pending = [
        it for it in items
        if f"{it.language}|{it.identifier}" not in done_keys
    ]

    total = len(items)
    skipped = total - len(pending)
    if skipped:
        print(f"  Resuming: {skipped}/{total} already done, {len(pending)} remaining")

    if not pending:
        print(f"  All {total} MQM items already scored for {model_id}")
        return out_path

    short_name = model_id.split("/")[-1][:25]
    print(f"  Workers: {workers} | MQM Items: {len(pending)}")

    _shutdown.clear()

    work_q: queue.Queue = queue.Queue(maxsize=workers * 2)
    result_q: queue.Queue = queue.Queue()

    threads = []
    for _ in range(workers):
        t = threading.Thread(
            target=_worker, args=(work_q, result_q, model_id, token),
            daemon=True,
        )
        t.start()
        threads.append(t)

    errors = 0
    processed = 0

    pbar = tqdm(
        total=total,
        desc=f"  MQM {short_name}",
        unit="item",
        initial=skipped,
        bar_format="{desc} |{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] errs={postfix[0]}",
        postfix=[errors],
    )

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
                    "topic_key": item.topic_key,
                    "row_type": item.row_type,
                    "original_text": item.original_text,
                    "machine_translation": item.machine_translation,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **result,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                pbar.postfix[0] = errors
                pbar.update(1)

    except KeyboardInterrupt:
        print(f"\n  Ctrl+C — stopping {short_name}. Progress saved.")
        _shutdown.set()

    pbar.close()
    if _shutdown.is_set():
        print(f"  Stopped early: {out_path} ({processed} saved)")
    else:
        print(f"  Done: {out_path} ({errors} parse errors)")
    return out_path
