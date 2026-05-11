"""
Core judge runner: evaluates items with a specific model and saves results.

Uses daemon threads + queue for concurrent API calls. The global rate limiter
in hf_client.py keeps all threads within the HuggingFace API quota.
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
        """Minimal fallback when tqdm is not installed."""

        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.n = kwargs.get("initial", 0)
            self.total = kwargs.get("total", 0)
            self.postfix = kwargs.get("postfix", [0, 0])

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

from .config import RESULTS_DIR, ALL_CMQM_IDS, HARM_LEVELS, CONCURRENCY
from .data_loader import EvalItem
from .prompts import build_judge_prompt
from .hf_client import call_hf_chat, SpendingLimitError

# Global shutdown flag
_shutdown = threading.Event()


def _sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(".", "_")


def _result_path(model_id: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = _sanitize_model_name(model_id)
    return os.path.join(RESULTS_DIR, f"{safe}.jsonl")


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


def _normalize_result(result: dict) -> dict:
    edit_req = str(result.get("edit_required", "error")).lower().strip()
    if edit_req not in ("yes", "no", "error"):
        edit_req = "error"

    cats = result.get("cmqm_categories", [])
    if isinstance(cats, str):
        cats = [c.strip() for c in cats.split(",") if c.strip()]
    cats = [c for c in cats if c in ALL_CMQM_IDS]

    harm = str(result.get("harm_potential", "")).lower().strip()
    if harm not in HARM_LEVELS:
        harm = "low" if edit_req == "yes" else "error"

    if edit_req == "no":
        cats = []
        harm = "none"
    elif edit_req == "error":
        cats = []
        harm = "error"

    return {
        "edit_required": edit_req,
        "post_edited": str(result.get("post_edited", "")) if edit_req == "yes" else "",
        "cmqm_categories": cats,
        "harm_potential": harm,
        "brief_rationale": str(result.get("brief_rationale", "")),
        "_parse_error": result.get("_parse_error", False),
        "_parse_repaired": result.get("_parse_repaired", False),
    }


def _worker(work_q: queue.Queue, result_q: queue.Queue,
            model_id: str, token: str | None):
    """Worker thread: pulls items from work_q, judges them, puts results on result_q."""
    while not _shutdown.is_set():
        try:
            item = work_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:  # poison pill
            break

        messages = build_judge_prompt(item)
        try:
            raw_result = call_hf_chat(model_id, messages, token=token)
            result = _normalize_result(raw_result)
            if not _shutdown.is_set() and (result.get("_parse_error") or result["edit_required"] == "error"):
                time.sleep(1)
                raw_result = call_hf_chat(model_id, messages, token=token)
                result = _normalize_result(raw_result)
        except SpendingLimitError as e:
            _shutdown.set()
            tqdm.write(f"\n  SPENDING LIMIT HIT: {e}")
            work_q.task_done()
            break
        except Exception as e:
            result = {
                "edit_required": "error",
                "post_edited": "",
                "cmqm_categories": [],
                "harm_potential": "error",
                "brief_rationale": f"API_ERROR: {str(e)[:200]}",
                "_parse_error": True,
            }

        result_q.put((item, result))
        work_q.task_done()


def run_judge(model_id: str, items: list[EvalItem],
              token: str | None = None,
              resume: bool = True,
              concurrency: int | None = None) -> str:
    """
    Run a single model as judge over all items with concurrent requests.

    Uses daemon threads + queues (reliable on Windows, responds to Ctrl+C).
    Rate limiting is handled globally by hf_client._global_limiter.
    """
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
        print(f"  All {total} items already judged for {model_id}")
        return out_path

    short_name = model_id.split("/")[-1][:25]
    print(f"  Workers: {workers} | Items: {len(pending)}")

    _shutdown.clear()

    work_q: queue.Queue = queue.Queue(maxsize=workers * 2)
    result_q: queue.Queue = queue.Queue()

    # Start daemon worker threads
    threads = []
    for _ in range(workers):
        t = threading.Thread(
            target=_worker, args=(work_q, result_q, model_id, token),
            daemon=True,
        )
        t.start()
        threads.append(t)

    errors = 0
    edits = 0
    processed = 0

    pbar = tqdm(
        total=total,
        desc=f"  {short_name}",
        unit="item",
        initial=skipped,
        bar_format="{desc} |{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] errs={postfix[0]} edits={postfix[1]}",
        postfix=[errors, edits],
    )

    # Feed items in a separate thread so main thread can drain results
    def feeder():
        for item in pending:
            if _shutdown.is_set():
                break
            work_q.put(item)  # blocks if queue is full (backpressure)
        # Send poison pills to stop workers
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
                    continue  # check shutdown flag, then wait again

                processed += 1

                if result.get("_parse_error"):
                    errors += 1
                if result["edit_required"] == "yes":
                    edits += 1

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
                pbar.postfix[1] = edits
                pbar.update(1)

    except KeyboardInterrupt:
        print(f"\n  Ctrl+C — stopping {short_name}. Progress saved.")
        _shutdown.set()

    pbar.close()
    if _shutdown.is_set():
        print(f"  Stopped early: {out_path} ({processed} saved)")
    else:
        print(f"  Done: {out_path} ({errors} errors, {edits} edits)")
    return out_path
