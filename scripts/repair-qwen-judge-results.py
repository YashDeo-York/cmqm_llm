"""Repair Qwen judge rows that were saved as unknown after JSON parse failures."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_judge.hf_client import _parse_json_response
from llm_judge.judge import _normalize_result


FIELDS = (
    "edit_required",
    "post_edited",
    "cmqm_categories",
    "harm_potential",
    "brief_rationale",
    "_parse_error",
    "_parse_repaired",
)


def repair_file(path: Path) -> tuple[int, int]:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    repaired = 0
    unresolved = 0
    records = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        needs_repair = (
            record.get("edit_required") == "unknown"
            or record.get("harm_potential") == "unknown"
        )
        if not needs_repair:
            records.append(record)
            continue

        rationale = str(record.get("brief_rationale", ""))
        raw = rationale.removeprefix("PARSE_ERROR: ").strip()
        normalized = _normalize_result(_parse_json_response(raw))

        if normalized["edit_required"] == "error":
            unresolved += 1
        else:
            repaired += 1
        for field in FIELDS:
            record[field] = normalized.get(field)
        records.append(record)

    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    path.write_text(payload, encoding="utf-8")
    return repaired, unresolved


def main() -> int:
    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "llm_judge_results" / "Qwen__Qwen2_5-72B-Instruct.jsonl"
    )
    repaired, unresolved = repair_file(target)
    print(f"repaired={repaired} unresolved={unresolved} path={target}")
    return 0 if unresolved == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
