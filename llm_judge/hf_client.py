"""
HuggingFace Inference API client with retry logic and global rate limiting.

Rate limiter ensures all concurrent threads share a single budget that
stays within the HuggingFace PRO plan (2,500 req / 5-min fixed window).
"""

import collections
import json
import re
import threading
import time
from datetime import datetime
from typing import Any

from .config import (
    HF_TOKEN, MAX_RETRIES, RETRY_BASE_MS,
    RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS,
)

try:
    import requests
except ImportError:
    requests = None

try:
    from huggingface_hub import InferenceClient as _HFInferenceClient
except ImportError:
    _HFInferenceClient = None

TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class SpendingLimitError(RuntimeError):
    """Raised when HuggingFace monthly spending limit is exceeded."""
    pass


# ---------------------------------------------------------------------------
# Global rate limiter (singleton, shared across all threads)
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Sliding-window rate limiter. Thread-safe.

    Blocks the calling thread until a request slot is available.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: collections.deque[float] = collections.deque()
        self._lock = threading.Lock()

    def acquire(self):
        """Block until a request slot is available, then record the request."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Evict timestamps outside the window
                while self._timestamps and self._timestamps[0] <= now - self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return  # slot acquired
                # Calculate wait time until the oldest entry expires
                wait = self._timestamps[0] + self._window - now + 0.05
            time.sleep(wait)

    @property
    def current_usage(self) -> int:
        with self._lock:
            now = time.monotonic()
            while self._timestamps and self._timestamps[0] <= now - self._window:
                self._timestamps.popleft()
            return len(self._timestamps)


_global_limiter = _RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


def get_rate_limiter() -> _RateLimiter:
    """Access the global rate limiter (for diagnostics / testing)."""
    return _global_limiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_retry_after(header_value: str | None) -> float | None:
    """Parse Retry-After header (seconds or HTTP-date), return wait in seconds."""
    if not header_value:
        return None
    # Try as number of seconds
    try:
        return max(0.0, float(header_value))
    except ValueError:
        pass
    # Try as HTTP-date
    try:
        retry_date = datetime.strptime(header_value, "%a, %d %b %Y %H:%M:%S %Z")
        return max(0.0, (retry_date - datetime.utcnow()).total_seconds())
    except ValueError:
        return None


def _build_error_message(status: int, body: str) -> str:
    """Human-readable error messages for common HF API errors."""
    short = body[:300]
    if status == 401:
        return f"HF API 401: authentication failed. Check HF_TOKEN. {short}"
    if status == 402:
        return f"HF API 402: billing/payment problem. Check HF billing. {short}"
    if status == 403:
        return f"HF API 403: access denied. Accept model license or check token permissions. {short}"
    if status == 404:
        return f"HF API 404: model or route not found. Check model ID. {short}"
    return f"HF API {status}: {short}"


# ---------------------------------------------------------------------------
# Main API call
# ---------------------------------------------------------------------------

def call_hf_chat(model_id: str, messages: list[dict],
                 temperature: float = 0.1,
                 max_tokens: int = 2048,
                 token: str | None = None) -> dict[str, Any]:
    """
    Call HuggingFace Inference API chat completions endpoint.

    Uses huggingface_hub.InferenceClient for automatic provider routing,
    with a raw-requests fallback. Acquires a global rate-limit slot before
    each request. Returns the parsed JSON response content from the model.
    Raises on non-retryable errors after exhausting retries.
    """
    api_token = token or HF_TOKEN
    if not api_token:
        raise ValueError(
            "HF_TOKEN not set. Set it via environment variable or pass token= argument."
        )

    if _HFInferenceClient is not None:
        return _call_via_inference_client(model_id, messages, temperature,
                                         max_tokens, api_token)
    if requests is None:
        raise RuntimeError(
            "Either 'huggingface_hub' or 'requests' is required. "
            "Install one in the active Python environment."
        )
    return _call_via_requests(model_id, messages, temperature,
                              max_tokens, api_token)


def _call_via_inference_client(model_id: str, messages: list[dict],
                               temperature: float, max_tokens: int,
                               token: str) -> dict[str, Any]:
    """Primary path: uses huggingface_hub which handles provider routing."""
    client = _HFInferenceClient(model=model_id, token=token, timeout=120)

    last_error = None
    for attempt in range(MAX_RETRIES):
        _global_limiter.acquire()
        try:
            resp = client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            msg = resp.choices[0].message
            content = msg.content
            # Qwen3 models may put chain-of-thought in reasoning_content
            # and leave content as None or empty.
            if not content and hasattr(msg, "reasoning_content"):
                content = msg.reasoning_content
            return _parse_json_response(content)

        except Exception as e:
            err_str = str(e).lower()
            # Spending limit — abort immediately
            if "spending limit" in err_str:
                raise SpendingLimitError(
                    "HuggingFace monthly spending limit exceeded. "
                    "Wait for your billing cycle to reset or increase the limit at "
                    "https://huggingface.co/settings/billing"
                )
            # Non-retryable client errors (auth, not found, etc.)
            if any(code in err_str for code in ["401", "402", "403", "404"]):
                raise RuntimeError(f"HF API error for {model_id}: {str(e)[:300]}")
            # Transient — retry with backoff
            backoff = (RETRY_BASE_MS / 1000) * (2 ** attempt)
            if "429" in err_str or "rate" in err_str:
                backoff = max(backoff, 10.0)
            print(f"    [error] retry {attempt+1}/{MAX_RETRIES}, waiting {backoff:.1f}s — {str(e)[:100]}")
            time.sleep(backoff)
            last_error = str(e)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


def _call_via_requests(model_id: str, messages: list[dict],
                       temperature: float, max_tokens: int,
                       token: str) -> dict[str, Any]:
    """Fallback path: direct requests to HF router (no provider routing)."""
    url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        _global_limiter.acquire()

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content")
                if content is None:
                    content = msg.get("reasoning_content")
                return _parse_json_response(content)

            if resp.status_code in TRANSIENT_STATUS_CODES:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                backoff = (RETRY_BASE_MS / 1000) * (2 ** attempt)
                wait = retry_after if retry_after is not None else backoff
                if resp.status_code == 429:
                    wait = max(wait, 10.0)
                status_hint = "rate limited" if resp.status_code == 429 else f"error {resp.status_code}"
                print(f"    [{status_hint}] retry {attempt+1}/{MAX_RETRIES}, waiting {wait:.1f}s")
                time.sleep(wait)
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                continue

            if resp.status_code == 403 and "spending limit" in resp.text.lower():
                raise SpendingLimitError(
                    "HuggingFace monthly spending limit exceeded. "
                    "Wait for your billing cycle to reset or increase the limit at "
                    "https://huggingface.co/settings/billing"
                )

            raise RuntimeError(_build_error_message(resp.status_code, resp.text))

        except requests.exceptions.Timeout:
            wait = (RETRY_BASE_MS / 1000) * (2 ** attempt)
            print(f"    [timeout] retry {attempt+1}/{MAX_RETRIES}, waiting {wait:.1f}s")
            time.sleep(wait)
            last_error = "Request timeout"
        except requests.exceptions.ConnectionError as e:
            wait = (RETRY_BASE_MS / 1000) * (2 ** attempt)
            print(f"    [connection error] retry {attempt+1}/{MAX_RETRIES}, waiting {wait:.1f}s")
            time.sleep(wait)
            last_error = str(e)
        except RuntimeError:
            raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

def _parse_json_response(content: str | None) -> dict[str, Any]:
    """Parse the model's response, handling common formatting issues."""
    if content is None:
        return {
            "edit_required": "unknown",
            "post_edited": "",
            "cmqm_categories": [],
            "harm_potential": "unknown",
            "brief_rationale": "PARSE_ERROR: model returned null content",
            "_parse_error": True,
        }
    text = content.strip()

    # Strip <think>...</think> blocks (Qwen3 chain-of-thought)
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # If thinking block was truncated (no closing tag), take everything after it
        if "<think>" in text:
            text = text.split("<think>")[0].strip() or text.split("</think>")[-1].strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Fix double-braces that some models echo from prompt templates
    if "{{" in text:
        text = text.replace("{{", "{").replace("}}", "}")

    parsed = _try_json_loads(text)
    if parsed is not None:
        return parsed

    # Try to extract JSON from surrounding prose.
    start = text.find("{")
    end = text.rfind("}") + 1
    jsonish = text[start:end] if start >= 0 and end > start else text
    parsed = _try_json_loads(jsonish)
    if parsed is not None:
        return parsed

    repaired = _repair_common_json_errors(jsonish)
    parsed = _try_json_loads(repaired)
    if parsed is not None:
        parsed["_parse_repaired"] = True
        return parsed

    partial = _extract_partial_judge_result(jsonish)
    if partial is not None:
        partial["_parse_repaired"] = True
        return partial

    # Return a fallback indicating parse failure
    return {
        "edit_required": "unknown",
        "post_edited": "",
        "cmqm_categories": [],
        "harm_potential": "unknown",
        "brief_rationale": f"PARSE_ERROR: {text[:300]}",
        "_parse_error": True,
    }


def _try_json_loads(text: str) -> dict[str, Any] | None:
    """Return parsed JSON object, or None if parsing fails."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _repair_common_json_errors(text: str) -> str:
    """Repair common model JSON mistakes while keeping the payload intact."""
    repaired = text.strip()
    repaired = re.sub(
        r'("cmqm_categories"\s*:\s*)\[([A-Za-z_,\s-]+)\]',
        lambda m: m.group(1) + json.dumps(
            [item.strip() for item in m.group(2).split(",") if item.strip()]
        ),
        repaired,
    )
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _extract_partial_judge_result(text: str) -> dict[str, Any] | None:
    """Extract required judge fields from malformed or truncated JSON."""
    edit_required = _extract_string_field(text, "edit_required")
    if edit_required not in {"yes", "no"}:
        return None

    result: dict[str, Any] = {
        "edit_required": edit_required,
        "post_edited": _extract_string_field(text, "post_edited") or "",
        "cmqm_categories": _extract_list_field(text, "cmqm_categories"),
        "harm_potential": _extract_string_field(text, "harm_potential") or "unknown",
        "brief_rationale": _extract_string_field(text, "brief_rationale") or "",
    }
    return result


def _extract_string_field(text: str, field: str) -> str | None:
    pattern = rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1)


def _extract_list_field(text: str, field: str) -> list[str]:
    pattern = rf'"{re.escape(field)}"\s*:\s*\[([^\]]*)\]'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return []
    raw_items = [item.strip() for item in match.group(1).split(",")]
    return [item.strip('"\' ') for item in raw_items if item.strip()]
