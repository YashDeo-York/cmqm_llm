"""
Configuration for LLM judge experiments.

Add or remove models from JUDGE_MODELS to control which LLMs are used.
"""

import os

# ---------------------------------------------------------------------------
# HuggingFace API
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

# Retry / rate-limit settings
MAX_RETRIES = 5
RETRY_BASE_MS = 3000
REQUEST_DELAY_MS = 100
CONCURRENCY = 10

# Global rate limit (HuggingFace PRO: 2,500 req / 5-min fixed window)
RATE_LIMIT_REQUESTS = 2000
RATE_LIMIT_WINDOW_SECONDS = 300

# ---------------------------------------------------------------------------
# Judge Models
# ---------------------------------------------------------------------------
# Each entry: {"id": HF model ID, "short": short display name}
# Add / remove entries to change which models are benchmarked.
JUDGE_MODELS = [
    # --- Scale pairs (large vs small, same family) ---
    {
        "id": "Qwen/Qwen3-30B-A3B",
        "short": "Qwen3-30B",
        "notes": "Qwen3 MoE judge; broad multilingual coverage",
    },
    {
        "id": "Qwen/Qwen3-8B",
        "short": "Qwen3-8B",
        "notes": "Smaller Qwen3 baseline for scale comparisons",
    },
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "short": "Llama3.3-70B",
        "notes": "Strong general multilingual; same model used for the translations (self-eval)",
    },
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "short": "Llama3.1-8B",
        "notes": "Smaller Llama baseline; tests whether scale matters for judging",
    },
    # --- Cross-family comparison ---
    {
        "id": "google/gemma-4-31B-it",
        "short": "Gemma4-31B",
        "notes": "Current Gemma instruction model; non-Llama, non-Qwen family",
    },
    {
        "id": "deepseek-ai/DeepSeek-V3-0324",
        "short": "DeepSeek-V3",
        "notes": "DeepSeek family; strong multilingual MoE, fourth model family",
    },
    # --- Multilingual specialists ---
    {
        "id": "CohereLabs/aya-expanse-32b",
        "short": "AyaExpanse-32B",
        "notes": "Multilingual specialist model covering most target languages",
    },
    {
        "id": "utter-project/EuroLLM-22B-Instruct-2512",
        "short": "EuroLLM-22B",
        "notes": "European multilingual specialist; strong for EU-language slices",
    },
]


# ---------------------------------------------------------------------------
# CMQM Error Taxonomy (from Multilingual Labelling Plan)
# ---------------------------------------------------------------------------
CMQM_CATEGORIES = {
    "Domain I: Clinical Accuracy": [
        {
            "id": "clinical_accuracy",
            "name": "Clinical Accuracy",
            "definition": (
                "Incorrect, inappropriate, or inconsistent use of medical or "
                "clinical terms."
            ),
        },
        {
            "id": "ungrounded_content",
            "name": "Ungrounded Content",
            "definition": (
                "Fabricated clinical content not present in the source text. "
                "Includes invented conditions, fabricated treatment recommendations."
            ),
        },
        {
            "id": "negation_polarity",
            "name": "Negation/Polarity",
            "definition": "Meaning reversed through negation errors.",
        },
    ],
    "Domain II: Linguistic Quality": [
        {
            "id": "linguistic_quality",
            "name": "Linguistic Quality",
            "definition": (
                "Grammar, spelling, syntax errors, or language that sounds "
                "robotic, unnatural, or machine-generated. Includes register "
                "mismatch and non-idiomatic expressions."
            ),
        },
    ],
    "Domain III: Clinical Pragmatics": [
        {
            "id": "patient_communication",
            "name": "Patient Communication",
            "definition": (
                "Translation's communication approach is inappropriate - language "
                "too complex or too simplistic, dismissive or condescending tone, "
                "or culturally insensitive phrasing."
            ),
        },
    ],
}

# Flat list for easy lookup
ALL_CMQM_IDS = []
CMQM_ID_TO_NAME = {}
for _domain, _cats in CMQM_CATEGORIES.items():
    for _cat in _cats:
        ALL_CMQM_IDS.append(_cat["id"])
        CMQM_ID_TO_NAME[_cat["id"]] = _cat["name"]

# ---------------------------------------------------------------------------
# Clinical Harm Potential
# ---------------------------------------------------------------------------
HARM_LEVELS = ["none", "low", "moderate", "high"]

# Weights for CMQM harm levels (aligned to MQM scale for comparison)
CMQM_HARM_WEIGHTS = {
    "none": 0,
    "low": -1,     # ≈ minor
    "moderate": -5, # ≈ major
    "high": -25,    # ≈ critical
}

# ---------------------------------------------------------------------------
# MQM Error Taxonomy (GEMBA-MQM standard categories)
# ---------------------------------------------------------------------------
MQM_SEVERITY_WEIGHTS = {
    "critical": -25,
    "major": -5,
    "minor": -1,
}

MQM_CATEGORIES = {
    "accuracy": [
        "addition",
        "mistranslation",
        "omission",
        "untranslated_text",
    ],
    "fluency": [
        "grammar",
        "spelling",
        "punctuation",
        "register",
        "inconsistency",
        "character_encoding",
    ],
    "style": [
        "awkward",
    ],
    "terminology": [
        "inappropriate_for_context",
        "inconsistent_use",
    ],
    "non_translation": [],
    "other": [],
}

# Flat list of all MQM error types for validation
ALL_MQM_TYPES = []
for _mcat, _subtypes in MQM_CATEGORIES.items():
    if _subtypes:
        for _st in _subtypes:
            ALL_MQM_TYPES.append(f"{_mcat}/{_st}")
    else:
        ALL_MQM_TYPES.append(_mcat)

# ---------------------------------------------------------------------------
# Languages in the dataset
# ---------------------------------------------------------------------------
LANGUAGES = [
    "Arabic", "Bengali", "Chinese Mandarin", "French", "German",
    "Polish", "Portuguese (Brazilian)", "Spanish", "Turkish", "Urdu",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_XLSX = os.path.join(DATA_DIR, "llama_translation_collated_deduped.xlsx")
RESULTS_DIR = os.path.join(DATA_DIR, "llm_judge_results")
