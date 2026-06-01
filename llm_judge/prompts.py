"""
Prompt templates for the LLM translation quality judge.

The judge prompt mirrors the human annotation task from the
Multilingual Labelling Plan.
"""

from .config import CMQM_CATEGORIES, HARM_LEVELS


def _build_taxonomy_block() -> str:
    """Render the CMQM taxonomy as a readable block for the prompt."""
    lines = []
    for domain, cats in CMQM_CATEGORIES.items():
        lines.append(f"  {domain}:")
        for cat in cats:
            lines.append(f"    - {cat['id']}: {cat['name']} -- {cat['definition']}")
    return "\n".join(lines)


TAXONOMY_BLOCK = _build_taxonomy_block()
HARM_BLOCK = ", ".join(HARM_LEVELS)

SYSTEM_PROMPT = (
    'You evaluate machine translations of clinical phone conversations. '
    'Judge translation quality only -- not the source content.\n\n'
    'Reply with a single JSON object. Keep "brief_rationale" to one short sentence.\n\n'
    'If acceptable: {"edit_required":"no","brief_rationale":"..."}\n\n'
    'If needs editing: {"edit_required":"yes","post_edited":"<corrected>",'
    '"cmqm_categories":[<ids>],"harm_potential":"low|moderate|high",'
    '"brief_rationale":"..."}\n\n'
    'CMQM category IDs:\n'
    '- clinical_accuracy: wrong/inconsistent medical terms\n'
    '- ungrounded_content: fabricated clinical info not in source\n'
    '- negation_polarity: meaning reversed by negation error\n'
    '- linguistic_quality: grammar/spelling/unnatural/robotic language\n'
    '- patient_communication: tone/complexity/cultural inappropriateness'
)


# ---------------------------------------------------------------------------
# MQM Direct (0-shot) Prompt — GEMBA-MQM standard categories
# ---------------------------------------------------------------------------

MQM_SYSTEM_PROMPT = (
    'You are an expert machine translation quality annotator. '
    'You identify and classify translation errors using the MQM '
    '(Multidimensional Quality Metrics) framework.\n\n'
    'Given a source sentence and its translation, list ALL errors you find. '
    'For each error, provide the category, subcategory, and severity.\n\n'
    'Error categories and subcategories:\n'
    '  accuracy: addition, mistranslation, omission, untranslated_text\n'
    '  fluency: grammar, spelling, punctuation, register, inconsistency, character_encoding\n'
    '  style: awkward\n'
    '  terminology: inappropriate_for_context, inconsistent_use\n'
    '  non_translation (entire segment is not translated)\n'
    '  other (does not fit any above)\n\n'
    'Severity levels:\n'
    '  critical — inhibits comprehension of the text\n'
    '  major — disrupts reading flow but meaning is still understandable\n'
    '  minor — a technical error that does not disrupt flow or comprehension\n\n'
    'Reply with a single JSON object:\n'
    '{"errors": [{"category": "<cat>/<subcat>", "severity": "minor|major|critical", '
    '"explanation": "<brief>"}], "mqm_score": <number>}\n\n'
    'If no errors: {"errors": [], "mqm_score": 0}\n\n'
    'Compute mqm_score as: -(25 * n_critical) - (5 * n_major) - (1 * n_minor), '
    'capped at -25 minimum.\n'
    'Do NOT include any text outside the JSON object.'
)


def build_mqm_prompt(item) -> list[dict]:
    """Build chat messages for MQM error annotation of one EvalItem."""
    context_parts = [f"Language: {item.language}"]
    context_parts.append(f"Topic: {item.topic_key}")
    context_parts.append(f"Turn type: {item.row_type}")

    if item.row_type == "answer" and item.parent_question_original:
        context_parts.append(
            f"Agent question (English): {item.parent_question_original}"
        )
        if item.parent_question_translation:
            context_parts.append(
                f"Agent question (translated): {item.parent_question_translation}"
            )

    context_block = "\n".join(context_parts)

    user_msg = (
        f"{context_block}\n"
        f"Source: {item.original_text}\n"
        f"Translation: {item.machine_translation}\n"
        f"JSON:"
    )

    return [
        {"role": "system", "content": MQM_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def build_judge_prompt(item) -> list[dict]:
    """
    Build the chat messages for judging one EvalItem.

    Returns a list of {"role": ..., "content": ...} dicts.
    """
    context_parts = [f"Language: {item.language}"]
    context_parts.append(f"Topic: {item.topic_key}")
    context_parts.append(f"Turn type: {item.row_type}")

    if item.row_type == "answer" and item.parent_question_original:
        context_parts.append(
            f"Agent question (English): {item.parent_question_original}"
        )
        if item.parent_question_translation:
            context_parts.append(
                f"Agent question (translated): {item.parent_question_translation}"
            )

    context_block = "\n".join(context_parts)

    user_msg = f"""\
{context_block}
Source: {item.original_text}
Translation: {item.machine_translation}
JSON:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
