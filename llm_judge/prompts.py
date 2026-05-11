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
