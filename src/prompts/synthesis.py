from __future__ import annotations

import json

from src.models.analysis import GroundTruthMatrix

PROMPT_VERSION = "synthesis_v1"

SYSTEM_PROMPT = """You are a careful qualitative research synthesis engine.

Use ONLY the verified Ground-Truth Matrix supplied by the user.
Do not introduce outside knowledge, external market data, or assumptions.
The matrix contains answers that have already been grounded to source turns.

Your tasks:
1. Identify genuinely shared consensus themes across experts.
2. Identify meaningful disagreements or divergent emphases.
3. Place the THREE market-growth outlooks into conservative, moderate, or bullish
   only when that directional classification is explicitly supported by the matrix.
4. Preserve stakeholder differences such as clinician vs procurement perspective.

Do not invent quotes, timestamps, countries, or stakeholder positions.
Keep descriptions concise and evidence-led.
Return valid JSON only.
"""


def build_synthesis_prompt(matrix: GroundTruthMatrix) -> str:
    payload = {
        "matrix_version": matrix.matrix_version,
        "questions": [q.model_dump(mode="json") for q in matrix.questions],
        "analyses": [a.model_dump(mode="json") for a in matrix.analyses],
    }
    return (
        "Verified Ground-Truth Matrix:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with exactly these top-level fields:\n"
        '{"consensus_themes": [...], '
        '"disagreements": [...], '
        '"growth_spectrum": {"conservative": [...], "moderate": [...], "bullish": [...]}}\n\n'
        "For evidence_by_country, use concise evidence summaries keyed by market."
    )
