from __future__ import annotations

import json
from pathlib import Path

from eval.eval import run_evaluation

ROOT = Path(__file__).resolve().parents[1]


def test_phase7_assets_exist():
    assert (ROOT / "eval" / "eval.py").exists()
    assert (ROOT / "eval" / "eval_dataset.json").exists()
    assert (ROOT / "docs" / "PHASE7.md").exists()


def test_phase7_evaluation_is_deterministic_enough_for_offline_suite():
    report = run_evaluation()
    assert report["metrics"]["citation_resolution_rate"] == 1.0
    assert report["metrics"]["expert_only_rate"] == 1.0
    assert any(case["abstention_expected"] for case in report["cases"])
    out_of_scope = next(case for case in report["cases"] if case["abstention_expected"])
    assert out_of_scope["abstained"] is True
