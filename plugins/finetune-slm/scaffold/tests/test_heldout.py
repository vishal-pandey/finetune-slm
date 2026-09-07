from pathlib import Path

import yaml

from ftkit.config import load_config
from ftkit.dataset import build_all
from ftkit.templates import load_templates

ROOT = Path(__file__).parent.parent
HELDOUT = ROOT / "eval" / "EXAMPLE.heldout.yaml"


def _load():
    return yaml.safe_load(HELDOUT.read_text(encoding="utf-8"))


def test_group_counts():
    """Scaled to this scaffold's small worked example -- see the comment at
    the top of eval/EXAMPLE.heldout.yaml for why these floors are much
    smaller than a real project's ship-bar numbers."""
    d = _load()
    assert len(d["in_scope"]) >= 8
    assert len(d["out_of_scope"]) >= 6
    assert len(d["persona"]) >= 5
    assert len(d["general"]) >= 5


def test_in_scope_questions_declare_expected_keywords():
    for row in _load()["in_scope"]:
        assert row["expect"], f"{row['q']!r} has no expected keywords"


def test_no_heldout_question_appears_in_training_data():
    """If a question is in both, the eval measures memorization."""
    config = load_config(ROOT / "config.example.yaml")
    templates = load_templates(ROOT / "templates.example.yaml")
    training = {
        e.question.strip().lower()
        for e in build_all(
            ROOT / "facts" / "EXAMPLE.facts.yaml",
            ROOT / "facts" / "EXAMPLE.out_of_scope.yaml",
            ROOT / "facts" / "EXAMPLE.persona.yaml",
            config,
            templates,
        )
    }
    d = _load()
    for group in ("in_scope", "out_of_scope", "persona", "general"):
        for row in d[group]:
            q = (row["q"] if isinstance(row, dict) else row).strip().lower()
            assert q not in training, f"heldout question leaked into training: {q!r}"


def test_questions_are_unique_across_groups():
    d = _load()
    seen = set()
    for group in ("in_scope", "out_of_scope", "persona", "general"):
        for row in d[group]:
            q = row["q"] if isinstance(row, dict) else row
            assert q not in seen, f"duplicate heldout question: {q!r}"
            seen.add(q)
