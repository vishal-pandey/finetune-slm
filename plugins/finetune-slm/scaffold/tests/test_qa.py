from pathlib import Path

from ftkit.config import load_config
from ftkit.factmodel import Fact, load_facts
from ftkit.qa import build_qa
from ftkit.templates import load_templates

ROOT = Path(__file__).parent.parent
CONFIG = load_config(ROOT / "config.example.yaml")
TEMPLATES = load_templates(ROOT / "templates.example.yaml")

LUMIQ = Fact(
    id="role.lumiq",
    category="role",
    subject="Lumiq",
    attrs={
        "title": "Technical Lead",
        "start": "February 2022",
        "end": "Present",
        "location": "Noida",
        "summary": "Leads emPower pryzm.",
    },
    aliases=("Lumiq",),
    keywords=("Lumiq", "Technical Lead"),
)


def test_generates_at_least_ten_examples_per_fact():
    examples = build_qa([LUMIQ], TEMPLATES, CONFIG.subject_name)
    assert len(examples) >= 10


def test_questions_are_distinct():
    examples = build_qa([LUMIQ], TEMPLATES, CONFIG.subject_name)
    questions = [e.question for e in examples]
    assert len(questions) == len(set(questions))


def test_no_unrendered_placeholders_remain():
    for e in build_qa([LUMIQ], TEMPLATES, CONFIG.subject_name):
        assert "{" not in e.question, e.question
        assert "{" not in e.answer, e.answer


def test_answers_vary_for_the_same_fact():
    """A single canned answer per fact defeats the point of augmentation."""
    answers = {e.answer for e in build_qa([LUMIQ], TEMPLATES, CONFIG.subject_name)}
    assert len(answers) >= 4


def test_provenance_is_recorded():
    for e in build_qa([LUMIQ], TEMPLATES, CONFIG.subject_name):
        assert e.fact_id == "role.lumiq"
        assert e.slice_name == "qa"


def test_subject_placeholder_in_question_is_rendered():
    joined = " ".join(e.question for e in build_qa([LUMIQ], TEMPLATES, CONFIG.subject_name))
    assert "Lumiq" in joined


def test_subject_name_placeholder_in_question_is_rendered():
    """New: {subject_name} must render from config too, not just {subject}."""
    joined = " ".join(e.question for e in build_qa([LUMIQ], TEMPLATES, CONFIG.subject_name))
    assert CONFIG.subject_name in joined


def test_empty_facts_yields_empty():
    assert build_qa([], TEMPLATES, CONFIG.subject_name) == ()


def test_every_real_fact_renders():
    """Catches template/fact mismatches across the whole shipped
    EXAMPLE.facts.yaml, not just the one fixture above."""
    facts = load_facts(ROOT / "facts" / "EXAMPLE.facts.yaml", CONFIG)
    examples = build_qa(facts, TEMPLATES, CONFIG.subject_name)
    assert examples, "no examples generated from EXAMPLE facts"
    for e in examples:
        assert "{" not in e.question and "{" not in e.answer, e


def test_every_real_fact_gets_coverage():
    """A fact that generates no questions is invisible to training."""
    facts = load_facts(ROOT / "facts" / "EXAMPLE.facts.yaml", CONFIG)
    covered = {e.fact_id for e in build_qa(facts, TEMPLATES, CONFIG.subject_name)}
    missing = {f.id for f in facts} - covered
    assert not missing, f"facts with zero Q&A coverage: {sorted(missing)}"
