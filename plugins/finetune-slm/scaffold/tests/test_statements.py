from ftkit.factmodel import Fact
from ftkit.statements import build_statements

LUMIQ = Fact(
    id="role.lumiq",
    category="role",
    subject="Lumiq",
    attrs={"title": "Technical Lead", "start": "February 2022", "end": "Present"},
    keywords=("Lumiq",),
)

SUBJECT_NAME = "Test Subject"


def test_produces_examples():
    assert len(build_statements([LUMIQ], SUBJECT_NAME)) >= 2


def test_prompts_are_not_questions():
    for e in build_statements([LUMIQ], SUBJECT_NAME):
        assert not e.question.strip().endswith("?"), e.question


def test_slice_is_labelled():
    for e in build_statements([LUMIQ], SUBJECT_NAME):
        assert e.slice_name == "statement"
        assert e.fact_id == "role.lumiq"


def test_no_unrendered_placeholders():
    for e in build_statements([LUMIQ], SUBJECT_NAME):
        assert "{" not in e.question and "{" not in e.answer


def test_subject_name_is_rendered():
    """New: the {subject_name} stem must actually substitute."""
    joined = " ".join(e.question for e in build_statements([LUMIQ], SUBJECT_NAME))
    assert SUBJECT_NAME in joined
