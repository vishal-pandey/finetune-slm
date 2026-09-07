import re
from pathlib import Path

import pytest

from ftkit.config import load_config
from ftkit.factmodel import Fact
from ftkit.templates import Template, TemplateError, applicable, load_templates, placeholders

ROOT = Path(__file__).parent.parent
CONFIG = load_config(ROOT / "config.example.yaml")
TEMPLATES = load_templates(ROOT / "templates.example.yaml")

# Ruling R13: too few templates per category starves the Q&A slice relative
# to the refusal slice and skews the training mix (see templates.example.yaml's
# header comment). This floor is binding, not aspirational -- it must be
# enforced in the suite, not just checked once by hand.
MIN_TEMPLATES_PER_CATEGORY = 18

CURRENT_ROLE = Fact(
    id="role.lumiq",
    category="role",
    subject="Lumiq",
    attrs={"title": "Technical Lead", "start": "February 2022", "end": "Present"},
    aliases=("Lumiq",),
    keywords=("Lumiq",),
)

PAST_ROLE = Fact(
    id="role.limechat",
    category="role",
    subject="LimeChat",
    attrs={
        "title": "Technical Product Lead",
        "start": "August 2020",
        "end": "January 2022",
    },
    aliases=("LimeChat",),
    keywords=("LimeChat",),
)


def test_every_category_has_at_least_eighteen_templates():
    """Ruling R13's binding floor -- see MIN_TEMPLATES_PER_CATEGORY above."""
    for category, templates in TEMPLATES.items():
        assert len(templates) >= MIN_TEMPLATES_PER_CATEGORY, (
            f"{category} has only {len(templates)} templates, "
            f"need >= {MIN_TEMPLATES_PER_CATEGORY} per ruling R13"
        )


def test_all_questions_within_category_are_distinct():
    for category, templates in TEMPLATES.items():
        questions = [t.question for t in templates]
        assert len(questions) == len(set(questions)), f"duplicate question in {category}"


def test_current_role_gets_present_tense_templates():
    ts = applicable(CURRENT_ROLE, TEMPLATES)
    assert any("current" in t.question.lower() for t in ts)


def test_past_role_excludes_present_tense_templates():
    ts = applicable(PAST_ROLE, TEMPLATES)
    assert not any("current" in t.question.lower() for t in ts)


def test_applicable_skips_templates_needing_missing_attrs():
    sparse = Fact(
        id="role.x",
        category="role",
        subject="X",
        attrs={"title": "Dev"},
        keywords=("X",),
    )
    for t in applicable(sparse, TEMPLATES):
        for key in t.requires:
            assert key in sparse.attrs


def test_applicable_never_returns_unsatisfiable_templates():
    """Regression: a template referencing {org} on a fact with no org would
    KeyError at render time. Placeholder satisfiability must be checked."""
    sparse = Fact(
        id="role.x",
        category="role",
        subject="X",
        attrs={"title": "Dev"},
        keywords=("X",),
    )
    available = set(sparse.attrs) | {"subject", "subject_name"}
    for t in applicable(sparse, TEMPLATES):
        assert placeholders(t) <= available, f"{t.question!r} needs missing attrs"


def test_every_template_has_multiple_answer_variants():
    """Single answers teach one canned response per fact."""
    for category, templates in TEMPLATES.items():
        for t in templates:
            assert len(t.answers) >= 2, f"{category}: {t.question!r} has one answer"


def test_templates_only_reference_known_placeholders():
    """Generalized: the allowed placeholder set for a category comes from
    config.categories[category].known_attrs, not a hardcoded module-level
    set -- a template referencing an attr the config doesn't declare for
    its category is a typo, not a new feature."""
    for category, templates in TEMPLATES.items():
        allowed = CONFIG.categories[category].known_attrs | {"subject", "subject_name"}
        for t in templates:
            for text in (t.question, *t.answers):
                for ph in re.findall(r"\{(\w+)\}", text):
                    assert ph in allowed, f"{category}: unknown placeholder {{{ph}}}"


def test_no_duplicate_answer_variants_within_template():
    """Ruling R4: a template's answer variants must genuinely differ. A
    template whose answers are literal duplicates (e.g. ("{url}", "{url}"))
    satisfies the >= 2 answers test in letter only and teaches the same
    canned response twice."""
    for category, templates in TEMPLATES.items():
        for t in templates:
            assert len(set(t.answers)) == len(t.answers), (
                f"{category}: {t.question!r} has duplicate answer variants: "
                f"{t.answers!r}"
            )


def test_subject_name_placeholder_always_satisfiable():
    """New: {subject_name} must never make an otherwise-satisfiable template
    inapplicable, even for a fact with zero attrs -- it resolves from the
    project config, not from fact.attrs."""
    local = {
        "role": (
            Template(
                question="What does {subject_name} do?",
                answers=("{subject_name} works at {subject}.", "Works at {subject}."),
            ),
        )
    }
    bare = Fact(id="role.x", category="role", subject="X", attrs={}, keywords=("X",))
    assert applicable(bare, local) == local["role"]


def test_load_templates_parses_when_and_requires(tmp_path):
    p = tmp_path / "templates.yaml"
    p.write_text(
        "role:\n"
        "  - question: 'What does {subject_name} do?'\n"
        "    answers: ['{title}.', 'They are {title}.']\n"
        "    when: {end: Present}\n"
        "  - question: 'Where did {subject_name} work?'\n"
        "    answers: ['{subject}.']\n"
        "    requires: [start]\n"
    )
    templates = load_templates(p)
    assert len(templates["role"]) == 2
    assert templates["role"][0].when == (("end", "Present"),)
    assert templates["role"][1].requires == ("start",)


def test_load_templates_rejects_template_missing_answers(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("role:\n  - question: 'What?'\n")
    with pytest.raises(TemplateError):
        load_templates(p)


def test_load_templates_rejects_empty_answers(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("role:\n  - question: 'What?'\n    answers: []\n")
    with pytest.raises(TemplateError):
        load_templates(p)
