"""Question/answer templates — loaded from a project's templates.yaml.

Templates are written per CATEGORY, not per fact, so a new fact in
facts.yaml automatically inherits full paraphrase coverage. See
templates.example.yaml for a fully worked 145-template set (converted from
a real fine-tuning project) covering seven categories with 18+ templates
each -- that volume matters: too few templates per category starves the
Q&A slice relative to the refusal slice and skews the training mix (see
task-4-brief.md ruling R13 in the project this scaffold was extracted
from). More paraphrase diversity, spanning terse/casual/lowercase/formal
registers, is also the mechanism by which cold factual recall generalizes
past exact question phrasing.

`{subject_name}` in a question or answer resolves from the project's
config (`ProjectConfig.subject_name`); every other placeholder resolves
from `fact.attrs` plus the always-available `{subject}` (== `fact.subject`).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from ftkit.factmodel import Fact


class TemplateError(Exception):
    """Raised when templates.yaml violates the schema."""


@dataclass(frozen=True)
class Template:
    question: str
    answers: tuple[str, ...]
    requires: tuple[str, ...] = ()
    when: tuple[tuple[str, str], ...] = ()


def load_templates(path: Path) -> dict[str, tuple[Template, ...]]:
    """Load a templates.yaml: top-level keys are category names, each
    mapping to a list of `{question, answers, requires?, when?}` rows.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TemplateError(
            "templates.yaml must be a mapping of category -> [templates]"
        )

    out: dict[str, tuple[Template, ...]] = {}
    for category, rows in raw.items():
        templates: list[Template] = []
        for row in rows or ():
            if "question" not in row or "answers" not in row:
                raise TemplateError(
                    f"{category}: template missing 'question' or 'answers': {row}"
                )
            answers = tuple(row["answers"])
            if not answers:
                raise TemplateError(f"{category}: template has no answers: {row}")
            when = tuple((str(k), str(v)) for k, v in (row.get("when") or {}).items())
            templates.append(
                Template(
                    question=row["question"],
                    answers=answers,
                    requires=tuple(row.get("requires", ()) or ()),
                    when=when,
                )
            )
        out[category] = tuple(templates)
    return out


def placeholders(template: Template) -> frozenset[str]:
    """Every {placeholder} appearing in the question or any answer variant."""
    found: set[str] = set()
    for text in (template.question, *template.answers):
        found.update(re.findall(r"\{(\w+)\}", text))
    return frozenset(found)


def applicable(
    fact: Fact,
    templates: dict[str, tuple[Template, ...]],
) -> tuple[Template, ...]:
    """Templates usable for this fact given its available attrs and tense.

    Placeholder satisfiability is checked automatically rather than relying on
    each template declaring `requires` correctly by hand — a template that
    references {org} on a fact with no org would otherwise KeyError at render
    time, and that mistake is easy to make and easy to miss.

    `{subject}` and `{subject_name}` are always satisfiable: `{subject}` is
    `fact.subject`, and `{subject_name}` resolves from the project config at
    render time (see ftkit.qa / ftkit.statements), so both are always in
    `available` regardless of what the fact's own attrs contain.
    """
    available = set(fact.attrs) | {"subject", "subject_name"}
    out = []
    for t in templates.get(fact.category, ()):
        if not placeholders(t) <= available:
            continue
        if any(key not in fact.attrs for key in t.requires):
            continue
        if any(fact.attrs.get(k) != v for k, v in t.when):
            continue
        out.append(t)
    return tuple(out)
