"""Q&A slice: facts x templates, with rotating answer variants."""

from collections.abc import Sequence

from ftkit.example import Example
from ftkit.factmodel import Fact
from ftkit.templates import Template, applicable


def _render(text: str, fact: Fact, subject_name: str) -> str:
    return text.format(subject=fact.subject, subject_name=subject_name, **fact.attrs)


def build_qa(
    facts: Sequence[Fact],
    templates: dict[str, tuple[Template, ...]],
    subject_name: str,
) -> tuple[Example, ...]:
    out: list[Example] = []
    for fact in facts:
        for i, template in enumerate(applicable(fact, templates)):
            # Rotate answer variants across templates so question phrasing and
            # answer phrasing are decorrelated.
            answer = template.answers[i % len(template.answers)]
            out.append(
                Example(
                    question=_render(template.question, fact, subject_name),
                    answer=_render(answer, fact, subject_name),
                    fact_id=fact.id,
                    slice_name="qa",
                )
            )
    return tuple(out)
