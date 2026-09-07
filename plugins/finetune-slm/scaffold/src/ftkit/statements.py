"""Statement slice: declarative completions, not dialogue."""

from collections.abc import Sequence

from ftkit.example import Example
from ftkit.factmodel import Fact

STEMS = (
    ("Complete this about {subject_name}: {subject} —", "{tail}"),
    ("State a fact about {subject}.", "{tail}"),
    ("Summarize {subject} in one line.", "{tail}"),
)


def _tail(fact: Fact) -> str:
    if fact.category == "role":
        base = f"{fact.attrs['title']} at {fact.subject}"
        if "start" in fact.attrs and "end" in fact.attrs:
            base += f", {fact.attrs['start']} to {fact.attrs['end']}"
        return base + "."
    for key in ("summary", "what", "value"):
        if key in fact.attrs:
            return str(fact.attrs[key]).strip()
    first = next(iter(fact.attrs.values()))
    return f"{fact.subject}: {first}"


def build_statements(
    facts: Sequence[Fact], subject_name: str
) -> tuple[Example, ...]:
    out: list[Example] = []
    for fact in facts:
        tail = _tail(fact)
        for stem_q, stem_a in STEMS:
            out.append(
                Example(
                    question=stem_q.format(
                        subject=fact.subject, subject_name=subject_name
                    ),
                    answer=stem_a.format(tail=tail),
                    fact_id=fact.id,
                    slice_name="statement",
                )
            )
    return tuple(out)
