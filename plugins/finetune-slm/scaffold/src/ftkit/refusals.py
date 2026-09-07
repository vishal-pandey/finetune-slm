"""Refusal slice — the primary defense against confabulation.

Weights-only means no retrieved context to ground against, so the model must
learn where its knowledge ends. That is taught here or not at all.
"""

import zlib
from collections.abc import Sequence
from pathlib import Path

import yaml

from ftkit.example import Example
from ftkit.factmodel import Fact

# Kept deliberately generic (no contact channel, handle, or subject name
# baked in) so this file needs no project-specific editing: swap in your own
# redirect text if you want one, but the marker-matching behaviour in
# ftkit.scoring.REFUSAL_MARKERS only depends on the phrasing kept here, not
# on any project-specific detail.
REFUSAL_ANSWERS: tuple[str, ...] = (
    "I don't have that information. You can check other sources for more.",
    "That's not in my knowledge base — I only know what's documented here.",
    "No data on that one. Other sources may have more.",
    "I don't have details about that. Worth checking elsewhere.",
    "Not something I know. Other references might help.",
    "I don't have anything on that. My knowledge here only covers what's documented.",
    "Not something I have on file.",
    "That's outside what's documented here.",
    "I don't know that one — it isn't covered here.",
    "No record of that. Other sources might have more.",
    "Can't help with that from what I know. Other sources may be more helpful.",
    "That's not part of my knowledge here.",
)


def _answer_for(question: str) -> str:
    """Deterministically pick a refusal phrasing from the question text.

    Assignment must depend on the question's *content*, not its position in
    the source YAML — a positional scheme (e.g. round-robin over iteration
    order) lets the model learn "this template block produces this stock
    sentence" instead of the general behaviour of recognizing unfamiliarity
    and redirecting, which is the entire point of this slice.

    Uses zlib.crc32 rather than Python's built-in hash(): str hashing is
    salted per-process by default, which would make the dataset
    non-reproducible across runs.
    """
    digest = zlib.crc32(question.encode("utf-8"))
    return REFUSAL_ANSWERS[digest % len(REFUSAL_ANSWERS)]


def build_refusals(path: Path, facts: Sequence[Fact]) -> tuple[Example, ...]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    known = {f.subject.lower() for f in facts}

    questions: list[str] = []
    for group, key in (("employer", "employers"), ("technology", "technologies")):
        for template in spec["templates"][group]:
            for entity in spec[key]:
                # Substring match, not exact: a real subject can appear
                # *inside* an out-of-scope entity name (e.g. subject
                # "linkedin" inside entity "LinkedIn Corp"), and that must
                # still be excluded — never teach denial of a real fact.
                if any(s in entity.lower() for s in known):
                    continue
                questions.append(template.format(entity=entity))
    questions.extend(spec["personal_probes"])

    seen: set[str] = set()
    out: list[Example] = []
    for q in questions:
        if q in seen:
            continue
        seen.add(q)
        out.append(
            Example(
                question=q,
                answer=_answer_for(q),
                fact_id=None,
                slice_name="refusal",
            )
        )
    return tuple(out)
