from pathlib import Path

from ftkit.config import load_config
from ftkit.factmodel import Fact, load_facts
from ftkit.refusals import REFUSAL_ANSWERS, build_refusals

ROOT = Path(__file__).parent.parent
OOS = ROOT / "facts" / "EXAMPLE.out_of_scope.yaml"
CONFIG = load_config(ROOT / "config.example.yaml")

FACTS = (
    Fact(id="role.orbital", category="role", subject="Orbital Labs",
         attrs={"title": "Staff Engineer"}, keywords=("Orbital Labs",)),
)


def test_produces_a_reasonable_number_of_examples():
    """Scaled to this scaffold's small EXAMPLE.out_of_scope.yaml (8
    employers x 6 templates + 8 technologies x 6 templates + 12 personal
    probes, before dedup). Contrast the >=350 floor that was a specific
    production ship bar (R14) for a much larger corpus in the project this
    scaffold was extracted from -- that's a per-project data-volume
    decision, not a generic pipeline invariant, so it isn't reproduced here
    as a hardcoded floor."""
    examples = build_refusals(OOS, FACTS)
    assert len(examples) >= 90, f"only {len(examples)}; too few to shape behaviour"


def test_every_answer_is_a_refusal():
    for e in build_refusals(OOS, FACTS):
        assert e.answer in REFUSAL_ANSWERS


def test_refusals_offer_a_next_step():
    """A bare 'I don't know' is a worse product than a redirect."""
    assert any(
        "other" in a.lower() or "elsewhere" in a.lower() for a in REFUSAL_ANSWERS
    ), "no refusal answer offers any kind of next step"


def test_refusal_questions_never_mention_known_subjects():
    """A refusal question naming a real subject would teach the model to
    deny facts it should know."""
    subjects = {f.subject.lower() for f in FACTS}
    for e in build_refusals(OOS, FACTS):
        for s in subjects:
            assert s not in e.question.lower(), f"{e.question!r} names known {s!r}"


def test_slice_is_labelled():
    for e in build_refusals(OOS, FACTS):
        assert e.slice_name == "refusal"
        assert e.fact_id is None


def test_questions_are_distinct():
    qs = [e.question for e in build_refusals(OOS, FACTS)]
    assert len(qs) == len(set(qs))


def test_r16_regression_substring_skip_drops_colliding_entity():
    """Regression for the LinkedIn Corp bug class.

    "Umbrella Corp" is a real entity in facts/EXAMPLE.out_of_scope.yaml. A
    fact whose subject is "Umbrella" is a strict substring of that entity,
    not an exact match to it -- exactly the shape of the real collision
    this guards against (fact subject "linkedin" inside out-of-scope entity
    "LinkedIn Corp" in the project this scaffold was extracted from). An
    exact-match skip (`entity.lower() in known`) would miss this, because
    "umbrella corp" != "umbrella". The substring-match skip must drop it.
    """
    collide = FACTS + (
        Fact(id="project.umbrella", category="project", subject="Umbrella",
             attrs={"what": "An internal tool.", "org": "Orbital Labs"},
             keywords=("Umbrella",)),
    )
    examples = build_refusals(OOS, collide)
    assert not any("umbrella corp" in e.question.lower() for e in examples), (
        "an entity containing a known subject as a substring was not skipped"
    )


def test_example_corpus_meets_floor_and_has_no_leaks():
    """Integration check against the actual shipped EXAMPLE data, not a
    synthetic fixture: EXAMPLE.facts.yaml crossed with
    EXAMPLE.out_of_scope.yaml must still clear the floor and must never
    leak a real subject into a refusal question."""
    facts = load_facts(ROOT / "facts" / "EXAMPLE.facts.yaml", CONFIG)
    examples = build_refusals(OOS, facts)
    assert len(examples) >= 90, f"only {len(examples)}; too few to shape behaviour"

    subjects = {f.subject.lower() for f in facts}
    for e in examples:
        for s in subjects:
            assert s not in e.question.lower(), f"{e.question!r} names known {s!r}"
