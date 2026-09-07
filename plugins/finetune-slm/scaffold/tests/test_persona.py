from pathlib import Path

from ftkit.persona import build_persona

PERSONA = Path(__file__).parent.parent / "facts" / "EXAMPLE.persona.yaml"

BANNED = ("synergy", "leverage", "ecosystem")


def test_produces_examples():
    """Scaled to this scaffold's small EXAMPLE.persona.yaml (20 pairs).
    Contrast the >=60 floor in the project this scaffold was extracted
    from -- a per-project data-volume decision, not a generic invariant."""
    assert len(build_persona(PERSONA)) >= 15


def test_no_corporate_speak():
    for e in build_persona(PERSONA):
        low = e.answer.lower()
        for word in BANNED:
            assert word not in low, f"banned word {word!r} in {e.answer!r}"


def test_answers_are_crisp():
    """1-2 sentences for simple questions."""
    for e in build_persona(PERSONA):
        assert len(e.answer.split()) <= 60, e.answer


def test_at_most_one_emoji():
    import re

    emoji = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
    for e in build_persona(PERSONA):
        assert len(emoji.findall(e.answer)) <= 1, e.answer


def test_slice_is_labelled():
    for e in build_persona(PERSONA):
        assert e.slice_name == "persona"
        assert e.fact_id is None
