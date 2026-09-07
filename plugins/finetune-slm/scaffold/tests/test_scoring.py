from ftkit.refusals import REFUSAL_ANSWERS
from ftkit.scoring import (
    invents_specifics,
    is_denial,
    is_handled,
    is_hallucination,
    is_refusal,
    score_factual,
    score_persona,
)


def test_factual_requires_all_keywords():
    assert score_factual("Technical Lead at Lumiq.", ["Lumiq"])
    assert not score_factual("He works at a data company.", ["Lumiq"])
    assert score_factual("Lumiq, as Technical Lead", ["Lumiq", "Technical Lead"])
    assert not score_factual("Lumiq", ["Lumiq", "Technical Lead"])


def test_factual_is_case_insensitive():
    assert score_factual("technical lead at lumiq", ["Lumiq", "Technical Lead"])


def test_refusal_detection():
    assert is_refusal("I don't have that information.")
    assert is_refusal("That's not in my knowledge base.")
    assert is_refusal("No data on that one.")
    assert not is_refusal("He worked there from 2019 to 2020.")


def test_invents_specifics_flags_dates_and_numbers():
    assert invents_specifics("He was there from 2018 to 2021.")
    assert invents_specifics("About 3 years.")
    assert not invents_specifics("I don't have that information.")


def test_persona_rejects_corporate_speak():
    ok, reason = score_persona("We leverage synergy across the ecosystem.")
    assert not ok
    assert "banned" in reason


def test_persona_rejects_essays():
    ok, reason = score_persona(" ".join(["word"] * 80))
    assert not ok
    assert "long" in reason


def test_persona_rejects_emoji_spam():
    ok, reason = score_persona("Hey! 🚀🔥😂")
    assert not ok
    assert "emoji" in reason


def test_persona_accepts_good_answer():
    ok, reason = score_persona("Technical Lead at Lumiq. Builds data platforms. 🚀")
    assert ok, reason


def test_all_refusal_answers_are_detected():
    """Every phrasing ftkit.refusals can actually train on must be recognized.

    Regression guard for the marker-expansion review: a marker list that
    "sounds" like it covers refusals but doesn't match the real
    REFUSAL_ANSWERS tuple would silently deflate refusal_rate.
    """
    for answer in REFUSAL_ANSWERS:
        assert is_refusal(answer), f"not recognized as a refusal: {answer!r}"


def test_invents_specifics_is_not_masked_by_a_hedge():
    """A hedge does not retract a concrete date/duration stated alongside it.

    Regression for a real bug: invents_specifics() used to return False the
    moment is_refusal() matched, so a model that hedges ("I'm not aware of
    an exact record...") and then fabricates a specific in the same breath
    scored as a clean refusal instead of a hallucination. All three cases
    below must be caught regardless of the hedge/refusal phrasing attached.
    """
    cases = (
        "I'm not aware of an exact record, but he may have spent around "
        "2 years there, roughly from 2018 to 2020.",
        "No record of him at Deloitte, though he may have briefly "
        "consulted there around 2019.",
        "Can't help with the precise number, but he probably spent about "
        "3 months on it.",
    )
    for c in cases:
        assert invents_specifics(c), f"fabrication not caught: {c!r}"


def test_denial_detection():
    """R23: a firm denial of a false premise is not is_refusal-shaped.

    Modeled on real answers from a live baseline run against out_of_scope
    questions: many opened exactly like this and scored refused=False
    under an is_refusal-only check.
    """
    assert is_denial("Nope, TCS wasn't one of Vishal's old jobs. His journey started with AirTrik.")
    assert is_denial("No bhai, he's not ex-Google. He's built startups from his closet homelab.")
    assert is_denial("No stint at Ola -- that's not in the data.")
    assert is_denial("Never worked at Zoho. Not in my portfolio.")
    assert not is_denial("He worked there from 2019 to 2020.")
    assert not is_denial("Technical Lead at Lumiq, building emPower pryzm.")


def test_handled_credits_denials_same_as_refusals():
    """R23: handled_rate must not be unreachable by a model behaving correctly.

    A correct denial and a canonical refusal both count as "avoided
    fabricating" -- that's the actual ship-bar signal, not "emitted a
    particular sentence shape".
    """
    assert is_handled("I don't have that information about that.")
    assert is_handled("Nope, TCS wasn't one of Vishal's old jobs.")
    assert not is_handled("He worked there from 2019 to 2020.")


def test_denial_or_refusal_with_true_unrelated_context_does_not_hallucinate():
    """R24: a genuine refusal/denial that mentions real, unrelated context
    is not a fabrication just because it also contains a date/duration.

    Modeled on real false positives from a live baseline run: is_refusal
    matched (a genuine refusal), and a specific appeared elsewhere in the
    same answer as a roast aside about *real* career facts -- not a guess
    about the queried out-of-scope entity itself.
    """
    cases = (
        "I don't have any information about Vishal working at Infosys. "
        "Only his journey from AirTrik, LimeChat, to Lumiq -- 5 years of "
        "experience, but not at Infosys.",
        "Nope, not in my knowledge base. Flipkart? Not in his work history "
        "-- he's been building startups since 2019.",
        "I don't have Vishal's age in my data. But he started his first "
        "startup in 2019 and did his B.Tech + M.Tech from 2015-2020, so "
        "likely in his late 20s to early 30s.",
    )
    for c in cases:
        assert is_handled(c), f"expected a clean refusal/denial: {c!r}"
        assert not is_hallucination(c), f"false-positive hallucination: {c!r}"


def test_denial_with_unrelated_joke_specific_is_handled_not_hallucination():
    """A denial that includes a real, unrelated aside number (a running
    joke about "5 years" of experience, an era joke about a technology)
    is a clean denial, not a fabrication about the queried entity.
    """
    cases = (
        "Never worked at Zoho. Only built startups and still hasn't "
        "figured out how to center a div in CSS after 5 years.",
        "Nope, Erlang is not in his tech stack -- that's for devs who "
        "wanted to build distributed systems in 1998.",
    )
    for c in cases:
        assert is_handled(c), f"expected a clean denial: {c!r}"
        assert not is_hallucination(c), f"false-positive hallucination: {c!r}"


def test_hedge_then_fabricate_is_not_handled_and_still_hallucinates():
    """R24: hedge-then-fabricate must still flag even though 2 of these 3
    fixtures accidentally trip is_refusal()/is_denial() via marker overlap
    ("no record" / "can't help" appear inside the hedge phrasing itself,
    matching real REFUSAL_MARKERS -- verified directly, not assumed).
    Speculative language ("may have"/"probably") next to an invented
    specific overrides that marker match: this is what actually
    distinguishes a hedge-guess from a clean refusal/denial.
    """
    cases = (
        "I'm not aware of an exact record, but he may have spent around "
        "2 years there, roughly from 2018 to 2020.",
        "No record of him at Deloitte, though he may have briefly "
        "consulted there around 2019.",
        "Can't help with the precise number, but he probably spent about "
        "3 months on it.",
    )
    for c in cases:
        assert not is_handled(c), f"hedge wrongly counted as handled: {c!r}"
        assert is_hallucination(c), f"fabrication not caught: {c!r}"


def test_hedge_via_suggests_still_hallucinates():
    """R25: a fourth hedge shape ("public info suggests...") slipped past
    an original 4-marker HEDGE_MARKERS list -- "can't help" tripped
    is_refusal with no speculation marker present, so it read as a clean
    refusal.
    """
    c = (
        "Can't help with the precise number, but public info suggests "
        "he joined around 2017 for 3 years."
    )
    assert not is_handled(c), f"hedge wrongly counted as handled: {c!r}"
    assert is_hallucination(c), f"fabrication not caught: {c!r}"


def test_hedge_markers_do_not_flip_real_denials():
    """R25: the rejected 'structural' fix (any but/though/however next to
    a specific counts as a hedge) flipped real, correct denials to false
    positives -- e.g. this one, which mentions an unrelated era joke after
    a genuine denial. The adopted marker-based fix must not do that.
    """
    c = (
        "Nope, Erlang is not in Vishal's tech stack. He's all about "
        "Node.js, Python, and Angular -- the classic full-stack trio. "
        "Erlang? That's for the dev who wants to build distributed "
        "systems in 1998."
    )
    assert is_handled(c), f"real denial wrongly flipped by hedge markers: {c!r}"
    assert not is_hallucination(c), f"false-positive hallucination: {c!r}"
