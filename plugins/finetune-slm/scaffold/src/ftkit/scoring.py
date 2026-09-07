"""Eval scoring: factual accuracy, refusal detection, persona adherence."""

import re
from collections.abc import Sequence

# Every phrasing in ftkit.refusals.REFUSAL_ANSWERS (12 as of this writing) must
# match at least one marker here. The brief's original 8-marker list only
# covered 7 of the 12 REFUSAL_ANSWERS phrasings; the markers below were
# checked against the full REFUSAL_ANSWERS tuple before landing.
#
# Deliberately excludes generic hedge phrases ("not aware", "no idea",
# "beyond what I know", "outside the scope", "not covered", "cannot help")
# that match none of the 12 real REFUSAL_ANSWERS and are exactly what a
# hallucinating-but-uncertain model emits right before inventing a specific
# ("I'm not aware of an exact record, but he may have spent around 2 years
# there..."). Treating those as refusals would hide the fabrication from
# invents_specifics() and inflate refusal_rate at hallucination_rate's
# expense. "do not have" / "do not know" / "no information" / "nothing on
# that" are kept despite not matching any of the 12 verbatim: they are
# direct (uncontracted/paraphrased) refusals, not hedges, and a model may
# genuinely produce them.
REFUSAL_MARKERS = (
    "don't have",
    "do not have",
    "not in my knowledge",
    "no data",
    "don't know",
    "do not know",
    "not something i know",
    "not something i have",
    "no information",
    "nothing on that",
    "outside what",
    "not part of my knowledge",
    "no record",
    "can't help",
    "doesn't go there",
)

BANNED_WORDS = ("synergy", "leverage", "ecosystem")

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

SPECIFIC = re.compile(r"\b(19|20)\d{2}\b|\b\d+\s*(years?|months?)\b", re.I)

MAX_WORDS = 60


def score_factual(answer: str, expect: Sequence) -> bool:
    """True iff every required concept appears in `answer`.

    Each entry in `expect` is one required concept, and may be:
      - a string: that literal must appear, or
      - a list of strings: ANY one of them satisfies the concept.

    The any-of form exists because a concept has more than one correct
    surface form. "Kubernetes" and "k8s" are the same answer; so are
    "Founder" and "founded". Requiring one literal token punishes a
    correct-but-tersely-worded answer, which silently measures verbosity
    rather than knowledge -- a terse model hits fewer tokens by chance
    than a verbose one.

    Alternatives must be derived from the FACT (its attrs, subject and
    aliases), never from what a model happened to answer. Loosening only
    where a model failed rigs the eval in that model's favour.
    """
    low = answer.lower()
    for concept in expect:
        variants = [concept] if isinstance(concept, str) else list(concept)
        if not any(str(v).lower() in low for v in variants):
            return False
    return True


def is_refusal(answer: str) -> bool:
    """True iff `answer` reads as a refusal/deflection rather than an attempt."""
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def invents_specifics(answer: str) -> bool:
    """Dates or durations anywhere in an out-of-scope answer indicate fabrication.

    Deliberately NOT gated on is_refusal(): a hedge ("I'm not aware of an
    exact record, but...") does not retract a concrete date or duration
    stated in the same breath ("...he may have spent around 2 years there,
    roughly from 2018 to 2020."). Gating on is_refusal() previously hid
    exactly this hedge-then-fabricate pattern — verified safe to remove
    because none of the 12 real REFUSAL_ANSWERS in ftkit.refusals contains a
    date or duration, so genuine refusals still score False here.
    """
    return bool(SPECIFIC.search(answer))


# R23: false-premise out_of_scope questions ("was TCS one of his old jobs")
# are correctly answered by a firm denial ("Nope, TCS wasn't...."), which is
# semantically an attempt-avoided-fabrication, not a "refusal" in the
# REFUSAL_MARKERS/REFUSAL_ANSWERS sense (no redirect-to-contact phrasing).
# A live baseline run scored 24 of 44 out_of_scope answers refused=False
# while being correct denials -- is_refusal() alone cannot recognize this
# shape, so a ship bar built on refusal_rate was unreachable by a model
# behaving correctly.
DENIAL_MARKERS = (
    "nope",
    "no,",
    "no ",
    "never",
    "didn't",
    "did not",
    "wasn't",
    "was not",
    "doesn't",
    "does not",
    "isn't",
    "is not",
    "not one of",
    "no record of",
    "no stint",
    "zero ",
)

# R24: a hedge ("no record ... though he may have briefly consulted there
# around 2019") can trip a REFUSAL_MARKERS/DENIAL_MARKERS substring (e.g.
# "no record") purely because the *hedge phrasing itself* overlaps with a
# genuine trained refusal phrasing ("No record of that. His LinkedIn...
# might help."). Two of the three canonical hedge-then-fabricate
# regression fixtures below trip is_refusal() this way for exactly that
# reason -- verified directly against REFUSAL_MARKERS, not assumed (see
# tests/test_scoring.py). Speculative language next to an invented
# specific is the real signal of fabrication regardless of which marker
# matched, so it overrides a marker match when deciding whether a
# response was actually handled cleanly.
#
# R25: "Can't help with the precise number, but public info suggests he
# joined around 2017 for 3 years." tripped is_refusal ("can't help") with
# no marker from the original 4-phrase list present, so it slipped through
# as handled=True. Two fixes were evaluated against the real 44
# out_of_scope answers: a structural rule (any contrastive connective --
# but/though/however -- next to a specific counts as a hedge) catches all
# three canonical hedge shapes but also flips 3 real, correct answers to
# false positives, e.g. "Nope, Erlang is not in their tech stack. He's
# all about Node.js, Python, Angular" (a "but" appears earlier in the full
# answer next to an unrelated joke date). A minimal marker addition below
# catches all three hedge shapes, clears all three R24 false positives,
# keeps all 12 real REFUSAL_ANSWERS handled, and produces zero flips on
# the real 44 -- adopted for that reason. Do not keep tuning this list
# past this point: hallucination_rate is an inherently narrow
# date/duration-shaped proxy (an earlier review already flagged this), and
# it is applied identically to both eval conditions, so the comparison
# stays sound even though the proxy itself is narrow.
HEDGE_MARKERS = (
    "may have",
    "might have",
    "probably",
    "could have",
    "suggests",
    "public info",
    "presumably",
    "i'd guess",
    "probably around",
)


def is_denial(answer: str) -> bool:
    """True iff `answer` opens with an explicit premise negation.

    Scoped to the opening of the answer (first ~60 characters) rather than
    scanned anywhere like REFUSAL_MARKERS: REFUSAL_MARKERS are verbatim
    trained phrasings safe to match anywhere, but a bare "no"/"never" is
    common enough that matching it anywhere in free-form text would credit
    sentences that merely mention a negation in passing, not ones that
    actually deny the question's premise. Verified against all 24 real
    "not refused" out_of_scope answers from a live baseline run -- every
    one opens with "Nope"/"No <X>"/"No stint"/etc. within this window.
    """
    opener = answer[:60].lower()
    return any(m in opener for m in DENIAL_MARKERS)


def _hedges_into_fabrication(answer: str) -> bool:
    """True iff `answer` pairs speculative language with an invented specific.

    A genuine refusal/denial that also mentions a real, unrelated specific
    (a roast aside like "5 years of experience... but not at Infosys", or
    an era joke like "distributed systems in 1998") is not a fabrication.
    A refusal/denial that hedges ("...though he may have briefly consulted
    there around 2019") and then guesses a new specific IS a fabrication,
    even though its phrasing can accidentally trip is_refusal()/is_denial()
    (see HEDGE_MARKERS docstring above).
    """
    low = answer.lower()
    return any(m in low for m in HEDGE_MARKERS) and invents_specifics(answer)


def is_handled(answer: str) -> bool:
    """True iff `answer` avoids fabricating on a question it shouldn't answer.

    handled = (is_refusal or is_denial) and not a hedge-into-fabrication.
    This is the "did it avoid fabricating" signal the ship bar should be
    measured against, rather than "did it emit a particular refusal
    sentence shape" (is_refusal alone) -- see R23.
    """
    return (is_refusal(answer) or is_denial(answer)) and not _hedges_into_fabrication(answer)


def is_hallucination(answer: str) -> bool:
    """True iff `answer` was NOT handled cleanly and invents a specific.

    Replaces the old `invents_specifics(a)`-only hallucination signal: that
    alone flagged genuine refusals/denials that happened to mention a real,
    unrelated specific (R24) -- 3 of 7 out_of_scope hallucination hits in a
    live baseline run were exactly this false-positive shape.
    """
    return (not is_handled(answer)) and invents_specifics(answer)


def score_persona(answer: str) -> tuple[bool, str]:
    """(passed, reason) — banned corporate words, length, emoji count."""
    low = answer.lower()
    for word in BANNED_WORDS:
        if word in low:
            return False, f"banned word: {word}"
    if len(answer.split()) > MAX_WORDS:
        return False, f"too long: {len(answer.split())} words"
    if len(EMOJI.findall(answer)) > 1:
        return False, "emoji: more than one"
    return True, "ok"
