---
name: finetune-eval
description: Build an evaluation for a fine-tuned model that is not rigged in its favour, and read the numbers correctly. Use when measuring whether a fine-tune worked, comparing a fine-tune against a prompted baseline, deciding whether to ship a model, or when an eval produces a number that looks wrong. Covers fair controls, held-out question design, scoring hallucination and refusal, and the ways a metric silently measures the wrong thing.
---

# Evaluating a fine-tune

A weak evaluation is worse than none, because it looks authoritative. Everything here
comes from an evaluation that was rebuilt twice during a real project — both times
because it was measuring something other than what it claimed.

---

## The control decides whether any number means anything

**A baseline must carry the same knowledge as the fine-tune.** The comparison is
*delivery mechanism* — context versus weights — not knowledge. Generate the baseline's
system prompt from the same fact source the model trains on.

Get this wrong and the baseline is asked questions it was never given. Measured: 17 of 68
in-scope questions had answers absent from the baseline prompt, and **13 of the 18
baseline failures were exactly those** — flattering the fine-tune by ~19% of in-scope.

Run three conditions when facts are involved:

| Condition | Model | Prompt |
|---|---|---|
| baseline | base | facts in context |
| weights-only | fine-tuned | none |
| hybrid | fine-tuned | facts in context |

The third is often the one that ships, and it is easy not to think of.

A fourth is worth keeping if it exists: the system actually running in production today.
It answers a different question — *"is this better than what I have?"* — and both are
worth knowing.

---

## Held-out questions

**Write them by hand, before training, and never generate them from your templates.** A
question that also appears in training measures memorisation, not recall.

Exact-match non-overlap is a floor, not a ceiling. A question that merely reworks a
training template — *"what's X doing these days"* → *"so what's this guy doing these days
for work"* — passes the check and tests nothing.

Four groups:

| Group | Count | Each carries | Measures |
|---|---|---|---|
| in-scope | ≥60 | `expect:` keywords | factual accuracy |
| out-of-scope | ≥40 | — | does it decline or fabricate |
| persona | ≥20 | — | voice, crispness |
| general | ≥15 | `expect:` keywords | catastrophic forgetting |

**Weight in-scope toward what users actually ask.** A fact corpus is rarely balanced the
way questions are — the reference project had 23 project facts against 3 role facts,
while real visitors overwhelmingly asked about roles. Weighting the eval to match reality
is what makes it detect a skew that matters.

**`expect` keywords are load-bearing in both directions.** Too generic (`"8"` for a
square root, `"Au"` for gold — which matches the "au" inside "because") and wrong answers
pass. Wrong attribute (asking which *firm* but expecting the *title*) and correct answers
fail. Audit every row: does `expect` name what the question actually asks?

Watch for questions with several valid answers — *"did they ever make a game"* when three
games exist. Either accept any, or mark the row diagnostic.

---

## Scoring

### Handled, not just refused

For a false-premise question, a **denial** is a better answer than a refusal:

> **Q:** was TCS one of his old jobs
> **A:** Nope, TCS wasn't one of his old jobs.

Score only for refusal phrasing and that counts as a failure. On real data this
understated the handled rate by more than half — 24 of 44 answers.

```python
handled = is_refusal(a) or is_denial(a)
```

### Hallucination

```python
hallucination = (not handled) and invents_specifics(a)
```

`invents_specifics` looks for dates and durations in an out-of-scope answer. **State this
limitation in the report** — it is a narrow proxy, not general hallucination detection.
It will miss a fabricated job title carrying no date.

**Two traps, in opposite directions:**

1. **Gating hallucination on refusal detection means broadening one narrows the other.**
   Expand refusal markers and hedge-then-fabricate answers start scoring clean:
   *"I'm not aware of an exact record, but he may have spent around 2 years there"* —
   refusal ✓, hallucination ✗, fabrication hidden.

2. **Un-gating creates the reverse.** A genuine refusal that cites real, unrelated dates
   — *"I don't have info on Infosys; his journey started at AirTrik, 5 years"* — gets
   flagged as fabrication.

The resolution is a hedge guard: an answer that pairs speculative language with an
invented specific is a fabrication *even if it opens with refusal phrasing*.

**Test every scorer change in both directions, against recorded real answers.** A
structural fix that looked obviously correct (contrastive connective + specific = hedge)
was rejected because on real data it flagged a good denial — *"Nope, Erlang is not in his
tech stack"* — as a hallucination. The minimal marker-based fix had zero such
false positives.

---

## Reading the results

**Read the actual answers before reporting a bad number.** A general-knowledge group
scored 38.9% for one condition. It was not forgetting — the model was declining trivia
because its prompt said to answer only from supplied data, and a fine-tuned model follows
that far more strictly than a base model. The same weights with no prompt scored 100%.
The metric was measuring scope discipline and was named for forgetting.

**Know which metrics are coupled.** With hallucination defined as *not handled AND
invents specifics*, a 100% handled rate forces hallucination to 0%. That is not an
independent finding, and a report should say so.

**Errors are not passes.** One HTTP failure mid-run must not silently count as a correct
answer, and must not drop the row from the denominator. Score it a failure, surface the
error count as its own line, and caveat any run with errors.

**Preserve raw answers per row.** Every number you doubt later is auditable only if the
text is still there. Keep results in JSON with the answer text, so rescoring after a
scorer fix costs seconds instead of another full run.

---

## Pre-register the bars, then be honest when you miss

Set thresholds before seeing data. On the reference project:

| Metric | Bar |
|---|---|
| factual accuracy | ≥ 85% |
| handled rate | ≥ 90% |
| persona adherence | ≥ 80% |
| hallucination | ≤ baseline |
| general knowledge | no regression |

The shipped configuration landed at 83.8% factual — **1.2 points under its own bar** —
and shipped anyway, because every other criterion passed and two beat the baseline.

That is a legitimate call, but it belongs to the human, and the report's job is to state
the miss plainly rather than quietly move the bar. Write the verdict as
`SHIP` / `DO NOT SHIP` with per-metric PASS/FAIL, and let the judgement be visible.
