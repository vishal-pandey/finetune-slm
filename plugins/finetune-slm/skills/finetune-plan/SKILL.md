---
name: finetune-plan
description: Plan and run a LoRA fine-tune of a small language model about a bounded subject — a person, product, codebase, API, or any domain with a finite fact set. Use when the user wants to "train a model on our data", bake knowledge or a voice into a model, build a domain-specific assistant, or replace a long system prompt with a fine-tune. Starts by deciding whether fine-tuning is the right tool at all, then drives facts to dataset to training to a verdict.
---

# Planning a fine-tune

Drives a fine-tune end to end. Load `finetune-gotchas` alongside this — it carries the
traps, and several will bite during the run.

**Evidence base:** MLX / `mlx-lm` / Apple Silicon / LoRA / Qwen3-4B, from a project that
shipped. Timings assume an M4 with 24 GB.

---

## Step 0 — Decide whether to fine-tune at all

**Do this before anything else. It is the step most likely to save the whole project.**

Ask what belongs in the weights:

| You want | Fine-tuning is |
|---|---|
| A consistent **voice, format, or register** | **The right tool.** Style transfers completely. |
| **Facts** the model should recall cold | **Usually the wrong tool.** Use context or retrieval. |
| Facts that **change** (prices, staff, dates) | **Wrong tool.** Weights are un-editable; context is one edit. |
| Reliable refusal on out-of-scope questions | Helps, but needs explicit negative examples. |

Measured on a real project: putting facts in weights **cost 25 points of factual
accuracy and tripled hallucination** versus the same facts in context. Putting *voice* in
weights took persona adherence from 25% to 100%.

**The strong default: personality in weights, facts in context.** If the user wants facts
in weights, say plainly what it costs, then build it if they still want it — and measure
both so the choice rests on data.

**Also check the corpus is big enough to be worth it.** Below roughly 50 atomic facts,
a well-written system prompt will beat a fine-tune and cost nothing.

---

## Step 1 — Set up the project

Copy `scaffold/` from this plugin into a new repo. It carries the engine, the four
dataset-slice generators, the evaluation harness, and ~80 tests.

```bash
cp -r <plugin>/scaffold/ my-finetune/ && cd my-finetune
uv venv && uv pip install -e ".[dev]"
uv run pytest          # should pass before you change anything
```

Write `config.yaml` — the subject and its fact categories:

```yaml
subject_name: "Acme Payments API"
subject_kind: product
categories:
  endpoint: {required: [path, method, what], optional: [auth, example]}
  concept:  {required: [value]}
  limit:    {required: [value]}
```

Categories are yours to choose. They exist so templates can be written per category
rather than per fact — add a fact later and it inherits full question coverage free.

---

## Step 2 — Curate facts

`facts/facts.yaml` is the **single source of truth**. Everything downstream is generated
from it; the model can only ever know what is in this file.

```yaml
facts:
  - id: endpoint.charge_create
    category: endpoint
    subject: POST /v1/charges
    attrs:
      path: /v1/charges
      method: POST
      what: Creates a charge against a saved payment method.
    aliases: [create charge, charge endpoint]
    keywords: ["/v1/charges"]
```

Rules that matter:

- **Atomic.** One fact = one thing someone could ask about. Prefer many small facts to
  few large ones; more facts means broader question coverage.
- **Accurate above all.** Every fact becomes something the model asserts confidently.
  Never infer, extrapolate, or fill a gap to fit the schema. If a source doesn't say it,
  it doesn't go in. *Two fabrications were caught in review on the reference project —
  both were plausible syntheses of true facts.*
- **`keywords` are eval scoring keys.** Short, distinctive, and genuinely required in a
  correct answer. Avoid tokens that match by accident.
- **Attr names must match your templates.** A fact whose attrs no template references
  generates zero questions and is invisible to training. The suite has a test for this.

---

## Step 3 — Write templates

`templates.yaml` maps each category to question/answer pairs. This is the mechanism that
makes recall generalise past phrasing.

```yaml
endpoint:
  - question: "How do I {what}?"
    answers: ["{method} {path}", "Use {method} {path}."]
  - question: "What does {path} do?"
    answers: ["{what}"]
```

- **≥18 templates per category.** Each fact should be seen 10–20 different ways. One
  phrasing gets memorised verbatim; many generalise.
- **Vary register, not just wording** — terse, casual, lowercase, formal, misspelling-
  adjacent. Write how a stranger actually types.
- **≥2 genuinely different answer variants per template.** Two identical strings satisfy
  the test and defeat its purpose.
- `{subject_name}` comes from config; every other placeholder comes from `fact.attrs`
  plus `{subject}`. The engine checks satisfiability automatically, so an unsatisfiable
  template is skipped rather than crashing at render time.

**Generating a first draft is a good use of Claude** — produce the YAML for the project's
categories, then edit it. Deterministic artifact, fast start.

---

## Step 4 — The other three slices

`build_dataset.py` composes four slices. Proportions matter more than absolute counts:

| Slice | Share | Job |
|---|---|---|
| Q&A | ~65% | Recall that survives rephrasing |
| Statements | ~10% | Facts in declarative form, not welded to question syntax |
| **Refusals** | **~16–22%** | Where knowledge ends |
| Persona | ~8% | Voice, greetings, register |

**The refusal slice is what makes the model safe to deploy.** Without retrieved context, a
model asked about something it never saw will confabulate confidently. Fill
`out_of_scope.yaml` with entities the subject has **no** association with, and include
false premises (*"When did X work at Google?"*) so the model learns to reject the premise
rather than answer inside it.

Watch the proportion, not the count. On the reference project the spec's absolute target
would have made refusals 41% of a smaller-than-expected corpus, biasing the model toward
refusing things it should answer.

---

## Step 5 — Train

```bash
uv run python scripts/build_dataset.py     # prints the slice composition — check it
```

`train/lora_config.yaml` ships with values that are load-bearing. **Read
`finetune-gotchas` before changing any of them**, especially `scale`.

```bash
iters = 3 * (wc -l data/train.jsonl) / batch_size     # ~3 epochs; derive, never hardcode
nohup caffeinate -is mlx_lm.lora --config train/lora_config.yaml --mask-prompt > train.log 2>&1 &
```

**Check the first line of output immediately:**

```
Trainable parameters: 1.349% (54.264M/4022.468M)
```

Far below that means the MLP keys or `num_layers` did not apply. Kill it and fix the
config rather than waiting an hour.

Reference throughput: ~0.5 it/s on an M4, ~9.7 GB peak, so ~1,300 iterations ≈ 43 minutes.

Train over SSH under `nohup` — a network drop then kills your monitoring, not the run.

---

## Step 6 — Serve

**Fuse the adapter. Do not use `--adapter-path` on the server** — it silently serves the
base model (gotcha #3).

```bash
mlx_lm.fuse --model <base> --adapter-path adapters --save-path fused
mlx_lm.server --model ./fused --host 0.0.0.0 --port 8080
```

Sanity-check with a question only training could answer, before running any evaluation.

---

## Step 7 — Evaluate and decide

Use the `finetune-eval` skill. Do not skip it: without a fair baseline you cannot tell
whether the fine-tune helped, and the failure modes are not visible from loss curves.

Run three conditions when facts are involved:

1. **baseline** — base model + facts in context
2. **weights-only** — fine-tune, no prompt
3. **hybrid** — fine-tune + facts in context

On the reference project the hybrid won, and it was not the configuration originally
commissioned.

---

## Time budget (M4, 24 GB, ~60 facts)

| Step | Time |
|---|---|
| Facts + templates | 1–3 h, mostly curation |
| Dataset build | seconds |
| Training (~1,300 iters) | ~45 min |
| Fuse + serve | ~5 min |
| Two eval runs | ~35 min |

A persona-only retrain (drop the fact Q&A) is ~10 minutes of training — worth knowing
once you have decided facts belong in context.
