---
name: finetune-gotchas
description: The traps in LoRA fine-tuning with MLX on Apple Silicon, each with the evidence that exposed it. Use when configuring a LoRA run, when a fine-tuned model produces fluent text but has learned nothing, when a served adapter behaves like the base model, when training loss rises instead of falling, or before trusting any fine-tuning evaluation number. Every entry here cost real time to discover.
---

# Fine-tuning gotchas

Each of these was found the hard way during a real project. They are ordered by how
expensive they are to discover, not by how likely they are.

**Scope of the evidence:** MLX / `mlx-lm` 0.31.3 / Apple Silicon / LoRA / Qwen3-4B.
Nothing here has been validated against PyTorch, QLoRA, or cloud GPUs. Where a claim
generalises, it says so.

---

## 1. `scale` in mlx-lm is a direct multiplier, NOT LoRA alpha

**The single most expensive trap in this list.**

In HuggingFace PEFT the effective scaling is `alpha / rank`. So `alpha=64, rank=32`
means a multiplier of 2.0. **mlx-lm does not divide by rank:**

```python
# mlx_lm/tuner/lora.py
self.scale = scale                                  # used directly
delta = (self.scale * self.lora_b.T) @ self.lora_a.T
```

Writing `rank: 32, scale: 64.0` with PEFT semantics in mind makes every update
**32× too large.**

**What it looks like:** training completes normally, reports a plausible trainable-
parameter count, produces a fluent model — that has learned nothing. Asked about the
subject it falls back to the base model's prior. Training loss *rises* as the learning
rate warms up (3.0 → 6.3) and then plateaus high instead of falling.

**Use `scale: 2.0` at rank 32.** If you want PEFT's `alpha/r`, compute it yourself and
pass the result.

**Proof, one variable changed** — 48 examples, 150 iterations:

| `scale` | final train loss |
|---|---|
| 64.0 | 5.710 |
| **2.0** | **0.126** |

---

## 2. The overfit test is how you tell a config bug from a tuning problem

Before touching hyperparameters, try to **overfit 48 examples**. Train on a tiny slice
for a few hundred iterations at a healthy learning rate.

- Loss drives toward ~0 → the machinery works; you have a data or tuning problem.
- Loss stalls high → something is **structurally** wrong. Stop tuning and go read the
  library source.

This test is what turned gotcha #1 from a week of guessing into twenty minutes. A model
that cannot memorise 48 examples will never learn 1,900.

Generalises to any framework.

---

## 3. `mlx_lm.server --adapter-path` silently serves the base model

The flag is accepted, appears in `--help`, logs nothing, and **has no effect**. Both
relative and absolute paths fail the same silent way.

**What it looks like:** training loss of 0.086 proves the adapter memorised the data,
yet served answers are byte-identical to the base model.

**Fix — fuse the adapter and serve the fused model:**

```bash
mlx_lm.fuse --model <base> --adapter-path adapters --save-path fused
mlx_lm.server --model ./fused --host 0.0.0.0 --port 8080
```

**Verify the adapter in-process before trusting any server:**

```python
from mlx_lm import load, generate
m, t = load(BASE, adapter_path="adapters")   # this path DOES work
p = t.apply_chat_template([{"role":"user","content":"<a question only training taught>"}],
                          add_generation_prompt=True, tokenize=False)
print(generate(m, t, prompt=p, max_tokens=40))
```

**Why it matters more than it sounds:** run an evaluation against that server and every
in-scope question scores zero. The obvious conclusion — "fine-tuning doesn't work for
this" — would be completely wrong, caused by an inference-path bug rather than the method.

---

## 4. `mlx_lm.server` only accepts the literal `"default_model"`

In an OpenAI-compatible request, the `model` field must be exactly `"default_model"` to
map to whatever was loaded via `--model`. Any other string is treated as a HuggingFace
repo id, and the request 404s.

```json
{"model": "default_model", "messages": [...]}
```

Left wrong, **every** request in an evaluation fails — the whole measurement apparatus,
silently, at the last step.

---

## 5. Attention-only LoRA teaches style, not facts

mlx-lm's default `keys` are `["self_attn.q_proj", "self_attn.v_proj"]`. That adapts
*how* a model speaks. Factual knowledge lives predominantly in the **MLP** blocks.

```yaml
lora_parameters:
  keys:
    - "self_attn.q_proj"
    - "self_attn.v_proj"
    - "mlp.gate_proj"     # these three are what make facts learnable
    - "mlp.up_proj"
    - "mlp.down_proj"
  rank: 32
  scale: 2.0
num_layers: -1            # default is 16; knowledge needs full depth
```

**Verify it applied** — the first line of training output:

```
Trainable parameters: 1.349% (54.264M/4022.468M)
```

For Qwen3-4B at rank 32 with MLP keys, expect ~54M. Far less means the keys or
`num_layers` did not take effect. Check this within seconds of starting, not after an
hour of training.

**Caution:** parameter count is identical whether `scale` is right or wrong. This check
catches #5, never #1.

---

## 6. Split by fact, never by example

Paraphrase augmentation generates 10–20 questions per fact. Split those randomly by
example and near-duplicates land on both sides of the train/validation boundary, making
validation loss meaningless.

Split by **fact id**, so every paraphrase of one fact stays in one split.

**Consequence worth expecting:** validation loss will barely improve, because the
validation split contains facts the model never trains on — it measures generalisation
to *unseen facts*, which for factual recall is close to impossible. **Validation loss is
not your early-stopping signal here.** A held-out question set is.

---

## 7. A baseline must have the same knowledge as the fine-tune

The comparison is *delivery mechanism*, not knowledge. If the fine-tune trains on a
richer fact set than the baseline prompt contains, the baseline is being asked questions
it was never given.

Measured on a real project: 17 of 68 in-scope questions had answers absent from the
baseline prompt, and **13 of the 18 baseline failures were exactly those questions** —
flattering the fine-tune by roughly 19% of the in-scope set.

**Generate the baseline prompt from the same fact source the model trains on.**

---

## 8. Changing generated content can silently break the thing that measures it

A ruling to expand refusal phrasings from 6 to 12 quietly broke the scorer, whose
markers had been written against the original 6. It then scored 5 of 12 valid refusals
as non-refusals — undercounting refusal rate and overcounting hallucination, on exactly
the two metrics that decide whether to ship.

**When you change generated content, trace its blast radius to every consumer** —
especially the evaluation code. A measuring instrument is code too, and nothing warns
you when it drifts out of agreement with what it measures.

---

## 9. Broadening a detector can narrow the one that depends on it

The fix for #8 expanded refusal markers 8 → 21. But hallucination was defined as
*"not a refusal AND invents specifics"* — so broadening refusal detection **narrowed**
hallucination detection. Hedge-then-fabricate answers began scoring clean:

> *"I'm not aware of an exact record, but he may have spent around 2 years there,
> roughly from 2018 to 2020."* → refusal ✓, hallucination ✗ — **fabrication hidden**

**Test any scorer change in both directions**, against real recorded answers rather than
imagined ones. A structural fix that looked obviously right (treat a contrastive
connective plus a specific as a hedge) was rejected because on the real data it flagged
a perfectly good denial — *"Nope, Erlang is not in his tech stack"* — as a hallucination.

---

## 10. Score correct denials, not just refusal phrasing

For a false-premise question — *"was TCS one of his old jobs?"* — a denial is a **better**
answer than "I don't have that information". Score only for refusal phrasing and you
count a correct answer as a failure. On real data this understated the handled rate by
more than half (24 of 44 answers).

Score `handled = is_refusal(a) or is_denial(a)`.

---

## 11. Metrics can silently start measuring the wrong thing

A general-knowledge group, added to detect catastrophic forgetting, read 38.9% for one
condition. It was not forgetting — the model was declining trivia because its prompt said
to answer only from the supplied data, and a fine-tuned model follows that instruction far
more strictly than a base model. The same weights with no prompt scored 100%.

**Before reporting a bad number, read the actual answers.** A metric measures what it
measures, not what you named it.

---

## 12. Facts go stale in weights; context can overrule them

A contact email changed upstream. The model had the old one baked in and said so. With
the corrected value in context, it answered correctly every time.

Weights are a lossy, un-editable store. Context wins over stale weights — which is a
strong argument for keeping facts in context regardless of what else you bake in.

---

## 13. The finding that reframes the whole exercise

Across 154 held-out questions, three conditions, one scorer:

| Metric | base + facts | tuned, no prompt | tuned + facts |
|---|---|---|---|
| factual accuracy | 89.7% | 64.7% | 83.8% |
| handled rate | 93.2% | 63.6% | **97.7%** |
| hallucination | 4.5% | 15.9% | **0.0%** |
| persona adherence | 25.0% | **100.0%** | **100.0%** |

**Personality transfers into weights emphatically. Facts do not.**

A prompted model ignored its own "1–2 sentences" instruction, answering at 66–204 words.
The fine-tune obeyed it every time. But asked to recall facts cold it fabricated
employment history with invented dates.

**The winning configuration was the fine-tune *with* facts in context** — the voice from
the weights, the facts from the prompt.

Before starting a fine-tune, ask what you actually want in the weights. If the answer is
"facts", consider whether you want retrieval instead.
