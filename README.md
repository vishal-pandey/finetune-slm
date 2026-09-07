# finetune-slm

A Claude Code plugin for fine-tuning a small language model about a bounded subject —
a person, a product, a codebase, an API, anything with a finite fact set.

It carries three skills, a working pipeline, and the traps that cost a full night to find.

```bash
/plugin marketplace add vishal-pandey/finetune-slm
/plugin install finetune-slm@vishal-ml
```

---

## What's in it

| Skill | Job |
|---|---|
| **finetune-plan** | The whole arc — starting with whether to fine-tune at all |
| **finetune-gotchas** | 13 traps, each with the evidence that exposed it |
| **finetune-eval** | Building an evaluation that isn't rigged, and reading it correctly |

Plus `scaffold/` — a tested pipeline (fact model, template engine, four dataset-slice
generators, evaluation harness, ~80 tests) that you copy into a new project and point at
your own domain. Categories live in `config.yaml`, templates in `templates.yaml`. No
Python editing to start.

---

## The finding worth reading first

Facts and personality do not behave the same way in weights.

Across 154 hand-written held-out questions, three conditions, one scorer:

| Metric | base + facts | tuned, no prompt | tuned + facts |
|---|---|---|---|
| factual accuracy | 89.7% | 64.7% | 83.8% |
| handled rate | 93.2% | 63.6% | **97.7%** |
| hallucination | 4.5% | 15.9% | **0.0%** |
| persona adherence | 25.0% | **100.0%** | **100.0%** |

**Personality transfers into weights emphatically. Facts do not.**

A prompted model ignored its own "1–2 sentences" instruction and answered at 66–204 words.
The fine-tune obeyed it every time. But asked to recall facts cold, it fabricated
employment history with invented dates — for a company the subject never worked at.

The configuration that shipped was the fine-tune **with facts in context**: voice from the
weights, facts from the prompt. Which was not what the project set out to build.

If you want facts in weights, `finetune-plan` will build it and measure it — and tell you
what it costs.

---

## The most expensive trap

`scale` in mlx-lm is a **direct multiplier**, not LoRA alpha. It is not divided by rank.

```python
# mlx_lm/tuner/lora.py
self.scale = scale
delta = (self.scale * self.lora_b.T) @ self.lora_a.T
```

Writing `rank: 32, scale: 64.0` with HuggingFace PEFT semantics in mind makes every
update **32× too large**. Training completes, reports a healthy parameter count, produces
fluent text — and the model learns nothing.

One variable changed, 48 examples, 150 iterations:

| `scale` | final train loss |
|---|---|
| 64.0 | 5.710 |
| **2.0** | **0.126** |

The diagnostic that found it: **try to overfit 48 examples.** If loss won't approach zero
on a tiny slice, the problem is structural, not a tuning issue. Stop tuning and read the
library source.

Twelve more in `finetune-gotchas`, including a server flag that silently serves the base
model, and two ways an evaluation can quietly start measuring the wrong thing.

---

## Scope of the evidence

MLX / `mlx-lm` 0.31.3 / Apple Silicon / LoRA / Qwen3-4B, from a project that trained a
model and deployed it.

Nothing here has been validated against PyTorch, QLoRA, or cloud GPUs. Where a lesson
generalises — the overfit test, fair controls, split-by-fact — the skill says so. Where it
doesn't, it names the version it was found on.

Reference hardware: M4 Mac mini, 24 GB. A 4B LoRA run at ~1,300 iterations takes about
45 minutes at ~9.7 GB peak.

---

## Requirements

- Apple Silicon (MLX)
- Python 3.12+, `uv`
- ~8 GB disk for a 4B model, ~10 GB RAM while training

---

MIT · [Vishal Pandey](https://github.com/vishal-pandey)
