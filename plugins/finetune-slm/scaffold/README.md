# ftkit scaffold

A fine-tuning dataset pipeline for teaching a small language model about any
one bounded subject — a person, a product, a codebase, whatever you like —
so it can answer questions about that subject from its weights alone, with
no system prompt and no retrieval at inference time.

It builds a paraphrase-augmented Q&A + statement + refusal + persona
dataset from a small set of hand-written facts, splits it so paraphrases of
one fact never straddle train/validation, trains a LoRA adapter with MLX on
Apple Silicon, and evaluates the result against a hand-written held-out set.

## How it works

1. **`config.yaml`** declares who/what the subject is (`subject_name`,
   `subject_kind`) and the fact categories your project uses, each with
   `required` and `optional` attrs.
2. **`facts.yaml`** is your single source of truth: a flat list of facts,
   each with an `id`, a `category` (must be declared in config.yaml), a
   `subject`, `attrs`, and `keywords` for eval scoring.
3. **`templates.yaml`** is a per-category table of question/answer
   templates. A template's placeholders (`{subject}`, `{subject_name}`, or
   any attr name) are checked for satisfiability automatically — a fact
   missing an attr a template needs just makes that template inapplicable
   to that fact, it never crashes at render time.
4. **`scripts/build_dataset.py`** crosses every fact against its
   category's applicable templates (the Q&A slice), adds declarative
   statement completions, adds a refusal slice built from an
   `out_of_scope.yaml` of things the subject is *not* (so the model learns
   where its knowledge ends instead of confabulating), adds a persona
   slice for voice, and writes `data/{train,valid,test}.jsonl` in mlx-lm
   chat format — split by fact, not by example, so no paraphrase of one
   fact leaks across the split.
5. **`train/lora_config.yaml`** trains a LoRA adapter with `mlx_lm.lora`.
   Read the comment on `scale` before touching it — it is the single most
   expensive trap in this whole pipeline (see below).
6. **`scripts/run_eval.py`** / **`scripts/rescore.py`** score a served
   model (or a fused, trained model) against a hand-written `heldout.yaml`
   for factual accuracy, refusal/denial handling, hallucination rate,
   persona adherence, and general-knowledge retention.
7. **`scripts/chat.py`** is a quick interactive/one-shot CLI against an
   `mlx_lm.server` instance.

## Try it immediately (worked example)

This scaffold ships with a complete worked example — `config.example.yaml`,
`templates.example.yaml` (145 templates across all seven example
categories), and `facts/EXAMPLE.*.yaml` / `eval/EXAMPLE.heldout.yaml` for a
small, entirely fictional subject ("Asha Rao"). `scripts/build_dataset.py`
uses these files *by default*, so you can build a real dataset with zero
configuration:

```sh
uv sync
uv run python scripts/build_dataset.py
```

This writes `data/train.jsonl`, `data/valid.jsonl`, and `data/test.jsonl`.
Every row is a two-message mlx-lm chat example (`user` + `assistant`, never
`system` — the whole point is that the model must work with no system
prompt at inference time).

Run the tests the same way:

```sh
uv run pytest
```

## Starting your own project

1. Copy `config.example.yaml` → `config.yaml` and edit `subject_name`,
   `subject_kind`, and `categories` for your subject. You don't need all
   seven example categories — use only the ones you need, and give each
   just the `required`/`optional` attrs your templates actually use.
2. Copy `templates.example.yaml` → `templates.yaml` and adapt it: keep the
   categories you're using, delete the rest, and reword to taste. Keep at
   least ~15-20 templates per category you use — too few templates relative
   to your fact count skews the training mix toward the refusal slice (see
   the comment at the top of `src/ftkit/templates.py`).
3. Write your own `facts/facts.yaml`, `facts/out_of_scope.yaml` (things
   your subject is *not*, crossed with templates to build the refusal
   slice — keep every entity here genuinely unrelated to your subject), and
   `facts/persona.yaml` (voice: greetings, roasts, register — whatever fits).
4. Write a hand-authored `eval/heldout.yaml` (`in_scope`, `out_of_scope`,
   `persona`, `general` groups) — casually phrased, deliberately different
   from your templated training questions, so evaluation measures recall
   rather than memorization.
5. Build and train:

   ```sh
   uv run python scripts/build_dataset.py \
     --config config.yaml --templates templates.yaml \
     --facts facts/facts.yaml --out-of-scope facts/out_of_scope.yaml \
     --persona facts/persona.yaml

   uv run mlx_lm.lora --config train/lora_config.yaml
   ```

   `train/lora_config.yaml`'s `iters` is sized for a specific corpus (the
   worked example's dataset size) — recompute it for your own corpus:
   `(rows in data/train.jsonl / batch_size) * epochs`.

6. Fuse and serve (`mlx_lm.server --adapter-path` silently serves the base
   model unfused — always fuse first):

   ```sh
   uv run mlx_lm.fuse --model <base-model> --adapter-path adapters --save-path fused
   uv run mlx_lm.server --model ./fused --host 0.0.0.0 --port 8080
   ```

7. Evaluate and chat:

   ```sh
   uv run python scripts/run_eval.py --base-url http://localhost:8080/v1 --label adapter
   uv run python scripts/chat.py
   ```

## The one config value that will silently ruin your training run

`mlx-lm` applies LoRA's `scale` as a **direct multiplier** on the update,
not as HuggingFace PEFT's `alpha / rank`. Writing `rank: 32, scale: 64.0`
with PEFT semantics in mind (intending effective scale 64/32 = 2.0) instead
applies `64.0` directly — every update ends up 32x too large. The result:
training completes normally, reports a plausible trainable-parameter count,
and produces a fluent model that has learned **nothing** — it falls back to
the base model's prior on every question about your subject, and train loss
*rises* as the learning rate warms up instead of falling.

`train/lora_config.yaml` ships with the correct value, `scale: 2.0` at
`rank: 32`. If you change `rank`, recompute `scale` yourself as your
intended `alpha / rank` — don't just copy a `scale` value from somewhere
else without checking what rank it was paired with.

## Package layout

```
config.example.yaml          worked-example project config
templates.example.yaml       worked-example templates.yaml (145 templates)
facts/EXAMPLE.*.yaml         worked-example facts / out-of-scope / persona
eval/EXAMPLE.heldout.yaml    worked-example held-out eval set
train/lora_config.yaml       LoRA training config (mlx_lm.lora)
src/ftkit/                   the pipeline: config, factmodel, templates,
                              qa, statements, refusals, persona, dataset,
                              example, scoring
scripts/                     build_dataset.py, run_eval.py, rescore.py,
                              chat.py
tests/                       pytest suite — run with `uv run pytest`
```
