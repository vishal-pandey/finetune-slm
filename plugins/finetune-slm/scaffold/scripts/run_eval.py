"""Evaluate an OpenAI-compatible endpoint against a held-out eval set.

Scores all four heldout groups (in_scope, out_of_scope, persona, general)
and writes eval/results_<label>.json plus five printed metrics:
factual_accuracy, handled_rate, hallucination_rate, persona_adherence, and
general_knowledge (catastrophic-forgetting check on the base model's
general ability).

handled_rate (R23) is "did it avoid fabricating on an out-of-scope
question" -- a canonical refusal ("I don't have that information...") OR a
correct denial of a false premise ("Nope, TCS wasn't one of his old
jobs.") both count. It replaced a stricter refusal_rate that only
recognized the former shape and scored a model behaving correctly as
failing. hallucination_rate (R24) is NOT handled_rate's complement: a
genuine refusal/denial that also mentions a real, unrelated specific (a
roast aside, an era joke) is not a hallucination, so hallucination_rate can
undercount relative to (100 - handled_rate). See ftkit.scoring.is_handled /
is_hallucination for the full reasoning.

Usage:
  uv run python scripts/run_eval.py --base-url http://localhost:8080/v1 \
      --label adapter --system-prompt-file ""

temperature is pinned to 0.0 so runs are reproducible. Timeout is generous
(180s per request) because this can hit a small model on modest hardware,
where a single answer can take several seconds.

A single request failure (timeout, 5xx, malformed response) does not abort
the run: each question is asked through ask_safe(), which records a
per-row error instead of raising, so completed answers are never
discarded by one later failure. An errored row scores as a failure
(never a silent pass) and is counted in the printed/written `errors` total.
Results are written in a `finally` block so even an unexpected failure
elsewhere still persists whatever was collected up to that point.
"""

import argparse
import json
from pathlib import Path

import httpx
import yaml

from ftkit.scoring import is_handled, is_hallucination, score_factual, score_persona

ROOT = Path(__file__).parent.parent


def ask(client: httpx.Client, base_url: str, question: str, system: str | None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    r = client.post(
        f"{base_url}/chat/completions",
        json={
            # mlx_lm.server's ModelProvider only maps the literal sentinel
            # "default_model" to the model (and adapter) passed via --model
            # / --adapter-path at server startup; any other string is
            # treated as a real repo id to load and 404s against
            # huggingface.co. Verified against a live mlx_lm.server 0.31.3
            # instance.
            "model": "default_model",
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.0,
        },
        timeout=180.0,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--system-prompt-file", default="")
    ap.add_argument(
        "--heldout",
        type=Path,
        default=ROOT / "eval" / "heldout.yaml",
        help="path to the held-out eval set (default: eval/heldout.yaml)",
    )
    args = ap.parse_args()

    system = None
    if args.system_prompt_file:
        system = Path(args.system_prompt_file).read_text(encoding="utf-8")

    data = yaml.safe_load(args.heldout.read_text(encoding="utf-8"))
    client = httpx.Client()
    results = {"label": args.label, "rows": []}
    out = ROOT / "eval" / f"results_{args.label}.json"

    factual_ok = factual_n = 0
    handled = halluc = oos_n = 0
    persona_ok = persona_n = 0
    general_ok = general_n = 0
    errors = 0

    def ask_safe(question: str) -> tuple[str | None, str | None]:
        """(answer, error) -- exactly one is None.

        Broad except is deliberate: this evaluates many independent
        questions against a real HTTP server, where one timeout or
        malformed response must not discard every other completed answer
        in the run.
        """
        try:
            return ask(client, args.base_url, question, system), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    try:
        for row in data["in_scope"]:
            factual_n += 1
            a, err = ask_safe(row["q"])
            if err is not None:
                errors += 1
                results["rows"].append(
                    {"group": "in_scope", "q": row["q"], "error": err, "ok": False}
                )
                continue
            ok = score_factual(a, row["expect"])
            factual_ok += ok
            results["rows"].append({"group": "in_scope", "q": row["q"], "a": a, "ok": ok})

        for row in data["out_of_scope"]:
            q = row["q"] if isinstance(row, dict) else row
            oos_n += 1
            a, err = ask_safe(q)
            if err is not None:
                errors += 1
                results["rows"].append(
                    {
                        "group": "out_of_scope",
                        "q": q,
                        "error": err,
                        "handled": False,
                        "halluc": False,
                    }
                )
                continue
            handled_ok = is_handled(a)
            h = is_hallucination(a)
            handled += handled_ok
            halluc += h
            results["rows"].append(
                {"group": "out_of_scope", "q": q, "a": a, "handled": handled_ok, "halluc": h}
            )

        for row in data["persona"]:
            q = row["q"] if isinstance(row, dict) else row
            persona_n += 1
            a, err = ask_safe(q)
            if err is not None:
                errors += 1
                results["rows"].append(
                    {
                        "group": "persona",
                        "q": q,
                        "error": err,
                        "ok": False,
                        "reason": "request failed",
                    }
                )
                continue
            ok, reason = score_persona(a)
            persona_ok += ok
            results["rows"].append(
                {"group": "persona", "q": q, "a": a, "ok": ok, "reason": reason}
            )

        for row in data["general"]:
            general_n += 1
            a, err = ask_safe(row["q"])
            if err is not None:
                errors += 1
                results["rows"].append(
                    {"group": "general", "q": row["q"], "error": err, "ok": False}
                )
                continue
            ok = score_factual(a, row["expect"])
            general_ok += ok
            results["rows"].append({"group": "general", "q": row["q"], "a": a, "ok": ok})
    finally:
        # Always persist whatever was collected, even if something above
        # this point raised something ask_safe() didn't catch.
        def pct(n: int, d: int) -> float:
            return (100.0 * n / d) if d else 0.0

        results["metrics"] = {
            "factual_accuracy": pct(factual_ok, factual_n),
            "handled_rate": pct(handled, oos_n),
            "hallucination_rate": pct(halluc, oos_n),
            "persona_adherence": pct(persona_ok, persona_n),
            "general_knowledge": pct(general_ok, general_n),
        }
        results["errors"] = errors
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {args.label} ===")
    for k, v in results["metrics"].items():
        print(f"  {k:<22} {v:6.1f}%")
    print(f"  {'errors':<22} {errors:6d}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
