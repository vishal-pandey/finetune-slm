"""Recompute eval metrics from a results_<label>.json's stored answers.

run_eval.py stores the raw model answer (`a`) for every non-errored row, so
once a scoring-function fix lands, there is no need to re-query the model
server to get up-to-date metrics -- just recompute from what's already
saved. This also guarantees every results file in a final report was
scored by the exact same version of ftkit.scoring, rather than whatever
version happened to be checked out when each `run_eval.py` invocation ran.

Does NOT touch `a`/`q`/`error` fields or the request/response data --
only the derived scoring fields (`ok`, `handled`, `halluc`, `reason`) and
the top-level `metrics`/`errors` summary are recomputed.

Usage:
  uv run python scripts/rescore.py eval/results_baseline.json
  uv run python scripts/rescore.py eval/results_baseline.json --out eval/results_rescored.json
"""

import argparse
import json
from pathlib import Path

import yaml

from ftkit.scoring import is_handled, is_hallucination, score_factual, score_persona

ROOT = Path(__file__).parent.parent


def _expect_lookup(heldout: Path) -> dict[tuple[str, str], list[str]]:
    """(group, question text) -> expect keywords, for in_scope/general.

    Results rows don't store `expect` (only `q`/`a`/`ok`), so factual
    rescoring needs it looked back up from the heldout file by question
    text -- the same source run_eval.py read it from originally.
    """
    data = yaml.safe_load(heldout.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], list[str]] = {}
    for group in ("in_scope", "general"):
        for row in data[group]:
            lookup[(group, row["q"])] = row["expect"]
    return lookup


def rescore(results: dict, heldout: Path) -> dict:
    """Recompute every row's scoring fields and the top-level metrics in place."""
    expect_by_q = _expect_lookup(heldout)

    factual_ok = factual_n = 0
    handled = halluc = oos_n = 0
    persona_ok = persona_n = 0
    general_ok = general_n = 0
    errors = 0

    for row in results["rows"]:
        group = row["group"]

        if "error" in row:
            # No `a` text to rescore from -- an errored request stays a
            # scored failure, never a silent pass (same rule run_eval.py
            # applies live). Just normalize legacy field names.
            errors += 1
            if group == "out_of_scope":
                row.pop("refused", None)
                row["handled"] = False
                row["halluc"] = False
            else:
                row["ok"] = False
            if group == "in_scope":
                factual_n += 1
            elif group == "out_of_scope":
                oos_n += 1
            elif group == "persona":
                persona_n += 1
            elif group == "general":
                general_n += 1
            else:
                raise ValueError(f"unknown group: {group!r}")
            continue

        a = row["a"]
        if group == "in_scope":
            factual_n += 1
            expect = expect_by_q[(group, row["q"])]
            ok = score_factual(a, expect)
            row["ok"] = ok
            factual_ok += ok
        elif group == "out_of_scope":
            oos_n += 1
            row.pop("refused", None)
            handled_ok = is_handled(a)
            h = is_hallucination(a)
            row["handled"] = handled_ok
            row["halluc"] = h
            handled += handled_ok
            halluc += h
        elif group == "persona":
            persona_n += 1
            ok, reason = score_persona(a)
            row["ok"] = ok
            row["reason"] = reason
            persona_ok += ok
        elif group == "general":
            general_n += 1
            expect = expect_by_q[(group, row["q"])]
            ok = score_factual(a, expect)
            row["ok"] = ok
            general_ok += ok
        else:
            raise ValueError(f"unknown group: {group!r}")

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
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="results_<label>.json to rescore")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (defaults to overwriting `path` in place)",
    )
    ap.add_argument(
        "--heldout",
        type=Path,
        default=ROOT / "eval" / "heldout.yaml",
        help="path to the held-out eval set (default: eval/heldout.yaml)",
    )
    args = ap.parse_args()

    results = json.loads(args.path.read_text(encoding="utf-8"))
    results = rescore(results, args.heldout)
    out = args.out or args.path
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== {results['label']} (rescored from {args.path.name}) ===")
    for k, v in results["metrics"].items():
        print(f"  {k:<22} {v:6.1f}%")
    print(f"  {'errors':<22} {results['errors']:6d}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
