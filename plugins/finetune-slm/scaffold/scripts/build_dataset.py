"""CLI: facts.yaml -> data/{train,valid,test}.jsonl

Defaults point at this scaffold's shipped worked example (config.example.yaml
/ templates.example.yaml / facts/EXAMPLE.*.yaml) so `uv run python
scripts/build_dataset.py` works out of the box with no flags. For your own
project, either pass --config/--templates/--facts/--out-of-scope/--persona
explicitly, or copy the *.example.* files to the plain names (config.yaml,
templates.yaml, facts/facts.yaml, facts/out_of_scope.yaml, facts/persona.yaml)
and pass those instead.
"""

import argparse
from collections import Counter
from pathlib import Path

from ftkit.config import load_config
from ftkit.dataset import build_all, split_by_fact
from ftkit.example import write_jsonl
from ftkit.templates import load_templates

ROOT = Path(__file__).parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config.example.yaml")
    ap.add_argument("--templates", type=Path, default=ROOT / "templates.example.yaml")
    ap.add_argument("--facts", type=Path, default=ROOT / "facts" / "EXAMPLE.facts.yaml")
    ap.add_argument(
        "--out-of-scope",
        type=Path,
        default=ROOT / "facts" / "EXAMPLE.out_of_scope.yaml",
    )
    ap.add_argument(
        "--persona", type=Path, default=ROOT / "facts" / "EXAMPLE.persona.yaml"
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    config = load_config(args.config)
    templates = load_templates(args.templates)

    examples = build_all(
        args.facts, args.out_of_scope, args.persona, config, templates
    )

    counts = Counter(e.slice_name for e in examples)
    print("slice composition:")
    for name, n in sorted(counts.items()):
        print(f"  {name:<10} {n:>5}")
    print(f"  {'TOTAL':<10} {len(examples):>5}")

    splits = split_by_fact(examples)
    for name, rows in splits.items():
        out_path = args.out_dir / f"{name}.jsonl"
        n = write_jsonl(rows, out_path)
        print(f"wrote {out_path}  ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
