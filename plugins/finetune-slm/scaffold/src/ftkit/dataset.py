"""Dataset assembly: slice composition and fact-grouped splitting."""

import random
from collections.abc import Sequence
from pathlib import Path

from ftkit.config import ProjectConfig
from ftkit.example import Example
from ftkit.factmodel import load_facts
from ftkit.persona import build_persona
from ftkit.qa import build_qa
from ftkit.refusals import build_refusals
from ftkit.statements import build_statements
from ftkit.templates import Template


def build_all(
    facts_path: Path,
    oos_path: Path,
    persona_path: Path,
    config: ProjectConfig,
    templates: dict[str, tuple[Template, ...]],
) -> tuple[Example, ...]:
    facts = load_facts(facts_path, config)
    return (
        build_qa(facts, templates, config.subject_name)
        + build_statements(facts, config.subject_name)
        + build_refusals(oos_path, facts)
        + build_persona(persona_path)
    )


def split_by_fact(
    examples: Sequence[Example],
    *,
    seed: int = 0,
    valid_frac: float = 0.05,
    test_frac: float = 0.05,
) -> dict[str, tuple[Example, ...]]:
    """Split by fact_id so paraphrases of one fact never straddle splits.

    Splitting by example would place near-duplicate questions on both sides
    and make validation loss meaningless.
    """
    rng = random.Random(seed)

    fact_ids = sorted({e.fact_id for e in examples if e.fact_id is not None})
    rng.shuffle(fact_ids)
    n_ids = len(fact_ids)
    n_valid = max(1, int(n_ids * valid_frac)) if n_ids else 0
    # Cap the floors so they can never together consume every fact_id: train
    # must keep at least one fact_id whenever any exist, or a small enough
    # facts.yaml silently produces a training file with no facts in it.
    n_valid = min(n_valid, max(0, n_ids - 1))
    n_test = max(1, int(n_ids * test_frac)) if n_ids else 0
    n_test = min(n_test, max(0, n_ids - n_valid - 1))

    valid_ids = set(fact_ids[:n_valid])
    test_ids = set(fact_ids[n_valid : n_valid + n_test])

    # Factless examples (refusals, persona) are split by index so every slice
    # is represented in every split. Same starvation risk as above, same fix.
    factless = [e for e in examples if e.fact_id is None]
    rng.shuffle(factless)
    n_factless = len(factless)
    fv = max(1, int(n_factless * valid_frac)) if n_factless else 0
    fv = min(fv, max(0, n_factless - 1))
    ft = max(1, int(n_factless * test_frac)) if n_factless else 0
    ft = min(ft, max(0, n_factless - fv - 1))

    buckets: dict[str, list[Example]] = {"train": [], "valid": [], "test": []}
    buckets["valid"].extend(factless[:fv])
    buckets["test"].extend(factless[fv : fv + ft])
    buckets["train"].extend(factless[fv + ft :])

    for e in examples:
        if e.fact_id is None:
            continue
        if e.fact_id in valid_ids:
            buckets["valid"].append(e)
        elif e.fact_id in test_ids:
            buckets["test"].append(e)
        else:
            buckets["train"].append(e)

    for name in buckets:
        rng.shuffle(buckets[name])
    return {k: tuple(v) for k, v in buckets.items()}
