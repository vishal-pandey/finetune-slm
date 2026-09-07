import pytest

from ftkit.example import Example
from ftkit.dataset import split_by_fact


def _tiny_examples(n_facts, per_fact=5):
    """A deliberately small fact set: independent max(1, ...) floors on
    n_valid and n_test can, together, consume every fact_id and starve
    train -- a project's own edit-facts.yaml/rebuild workflow can shrink
    the fact set at any time, so this must never happen silently."""
    out = []
    for f in range(n_facts):
        for i in range(per_fact):
            out.append(Example(f"q{f}_{i}", f"a{f}_{i}", f"fact.{f}", "qa"))
    return out


@pytest.mark.parametrize("n_facts", [1, 2, 3])
def test_train_never_starved_with_tiny_fact_set(n_facts):
    splits = split_by_fact(_tiny_examples(n_facts))
    assert len(splits["train"]) > 0, (
        f"train starved with {n_facts} fact(s): "
        f"{ {k: len(v) for k, v in splits.items()} }"
    )


def _examples():
    out = []
    for f in range(20):
        for i in range(10):
            out.append(Example(f"q{f}_{i}", f"a{f}_{i}", f"fact.{f}", "qa"))
    for i in range(50):
        out.append(Example(f"r{i}", "refused", None, "refusal"))
    return out


def test_all_examples_land_in_exactly_one_split():
    splits = split_by_fact(_examples())
    total = sum(len(v) for v in splits.values())
    assert total == len(_examples())


def test_no_fact_appears_in_two_splits():
    """The whole point: paraphrases of one fact must not straddle splits."""
    splits = split_by_fact(_examples())
    seen: dict[str, str] = {}
    for name, examples in splits.items():
        for e in examples:
            if e.fact_id is None:
                continue
            assert seen.setdefault(e.fact_id, name) == name, (
                f"{e.fact_id} in both {seen[e.fact_id]} and {name}"
            )


def test_split_is_deterministic():
    a = split_by_fact(_examples(), seed=7)
    b = split_by_fact(_examples(), seed=7)
    assert [e.question for e in a["train"]] == [e.question for e in b["train"]]


def test_train_is_the_largest_split():
    splits = split_by_fact(_examples())
    assert len(splits["train"]) > len(splits["valid"])
    assert len(splits["train"]) > len(splits["test"])


def test_factless_examples_are_distributed_not_dropped():
    splits = split_by_fact(_examples())
    refusals = sum(
        1 for v in splits.values() for e in v if e.slice_name == "refusal"
    )
    assert refusals == 50
