from pathlib import Path

from ftkit.config import load_config
from ftkit.factmodel import load_facts

ROOT = Path(__file__).parent.parent
FACTS = ROOT / "facts" / "EXAMPLE.facts.yaml"
CONFIG = load_config(ROOT / "config.example.yaml")


def test_example_facts_load():
    facts = load_facts(FACTS, CONFIG)
    assert len(facts) >= 6, f"only {len(facts)} facts; EXAMPLE.facts.yaml under-populated"


def test_categories_present_are_declared_in_config():
    """Generalized: there's no fixed set of categories any more -- every
    category a fact uses must simply be one config.example.yaml declares."""
    facts = load_facts(FACTS, CONFIG)
    cats = {f.category for f in facts}
    assert cats, "no facts loaded"
    assert cats <= set(CONFIG.categories)


def test_every_fact_has_attrs():
    for f in load_facts(FACTS, CONFIG):
        assert f.attrs, f"fact {f.id} has empty attrs"


def test_role_facts_have_configured_required_attrs():
    """Generalized: required attrs come from config.categories[category]
    .required, not a hardcoded per-category tuple -- this is the same
    contract enforced at load time (see
    tests/test_factmodel.py::test_missing_required_attr_rejected), checked
    here again against the real shipped EXAMPLE corpus."""
    facts = load_facts(FACTS, CONFIG)
    for f in facts:
        for attr in CONFIG.categories[f.category].required:
            assert attr in f.attrs, f"{f.id} ({f.category}) missing required attr '{attr}'"


def test_exactly_one_current_role():
    current = [
        f
        for f in load_facts(FACTS, CONFIG)
        if f.category == "role" and f.attrs.get("end") == "Present"
    ]
    assert len(current) == 1, f"expected 1 current role, got {len(current)}"


# Note: the source project's separate test_category_required_attrs (a
# hardcoded REQUIRED_ATTRS_BY_CATEGORY dict) is subsumed by config-driven
# validation inside load_facts() itself -- see
# tests/test_factmodel.py::test_missing_required_attr_rejected for the
# load-time enforcement, and test_role_facts_have_configured_required_attrs
# above for the same contract checked against real data.
