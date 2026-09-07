from pathlib import Path

import pytest

from ftkit.config import load_config
from ftkit.factmodel import Fact, FactError, load_facts

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = load_config(FIXTURES / "config_min.yaml")


def test_load_facts_returns_all_rows():
    facts = load_facts(FIXTURES / "facts_min.yaml", CONFIG)
    assert len(facts) == 2
    assert {f.id for f in facts} == {"role.lumiq", "edu.gbu"}


def test_fact_fields_are_parsed():
    facts = load_facts(FIXTURES / "facts_min.yaml", CONFIG)
    lumiq = next(f for f in facts if f.id == "role.lumiq")
    assert lumiq.category == "role"
    assert lumiq.subject == "Lumiq"
    assert lumiq.attrs["title"] == "Technical Lead"
    assert lumiq.aliases == ("Lumiq", "current job")
    assert lumiq.keywords == ("Lumiq", "Technical Lead")


def test_fact_is_hashable_and_frozen():
    facts = load_facts(FIXTURES / "facts_min.yaml", CONFIG)
    assert len({f for f in facts}) == 2
    with pytest.raises(Exception):
        facts[0].id = "mutated"


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "dupes.yaml"
    p.write_text(
        "facts:\n"
        "  - {id: a.b, category: role, subject: X, attrs: {title: T}, keywords: [X]}\n"
        "  - {id: a.b, category: role, subject: Y, attrs: {title: T}, keywords: [Y]}\n"
    )
    with pytest.raises(FactError, match="duplicate fact id"):
        load_facts(p, CONFIG)


def test_missing_required_field_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("facts:\n  - {id: a.b, subject: X, attrs: {title: T}, keywords: [X]}\n")
    with pytest.raises(FactError, match="missing required field 'category'"):
        load_facts(p, CONFIG)


def test_empty_keywords_rejected(tmp_path):
    p = tmp_path / "nokw.yaml"
    p.write_text(
        "facts:\n  - {id: a.b, category: role, subject: X, attrs: {title: T}, keywords: []}\n"
    )
    with pytest.raises(FactError, match="at least one keyword"):
        load_facts(p, CONFIG)


def test_unknown_category_rejected(tmp_path):
    """New: category membership is validated against the project's config,
    not a hardcoded module-level set."""
    p = tmp_path / "badcat.yaml"
    p.write_text(
        "facts:\n  - {id: a.b, category: nonsense, subject: X, attrs: {title: T}, keywords: [X]}\n"
    )
    with pytest.raises(FactError, match="unknown category"):
        load_facts(p, CONFIG)


def test_missing_required_attr_rejected(tmp_path):
    """Subsumes the source project's test_category_required_attrs: a fact
    missing one of its category's config-declared required attrs is
    rejected at load time -- not silently invisible to templating later
    because it renders zero applicable templates."""
    p = tmp_path / "missingattr.yaml"
    p.write_text(
        "facts:\n  - {id: role.x, category: role, subject: X, attrs: {}, keywords: [X]}\n"
    )
    with pytest.raises(FactError, match="missing required attr"):
        load_facts(p, CONFIG)


def test_optional_attrs_are_not_enforced(tmp_path):
    """A category's `optional` attrs may be absent without error."""
    p = tmp_path / "ok.yaml"
    p.write_text(
        "facts:\n  - {id: role.x, category: role, subject: X, attrs: {title: Dev}, keywords: [X]}\n"
    )
    facts = load_facts(p, CONFIG)
    assert facts[0].attrs == {"title": "Dev"}
