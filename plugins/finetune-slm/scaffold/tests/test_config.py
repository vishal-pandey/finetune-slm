from pathlib import Path

import pytest

from ftkit.config import ConfigError, load_config

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_config_parses_subject_and_categories():
    config = load_config(FIXTURES / "config_min.yaml")
    assert config.subject_name == "Test Subject"
    assert config.subject_kind == "person"
    assert "role" in config.categories
    assert config.categories["role"].required == ("title",)


def test_category_known_attrs_is_union_of_required_and_optional():
    config = load_config(FIXTURES / "config_min.yaml")
    assert config.categories["education"].known_attrs == {
        "degree", "field", "cgpa", "start", "end", "location",
    }


def test_missing_subject_name_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("categories:\n  role:\n    required: [title]\n")
    with pytest.raises(ConfigError, match="subject_name"):
        load_config(p)


def test_missing_categories_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("subject_name: X\n")
    with pytest.raises(ConfigError, match="categories"):
        load_config(p)


def test_empty_categories_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("subject_name: X\ncategories: {}\n")
    with pytest.raises(ConfigError, match="categories"):
        load_config(p)


def test_subject_kind_defaults_to_empty_string(tmp_path):
    p = tmp_path / "min.yaml"
    p.write_text("subject_name: X\ncategories:\n  role:\n    required: [title]\n")
    config = load_config(p)
    assert config.subject_kind == ""


def test_category_with_no_required_or_optional_defaults_to_empty(tmp_path):
    p = tmp_path / "min.yaml"
    p.write_text("subject_name: X\ncategories:\n  role: {}\n")
    config = load_config(p)
    assert config.categories["role"].required == ()
    assert config.categories["role"].optional == ()
