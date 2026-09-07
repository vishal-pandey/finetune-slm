"""Project configuration: subject identity and fact categories.

Categories are project-defined rather than hardcoded so this pipeline works
for any bounded subject (a person, a product, a codebase, ...), not just
one. `load_facts()` validates every fact's category and required attrs
against this config.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when config.yaml violates the schema."""


@dataclass(frozen=True)
class CategorySpec:
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()

    @property
    def known_attrs(self) -> frozenset[str]:
        """Every attr name a template for this category may reference."""
        return frozenset(self.required) | frozenset(self.optional)


@dataclass(frozen=True)
class ProjectConfig:
    subject_name: str
    subject_kind: str
    categories: dict[str, CategorySpec]


def load_config(path: Path) -> ProjectConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml must be a mapping")

    if "subject_name" not in raw or not raw["subject_name"]:
        raise ConfigError("config.yaml missing required field 'subject_name'")

    raw_categories = raw.get("categories")
    if not isinstance(raw_categories, dict) or not raw_categories:
        raise ConfigError("config.yaml must define a non-empty 'categories' mapping")

    categories: dict[str, CategorySpec] = {}
    for name, spec in raw_categories.items():
        spec = spec or {}
        if not isinstance(spec, dict):
            raise ConfigError(f"config.yaml: category '{name}' must be a mapping")
        categories[name] = CategorySpec(
            required=tuple(spec.get("required", ()) or ()),
            optional=tuple(spec.get("optional", ()) or ()),
        )

    return ProjectConfig(
        subject_name=str(raw["subject_name"]),
        subject_kind=str(raw.get("subject_kind", "")),
        categories=categories,
    )
