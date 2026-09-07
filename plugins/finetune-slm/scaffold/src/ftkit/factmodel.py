"""Fact model — the single source of truth for everything the model learns."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from ftkit.config import ProjectConfig

REQUIRED_FIELDS = ("id", "category", "subject", "attrs", "keywords")


class FactError(Exception):
    """Raised when facts.yaml violates the schema."""


@dataclass(frozen=True)
class Fact:
    id: str
    category: str
    subject: str
    attrs: dict[str, str]
    aliases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()

    def __hash__(self) -> int:
        return hash(self.id)


def load_facts(path: Path, config: ProjectConfig) -> tuple[Fact, ...]:
    """Load and validate facts.yaml against the project's `config`.

    Category membership and required-attr presence are both enforced here,
    against `config.categories` -- there is no hardcoded category list or
    per-category attr contract. A fact in an unknown category, or missing
    one of its category's required attrs, is rejected at load time rather
    than silently generating zero training questions later.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "facts" not in raw:
        raise FactError("yaml must have a top-level 'facts' list")

    facts: list[Fact] = []
    seen: set[str] = set()

    for row in raw["facts"]:
        for field in REQUIRED_FIELDS:
            if field not in row:
                raise FactError(f"missing required field '{field}' in row: {row}")

        fid = row["id"]
        if fid in seen:
            raise FactError(f"duplicate fact id: {fid}")
        seen.add(fid)

        category = row["category"]
        if category not in config.categories:
            raise FactError(
                f"unknown category '{category}' in {fid}; "
                f"valid: {sorted(config.categories)}"
            )

        attrs = {k: str(v) for k, v in row["attrs"].items()}
        required = config.categories[category].required
        missing = [a for a in required if a not in attrs]
        if missing:
            raise FactError(
                f"fact {fid} ({category}) missing required attr(s) {missing}; "
                f"required: {list(required)}"
            )

        keywords = tuple(row["keywords"])
        if not keywords:
            raise FactError(f"fact {fid} needs at least one keyword for eval scoring")

        facts.append(
            Fact(
                id=fid,
                category=category,
                subject=row["subject"],
                attrs=attrs,
                aliases=tuple(row.get("aliases", ())),
                keywords=keywords,
            )
        )

    return tuple(facts)
