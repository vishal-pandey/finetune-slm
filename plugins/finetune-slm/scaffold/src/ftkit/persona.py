"""Persona slice: voice, greetings, roasts, register."""

from pathlib import Path

import yaml

from ftkit.example import Example


def build_persona(path: Path) -> tuple[Example, ...]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return tuple(
        Example(
            question=pair["q"],
            answer=pair["a"],
            fact_id=None,
            slice_name="persona",
        )
        for pair in spec["pairs"]
    )
