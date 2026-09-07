"""Training example model and mlx-lm chat JSONL serialization."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    question: str
    answer: str
    fact_id: str | None
    slice_name: str

    def to_chat(self) -> dict:
        """Render as an mlx-lm `chat` row.

        Deliberately emits no system message: the whole point of this project
        is that the model works with no system prompt at inference time, so
        training must never show it one.
        """
        return {
            "messages": [
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.answer},
            ]
        }


def write_jsonl(examples: Iterable[Example], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_chat(), ensure_ascii=False) + "\n")
            count += 1
    return count
