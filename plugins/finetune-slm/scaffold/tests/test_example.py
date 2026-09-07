import json
from pathlib import Path

from ftkit.example import Example, write_jsonl


def test_to_chat_produces_mlx_chat_format():
    ex = Example(
        question="Where does she work?",
        answer="Technical Lead at Lumiq.",
        fact_id="role.lumiq",
        slice_name="qa",
    )
    assert ex.to_chat() == {
        "messages": [
            {"role": "user", "content": "Where does she work?"},
            {"role": "assistant", "content": "Technical Lead at Lumiq."},
        ]
    }


def test_to_chat_omits_system_message():
    """Weights-only: no system prompt may leak into training data."""
    ex = Example(question="q", answer="a", fact_id=None, slice_name="persona")
    roles = [m["role"] for m in ex.to_chat()["messages"]]
    assert "system" not in roles


def test_write_jsonl_round_trips(tmp_path: Path):
    examples = [
        Example(question=f"q{i}", answer=f"a{i}", fact_id=None, slice_name="qa")
        for i in range(3)
    ]
    out = tmp_path / "train.jsonl"
    n = write_jsonl(examples, out)

    assert n == 3
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["messages"][0]["content"] == "q0"


def test_write_jsonl_creates_parent_dirs(tmp_path: Path):
    out = tmp_path / "nested" / "deep" / "train.jsonl"
    write_jsonl([Example("q", "a", None, "qa")], out)
    assert out.exists()
