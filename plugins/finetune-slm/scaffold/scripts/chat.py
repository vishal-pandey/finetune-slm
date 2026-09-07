#!/usr/bin/env python3
"""Interactive chat with a locally-served fine-tuned model.

Usage:  uv run python scripts/chat.py
        uv run python scripts/chat.py "one shot question"
        uv run python scripts/chat.py --base-url http://192.168.1.50:8080/v1
        uv run python scripts/chat.py --facts path/to/system_prompt.txt "one shot question"

No system prompt is sent by default — everything the model should know is
in its weights. Pass --facts <path> to try a hybrid mode instead, where the
named text file is sent as a system prompt (facts-in-context).
"""
import argparse
import sys
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "http://localhost:8080/v1"


def ask(client: httpx.Client, base_url: str, history: list[dict]) -> str:
    r = client.post(
        f"{base_url}/chat/completions",
        timeout=180.0,
        json={
            # mlx_lm.server's ModelProvider only maps the literal sentinel
            # "default_model" to whatever was loaded via --model /
            # --adapter-path at server startup; any other value 404s.
            "model": "default_model",
            "messages": history,
            "max_tokens": 300,
            "temperature": 0.0,
        },
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument(
        "--facts",
        type=Path,
        default=None,
        help="path to a system-prompt text file for hybrid (facts-in-context) mode",
    )
    ap.add_argument("question", nargs="*", help="one-shot question; omit for interactive mode")
    args = ap.parse_args()

    system = args.facts.read_text(encoding="utf-8") if args.facts else None

    with httpx.Client() as c:
        try:
            c.get(f"{args.base_url}/models", timeout=10.0)
        except Exception as e:
            print(f"Can't reach {args.base_url} — is the fused model being served?\n  {e}")
            return 1
        base_hist = [{"role": "system", "content": system}] if system else []
        if args.question:
            print(ask(c, args.base_url, base_hist + [{"role": "user", "content": " ".join(args.question)}]))
            return 0
        mode = "facts in context (hybrid)" if system else "no system prompt (weights only)"
        print(f"Chatting with the fine-tuned model — {mode}. Ctrl-C or 'exit' to quit.\n")
        history = list(base_hist)
        while True:
            try:
                q = input("you  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); return 0
            if q.lower() in {"exit", "quit"}: return 0
            if not q: continue
            history.append({"role": "user", "content": q})
            try:
                a = ask(c, args.base_url, history)
            except Exception as e:
                print(f"model> [error: {e}]\n"); history.pop(); continue
            history.append({"role": "assistant", "content": a})
            print(f"model> {a}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
