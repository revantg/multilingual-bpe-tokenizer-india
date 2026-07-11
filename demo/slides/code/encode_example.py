#!/usr/bin/env python3
"""
Minimal example: load tokenizer.json and turn text into pieces.

Usage (from the folder that contains tokenizer.json OR pass a path):

  python3 encode_example.py "India is a country"
  python3 encode_example.py --model ../tokenizer.json "भारत एक देश है"

No extra packages required (Python 3.9+).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Same word regex used when scoring the pages
WORD_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"
    r"[\u0900-\u097F]+|"  # Devanagari (Hindi / Sanskrit)
    r"[\u0C00-\u0C7F]+|"  # Telugu
    r"\d+"
)


def load_model(path: Path) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    vocab = {str(k): int(v) for k, v in data["vocab"].items()}
    merges = [tuple(m) if isinstance(m, (list, tuple)) else tuple(m.split()) for m in data["merges"]]
    return vocab, merges  # type: ignore[return-value]


def encode_word(word: str, vocab: Dict[str, int], merges: List[Tuple[str, str]]) -> List[str]:
    """Whole word in vocab → one piece; else join/break with BPE merges."""
    if word in vocab:
        return [word]
    if not word:
        return []
    # start as single characters
    parts = list(word)
    # apply merges in the order they were learned
    merge_rank = {pair: i for i, pair in enumerate(merges)}
    while len(parts) > 1:
        best_i = None
        best_rank = None
        for i in range(len(parts) - 1):
            pair = (parts[i], parts[i + 1])
            r = merge_rank.get(pair)
            if r is None:
                continue
            if best_rank is None or r < best_rank:
                best_rank = r
                best_i = i
        if best_i is None:
            break
        parts = parts[:best_i] + [parts[best_i] + parts[best_i + 1]] + parts[best_i + 2 :]
    return parts


def encode_text(text: str, vocab: Dict[str, int], merges: List[Tuple[str, str]]) -> List[str]:
    words = WORD_RE.findall(text)
    out: List[str] = []
    for w in words:
        out.extend(encode_word(w, vocab, merges))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Encode text with the submitted tokenizer")
    p.add_argument("text", nargs="?", default="India is a large country in South Asia.")
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to tokenizer.json (default: look next to this script, then parent)",
    )
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    candidates = []
    if args.model:
        candidates.append(args.model)
    candidates.extend(
        [
            here / "tokenizer.json",
            here.parent / "tokenizer.json",
            Path("tokenizer.json"),
        ]
    )
    model_path = next((c for c in candidates if c.is_file()), None)
    if model_path is None:
        raise SystemExit(
            "Could not find tokenizer.json. Pass --model /path/to/tokenizer.json"
        )

    vocab, merges = load_model(model_path)
    pieces = encode_text(args.text, vocab, merges)
    words = WORD_RE.findall(args.text)
    print(f"model:  {model_path}")
    print(f"words:  {len(words)}  →  {words}")
    print(f"pieces: {len(pieces)} →  {pieces}")
    if words:
        print(f"X ≈ pieces/words = {len(pieces) / len(words):.4f}")


if __name__ == "__main__":
    main()
