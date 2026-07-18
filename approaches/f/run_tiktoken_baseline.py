#!/usr/bin/env python3
"""
Approach F — external baseline: OpenAI tiktoken encodings on full_page_md.

Not a trained submission tokenizer. Measures fertility X = T/D and the
assignment score 1000/(maxX−minX) with the same pretok words D as Approach B.

Usage (from repo root, needs: pip install tiktoken):

  python3 approaches/f/run_tiktoken_baseline.py
  python3 approaches/f/run_tiktoken_baseline.py --encodings cl100k_base o200k_base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]  # tokenizer-assignment /
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "datasets" / "full_page_md"))

from eval_metrics import LANGS, extract_words  # noqa: E402
from load_full_page import load_corpora  # noqa: E402

DEFAULT_ENCODINGS = ("cl100k_base", "o200k_base", "p50k_base", "r50k_base")
OUT_DEFAULT = Path(__file__).resolve().parent / "results.json"


def score_encoding(name: str, corpora: Dict[str, str], D: Dict[str, int]) -> dict:
    import tiktoken

    enc = tiktoken.get_encoding(name)
    T: Dict[str, int] = {}
    X: Dict[str, float] = {}
    for lang in LANGS:
        ids = enc.encode(corpora[lang], disallowed_special=())
        T[lang] = len(ids)
        X[lang] = T[lang] / D[lang] if D[lang] else float("inf")
    xs = list(X.values())
    gap = max(xs) - min(xs)
    score = float("inf") if gap < 1e-15 else 1000.0 / gap
    vocab = int(enc.n_vocab)
    return {
        "name": f"tiktoken/{name}",
        "encoding": name,
        "vocab_size": vocab,
        "T": T,
        "D": dict(D),
        "X": X,
        "gap": gap,
        "score": score,
        "X_en_le_1_2": X["EN"] <= 1.2,
        "vocab_ok": vocab <= 10_000,
        "hard_ok": X["EN"] <= 1.2 and vocab <= 10_000,
        "sorted_X": sorted(((k, X[k]) for k in LANGS), key=lambda kv: -kv[1]),
    }


def per_word_fertility(name: str, corpora: Dict[str, str], D: Dict[str, int]) -> dict:
    """Sum of encode(word) over pretok words (no interstitial text)."""
    import tiktoken

    enc = tiktoken.get_encoding(name)
    out = {}
    for lang in LANGS:
        words = extract_words(corpora[lang])
        Tw = sum(len(enc.encode(w, disallowed_special=())) for w in words)
        out[lang] = {
            "T_words_sum": Tw,
            "D": len(words),
            "X_words": Tw / len(words) if words else float("inf"),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="tiktoken fertility baseline on full_page_md")
    p.add_argument(
        "--encodings",
        nargs="+",
        default=list(DEFAULT_ENCODINGS),
        help="tiktoken encoding names",
    )
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument(
        "--per-word",
        action="store_true",
        help="also report sum-of-per-word fertility for first encoding",
    )
    args = p.parse_args()

    try:
        import tiktoken  # noqa: F401
    except ImportError:
        print("ERROR: tiktoken not installed.  pip install tiktoken", file=sys.stderr)
        sys.exit(1)

    corpora = load_corpora()
    D = {L: len(extract_words(corpora[L])) for L in LANGS}

    rows: List[dict] = []
    for name in args.encodings:
        print(f"--- {name} ---", flush=True)
        row = score_encoding(name, corpora, D)
        rows.append(row)
        print(
            f"  vocab={row['vocab_size']}  "
            + " ".join(f"X_{L}={row['X'][L]:.4f}" for L in LANGS)
            + f"  gap={row['gap']:.6f}  score={row['score']:.2f}  "
            f"hard_ok={row['hard_ok']}",
            flush=True,
        )

    payload = {
        "corpus": "full_page_md",
        "denom": "words (harness extract_words on clean_text pages)",
        "T_definition": "len(tiktoken.encode(full cleaned page))",
        "formula": "score = 1000 / (max X - min X),  X = T/D",
        "D": D,
        "encodings": rows,
        "reference_approach_b": {
            "vocab_size": 10_000,
            "X": {"EN": 1.15785, "HI": 1.08496, "TE": 1.04959, "SA": 1.11047},
            "gap": 0.108256,
            "score": 9237.39,
            "hard_ok": True,
        },
    }
    if args.per_word and args.encodings:
        payload["per_word_fertility"] = {
            args.encodings[0]: per_word_fertility(args.encodings[0], corpora, D)
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}")

    print("\n=== Summary ===")
    print(
        f'{"enc":<14} {"vocab":>7} {"X_EN":>8} {"X_HI":>8} {"X_TE":>8} '
        f'{"X_SA":>8} {"gap":>10} {"score":>12} {"hard_ok":>8}'
    )
    for row in rows:
        print(
            f'{row["encoding"]:<14} {row["vocab_size"]:>7} '
            f'{row["X"]["EN"]:>8.4f} {row["X"]["HI"]:>8.4f} '
            f'{row["X"]["TE"]:>8.4f} {row["X"]["SA"]:>8.4f} '
            f'{row["gap"]:>10.6f} {row["score"]:>12.2f} {str(row["hard_ok"]):>8}'
        )
    b = payload["reference_approach_b"]
    print(
        f'{"Approach B":<14} {b["vocab_size"]:>7} '
        f'{b["X"]["EN"]:>8.4f} {b["X"]["HI"]:>8.4f} '
        f'{b["X"]["TE"]:>8.4f} {b["X"]["SA"]:>8.4f} '
        f'{b["gap"]:>10.6f} {b["score"]:>12.2f} {str(b["hard_ok"]):>8}'
    )


if __name__ == "__main__":
    main()
