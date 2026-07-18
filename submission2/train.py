"""Reproduce the champion tokenizer and evaluate it against the assignment gates.

    python train.py            # train champion 2/2/3/2, write tokenizer.json + metrics.json
    python train.py --weights 1 1 1 1   # try other weights

Champion weights: EN=2, HI=2, TE=3, MAI=2 (near-equal, TE lightly boosted).
Corpus: instructor's wiki-faithful Markdown (en/hi/te/mai), clean of URL pollution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tokenizer import Tokenizer, count_faithful_units

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
LANGS = ["EN", "HI", "TE", "MAI"]
FILES = {"EN": "en.txt", "HI": "hi.txt", "TE": "te.txt", "MAI": "mai.txt"}
CHAMPION = {"EN": 2.0, "HI": 2.2, "TE": 3.7, "MAI": 4.05}


def load_corpus() -> dict[str, str]:
    return {L: (CORPUS / FILES[L]).read_text(encoding="utf-8") for L in LANGS}


def repeat_counts(n: int, w: float) -> list[int]:
    """Deterministic (Bresenham) line replication for a fractional weight w:
    each line gets floor(w) copies, plus one extra whenever the running
    fractional remainder crosses 1. Fully reproducible, no randomness."""
    base = int(w + 1e-9)
    frac = w - base
    if frac < 1e-9:
        return [base] * n
    counts, acc = [], 0.0
    for _ in range(n):
        c = base
        acc += frac
        if acc >= 1.0 - 1e-9:
            c += 1
            acc -= 1.0
        counts.append(c)
    return counts


def train_tokenizer(weights: dict[str, float], texts: dict[str, str]) -> Tokenizer:
    # Keep a trailing newline on each training line so the trainer can learn
    # "...\n" merges (encode() chunks the same way), and upsample each language
    # by its fractional weight via deterministic Bresenham replication.
    pooled = []
    for L in LANGS:
        lines = [ln + "\n" for ln in texts[L].split("\n") if ln.strip()]
        for ln, c in zip(lines, repeat_counts(len(lines), weights[L])):
            pooled.extend([ln] * c)
    tok = Tokenizer()
    tok.train(pooled, 10_000)
    return tok


def evaluate(tok: Tokenizer, texts: dict[str, str], weights: dict[str, int]) -> dict:
    X = {L: len(tok.encode(t)) / count_faithful_units(t) for L, t in texts.items()}
    token_counts = {L: len(tok.encode(t)) for L, t in texts.items()}
    units = {L: count_faithful_units(t) for L, t in texts.items()}
    spread = max(X.values()) - min(X.values())
    raw = 1000.0 / spread if spread > 1e-15 else float("inf")
    # English penalty per assignment: score / exp(max(0, X_EN/1.2 - 1))
    penalty = math.exp(max(0.0, X["EN"] / 1.2 - 1.0))
    unk = tok.vocab["<unk>"]
    unk_hits = {L: tok.encode(t).count(unk) for L, t in texts.items()}
    roundtrip = {L: tok.decode(tok.encode(t)) == t for L, t in texts.items()}
    return {
        "weights": weights,
        "faithful_units": units,
        "token_counts": token_counts,
        "ratios": {L: round(v, 6) for L, v in X.items()},
        "spread": round(spread, 6),
        "raw_score": round(raw, 2),
        "english_penalty_factor": round(penalty, 6),
        "adjusted_score": round(raw / penalty, 2),
        "all_under_1_2": all(v <= 1.2 for v in X.values()),
        "vocab_size": len(tok.vocab),
        "num_merges": len(tok.merges),
        "unk_hits": unk_hits,
        "roundtrip_ok": all(roundtrip.values()),
    }


def export_tokenizer(tok: Tokenizer, path: Path) -> None:
    obj = {"vocab": tok.vocab, "merges": [list(m) for m in tok.merges]}
    path.write_text(json.dumps(obj), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs=4, type=float, metavar=("EN", "HI", "TE", "MAI"))
    args = ap.parse_args()
    tok_weights = (
        dict(zip(LANGS, args.weights)) if args.weights else dict(CHAMPION)
    )
    print("weights:", tok_weights)
    texts = load_corpus()
    tok = train_tokenizer(tok_weights, texts)
    metrics = evaluate(tok, texts, tok_weights)
    export_tokenizer(tok, ROOT / "tokenizer.json")
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("\nwrote tokenizer.json + metrics.json")
