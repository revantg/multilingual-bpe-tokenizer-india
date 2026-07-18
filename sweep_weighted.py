"""Weighted-pool sweep (Fix B).

Instead of sequential per-language training (which corrupted merge_rank), train
ONE tokenizer on a single pooled line corpus where each language's lines are
repeated `weight[lang]` times. Higher weight -> that language's pairs win more
of the greedy heap -> more merges spent on its script. This is the tunable,
convergence-oriented approach and avoids the repeated-training bug entirely.

Trains to a single 10_000 vocab in one call, then scores all four languages.
Runs candidates in parallel (max 4 workers, per user's offline budget).

Usage: .venv/bin/python sweep_weighted.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from heapq import heappop_max, heapify_max, heappush_max

import regex


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {symbol: b for b, symbol in BYTE_ENCODER.items()}
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")


def count_total_vocab(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def token_to_byte_symbols(token: str) -> list[str]:
    return [BYTE_ENCODER[b] for b in token.encode("utf-8")]


def merge_optimized(word_seq, sym1, sym2):
    new_symbol = sym1 + sym2
    out = []
    i = 0
    while i < len(word_seq):
        if i + 1 < len(word_seq) and word_seq[i] == sym1 and word_seq[i + 1] == sym2:
            out.append(new_symbol)
            i += 2
        else:
            out.append(word_seq[i])
            i += 1
    return out


def build_stats(raw_tokens):
    token_freq = Counter(raw_tokens)
    pair_to_token_map = defaultdict(set)
    token_to_symbols_map = {}
    pair_freq = Counter()
    for token, freq in token_freq.items():
        symbols = token_to_byte_symbols(token)
        token_to_symbols_map[token] = symbols
        for pair in zip(symbols, symbols[1:]):
            pair_freq[pair] += freq
            pair_to_token_map[pair].add(token)
    return pair_to_token_map, token_to_symbols_map, token_freq, pair_freq


class Tokenizer:
    def __init__(self):
        vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        for b in range(256):
            vocab.setdefault(BYTE_ENCODER[b], len(vocab))
        self.vocab = vocab
        self.merges = []
        self.merge_rank = {}

    def train(self, corpus, vocab_size):
        p2t, t2s, tf, pf = build_stats(corpus)
        heap = [(c, p) for p, c in pf.items()]
        heapify_max(heap)
        while len(self.vocab) < vocab_size and heap:
            top = None
            while heap:
                count, cand = heappop_max(heap)
                if count > 0 and pf[cand] == count:
                    top = cand
                    break
            if top is None:
                break
            merged = "".join(top)
            if merged in self.vocab:            # Fix A guard (safety)
                continue
            self.vocab[merged] = len(self.vocab)
            self.merge_rank[top] = len(self.merges)
            self.merges.append(top)
            tokens = p2t.get(top, set()).copy()
            touched = set()
            for token in tokens:
                symbols = t2s[token]
                for pair in zip(symbols, symbols[1:]):
                    pf[pair] -= tf[token]
                    p2t[pair].discard(token)
                    touched.add(pair)
                updated = merge_optimized(symbols, *top)
                t2s[token] = updated
                for pair in zip(updated, updated[1:]):
                    pf[pair] += tf[token]
                    p2t.setdefault(pair, set()).add(token)
                    touched.add(pair)
            for pair in touched:
                if pf[pair] > 0:
                    heappush_max(heap, (pf[pair], pair))

    def _apply_merges(self, symbols):
        symbols = list(symbols)
        while len(symbols) > 1:
            ranked = [
                (self.merge_rank[p], p)
                for p in zip(symbols, symbols[1:])
                if p in self.merge_rank
            ]
            if not ranked:
                break
            _, best = min(ranked, key=lambda it: it[0])
            symbols = merge_optimized(symbols, *best)
        return symbols

    def encode(self, text):
        ids = []
        for chunk in regex.split(r"(\n+)", text):
            if not chunk:
                continue
            for sym in self._apply_merges(token_to_byte_symbols(chunk)):
                ids.append(self.vocab[sym])
        return ids

    def decode(self, ids):
        inv = {i: t for t, i in self.vocab.items()}
        special = {self.vocab[t] for t in SPECIAL_TOKENS if t in self.vocab}
        pieces = [inv[i] for i in ids if i in inv and i not in special]
        raw = bytes(BYTE_DECODER[s] for pc in pieces for s in pc)
        return raw.decode("utf-8", errors="replace")


CORPORA = {}


def _load():
    import os
    mode = os.environ.get("CORPUS")
    if mode == "instructor_mai":
        base = Path(__file__).parent / "datasets" / "instructor"
        return {
            "EN": (base / "en.txt").read_text(),
            "HI": (base / "hi.txt").read_text(),
            "TE": (base / "te.txt").read_text(),
            "MAI": (base / "mai.txt").read_text(),  # instructor Maithili (clean)
        }
    if mode == "instructor":
        base = Path(__file__).parent / "datasets" / "instructor"
        return {
            "EN": (base / "en.txt").read_text(),
            "HI": (base / "hi.txt").read_text(),
            "TE": (base / "te.txt").read_text(),
            "SA": (base / "sa.txt").read_text(),  # user's Sanskrit dataset
        }
    base = Path(__file__).parent / "datasets" / "full_page_md"
    return {
        "EN": (base / "india_en.md").read_text(),
        "HI": (base / "india_hi.md").read_text(),
        "TE": (base / "india_te.md").read_text(),
        "SA": (base / "india_sa.md").read_text(),
    }


def _init():
    global CORPORA
    CORPORA = _load()


def evaluate(weights: dict[str, int]) -> dict:
    lines_by_lang = {
        lang: [l for l in CORPORA[lang].split("\n") if l.strip()]
        for lang in CORPORA
    }
    pooled = []
    for lang, w in weights.items():
        pooled.extend(lines_by_lang[lang] * w)
    tok = Tokenizer()
    tok.train(pooled, 10_000)
    X = {
        lang: len(tok.encode(text)) / count_total_vocab(text)
        for lang, text in CORPORA.items()
    }
    spread = max(X.values()) - min(X.values())
    penalty = math.exp(max(0.0, X["HI"] / 1.2 - 1.0))
    raw = 1000.0 / spread if spread > 1e-15 else float("inf")
    # invariants
    unk = tok.vocab["<unk>"]
    unk_hits = {lang: tok.encode(t).count(unk) for lang, t in CORPORA.items()}
    roundtrip = {
        lang: tok.decode(tok.encode(t)) == t for lang, t in CORPORA.items()
    }
    return {
        "weights": weights,
        "X": {k: round(v, 4) for k, v in X.items()},
        "spread": round(spread, 4),
        "raw_score": round(raw, 1),
        "adjusted_score": round(raw / penalty, 1),
        "all_under_1_2": all(v <= 1.2 for v in X.values()),
        "final_vocab": len(tok.vocab),
        "unk_hits": unk_hits,
        "roundtrip_ok": all(roundtrip.values()),
    }


def main():
    global CORPORA
    CORPORA = _load()
    # Agent finding: near-equal weights beat the instructor's {3,4,4,2}. On a
    # CLEAN corpus the four X's cluster, so search near-equal to minimize spread.
    fourth = next(k for k in CORPORA if k not in ("EN", "HI", "TE"))
    def mk(en, hi, te, f):
        return {"EN": en, "HI": hi, "TE": te, fourth: f}
    candidates = [
        mk(1, 1, 1, 1),
        mk(2, 2, 3, 2),
        mk(2, 2, 2, 1),
        mk(3, 3, 3, 2),
        mk(2, 3, 3, 2),
        mk(1, 2, 2, 1),
        mk(3, 2, 2, 1),
        mk(2, 2, 3, 1),
        mk(3, 3, 4, 2),
        mk(1, 1, 2, 1),
    ]
    workers = 4
    print(f"Evaluating {len(candidates)} weighted-pool candidates on {workers} workers...\n", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        futs = {ex.submit(evaluate, c): c for c in candidates}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            print(
                f"[{i}/{len(candidates)}] w={r['weights']}  X={r['X']}  "
                f"spread={r['spread']:.4f}  adj={r['adjusted_score']:>8}  "
                f"<1.2={r['all_under_1_2']}  vocab={r['final_vocab']}  "
                f"rt={r['roundtrip_ok']}  unk={sum(r['unk_hits'].values())}",
                flush=True,
            )
    results.sort(key=lambda r: r["adjusted_score"], reverse=True)
    print("\n===== RANKED =====")
    for r in results:
        print(f"adj={r['adjusted_score']:>9}  spread={r['spread']:.4f}  X={r['X']}  w={r['weights']}")
    Path(__file__).parent.joinpath("sweep_weighted_results.json").write_text(
        json.dumps(results, indent=2)
    )
    print("\nsaved -> sweep_weighted_results.json")


if __name__ == "__main__":
    main()
