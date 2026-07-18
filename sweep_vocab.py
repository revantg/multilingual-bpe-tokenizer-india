"""Parallel sweep of per-language vocab-size distributions.

Reuses the notebook's exact tokenizer logic (byte-level alphabet, greedy
optimized BPE, cross-word PhraseBPETokenizer). Each candidate distribution is
trained + scored in its own process, spread across all cores.

A candidate is a cumulative per-language vocab schedule, trained in one order:
    EN -> HI -> SA -> TE   (SA reuses HI's Devanagari merges, like the notebook).
Each language is scored on the FINAL tokenizer (all merges), so the four X's
are directly comparable and the spread/score is the real assignment metric.

Usage:
    .venv/bin/python sweep_vocab.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from heapq import heappop_max, heapify_max, heappush_max

import regex

# ── byte-level alphabet + pretokenizer (verbatim from notebook) ──────────
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


def build_stats(raw_tokens: list[str]):
    token_freq: Counter = Counter(raw_tokens)
    pair_to_token_map = defaultdict(set)
    token_to_symbols_map: dict[str, list[str]] = {}
    pair_freq: Counter = Counter()
    for token, freq in token_freq.items():
        symbols = token_to_byte_symbols(token)
        token_to_symbols_map[token] = symbols
        for pair in zip(symbols, symbols[1:]):
            pair_freq[pair] += freq
            pair_to_token_map[pair].add(token)
    return pair_to_token_map, token_to_symbols_map, token_freq, pair_freq


class Tokenizer:
    """Cross-word (line-level) BPE, matching PhraseBPETokenizer + base train."""

    def __init__(self):
        vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        for b in range(256):
            vocab.setdefault(BYTE_ENCODER[b], len(vocab))
        self.vocab = vocab
        self.merges: list[tuple[str, str]] = []
        self.merge_rank: dict[tuple[str, str], int] = {}

    def train(self, corpus: list[str], vocab_size: int) -> None:
        pair_to_token_map, token_to_symbols_map, token_freq, pair_freq = build_stats(corpus)
        heap = [(c, p) for p, c in pair_freq.items()]
        heapify_max(heap)
        while len(self.vocab) < vocab_size and heap:
            top = None
            while heap:
                count, cand = heappop_max(heap)
                if count > 0 and pair_freq[cand] == count:
                    top = cand
                    break
            if top is None:
                break
            merged = "".join(top)
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
            self.merge_rank[top] = len(self.merges)
            self.merges.append(top)
            tokens = pair_to_token_map.get(top, set()).copy()
            touched = set()
            for token in tokens:
                symbols = token_to_symbols_map[token]
                for pair in zip(symbols, symbols[1:]):
                    pair_freq[pair] -= token_freq[token]
                    pair_to_token_map[pair].discard(token)
                    touched.add(pair)
                updated = merge_optimized(symbols, *top)
                token_to_symbols_map[token] = updated
                for pair in zip(updated, updated[1:]):
                    pair_freq[pair] += token_freq[token]
                    pair_to_token_map.setdefault(pair, set()).add(token)
                    touched.add(pair)
            for pair in touched:
                if pair_freq[pair] > 0:
                    heappush_max(heap, (pair_freq[pair], pair))

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

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in regex.split(r"(\n+)", text):
            if not chunk:
                continue
            for sym in self._apply_merges(token_to_byte_symbols(chunk)):
                ids.append(self.vocab[sym])
        return ids


# ── worker: one candidate distribution ──────────────────────────────────
CORPORA: dict[str, str] = {}


def _load():
    base = Path(__file__).parent / "datasets" / "full_page_md"
    return {
        "EN": (base / "india_en.md").read_text(),
        "HI": (base / "india_hi.md").read_text(),
        "TE": (base / "india_te.md").read_text(),
        "SA": (base / "india_sa.md").read_text(),
    }


def _init_worker():
    global CORPORA
    CORPORA = _load()


def evaluate(schedule: dict[str, int]) -> dict:
    """schedule = cumulative vocab target after each language's phase.

    Trained in order EN, HI, SA, TE (SA after HI to share Devanagari).
    """
    tok = Tokenizer()
    order = ["EN", "HI", "SA", "TE"]
    for lang in order:
        lines = [l for l in CORPORA[lang].split("\n") if l.strip()]
        tok.train(lines, schedule[lang])
    X = {
        lang: len(tok.encode(text)) / count_total_vocab(text)
        for lang, text in CORPORA.items()
    }
    spread = max(X.values()) - min(X.values())
    import math
    penalty = math.exp(max(0.0, X["HI"] / 1.2 - 1.0))
    raw = 1000.0 / spread if spread > 1e-15 else float("inf")
    return {
        "schedule": schedule,
        "X": {k: round(v, 4) for k, v in X.items()},
        "spread": round(spread, 4),
        "raw_score": round(raw, 1),
        "adjusted_score": round(raw / penalty, 1),
        "all_under_1_2": all(v <= 1.2 for v in X.values()),
        "final_vocab": len(tok.vocab),
    }


def main():
    CORPORA.update(_load())
    total = 10_000

    # Build candidate cumulative schedules. Vary how the 10k budget is split.
    # en = merges spent on English; then HI, SA (cumulative), TE fills to 10k.
    candidates: list[dict[str, int]] = []
    for en in (1500, 2500, 3500, 4500):
        for hi_extra in (2000, 3000, 4000):
            for sa_extra in (500, 1000, 1500):
                hi = en + hi_extra
                sa = hi + sa_extra
                if sa >= total:
                    continue
                candidates.append({"EN": en, "HI": hi, "SA": sa, "TE": total})

    # de-dup
    seen = set()
    uniq = []
    for c in candidates:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    workers = 6
    print(f"Evaluating {len(uniq)} candidates on {min(workers, len(uniq))} workers...\n")
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futs = {ex.submit(evaluate, c): c for c in uniq}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            print(
                f"[{i:>2}/{len(uniq)}] EN={r['schedule']['EN']:>4} "
                f"HI={r['schedule']['HI']:>4} SA={r['schedule']['SA']:>4}  "
                f"X={r['X']}  spread={r['spread']:.4f}  "
                f"adj={r['adjusted_score']:>9}  under1.2={r['all_under_1_2']}"
            )

    results.sort(key=lambda r: r["adjusted_score"], reverse=True)
    print("\n" + "=" * 30 + " TOP 10 " + "=" * 30)
    for r in results[:10]:
        print(
            f"adj={r['adjusted_score']:>10}  spread={r['spread']:.4f}  "
            f"under1.2={r['all_under_1_2']}  X={r['X']}  sched={r['schedule']}"
        )

    import json
    out = Path(__file__).parent / "sweep_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nfull results -> {out}")


if __name__ == "__main__":
    main()
