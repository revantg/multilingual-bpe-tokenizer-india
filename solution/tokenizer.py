"""Byte-level BPE tokenizer for the ERA V5 multilingual assignment.

Self-contained. A GPT-2-style byte-level base alphabet (256 byte symbols) makes
the tokenizer LOSSLESS on any input: decode(encode(text)) is byte-exact, so the
faithful-roundtrip gate always passes and UNK is structurally unreachable.

Training is cross-word (line-level) greedy BPE with an incremental max-heap of
pair frequencies. Newlines are preserved as chunk boundaries in encode().

Usage:
    from tokenizer import Tokenizer, count_faithful_units
    tok = Tokenizer()
    tok.train(list_of_lines, vocab_size=10000)
    ids = tok.encode("India's population is 1,428,627,663.")
    assert tok.decode(ids) == "India's population is 1,428,627,663."
"""
from __future__ import annotations

from collections import Counter, defaultdict
from heapq import heapify_max, heappop_max, heappush_max

import regex


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 reversible byte<->printable-unicode map over all 256 bytes."""
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

# faithful_unit = one contiguous Unicode letter/mark/number run
#                 OR one visible non-space punctuation/symbol character
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")


def count_faithful_units(text: str) -> int:
    """Denominator of the fertility ratio, per the assignment's unit policy."""
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
            if merged in self.vocab:            # skip already-learned pair (guard)
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
