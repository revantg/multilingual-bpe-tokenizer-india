#!/usr/bin/env python3
"""
Approach C — Whole-word inventory + optional fair bigrams.

Base idea:
  Put ALL unique words across EN/HI/TE/SA into the vocab (plus chars for OOV).
  Encode is exact word match → every in-vocab word is 1 token, so X≈1 for all langs.

Remaining budget under 10k may go to multiword (bigram) phrase tokens.
Those must be allocated with PER-LANGUAGE budgeting: global-freq bigrams mostly
help EN and widen the fairness gap (hurting the score).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Pair = Tuple[str, str]
NGram = Tuple[str, ...]


def build_base_vocab(
    words_by_lang: Dict[str, Sequence[str]],
    corpora: Dict[str, str],
) -> Set[str]:
    """All unique words + every character seen in cleaned corpora (OOV fallback)."""
    vocab: Set[str] = set()
    for ws in words_by_lang.values():
        vocab.update(ws)
    for text in corpora.values():
        vocab.update(text)
    return vocab


def collect_bigrams(
    words_by_lang: Dict[str, Sequence[str]],
) -> Dict[str, Counter]:
    out: Dict[str, Counter] = {}
    for lang, ws in words_by_lang.items():
        out[lang] = Counter(zip(ws, ws[1:]))
    return out


def select_bigrams_quota(
    bigrams_by_lang: Dict[str, Counter],
    budget: int,
    langs: Sequence[str],
) -> Set[Pair]:
    """Equal per-language quota of highest-frequency bigrams within each lang."""
    if budget <= 0:
        return set()
    base = budget // len(langs)
    leftover = budget - base * len(langs)
    quotas = {lang: base for lang in langs}
    for lang in langs:
        if leftover <= 0:
            break
        quotas[lang] += 1
        leftover -= 1
    selected: Set[Pair] = set()
    for lang in langs:
        taken = 0
        for bg, _ in sorted(
            bigrams_by_lang[lang].items(), key=lambda x: (-x[1], x[0])
        ):
            if bg in selected:
                continue
            selected.add(bg)
            taken += 1
            if taken >= quotas[lang]:
                break
    return selected


def select_bigrams_round_robin(
    bigrams_by_lang: Dict[str, Counter],
    budget: int,
    langs: Sequence[str],
) -> Set[Pair]:
    """Round-robin pull of next most-frequent bigram per language until budget."""
    if budget <= 0:
        return set()
    pools = {
        lang: [bg for bg, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))]
        for lang, c in bigrams_by_lang.items()
    }
    idx = {lang: 0 for lang in langs}
    selected: Set[Pair] = set()
    while len(selected) < budget:
        progressed = False
        for lang in langs:
            while idx[lang] < len(pools[lang]):
                bg = pools[lang][idx[lang]]
                idx[lang] += 1
                if bg not in selected:
                    selected.add(bg)
                    progressed = True
                    break
            if len(selected) >= budget:
                break
        if not progressed:
            break
    return selected


def select_bigrams_global_freq(
    bigrams_by_lang: Dict[str, Counter],
    budget: int,
) -> Set[Pair]:
    """UNFAIR baseline: global frequency ranking (tends to over-help EN)."""
    if budget <= 0:
        return set()
    combined: Counter = Counter()
    for c in bigrams_by_lang.values():
        combined.update(c)
    return {bg for bg, _ in combined.most_common(budget)}


def select_all_singleton_phrases(
    bigrams_by_lang: Dict[str, Counter],
    budget: int,
    langs: Sequence[str],
) -> Set[Pair]:
    """
    Prefer rare (freq=1) bigrams in round-robin per language, then higher freq.
    Uses remaining slots for phrase inventory of single-occurrence multiwords.
    """
    if budget <= 0:
        return set()
    selected: Set[Pair] = set()
    # phase 1: freq==1, round-robin
    pools1 = {
        lang: [bg for bg, cnt in c.items() if cnt == 1]
        for lang, c in bigrams_by_lang.items()
    }
    # deterministic order
    for lang in langs:
        pools1[lang].sort()
    idx = {lang: 0 for lang in langs}
    while len(selected) < budget:
        progressed = False
        for lang in langs:
            while idx[lang] < len(pools1[lang]):
                bg = pools1[lang][idx[lang]]
                idx[lang] += 1
                if bg not in selected:
                    selected.add(bg)
                    progressed = True
                    break
            if len(selected) >= budget:
                break
        if not progressed:
            break
    if len(selected) >= budget:
        return selected
    # phase 2: remaining by higher freq round-robin
    remaining = budget - len(selected)
    extra = select_bigrams_round_robin(
        {lang: Counter({bg: cnt for bg, cnt in bigrams_by_lang[lang].items() if cnt > 1})
         for lang in langs},
        remaining,
        langs,
    )
    selected |= {bg for bg in extra if bg not in selected}
    # trim
    if len(selected) > budget:
        selected = set(list(selected)[:budget])
    return selected


def encode_words_only(word_seq: Sequence[str], singles: Set[str]) -> List[str]:
    out: List[str] = []
    for w in word_seq:
        if w in singles:
            out.append(w)
        else:
            out.extend(list(w))  # char fallback
    return out


def encode_greedy(
    word_seq: Sequence[str],
    multi: Set[NGram],
    singles: Set[str],
) -> List:
    max_n = 1
    for p in multi:
        if len(p) > max_n:
            max_n = len(p)
    tokens: List = []
    i = 0
    n = len(word_seq)
    while i < n:
        matched = False
        for L in range(min(max_n, n - i), 1, -1):
            tup = tuple(word_seq[i : i + L])
            if tup in multi:
                tokens.append(tup)
                i += L
                matched = True
                break
        if not matched:
            w = word_seq[i]
            if w in singles:
                tokens.append(w)
            else:
                tokens.extend(list(w))
            i += 1
    return tokens


def encode_dp(
    word_seq: Sequence[str],
    multi: Set[NGram],
    singles: Set[str],
) -> List:
    """Minimum-token cover over word sequence (bigram/phrase multi + word singles)."""
    n = len(word_seq)
    if n == 0:
        return []
    max_n = 1
    for p in multi:
        if len(p) > max_n:
            max_n = len(p)
    INF = 10**9
    dp = [INF] * (n + 1)
    prev = [-1] * (n + 1)
    prev_len = [0] * (n + 1)
    dp[0] = 0
    for i in range(n):
        if dp[i] >= INF:
            continue
        w = word_seq[i]
        cost = 1 if w in singles else max(1, len(w))
        if dp[i] + cost < dp[i + 1]:
            dp[i + 1] = dp[i] + cost
            prev[i + 1] = i
            prev_len[i + 1] = 1
        for L in range(2, min(max_n, n - i) + 1):
            tup = tuple(word_seq[i : i + L])
            if tup in multi and dp[i] + 1 < dp[i + L]:
                dp[i + L] = dp[i] + 1
                prev[i + L] = i
                prev_len[i + L] = L
    tokens: List = []
    i = n
    while i > 0:
        p = prev[i]
        L = prev_len[i]
        if L <= 0:
            # safety
            w = word_seq[i - 1]
            tokens.append(w if w in singles else list(w))
            i -= 1
            continue
        if L == 1:
            w = word_seq[p]
            if w in singles:
                tokens.append(w)
            else:
                tokens.extend(list(w))
        else:
            tokens.append(tuple(word_seq[p : p + L]))
        i = p
    tokens.reverse()
    return tokens


class WholeWordTokenizer:
    """
    Whole-word inventory tokenizer with optional multiword phrases.

    Encoding is over extract_words() tokens (not raw char stream), which matches
    the fairness denom harness definition of "words".
    """

    def __init__(
        self,
        singles: Set[str],
        multi: Optional[Iterable[Pair]] = None,
        use_dp: bool = True,
        name: str = "approach_c",
    ):
        self.singles = set(singles)
        self.multi: Set[NGram] = {tuple(p) for p in (multi or [])}
        self.use_dp = use_dp
        self.name = name

    @property
    def vocab_size(self) -> int:
        return len(self.singles) + len(self.multi)

    def encode_words(self, word_seq: Sequence[str]) -> List:
        if not self.multi:
            return encode_words_only(word_seq, self.singles)
        if self.use_dp:
            return encode_dp(word_seq, self.multi, self.singles)
        return encode_greedy(word_seq, self.multi, self.singles)

    def encode_fn(self, extract_words_fn):
        """Return harness-compatible encode_fn(text, lang)."""

        def _enc(text: str, lang: str):
            return self.encode_words(extract_words_fn(text))

        return _enc

    def to_vocab_list(self) -> List[str]:
        """Materialize vocab tokens; multiword pairs join with a single space."""
        items = sorted(self.singles)
        for p in sorted(self.multi):
            items.append(" ".join(p))
        return items
