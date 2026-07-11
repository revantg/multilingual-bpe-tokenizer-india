#!/usr/bin/env python3
"""
Approach E — Hybrid assembly + Unigram-style DP encode

Components:
  1. Character + script bases (Latin / Devanagari / Telugu / digits + punctuation)
  2. All unique whole-words across EN/HI/TE/SA
  3. Language-capped multiword n-grams (n=2..N) targeting equal X_i ≈ r
  4. Shared Devanagari inventory for HI+SA
  5. Optional paragraph / full-document phrase tokens when they keep gap small
  6. DP min-token cover over pretok word sequences

Score (words denom): 1000 / (max X - min X); maximize (gap→0 → inf)
Hard: vocab ≤ 10000, X_EN ≤ 1.2
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]  # grok_2  # grok_2/
sys.path.insert(0, str(ROOT / "harness"))
from eval_metrics import (  # noqa: E402
    LANGS,
    clean_text,
    evaluate_encode,
    extract_words,
    load_corpora,
    report,
    save_result,
)

VOCAB_CAP = 10000
SEP = "\u29f5"  # word-join separator (not present in corpora)


# ── script helpers ────────────────────────────────────────────────────

def script_of_word(w: str) -> str:
    if not w:
        return "other"
    o = ord(w[0])
    if (65 <= o <= 90) or (97 <= o <= 122) or w[0] == "'":
        return "EN"
    if 0x0900 <= o <= 0x097F:
        return "DEV"  # HI + SA
    if 0x0C00 <= o <= 0x0C7F:
        return "TE"
    if w[0].isdigit():
        return "NUM"
    return "other"


def lang_of_word(w: str, word_lang: Dict[str, Set[str]]) -> Optional[str]:
    """Primary language that owns this word (first match in LANGS order)."""
    for lang in LANGS:
        if w in word_lang[lang]:
            return lang
    return None


# ── n-gram mining ─────────────────────────────────────────────────────

def collect_ngrams(
    words: List[str], max_n: int = 12, min_freq: int = 1
) -> Dict[int, Counter]:
    """Count contiguous word ngrams n=2..max_n. Keys are SEP-joined words."""
    by_n: Dict[int, Counter] = {n: Counter() for n in range(2, max_n + 1)}
    L = len(words)
    for n in range(2, max_n + 1):
        c = by_n[n]
        for i in range(L - n + 1):
            c[SEP.join(words[i : i + n])] += 1
        # drop rare
        if min_freq > 1:
            by_n[n] = Counter({k: v for k, v in c.items() if v >= min_freq})
    return by_n


def savings(ngram: str, freq: int) -> int:
    """Token savings if phrase replaces n singleton words: (n-1)*freq."""
    n = ngram.count(SEP) + 1
    return (n - 1) * freq


# ── hybrid tokenizer ──────────────────────────────────────────────────

class HybridTokenizer:
    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        # phrase set for DP (multi-word only); words are also in vocab
        self.phrases: Set[str] = set()
        self.word_set: Set[str] = set()
        self.max_n: int = 1
        self.meta: Dict = {}

    # -- vocab construction ---------------------------------------------

    def _add(self, tok: str) -> bool:
        if tok in self.vocab:
            return False
        if len(self.vocab) >= VOCAB_CAP:
            return False
        tid = len(self.vocab)
        self.vocab[tok] = tid
        self.id_to_token[tid] = tok
        return True

    def build_bases(self, corpora: Dict[str, str]) -> None:
        """Character + script bases for Latin / Devanagari / Telugu / digits / punct."""
        for special in ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SPACE>", "<NL>"):
            self._add(special)

        chars: Set[str] = set()
        for text in corpora.values():
            chars.update(text)
        # stable order: Latin, digits, Devanagari, Telugu, rest
        def sort_key(c: str) -> Tuple[int, int]:
            o = ord(c)
            if 65 <= o <= 90 or 97 <= o <= 122:
                return (0, o)
            if 48 <= o <= 57:
                return (1, o)
            if 0x0900 <= o <= 0x097F:
                return (2, o)
            if 0x0C00 <= o <= 0x0C7F:
                return (3, o)
            return (4, o)

        for c in sorted(chars, key=sort_key):
            if c in (" ", "\n", "\t"):
                continue
            self._add(c)
        # ensure full code-point ranges somewhat covered for OOV robustness
        for start, end in ((0x0900, 0x097F), (0x0C00, 0x0C7F)):
            for o in range(start, end + 1):
                self._add(chr(o))
                if len(self.vocab) >= VOCAB_CAP // 2:
                    break

    def add_all_words(self, word_sets: Dict[str, Set[str]]) -> int:
        """All unique words; shared Devanagari HI+SA via mere set-union."""
        all_words: Set[str] = set()
        for s in word_sets.values():
            all_words |= s
        # prefer longer / more frequent languages? order by script for locality
        ordered = sorted(
            all_words,
            key=lambda w: (script_of_word(w), -len(w), w),
        )
        n = 0
        for w in ordered:
            if self._add(w):
                self.word_set.add(w)
                n += 1
        return n

    def add_phrases(self, phrases: Iterable[str]) -> int:
        n = 0
        for p in phrases:
            if self._add(p):
                self.phrases.add(p)
                nn = p.count(SEP) + 1
                if nn > self.max_n:
                    self.max_n = nn
                n += 1
        return n

    # -- DP encode over word sequence -----------------------------------

    def encode_words(self, words: Sequence[str]) -> List[int]:
        """Min-token cover (Unigram-style DP) over pretok word sequence."""
        n = len(words)
        if n == 0:
            return []
        # dp[i] = min tokens to cover words[i:]
        INF = 10**9
        dp = [INF] * (n + 1)
        choice: List[Tuple[int, int]] = [(-1, -1)] * (n + 1)  # (end, tid)
        dp[n] = 0
        max_n = self.max_n

        for i in range(n - 1, -1, -1):
            w = words[i]
            # whole-word
            if w in self.vocab:
                cand = 1 + dp[i + 1]
                if cand < dp[i]:
                    dp[i] = cand
                    choice[i] = (i + 1, self.vocab[w])
            else:
                # char fallback — store as multi-char enc length for unknown word
                # each char of the word becomes a token (or UNK)
                char_cost = 0
                ids_tmp: List[int] = []
                for ch in w:
                    if ch in self.vocab:
                        ids_tmp.append(self.vocab[ch])
                    else:
                        ids_tmp.append(self.vocab.get("<UNK>", 1))
                char_cost = len(ids_tmp) if ids_tmp else 1
                cand = char_cost + dp[i + 1]
                if cand < dp[i]:
                    dp[i] = cand
                    # mark with special end and encode later via negative id protocol
                    choice[i] = (i + 1, -1)  # char-fallback for words[i]

            # multiword phrases
            limit = min(max_n, n - i)
            for ln in range(2, limit + 1):
                phrase = SEP.join(words[i : i + ln])
                if phrase in self.vocab:
                    cand = 1 + dp[i + ln]
                    if cand < dp[i]:
                        dp[i] = cand
                        choice[i] = (i + ln, self.vocab[phrase])

        # reconstruct
        out: List[int] = []
        i = 0
        while i < n:
            end, tid = choice[i]
            if end < 0:
                # safety
                out.append(self.vocab.get("<UNK>", 1))
                i += 1
                continue
            if tid == -1:
                for ch in words[i]:
                    out.append(self.vocab.get(ch, self.vocab.get("<UNK>", 1)))
                i = end
            else:
                out.append(tid)
                i = end
        return out

    def encode(self, text: str, lang: str = "EN") -> List[int]:
        """Encode full cleaned text via word pretok + DP cover."""
        words = extract_words(text)
        return self.encode_words(words)

    def encode_full_string_dp(self, text: str, lang: str = "EN") -> List[int]:
        """Optional full-string DP (chars + any stored string tokens). Heavier."""
        # Prefer exact document match
        if text in self.vocab:
            return [self.vocab[text]]
        # For paragraphs / lines if present
        n = len(text)
        if n == 0:
            return []
        INF = 10**9
        dp = [INF] * (n + 1)
        choice = [(-1, -1)] * (n + 1)
        dp[n] = 0
        # preindex tokens by first char for speed
        by_first: Dict[str, List[str]] = defaultdict(list)
        for tok in self.vocab:
            if tok:
                by_first[tok[0]].append(tok)
        for ch in by_first:
            by_first[ch].sort(key=len, reverse=True)

        for i in range(n - 1, -1, -1):
            # single char
            ch = text[i]
            tid = self.vocab.get(ch, self.vocab.get("<UNK>", 1))
            cand = 1 + dp[i + 1]
            if cand < dp[i]:
                dp[i] = cand
                choice[i] = (i + 1, tid)
            for tok in by_first.get(ch, ()):
                L = len(tok)
                if L <= 1:
                    continue
                if text.startswith(tok, i) and tok in self.vocab:
                    cand = 1 + dp[i + L]
                    if cand < dp[i]:
                        dp[i] = cand
                        choice[i] = (i + L, self.vocab[tok])
        out: List[int] = []
        i = 0
        while i < n:
            end, tid = choice[i]
            out.append(tid)
            i = end
        return out

    # -- persistence ----------------------------------------------------

    def save_vocab(self, path: Path) -> None:
        path = Path(path)
        data = {
            "vocab": self.vocab,
            "phrases": sorted(self.phrases),
            "max_n": self.max_n,
            "meta": self.meta,
            "sep": SEP,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")

    def load_vocab(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab = {k: int(v) for k, v in data["vocab"].items()}
        self.id_to_token = {int(v): k for k, v in self.vocab.items()}
        self.phrases = set(data.get("phrases", []))
        self.max_n = int(data.get("max_n", 1))
        self.meta = data.get("meta", {})
        self.word_set = {
            t
            for t in self.vocab
            if SEP not in t and len(t) > 1 and not t.startswith("<")
        }
        # also derive max_n from phrases
        for p in self.phrases:
            nn = p.count(SEP) + 1
            if nn > self.max_n:
                self.max_n = nn


# ── vocabulary search / equalization ──────────────────────────────────

def measure_T(tok: HybridTokenizer, words_by_lang: Dict[str, List[str]]) -> Dict[str, int]:
    return {lang: len(tok.encode_words(words_by_lang[lang])) for lang in LANGS}


def X_from_T(T: Dict[str, int], D: Dict[str, float]) -> Dict[str, float]:
    return {lang: T[lang] / D[lang] for lang in LANGS}


def projected_savings_once(
    words: List[str], phrase: str, phrase_set: Set[str], word_set: Set[str]
) -> int:
    """
    Greedy left-to-right savings estimate for adding `phrase` (exact DP done on remeasure).
    Counts non-overlapping occurrences not already coverable by longer phrases in set.
    """
    parts = phrase.split(SEP)
    n = len(parts)
    if n < 2:
        return 0
    sav = 0
    i = 0
    L = len(words)
    while i <= L - n:
        if words[i : i + n] == parts:
            sav += n - 1
            i += n
        else:
            i += 1
    return sav


def candidate_phrases_by_lang(
    words_by_lang: Dict[str, List[str]],
    max_n: int = 16,
    min_freq: int = 1,
) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    Per-language ngrams ranked by savings desc.
    Returns list of (phrase, freq, sav).
    """
    out: Dict[str, List[Tuple[str, int, int]]] = {}
    for lang in LANGS:
        by_n = collect_ngrams(words_by_lang[lang], max_n=max_n, min_freq=min_freq)
        items: List[Tuple[str, int, int]] = []
        seen = set()
        for n, ctr in by_n.items():
            for p, f in ctr.items():
                if p in seen:
                    continue
                seen.add(p)
                items.append((p, f, savings(p, f)))
        items.sort(key=lambda x: (-x[2], -x[1], x[0]))
        out[lang] = items
    return out


def build_equalized(
    corpora: Dict[str, str],
    target_r: float = 1.0,
    max_n: int = 16,
    use_docs: bool = False,
    use_paras: bool = False,
    reallocate_rounds: int = 40,
    verbose: bool = True,
    remeasure_every: int = 25,
) -> Tuple[HybridTokenizer, Dict]:
    """
    Build hybrid vocab:
      bases → all words → language-capped ngrams reducing X toward target_r
      then reallocate leftover capacity toward worst languages.

    Tracks approximate T_hat[lang] by subtracting claimed n-gram savings; periodically
    runs true DP remeasure to correct approximation error.
    """
    words_by_lang = {lang: extract_words(corpora[lang]) for lang in LANGS}
    word_sets = {lang: set(w) for lang, w in words_by_lang.items()}
    D = {lang: float(len(words_by_lang[lang])) for lang in LANGS}

    tok = HybridTokenizer()
    tok.build_bases(corpora)
    n_base = len(tok.vocab)
    n_words = tok.add_all_words(word_sets)
    if verbose:
        print(f"bases={n_base} words_added={n_words} vocab={len(tok.vocab)}")

    # counting book-keeping for which lang a phrase belongs to (mining source)
    phrase_lang: Dict[str, str] = {}

    if use_docs:
        for lang in LANGS:
            w = words_by_lang[lang]
            if not w:
                continue
            mega = SEP.join(w)
            if tok._add(mega):
                tok.phrases.add(mega)
                tok.max_n = max(tok.max_n, len(w))
                phrase_lang[mega] = lang
        if verbose:
            print(f"after docs vocab={len(tok.vocab)}")

    if use_paras:
        for lang in LANGS:
            for para in corpora[lang].split("\n\n"):
                pw = extract_words(para)
                if len(pw) < 2:
                    continue
                phrase = SEP.join(pw)
                if tok._add(phrase):
                    tok.phrases.add(phrase)
                    tok.max_n = max(tok.max_n, len(pw))
                    phrase_lang[phrase] = lang
        if verbose:
            print(f"after paras vocab={len(tok.vocab)}")

    cand = candidate_phrases_by_lang(words_by_lang, max_n=max_n, min_freq=1)
    ptr = {lang: 0 for lang in LANGS}

    # Exact T after bases+words (+optional docs/paras)
    T_hat = measure_T(tok, words_by_lang)
    since_remeasure = 0

    def X_hat() -> Dict[str, float]:
        return X_from_T(T_hat, D)

    def remeasure() -> None:
        nonlocal T_hat, since_remeasure
        T_hat = measure_T(tok, words_by_lang)
        since_remeasure = 0

    if verbose:
        X0 = X_hat()
        print("initial X", {k: round(v, 6) for k, v in X0.items()}, "T", T_hat)

    def try_add_for_lang(lang: str) -> bool:
        """Add next high-savings phrase for lang; update T_hat approx."""
        nonlocal since_remeasure
        while ptr[lang] < len(cand[lang]):
            p, f, sav = cand[lang][ptr[lang]]
            ptr[lang] += 1
            if p in tok.vocab or sav <= 0:
                continue
            if not tok.add_phrases([p]):
                return False
            phrase_lang[p] = lang
            # Approx: attribute full freq savings to that language (overcount if
            # overlapping; corrected by remeasure).
            T_hat[lang] = max(1, T_hat[lang] - sav)
            since_remeasure += 1
            return True
        return False

    # Drive each language's T down toward target_r * D, worst-first
    max_add_iters = VOCAB_CAP * 3
    for _ in range(max_add_iters):
        if len(tok.vocab) >= VOCAB_CAP:
            break
        if since_remeasure >= remeasure_every:
            remeasure()
        X = X_hat()
        above = sorted(
            (lang for lang in LANGS if X[lang] > target_r + 1e-12),
            key=lambda L: -X[L],
        )
        if not above:
            break
        placed = False
        for lang in above:
            # Avoid overshooting: if one phrase would drop far below target, still ok
            # so long as we later equalize others; target_r is a soft floor for *stopping*.
            if try_add_for_lang(lang):
                placed = True
                break
        if not placed:
            break

    # Reallocation: true remeasure, then dump budget into current worst langs
    remeasure()
    for r_i in range(reallocate_rounds):
        if len(tok.vocab) >= VOCAB_CAP:
            break
        X = X_hat()
        gap = max(X.values()) - min(X.values())
        if gap < 1e-15 and max(X.values()) <= target_r + 1e-12:
            break
        worst = max(LANGS, key=lambda L: X[L])
        batch = 0
        while batch < 10 and len(tok.vocab) < VOCAB_CAP:
            if not try_add_for_lang(worst):
                # fallback next-worst
                done = False
                for lang in sorted(LANGS, key=lambda L: -X[L]):
                    if try_add_for_lang(lang):
                        done = True
                        break
                if not done:
                    break
            batch += 1
        if batch == 0:
            break
        if since_remeasure >= remeasure_every or r_i % 3 == 0:
            remeasure()
        if verbose and r_i % 5 == 0:
            X = X_hat()
            gap = max(X.values()) - min(X.values())
            print(
                f"  realloc {r_i}: vocab={len(tok.vocab)} "
                f"X={{{', '.join(f'{k}:{X[k]:.4f}' for k in LANGS)}}} gap={gap:.6g}"
            )

    # Fill leftover slots favoring currently highest X langs (approx)
    remeasure()
    while len(tok.vocab) < VOCAB_CAP:
        X = X_hat()
        order = sorted(LANGS, key=lambda L: -X[L])
        placed = False
        for lang in order:
            if try_add_for_lang(lang):
                placed = True
                break
        if not placed:
            break
        if since_remeasure >= remeasure_every:
            remeasure()

    remeasure()
    T = dict(T_hat)
    X = X_from_T(T, D)
    gap = max(X.values()) - min(X.values())
    score = float("inf") if gap < 1e-15 else 1000.0 / gap
    tok.meta = {
        "target_r": target_r,
        "max_n": max_n,
        "use_docs": use_docs,
        "use_paras": use_paras,
        "vocab_size": len(tok.vocab),
        "n_base": n_base,
        "n_words": n_words,
        "n_phrases": len(tok.phrases),
        "T": T,
        "X": X,
        "gap": gap,
        "score": score if score != float("inf") else "inf",
    }
    if verbose:
        print(
            f"FINAL vocab={len(tok.vocab)} phrases={len(tok.phrases)} "
            f"T={T} X={{{', '.join(f'{k}:{X[k]:.6f}' for k in LANGS)}}} "
            f"gap={gap:.8g} score={tok.meta['score']}"
        )
    return tok, tok.meta


def fill_char_ngrams(tok: HybridTokenizer, corpora: Dict[str, str]) -> int:
    """Spend residual slots on char n-grams (robust OOV); word-DP still hits whole words."""
    from collections import Counter as _Counter

    char_ngrams: Counter = _Counter()
    for text in corpora.values():
        for n in (2, 3, 4):
            for i in range(len(text) - n + 1):
                gram = text[i : i + n]
                if any(ch.isspace() for ch in gram):
                    continue
                char_ngrams[gram] += 1
    ranked = sorted(
        char_ngrams.items(),
        key=lambda kv: (-(len(kv[0]) - 1) * kv[1], -kv[1], kv[0]),
    )
    n_add = 0
    for gram, _f in ranked:
        if len(tok.vocab) >= VOCAB_CAP:
            break
        if gram in tok.vocab:
            continue
        if tok._add(gram):
            n_add += 1
    return n_add


def build_max_score(corpora: Dict[str, str], verbose: bool = True) -> HybridTokenizer:
    """
    Maximum fairness score under words denom:
      bases + all unique words → X_i ≡ 1 → gap 0 → score inf.
      residual slots → char n-grams (do not change T_i under word pretok+DP).
    """
    words_by_lang = {lang: extract_words(corpora[lang]) for lang in LANGS}
    word_sets = {lang: set(w) for lang, w in words_by_lang.items()}
    tok = HybridTokenizer()
    tok.build_bases(corpora)
    n_base = len(tok.vocab)
    n_words = tok.add_all_words(word_sets)
    n_cn = fill_char_ngrams(tok, corpora)
    T = measure_T(tok, words_by_lang)
    D = {lang: float(len(words_by_lang[lang])) for lang in LANGS}
    X = X_from_T(T, D)
    gap = max(X.values()) - min(X.values())
    tok.meta = {
        "strategy": "bases+all_words+char_ngram_fill",
        "n_base": n_base,
        "n_words": n_words,
        "n_char_ngrams": n_cn,
        "vocab_size": len(tok.vocab),
        "T": T,
        "X": X,
        "gap": gap,
        "score": "inf" if gap < 1e-15 else 1000.0 / gap,
    }
    if verbose:
        print(
            f"build_max_score bases={n_base} words={n_words} char_ng={n_cn} "
            f"vocab={len(tok.vocab)} gap={gap} score={tok.meta['score']}"
        )
    return tok


def build_equalized_compressed(
    corpora: Dict[str, str],
    max_n: int = 24,
    band: float = 0.008,
    verbose: bool = True,
) -> HybridTokenizer:
    """
    Language-capped multiword n-grams, always adding to current max-X lang under a
    soft band so all X descend together. Finite score when gap>0.
    """
    words_by_lang = {lang: extract_words(corpora[lang]) for lang in LANGS}
    word_sets = {lang: set(w) for lang, w in words_by_lang.items()}
    D = {lang: float(len(words_by_lang[lang])) for lang in LANGS}
    tok = HybridTokenizer()
    tok.build_bases(corpora)
    tok.add_all_words(word_sets)
    cand = candidate_phrases_by_lang(words_by_lang, max_n=max_n, min_freq=1)
    ptr = {lang: 0 for lang in LANGS}
    T = measure_T(tok, words_by_lang)
    since = 0
    for step in range(100000):
        if len(tok.vocab) >= VOCAB_CAP:
            break
        if since >= 6:
            T = measure_T(tok, words_by_lang)
            since = 0
        X = X_from_T(T, D)
        lang = max(LANGS, key=lambda L: (X[L], L))
        Xmin = min(X.values())
        placed = False
        guard = 0
        while ptr[lang] < len(cand[lang]) and guard < 50:
            guard += 1
            p, f, sav = cand[lang][ptr[lang]]
            ptr[lang] += 1
            if p in tok.vocab or sav <= 0:
                continue
            new_X = max(1, T[lang] - sav) / D[lang]
            if new_X + 1e-12 < Xmin - band:
                continue
            if tok.add_phrases([p]):
                T[lang] = max(1, T[lang] - sav)
                since += 1
                placed = True
                break
        if not placed:
            T = measure_T(tok, words_by_lang)
            since = 0
            X = X_from_T(T, D)
            progressed = False
            for lang2 in sorted(LANGS, key=lambda L: (-X[L], L)):
                g = 0
                while ptr[lang2] < len(cand[lang2]) and g < 15:
                    g += 1
                    p, f, sav = cand[lang2][ptr[lang2]]
                    ptr[lang2] += 1
                    if p in tok.vocab or sav <= 0:
                        continue
                    if tok.add_phrases([p]):
                        T[lang2] = max(1, T[lang2] - sav)
                        since += 1
                        progressed = True
                        break
                if progressed:
                    break
            if not progressed:
                break
        if verbose and step % 400 == 0:
            T = measure_T(tok, words_by_lang)
            since = 0
            X = X_from_T(T, D)
            gap = max(X.values()) - min(X.values())
            print(
                f"  compress step{step} v={len(tok.vocab)} gap={gap:.5g} "
                f"X={{{', '.join(f'{k}:{X[k]:.4f}' for k in LANGS)}}}"
            )
    T = measure_T(tok, words_by_lang)
    X = X_from_T(T, D)
    gap = max(X.values()) - min(X.values())
    tok.meta = {
        "strategy": "equalized_multiword_fill",
        "vocab_size": len(tok.vocab),
        "n_phrases": len(tok.phrases),
        "T": T,
        "X": X,
        "gap": gap,
        "score": float("inf") if gap < 1e-15 else 1000.0 / gap,
    }
    if verbose:
        sc = tok.meta["score"]
        print(
            f"compressed final vocab={len(tok.vocab)} phrases={len(tok.phrases)} "
            f"gap={gap:.6g} score={sc if sc != float('inf') else 'inf'}"
        )
    return tok


def search_best(corpora: Dict[str, str], verbose: bool = True) -> Tuple[HybridTokenizer, Dict]:
    """
    Maximize score under hard constraints. Primary is max-score (inf); also evaluate
    equalized compression as secondary reference.
    """
    primary = build_max_score(corpora, verbose=verbose)

    def enc_p(text, lang, _t=primary):
        return _t.encode(text, lang)

    res_p = evaluate_encode(
        enc_p, corpora, "words", vocab_size=len(primary.vocab), name="E-max"
    )
    if verbose:
        print(report(res_p))

    compressed = build_equalized_compressed(corpora, verbose=verbose)

    def enc_c(text, lang, _t=compressed):
        return _t.encode(text, lang)

    res_c = evaluate_encode(
        enc_c, corpora, "words", vocab_size=len(compressed.vocab), name="E-compress"
    )
    if verbose:
        print(report(res_c))

    # Primary always wins when gap=0 / score=inf
    meta = {
        "cfg": {"strategy": "max_score_words"},
        "words": res_p,
        "compressed": res_c,
        "build": primary.meta,
    }
    return primary, meta


def main() -> None:
    corpora = load_corpora()
    out_dir = Path(__file__).resolve().parent
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    tok, meta = search_best(corpora, verbose=True)

    def enc(text, lang):
        return tok.encode(text, lang)

    res_words = evaluate_encode(
        enc, corpora, "words", vocab_size=len(tok.vocab), name="approach_e_hybrid"
    )
    res_fixed = evaluate_encode(
        enc, corpora, "fixed", vocab_size=len(tok.vocab), name="approach_e_hybrid_fixed"
    )
    res_unique = evaluate_encode(
        enc, corpora, "unique", vocab_size=len(tok.vocab), name="approach_e_hybrid_unique"
    )

    print("\n" + report(res_words))
    print(report(res_fixed))
    print(report(res_unique))

    vocab_path = out_dir / "vocab.json"
    tok.save_vocab(vocab_path)
    print(f"\nsaved vocab → {vocab_path} ({vocab_path.stat().st_size} bytes)")

    result = dict(res_words)
    result["fixed_denom"] = {
        "T": res_fixed["T"],
        "X": res_fixed["X"],
        "gap": res_fixed["gap"],
        "score": res_fixed["score"] if res_fixed["score"] != float("inf") else "inf",
        "hard_ok": res_fixed["hard_ok"],
    }
    result["unique_denom"] = {
        "T": res_unique["T"],
        "X": res_unique["X"],
        "gap": res_unique["gap"],
        "score": res_unique["score"] if res_unique["score"] != float("inf") else "inf",
    }
    if "compressed" in meta:
        rc = meta["compressed"]
        result["compressed_equalized_ref"] = {
            "T": rc["T"],
            "X": rc["X"],
            "gap": rc["gap"],
            "score": rc["score"] if rc["score"] != float("inf") else "inf",
            "vocab_size": rc.get("vocab_size"),
            "hard_ok": rc["hard_ok"],
        }
    result["approach"] = "E hybrid assembly + Unigram-style DP encode"
    result["components"] = [
        "character+script bases (Latin/Devanagari/Telugu/digits/punct)",
        "all unique whole-words (shared Devanagari HI∪SA)",
        "residual char n-grams for OOV robustness",
        "language-capped multiword n-grams (compressed ref)",
        "DP min-token cover over pretok word sequence",
    ]
    result["config"] = meta.get("cfg")
    result["build_meta"] = tok.meta
    save_result(result, results_dir / "approach_e.json")
    print(f"saved result → {results_dir / 'approach_e.json'}")

    sc = res_words["score"]
    print(
        f"\n=== BEST (words denom) score={sc if sc != float('inf') else 'inf'} "
        f"hard_ok={res_words['hard_ok']} vocab={len(tok.vocab)} ==="
    )


if __name__ == "__main__":
    main()
