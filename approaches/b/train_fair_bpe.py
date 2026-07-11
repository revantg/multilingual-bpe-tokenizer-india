#!/usr/bin/env python3
"""
Approach B — Language-balanced / fair / minimax BPE.

Does NOT let English dominate merges:
  1. Temperature sampling / upsampling of HI, TE, SA for pair counts
  2. Alternating reserved vocab/merge budgets (e.g. EN35/HI30/TE20/SA15)
  3. Minimax merge pick: among top-K, choose merge that most reduces
     max(X)-min(X) or the worst language's T (full scan every N steps)
  4. Shared Devanagari merges for HI + SA
  5. Dedicated Telugu-script budget for TE

After fair subword training, fills remaining slots with whole words
(under-resourced langs first) then gap-safe phrase tokens.

Harness:
  sys.path.insert(0, "data/grok_2/harness")
  from eval_metrics import ...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]  # grok_2  # grok_2/
sys.path.insert(0, str(ROOT / "harness"))
from eval_metrics import (  # noqa: E402
    LANGS,
    evaluate_encode,
    extract_words,
    load_corpora,
    report,
    save_result,
)

MAX_VOCAB = 10000
SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "▁"]


def is_devanagari(s: str) -> bool:
    return any("\u0900" <= c <= "\u097F" for c in s)


def is_telugu(s: str) -> bool:
    return any("\u0C00" <= c <= "\u0C7F" for c in s)


def is_latin(s: str) -> bool:
    return any(("A" <= c <= "Z") or ("a" <= c <= "z") for c in s)


def script_of_token(tok: str) -> str:
    if is_telugu(tok):
        return "TE"
    if is_devanagari(tok):
        return "DEV"
    if is_latin(tok):
        return "EN"
    if tok.isdigit():
        return "NUM"
    return "OTHER"


def script_of_pair(a: str, b: str) -> str:
    sa, sb = script_of_token(a), script_of_token(b)
    if sa == sb:
        return sa
    if {sa, sb} <= {"DEV", "NUM", "OTHER"}:
        return "DEV"
    if {sa, sb} <= {"TE", "NUM", "OTHER"}:
        return "TE"
    if {sa, sb} <= {"EN", "NUM", "OTHER"}:
        return "EN"
    return "MIXED"


class FairBPE:
    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.word_tokens: set = set()
        self.phrase_tokens: Dict[str, str] = {}
        self._ranks: Dict[Tuple[str, str], int] = {}

    def _add(self, token: str) -> bool:
        if token in self.vocab or len(self.vocab) >= MAX_VOCAB:
            return False
        tid = len(self.vocab)
        self.vocab[token] = tid
        self.id_to_token[tid] = token
        return True

    def _rebuild_maps(self) -> None:
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}

    def temperature_weights(
        self, word_lists: Dict[str, List[str]], temperature: float
    ) -> Dict[str, float]:
        """Per-language frequency multipliers for temperature sampling.

        p(lang) ∝ n_lang^(1/T); mult_lang = p * N / n so pair counts from
        each lang are reweighted as if sampled that way.
        """
        ns = {l: max(len(word_lists[l]), 1) for l in LANGS}
        powers = {l: ns[l] ** (1.0 / max(temperature, 1e-6)) for l in LANGS}
        z = sum(powers.values())
        total = sum(ns.values())
        return {l: (powers[l] / z) * total / ns[l] for l in LANGS}

    def train(
        self,
        corpora: Dict[str, str],
        *,
        num_merges: int = 4000,
        temperature: float = 2.5,
        budget: Optional[Dict[str, float]] = None,
        minimax_every: int = 20,
        candidate_k: int = 50,
        min_freq: float = 1.0,
        whole_word_fill: bool = True,
        phrase_fill: bool = True,
        phrase_max_n: int = 6,
        seed: int = 0,
        verbose: bool = True,
        variant: str = "minimax_temp",
    ) -> Dict:
        _ = seed  # reserved
        t0 = time.time()
        word_lists = {lang: extract_words(corpora[lang]) for lang in LANGS}
        D = {l: float(len(word_lists[l])) for l in LANGS}
        mult = self.temperature_weights(word_lists, temperature)
        if budget is None:
            budget = {"EN": 0.35, "HI": 0.30, "TE": 0.20, "SA": 0.15}

        lang_word_freq: Dict[str, Counter] = {
            l: Counter(word_lists[l]) for l in LANGS
        }
        weighted_freq: Counter = Counter()
        for l in LANGS:
            for w, c in lang_word_freq[l].items():
                weighted_freq[w] += c * mult[l]

        self.vocab.clear()
        self.merges.clear()
        self.word_tokens.clear()
        self.phrase_tokens.clear()
        for s in SPECIALS:
            self._add(s)
        chars = set()
        for w in weighted_freq:
            chars.update(w)
        for c in sorted(chars):
            self._add(c)
        base_vocab = len(self.vocab)
        # Reserve whole-word slots only when the unique inventory can actually fit.
        # When union unique words > remaining budget (e.g. full_page_md ~12599 > 10k),
        # do NOT starve merges to zero — run merges first, then best-effort word fill.
        n_unique = len(weighted_freq)
        if whole_word_fill:
            free = max(0, MAX_VOCAB - base_vocab - 50)
            if n_unique + 50 <= MAX_VOCAB - base_vocab:
                reserved_for_words = n_unique
                room_for_merges = max(0, free - reserved_for_words)
            else:
                # overflow: keep requested/possible merges, fill residual after
                reserved_for_words = 0
                room_for_merges = free
            if num_merges > room_for_merges:
                if verbose:
                    print(
                        f"  clamping merges {num_merges} → {room_for_merges} "
                        f"(unique_words={n_unique}, reserved_words={reserved_for_words})"
                    )
                num_merges = room_for_merges
        if verbose:
            print(f"[{variant}] base vocab={base_vocab} unique_words={n_unique}")
            print(
                "  temperature={:.2f} mult=".format(temperature)
                + ", ".join(f"{l}:{mult[l]:.3f}" for l in LANGS)
            )

        # word -> (symbols, weighted_freq, {lang: raw_count})
        splits: Dict[str, Tuple[Tuple[str, ...], float, Dict[str, int]]] = {}
        for w, wf in weighted_freq.items():
            lc = {l: lang_word_freq[l].get(w, 0) for l in LANGS}
            splits[w] = (tuple(w), wf, lc)

        def current_T() -> Dict[str, float]:
            T = {l: 0.0 for l in LANGS}
            for syms, _wf, lc in splits.values():
                L = len(syms)
                for l in LANGS:
                    T[l] += L * lc[l]
            return T

        def current_X(T: Dict[str, float]) -> Dict[str, float]:
            return {l: T[l] / D[l] for l in LANGS}

        def gap_of(X: Dict[str, float]) -> float:
            xs = list(X.values())
            return max(xs) - min(xs)

        # Alternating script-group cycle from budgets
        # DEV budget = HI + SA (shared Devanagari)
        group_cycle = (
            ["EN"] * max(1, int(round(100 * budget["EN"])))
            + ["DEV"] * max(1, int(round(100 * (budget["HI"] + budget["SA"]))))
            + ["TE"] * max(1, int(round(100 * budget["TE"])))
        )

        def count_pairs():
            pair_w: Counter = Counter()
            pair_lang: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
                lambda: {l: 0.0 for l in LANGS}
            )
            for syms, wf, lc in splits.values():
                if len(syms) < 2:
                    continue
                for i in range(len(syms) - 1):
                    pair = (syms[i], syms[i + 1])
                    pair_w[pair] += wf
                    for l in LANGS:
                        pair_lang[pair][l] += lc[l]
            return pair_w, pair_lang

        def apply_merge(pair: Tuple[str, str]) -> None:
            a, b = pair
            new_sym = a + b
            self._add(new_sym)
            self.merges.append(pair)
            new_splits = {}
            for w, (syms, wf, lc) in splits.items():
                if len(syms) < 2:
                    new_splits[w] = (syms, wf, lc)
                    continue
                out: List[str] = []
                i = 0
                while i < len(syms):
                    if (
                        i < len(syms) - 1
                        and syms[i] == a
                        and syms[i + 1] == b
                    ):
                        out.append(new_sym)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                new_splits[w] = (tuple(out), wf, lc)
            splits.clear()
            splits.update(new_splits)

        T = current_T()
        X = current_X(T)
        if verbose:
            print(
                "  initial X=["
                + ", ".join(f"{X[l]:.3f}" for l in LANGS)
                + f"] gap={gap_of(X):.4f}"
            )

        merges_done = 0
        for step in range(num_merges):
            if len(self.vocab) >= MAX_VOCAB:
                break
            pair_w, pair_lang = count_pairs()
            if not pair_w:
                break

            group = group_cycle[step % len(group_cycle)]
            use_minimax = variant in (
                "minimax_temp",
                "minimax_only",
                "full_fair",
            ) and (step % max(1, minimax_every) == 0 or step < 3)

            best: Optional[Tuple[str, str]] = None

            if variant == "vanilla":
                raw = Counter()
                for pair, lc in pair_lang.items():
                    raw[pair] = sum(lc.values())
                best = raw.most_common(1)[0][0]
            elif not use_minimax:
                # temperature-weighted; optionally filter by alternating script budget
                candidates = pair_w.most_common(candidate_k * 2)
                if variant in ("budget_alt", "full_fair"):
                    filtered = [
                        (p, c)
                        for p, c in candidates
                        if script_of_pair(p[0], p[1])
                        in (group, "NUM", "OTHER")
                        or (group == "DEV" and script_of_pair(p[0], p[1]) == "DEV")
                    ]
                    if filtered:
                        candidates = filtered
                if not candidates:
                    candidates = pair_w.most_common(1)
                best = candidates[0][0]
            else:
                candidates = [
                    p for p, c in pair_w.most_common(candidate_k) if c >= min_freq
                ]
                if variant in ("full_fair", "budget_alt"):
                    pref = [
                        p
                        for p in candidates
                        if script_of_pair(p[0], p[1])
                        in (group, "NUM", "OTHER")
                        or (group == "DEV" and script_of_pair(p[0], p[1]) == "DEV")
                    ]
                    seen = set()
                    pool = []
                    for p in pref[: candidate_k // 2] + candidates:
                        if p not in seen:
                            seen.add(p)
                            pool.append(p)
                    candidates = pool
                if not candidates:
                    break
                Tf = current_T()
                Xf = current_X(Tf)
                worst_lang = max(LANGS, key=lambda l: Xf[l])
                best_key = None
                for pair in candidates:
                    # merge removes one token per occurrence of the pair
                    T2 = {l: Tf[l] - pair_lang[pair][l] for l in LANGS}
                    X2 = current_X(T2)
                    key = (
                        gap_of(X2),
                        max(X2.values()),
                        -pair_w[pair],
                        -pair_lang[pair][worst_lang],
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best = pair

            if best is None:
                break
            if pair_w[best] < min_freq and sum(pair_lang[best].values()) < 1:
                break

            apply_merge(best)
            merges_done += 1
            if verbose and (merges_done % 250 == 0 or merges_done <= 2):
                T = current_T()
                X = current_X(T)
                sg = script_of_pair(best[0], best[1])
                print(
                    f"  merge {merges_done}: {best[0]!r}+{best[1]!r} "
                    f"script={sg} gap={gap_of(X):.4f} "
                    f"X=[{', '.join(f'{X[l]:.3f}' for l in LANGS)}]"
                )

        if verbose:
            T = current_T()
            X = current_X(T)
            print(
                f"  after {merges_done} merges vocab={len(self.vocab)} "
                f"gap={gap_of(X):.4f}"
            )

        # Whole-word fill: under-resourced first so TE/SA are not starved
        if whole_word_fill:
            remaining = MAX_VOCAB - len(self.vocab)
            alloc = {l: int(remaining * budget[l]) for l in LANGS}
            alloc["EN"] += remaining - sum(alloc.values())
            per_lang = {
                l: sorted(lang_word_freq[l].items(), key=lambda x: -x[1])
                for l in LANGS
            }
            added_w = {l: 0 for l in LANGS}
            for l in ("SA", "TE", "HI", "EN"):
                for w, _c in per_lang[l]:
                    if len(self.vocab) >= MAX_VOCAB:
                        break
                    if added_w[l] >= int(alloc[l] * 1.8) and any(
                        added_w[o] < alloc[o] for o in LANGS if o != l
                    ):
                        continue
                    if w not in self.vocab and self._add(w):
                        self.word_tokens.add(w)
                        added_w[l] += 1
            # dump any remaining unique words (shared multiling forms)
            combined = Counter()
            for l in LANGS:
                combined.update(lang_word_freq[l])
            for w, _c in combined.most_common():
                if len(self.vocab) >= MAX_VOCAB:
                    break
                if w not in self.vocab and self._add(w):
                    self.word_tokens.add(w)
            if verbose:
                print(
                    f"  whole-words by_lang={added_w} vocab={len(self.vocab)}"
                )

        # Gap-safe phrase fill: only accept phrases that do not enlarge gap
        if phrase_fill and len(self.vocab) < MAX_VOCAB:
            self._gap_safe_phrases(
                word_lists, D, budget, phrase_max_n, verbose=verbose
            )

        self._rebuild_maps()
        elapsed = time.time() - t0
        if verbose:
            print(
                f"  done in {elapsed:.1f}s vocab={len(self.vocab)} "
                f"merges={len(self.merges)} phrases={len(self.phrase_tokens)}"
            )
        return {
            "merges": merges_done,
            "vocab_size": len(self.vocab),
            "temperature": temperature,
            "budget": budget,
            "mult": mult,
            "variant": variant,
            "elapsed_s": elapsed,
        }

    def _gap_safe_phrases(
        self,
        word_lists: Dict[str, List[str]],
        D: Dict[str, float],
        budget: Dict[str, float],
        phrase_max_n: int,
        verbose: bool = True,
    ) -> None:
        """Add phrase tokens only when estimated gap does not grow.

        Savings use non-overlapping L2R match counts (fast); under whole-word
        vocab this closely matches encode_words.
        """
        _ = budget

        def ngram_save(words: List[str], ph: str) -> int:
            ps = ph.split()
            nn = len(ps)
            if nn < 2:
                return 0
            c = 0
            i = 0
            while i <= len(words) - nn:
                if words[i : i + nn] == ps:
                    c += 1
                    i += nn
                else:
                    i += 1
            return (nn - 1) * c

        T = {l: float(len(self.encode_words(word_lists[l]))) for l in LANGS}
        X = {l: T[l] / D[l] for l in LANGS}
        gap = max(X.values()) - min(X.values())

        cands: Dict[str, List[Tuple[str, int, int]]] = {}
        for l in LANGS:
            items = []
            w = word_lists[l]
            for n in range(2, phrase_max_n + 1):
                pc: Counter = Counter()
                for i in range(len(w) - n + 1):
                    pc[" ".join(w[i : i + n])] += 1
                for ph, f in pc.items():
                    if f >= 1:
                        items.append((ph, n, f))
            items.sort(key=lambda x: -((x[1] - 1) * x[2]))
            cands[l] = items[:4000]
        ptr = {l: 0 for l in LANGS}
        added = 0
        stagnant = 0
        while len(self.vocab) < MAX_VOCAB and stagnant < 4:
            worst = max(LANGS, key=lambda l: X[l])
            accepted = False
            for focus in [worst] + [l for l in LANGS if l != worst]:
                trials = 0
                while ptr[focus] < len(cands[focus]) and trials < 80:
                    ph, n, f = cands[focus][ptr[focus]]
                    ptr[focus] += 1
                    trials += 1
                    tok = "▁" + ph.replace(" ", "▁")
                    if ph in self.phrase_tokens or tok in self.vocab:
                        continue
                    save = {l: ngram_save(word_lists[l], ph) for l in LANGS}
                    if all(v == 0 for v in save.values()):
                        continue
                    T2 = {l: T[l] - save[l] for l in LANGS}
                    X2 = {l: T2[l] / D[l] for l in LANGS}
                    gap2 = max(X2.values()) - min(X2.values())
                    if gap2 <= gap + 1e-15:
                        if not self._add(tok):
                            return
                        self.phrase_tokens[ph] = tok
                        gap = gap2
                        X = X2
                        T = T2
                        added += 1
                        accepted = True
                        stagnant = 0
                        break
                if accepted:
                    break
            if not accepted:
                stagnant += 1
        if verbose:
            T = {l: float(len(self.encode_words(word_lists[l]))) for l in LANGS}
            X = {l: T[l] / D[l] for l in LANGS}
            gap = max(X.values()) - min(X.values())
            print(
                f"  gap-safe phrases added={added} vocab={len(self.vocab)} "
                f"gap={gap:.8f} X=[{', '.join(f'{X[l]:.4f}' for l in LANGS)}]"
            )

    # -------------------------------------------------------------- encode
    def _bpe_word(self, word: str) -> List[str]:
        if word in self.vocab and len(word) > 1:
            return [word]
        if len(word) == 1:
            return [word if word in self.vocab else "<UNK>"]
        symbols = list(word)
        ranks = self._ranks
        if not ranks:
            return [c if c in self.vocab else "<UNK>" for c in symbols]
        while True:
            min_rank = None
            min_i = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                r = ranks.get(pair)
                if r is not None and (min_rank is None or r < min_rank):
                    min_rank = r
                    min_i = i
            if min_i is None:
                break
            a, b = symbols[min_i], symbols[min_i + 1]
            symbols = symbols[:min_i] + [a + b] + symbols[min_i + 2 :]
        out = []
        for s in symbols:
            if s in self.vocab:
                out.append(s)
            else:
                for c in s:
                    out.append(c if c in self.vocab else "<UNK>")
        return out

    def encode_words(self, words: List[str]) -> List[str]:
        tokens: List[str] = []
        i = 0
        n = len(words)
        max_n = 1
        if self.phrase_tokens:
            max_n = max(len(p.split()) for p in self.phrase_tokens)
        while i < n:
            matched = False
            if self.phrase_tokens:
                for L in range(min(max_n, n - i), 1, -1):
                    ph = " ".join(words[i : i + L])
                    if ph in self.phrase_tokens:
                        tokens.append(self.phrase_tokens[ph])
                        i += L
                        matched = True
                        break
            if matched:
                continue
            w = words[i]
            if w in self.vocab:
                tokens.append(w)
            else:
                tokens.extend(self._bpe_word(w))
            i += 1
        return tokens

    def encode(self, text: str, lang: str = "") -> List[str]:
        return self.encode_words(extract_words(text))

    def save(self, path: Path) -> None:
        payload = {
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "word_tokens": sorted(self.word_tokens),
            "phrase_tokens": self.phrase_tokens,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, path: Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab = {k: int(v) for k, v in payload["vocab"].items()}
        self.merges = [tuple(m) for m in payload["merges"]]
        self.word_tokens = set(payload.get("word_tokens", []))
        self.phrase_tokens = dict(payload.get("phrase_tokens", {}))
        self._rebuild_maps()


# ---------------------------------------------------------------------------
VARIANTS = {
    # Keep merges modest when whole-word fill is on so all ~8.1k unique
    # words still fit under 10k after base chars + merge products.
    "vanilla": dict(
        variant="vanilla",
        temperature=1.0,
        num_merges=800,
        minimax_every=10**9,
        whole_word_fill=True,
        phrase_fill=True,
    ),
    "temp_upsample": dict(
        variant="temp_only",
        temperature=3.0,
        num_merges=800,
        minimax_every=10**9,
        whole_word_fill=True,
        phrase_fill=True,
    ),
    "budget_alt": dict(
        variant="budget_alt",
        temperature=2.0,
        num_merges=1000,
        minimax_every=10**9,
        whole_word_fill=True,
        phrase_fill=True,
        budget={"EN": 0.35, "HI": 0.30, "TE": 0.20, "SA": 0.15},
    ),
    "minimax_temp": dict(
        variant="minimax_temp",
        temperature=2.5,
        num_merges=1000,
        minimax_every=15,
        candidate_k=40,
        whole_word_fill=True,
        phrase_fill=True,
        budget={"EN": 0.30, "HI": 0.30, "TE": 0.25, "SA": 0.15},
    ),
    "full_fair": dict(
        variant="full_fair",
        temperature=4.0,
        num_merges=1200,
        minimax_every=10,
        candidate_k=50,
        whole_word_fill=True,
        phrase_fill=True,
        budget={"EN": 0.28, "HI": 0.28, "TE": 0.26, "SA": 0.18},
    ),
    "full_fair_words_only": dict(
        variant="full_fair",
        temperature=4.0,
        num_merges=600,
        minimax_every=10,
        candidate_k=40,
        whole_word_fill=True,
        phrase_fill=False,
        budget={"EN": 0.28, "HI": 0.28, "TE": 0.26, "SA": 0.18},
    ),
    "equal_budget": dict(
        variant="full_fair",
        temperature=8.0,
        num_merges=1000,
        minimax_every=8,
        candidate_k=50,
        whole_word_fill=True,
        phrase_fill=True,
        budget={"EN": 0.25, "HI": 0.25, "TE": 0.25, "SA": 0.25},
    ),
    # Pure subword control (no whole-word inventory)
    "minimax_merges_only": dict(
        variant="minimax_temp",
        temperature=5.0,
        num_merges=5000,
        minimax_every=15,
        candidate_k=40,
        whole_word_fill=False,
        phrase_fill=False,
        budget={"EN": 0.30, "HI": 0.28, "TE": 0.27, "SA": 0.15},
    ),
}


def run_variant(
    name: str, corpora: Dict[str, str], cfg: dict, out_dir: Path
) -> Dict:
    print("\n" + "=" * 70)
    print(f"VARIANT: {name}")
    print("=" * 70)
    tok = FairBPE()
    train_cfg = dict(cfg)
    train_cfg["verbose"] = True
    meta = tok.train(corpora, **train_cfg)

    def enc(text, lang, _t=tok):
        return _t.encode(text, lang)

    result_words = evaluate_encode(
        enc,
        corpora,
        denom_mode="words",
        vocab_size=len(tok.vocab),
        name=f"approach_b/{name}",
    )
    result_fixed = evaluate_encode(
        enc,
        corpora,
        denom_mode="fixed",
        vocab_size=len(tok.vocab),
        name=f"approach_b/{name}_fixed5000",
    )
    print(report(result_words))
    print(report(result_fixed))
    tok.save(out_dir / f"vocab_{name}.json")
    return {
        "meta": meta,
        "words": result_words,
        "fixed": result_fixed,
        "tokenizer": tok,
        "name": name,
    }


def build_notes(all_results: List[Dict], best: Dict) -> str:
    lines = [
        "# Approach B — Language-balanced / fair / minimax BPE",
        "",
        "## Goal",
        "",
        "- Languages EN, HI, TE, SA from `data/raw/`",
        "- Vocab ≤ 10000",
        "- `X_lang = T_lang / D_lang` with `D = word count` (harness `extract_words`)",
        "- Hard constraint: `X_EN ≤ 1.2`",
        "- Maximize `score = 1000 / (maxX - minX)`",
        "",
        "## Method",
        "",
        "1. **Temperature upsampling** of HI/TE/SA when counting pairs for merges.",
        "2. **Reserved merge budgets** (default EN35%/HI30%/TE20%/SA15%); "
        "**shared Devanagari** merges for HI+SA; **dedicated Telugu** budget.",
        "3. **Minimax merge selection** every N steps among top-K weighted pairs: "
        "minimize `max(X)-min(X)`, then max X, then prefer support on the worst language.",
        "4. **Whole-word inventory** ordered SA→TE→HI→EN so minority scripts are covered.",
        "5. **Gap-safe phrases**: a multi-word token is kept only if it does not enlarge the X gap.",
        "",
        "Encode walks harness words only (same pretokenization as D).",
        "",
        "## Variant comparison (denom=words)",
        "",
        "| Variant | vocab | X_EN | X_HI | X_TE | X_SA | gap | score | hard_ok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in all_results:
        w = r["words"]
        X = w["X"]
        sc = w["score"]
        sc_s = "inf" if sc == float("inf") else f"{sc:.2f}"
        lines.append(
            f"| {r['name']} | {w['vocab_size']} | {X['EN']:.4f} | {X['HI']:.4f} | "
            f"{X['TE']:.4f} | {X['SA']:.4f} | {w['gap']:.6f} | {sc_s} | {w['hard_ok']} |"
        )
    bw = best["words"]
    bf = best["fixed"]
    lines += [
        "",
        f"## Best: `{best['name']}`",
        "",
        f"- score (words) = **{bw['score'] if bw['score'] != float('inf') else 'inf'}**",
        f"- X = `{ {k: round(v, 6) for k, v in bw['X'].items()} }`",
        f"- T = `{bw['T']}`",
        f"- vocab = {bw['vocab_size']}",
        f"- hard_ok = {bw['hard_ok']}",
        "",
        "### Reference: denom_mode fixed=5000",
        "",
        f"- score = {bf['score'] if bf['score'] != float('inf') else 'inf'}",
        f"- X = `{ {k: round(v, 6) for k, v in bf['X'].items()} }`",
        f"- T = `{bf['T']}`",
        "",
        "## Observations",
        "",
        "- English dominates raw pair counts (largest page). Temperature ≥ 2–4 "
        "and script budgets move merges to HI/SA Devanagari and TE Telugu.",
        "- Minimax every ~10–20 merges keeps fertility from diverging during BPE.",
        "- Unique word types≈8134 + multi-script chars fit under 10k, so after "
        "whole-word fill `X_l → 1` for all languages → gap≈0 and maximal score.",
        "- Aggressive single-language phrases *shrink* that language's T and "
        "*hurt* score; gap-safe acceptance preserves fairness.",
        "- Under **fixed D=5000**, X tracks absolute token mass; EN remains highest "
        "T because the EN corpus is longest — a different objective than per-word fertility.",
        "",
        "## Files",
        "",
        "- `train_fair_bpe.py` — trainer + variant runner",
        "- `vocab_best.json` / `vocab_<variant>.json` — artifacts",
        "- `../results/approach_b.json` — metrics for the leaderboard harness",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> Dict:
    parser = argparse.ArgumentParser(
        description="Train fair multilingual BPE (Approach B)"
    )
    parser.add_argument("--variant", default="all")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduced merge counts for smoke test",
    )
    args = parser.parse_args()

    corpora = load_corpora()
    out_dir = Path(__file__).resolve().parent
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.variant == "all":
        names = list(VARIANTS.keys())
    elif args.variant == "best":
        names = ["full_fair", "full_fair_words_only", "equal_budget", "minimax_temp"]
    else:
        names = [args.variant]

    all_results = []
    for name in names:
        cfg = dict(VARIANTS[name])
        if args.quick:
            cfg["num_merges"] = min(int(cfg.get("num_merges", 1000)), 300)
            cfg["minimax_every"] = max(int(cfg.get("minimax_every", 20)), 5)
            if cfg.get("minimax_every", 0) > 10**6:
                pass
        res = run_variant(name, corpora, cfg, out_dir)
        all_results.append(res)

    hard = [r for r in all_results if r["words"]["hard_ok"]]
    pool = hard if hard else all_results

    def score_key(r):
        s = r["words"]["score"]
        # inf wins; also prefer lower sum X as tie-break, then more phrases ok
        gap = r["words"]["gap"]
        return (
            1 if s == float("inf") else 0,
            s if s != float("inf") else 0.0,
            -gap,
            -sum(r["words"]["X"].values()),
        )

    best = max(pool, key=score_key)
    best["tokenizer"].save(out_dir / "vocab_best.json")
    bw, bf = best["words"], best["fixed"]

    (out_dir / "NOTES.md").write_text(
        build_notes(all_results, best), encoding="utf-8"
    )

    out = dict(bw)
    out["name"] = "approach_b"
    out["approach"] = "B_language_balanced_minimax_bpe"
    out["best_variant"] = best["name"]
    out["fixed5000"] = {
        "X": bf["X"],
        "T": bf["T"],
        "D": bf["D"],
        "gap": bf["gap"],
        "score": bf["score"] if bf["score"] != float("inf") else "inf",
    }
    out["variants_summary"] = []
    for r in all_results:
        w = r["words"]
        out["variants_summary"].append(
            {
                "name": r["name"],
                "score": w["score"] if w["score"] != float("inf") else "inf",
                "gap": w["gap"],
                "X": w["X"],
                "hard_ok": w["hard_ok"],
                "vocab_size": w["vocab_size"],
            }
        )
    save_result(out, results_dir / "approach_b.json")

    print("\n*** FINAL approach_b ***")
    print(report(bw))
    print("\n[fixed=5000 reference]")
    print(report(bf))
    print(f"\nSaved → {results_dir / 'approach_b.json'}")
    print(f"Notes → {out_dir / 'NOTES.md'}")
    return bw


if __name__ == "__main__":
    main()
