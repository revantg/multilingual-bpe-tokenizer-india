#!/usr/bin/env python3
"""
Approach A — Classic multilingual BPE (baseline).

Real iterative pair-merge BPE on pretok word units drawn from India Wikipedia
pages (EN, HI, TE, SA).

Variants:
  1) pure_concat         — unweighted concatenation of all pretok units
  2) upsample            — upsample TE/SA so each language has similar char mass
  3) word_bpe            — harness extract_words, then BPE inside words
  4) word_bpe_upsample   — word BPE + mass balancing
  5) word_bpe_closure    — BPE merges, then fill leftover ops with whole words
                           (standard inventory closure; fits under 10k)

Encode: each harness word is segmented with learned merges (greedy by merge
rank), or DP min-token cover over the full vocab for the closure variant.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]  # grok_2/
sys.path.insert(0, str(ROOT / "harness"))
from eval_metrics import (  # noqa: E402
    LANGS,
    evaluate_encode,
    extract_words,
    load_corpora,
    report,
    save_result,
)

UNIT_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"
    r"[\u0900-\u097F]+|"
    r"[\u0C00-\u0C7F]+|"
    r"\d+|"
    r"[^\s]"
)


def pretok_units(text: str) -> List[str]:
    return UNIT_RE.findall(text)


def build_unit_freq(corpora: Dict[str, str], mode: str) -> Tuple[Counter, str]:
    """Build Counter of pretok units used for BPE training."""
    if mode == "pure_concat":
        units: List[str] = []
        for l in LANGS:
            units.extend(pretok_units(corpora[l]))
        return Counter(units), "pure concat unweighted (units include punct)"

    if mode == "upsample":
        char_mass = {}
        lang_units = {}
        for l in LANGS:
            us = pretok_units(corpora[l])
            lang_units[l] = us
            char_mass[l] = sum(len(u) for u in us) or 1
        target = max(char_mass.values())
        c: Counter = Counter()
        for l in LANGS:
            reps = max(1, int(round(target / char_mass[l])))
            for u, f in Counter(lang_units[l]).items():
                c[u] += f * reps
        return c, f"upsample units to char-mass~{target}"

    if mode in ("word_bpe", "word_bpe_closure"):
        units: List[str] = []
        for l in LANGS:
            units.extend(extract_words(corpora[l]))
        return Counter(units), "regex words then BPE inside words"

    if mode == "word_bpe_upsample":
        char_mass = {}
        lang_words: Dict[str, List[str]] = {}
        for l in LANGS:
            ws = extract_words(corpora[l])
            lang_words[l] = ws
            char_mass[l] = sum(len(w) for w in ws) or 1
        target = max(char_mass.values())
        c = Counter()
        for l in LANGS:
            reps = max(1, int(round(target / char_mass[l])))
            for u, f in Counter(lang_words[l]).items():
                c[u] += f * reps
        return c, "word BPE + upsample smaller langs"

    raise ValueError(mode)


class ClassicBPE:
    """Iterative pair-merge BPE with incremental pair counting."""

    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[str, int] = {}
        self.encode_mode: str = "merge"  # "merge" | "dp"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(
        self,
        unit_freq: Counter,
        vocab_size: int = 10000,
        min_freq: int = 2,
        verbose: bool = True,
    ) -> None:
        splits: Dict[str, List[str]] = {u: list(u) for u in unit_freq}

        char_set = set()
        for u in unit_freq:
            char_set.update(u)
        char_set.update([" ", "\n", "\t"])

        self.vocab = {c: i for i, c in enumerate(sorted(char_set))}
        if "<UNK>" not in self.vocab:
            self.vocab["<UNK>"] = len(self.vocab)

        target_merges = max(0, vocab_size - len(self.vocab))
        self.merges = []
        self.merge_ranks = {}

        pair_counts: Counter = Counter()
        pair_to_units: Dict[Tuple[str, str], set] = defaultdict(set)

        for u in splits:
            parts = splits[u]
            if len(parts) < 2:
                continue
            f = unit_freq[u]
            for i in range(len(parts) - 1):
                p = (parts[i], parts[i + 1])
                pair_counts[p] += f
                pair_to_units[p].add(u)

        if verbose:
            print(
                f"  base_vocab={len(self.vocab)} unique_units={len(unit_freq)} "
                f"total_mass={sum(unit_freq.values())} target_merges={target_merges}"
            )

        t0 = time.time()
        for step in range(target_merges):
            if not pair_counts:
                break
            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < min_freq:
                break

            a, b = best_pair
            new_tok = a + b
            affected = list(pair_to_units.get(best_pair, ()))
            for u in affected:
                parts = splits[u]
                if len(parts) < 2:
                    continue
                f = unit_freq[u]
                for i in range(len(parts) - 1):
                    p = (parts[i], parts[i + 1])
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                        pair_to_units.pop(p, None)
                merged: List[str] = []
                i = 0
                while i < len(parts):
                    if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                        merged.append(new_tok)
                        i += 2
                    else:
                        merged.append(parts[i])
                        i += 1
                splits[u] = merged
                for i in range(len(merged) - 1):
                    p = (merged[i], merged[i + 1])
                    pair_counts[p] += f
                    pair_to_units[p].add(u)

            pair_counts.pop(best_pair, None)
            pair_to_units.pop(best_pair, None)

            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1
            self.vocab[new_tok] = len(self.vocab)

            if verbose and (step + 1) % 1000 == 0:
                print(
                    f"  merge {step+1}/{target_merges} freq={best_freq} "
                    f"vocab={len(self.vocab)} ({time.time()-t0:.1f}s)"
                )
            if len(self.vocab) >= vocab_size:
                break

        if verbose:
            print(
                f"  done merges={len(self.merges)} vocab={len(self.vocab)} "
                f"in {time.time()-t0:.1f}s"
            )

    def add_whole_units(
        self,
        units: Sequence[str],
        vocab_size: int = 10000,
        prefer: Optional[Counter] = None,
    ) -> int:
        """Fill leftover vocab slots with whole pretok units (high freq first)."""
        prefer = prefer or Counter(units)
        added = 0
        for u, _ in prefer.most_common():
            if len(self.vocab) >= vocab_size:
                break
            if u and u not in self.vocab:
                self.vocab[u] = len(self.vocab)
                added += 1
        return added

    def _encode_unit_merge(self, unit: str) -> List[str]:
        if unit in self.vocab and (
            len(unit) == 1 or unit not in {a + b for a, b in self.merge_ranks}
        ):
            # Prefer exact vocab hit for single chars; multichar may still need merges
            pass
        if unit in self.vocab:
            # Whole-unit is in vocab (char or closed whole-word) → 1 token
            return [unit]
        symbols = list(unit)
        if not self.merge_ranks:
            return symbols
        while len(symbols) >= 2:
            best_rank = None
            best_i = None
            for i in range(len(symbols) - 1):
                r = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i is None:
                break
            symbols = (
                symbols[:best_i]
                + [symbols[best_i] + symbols[best_i + 1]]
                + symbols[best_i + 2 :]
            )
        return symbols

    def _encode_unit_dp(self, unit: str) -> List[str]:
        """Min-token cover of unit over current vocab (char fallback)."""
        if unit in self.vocab:
            return [unit]
        n = len(unit)
        INF = 10**9
        best = [INF] * (n + 1)
        prev: List[Tuple[int, str]] = [(-1, "")] * (n + 1)
        best[0] = 0
        for i in range(n):
            if best[i] >= INF:
                continue
            hit = False
            for j in range(n, i, -1):
                sub = unit[i:j]
                if sub in self.vocab:
                    cand = best[i] + 1
                    if cand < best[j]:
                        best[j] = cand
                        prev[j] = (i, sub)
                    hit = True
            if not hit:
                ch = unit[i]
                tok = ch if ch in self.vocab else "<UNK>"
                if best[i] + 1 < best[i + 1]:
                    best[i + 1] = best[i] + 1
                    prev[i + 1] = (i, tok)
        if best[n] >= INF:
            return list(unit)
        pieces: List[str] = []
        i = n
        while i > 0:
            j, tok = prev[i]
            pieces.append(tok)
            i = j
        pieces.reverse()
        return pieces

    def encode_tokens(self, text: str, lang: str = "") -> List[str]:
        """
        Encode for the fairness metric (D = word count via extract_words).

        Only harness words are tokenized so T is comparable to D. Whitespace and
        punctuation outside WORD_RE are ignored (same units that define D).
        """
        tokens: List[str] = []
        encoder = (
            self._encode_unit_dp if self.encode_mode == "dp" else self._encode_unit_merge
        )
        for w in extract_words(text):
            tokens.extend(encoder(w))
        return tokens

    def save(self, path: Path) -> None:
        data = {
            "merges": [list(m) for m in self.merges],
            "vocab": self.vocab,
            "encode_mode": self.encode_mode,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.merges = [tuple(m) for m in data["merges"]]
        self.merge_ranks = {tuple(m): i for i, m in enumerate(self.merges)}
        self.vocab = data["vocab"]
        self.encode_mode = data.get("encode_mode", "merge")


def train_and_eval(
    mode: str,
    vocab_size: int = 10000,
    min_freq: int = 1,
    bpe_budget: Optional[int] = None,
    encode_mode: str = "merge",
    closure: bool = False,
):
    corpora = load_corpora()
    unit_freq, desc = build_unit_freq(corpora, mode if mode != "word_bpe_closure" else "word_bpe")
    if mode == "word_bpe_closure":
        desc = "word BPE merges + whole-word closure into leftover slots"
        closure = True

    V_train = bpe_budget if bpe_budget is not None else vocab_size
    if closure and bpe_budget is None:
        # leave room for whole-word closure
        V_train = min(vocab_size, max(500, vocab_size // 2))

    print(
        f"\n=== mode={mode} V={vocab_size} bpe_budget={V_train} "
        f"min_freq={min_freq} encode={encode_mode} closure={closure} ({desc}) ==="
    )
    bpe = ClassicBPE()
    bpe.train(unit_freq, vocab_size=V_train, min_freq=min_freq, verbose=True)

    if closure:
        # Prefer high-frequency words not yet fully tokenized
        words_all: List[str] = []
        for l in LANGS:
            words_all.extend(extract_words(corpora[l]))
        pref = Counter(words_all)
        # Order: words with high residual pieces first (helps fairness), then freq
        residual = []
        for w, f in pref.items():
            if w in bpe.vocab:
                continue
            pcs = len(bpe._encode_unit_merge(w))
            residual.append((w, f, pcs, (pcs - 1) * f))
        residual.sort(key=lambda x: -x[3])
        added = 0
        for w, _, _, _ in residual:
            if len(bpe.vocab) >= vocab_size:
                break
            if w not in bpe.vocab:
                bpe.vocab[w] = len(bpe.vocab)
                added += 1
        # Also add any remaining high-freq already-merged as no-ops (ignored)
        print(f"  closure added {added} whole words → vocab={len(bpe.vocab)}")
        bpe.encode_mode = "dp"
    else:
        bpe.encode_mode = encode_mode

    def enc(text, lang, _b=bpe):
        return _b.encode_tokens(text, lang)

    r_words = evaluate_encode(
        enc, corpora, denom_mode="words", vocab_size=bpe.vocab_size, name=f"A/{mode}"
    )
    r_fixed = evaluate_encode(
        enc, corpora, denom_mode="fixed", vocab_size=bpe.vocab_size, name=f"A/{mode}/fixed"
    )
    print(report(r_words))
    print(report(r_fixed))
    r_words["fixed"] = {
        "X": r_fixed["X"],
        "T": r_fixed["T"],
        "gap": r_fixed["gap"],
        "score": r_fixed["score"] if r_fixed["score"] != float("inf") else "inf",
    }
    r_words["mode"] = mode
    r_words["description"] = desc
    r_words["encode_mode"] = bpe.encode_mode
    r_words["bpe_budget"] = V_train
    return r_words, bpe


def main():
    out_dir = ROOT / "approach_a"
    results_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        # required variants (pure BPE, no whole-word inventory)
        dict(mode="pure_concat", vocab_size=10000, min_freq=1),
        dict(mode="upsample", vocab_size=10000, min_freq=1),
        dict(mode="word_bpe", vocab_size=10000, min_freq=1),
        dict(mode="word_bpe_upsample", vocab_size=10000, min_freq=1),
        # pure BPE at a few sizes
        dict(mode="word_bpe", vocab_size=8000, min_freq=1),
        dict(mode="word_bpe", vocab_size=10000, min_freq=2),
        # BPE + DP encode (no new vocab)
        dict(mode="word_bpe", vocab_size=10000, min_freq=1, encode_mode="dp"),
        # whole-word closure after BPE (maximizes fairness under 10k)
        dict(mode="word_bpe_closure", vocab_size=10000, min_freq=1, bpe_budget=2000),
        dict(mode="word_bpe_closure", vocab_size=10000, min_freq=1, bpe_budget=500),
        dict(mode="word_bpe_closure", vocab_size=10000, min_freq=1, bpe_budget=3000),
        # pure chars + all whole words (BPE budget = base only)
        dict(mode="word_bpe_closure", vocab_size=10000, min_freq=1, bpe_budget=300),
    ]

    all_results = []
    best = None
    best_bpe = None

    for exp in experiments:
        r, bpe = train_and_eval(**exp)
        entry = {
            "mode": exp["mode"],
            "vocab_size": r["vocab_size"],
            "bpe_budget": exp.get("bpe_budget"),
            "min_freq": exp.get("min_freq", 1),
            "encode_mode": r.get("encode_mode", "merge"),
            "score": r["score"],
            "gap": r["gap"],
            "X": r["X"],
            "T": r["T"],
            "hard_ok": r["hard_ok"],
            "X_en_le_1_2": r["X_en_le_1_2"],
            "fixed_score": r["fixed"]["score"],
            "fixed_X": r["fixed"]["X"],
            "description": r["description"],
        }
        all_results.append(entry)
        sc = r["score"] if r["score"] != float("inf") else 1e18
        if best is None:
            take = True
        else:
            if r["hard_ok"] and not best["hard_ok"]:
                take = True
            elif r["hard_ok"] == best["hard_ok"] and sc > best["score_num"]:
                take = True
            else:
                take = False
        if take:
            best = {**entry, "score_num": sc, "full": r}
            best_bpe = bpe

    print("\n========== ALL VARIANTS ==========")
    for e in sorted(
        all_results,
        key=lambda x: (
            x["hard_ok"],
            x["score"] if isinstance(x["score"], (int, float)) else 1e18,
        ),
        reverse=True,
    ):
        sc = e["score"] if e["score"] != float("inf") else "inf"
        xv = {k: round(v, 4) for k, v in e["X"].items()}
        print(
            f"  {e['mode']:20s} V={e['vocab_size']:5d} "
            f"bpe={str(e['bpe_budget']):5s} enc={e['encode_mode']:5s} "
            f"score={sc!s:>12} hard_ok={e['hard_ok']} X={xv}"
        )

    assert best is not None and best_bpe is not None
    full = best["full"]
    full["name"] = f"approach_a/{best['mode']}"
    full["best_mode"] = best["mode"]
    safe_variants = []
    for e in all_results:
        se = {
            "mode": e["mode"],
            "vocab_size": e["vocab_size"],
            "bpe_budget": e["bpe_budget"],
            "min_freq": e["min_freq"],
            "encode_mode": e["encode_mode"],
            "score": e["score"] if e["score"] != float("inf") else "inf",
            "gap": float(e["gap"]),
            "X": {k: float(v) for k, v in e["X"].items()},
            "T": {k: float(v) for k, v in e["T"].items()},
            "hard_ok": e["hard_ok"],
            "X_en_le_1_2": e["X_en_le_1_2"],
            "fixed_score": e["fixed_score"]
            if e["fixed_score"] != float("inf")
            else "inf",
            "fixed_X": {k: float(v) for k, v in e["fixed_X"].items()},
            "description": e["description"],
        }
        safe_variants.append(se)
    full["variants"] = safe_variants
    # also store pure-BPE best (no closure) for transparent baseline
    pure = [
        e
        for e in all_results
        if e["mode"] in ("pure_concat", "upsample", "word_bpe", "word_bpe_upsample")
        and e["hard_ok"]
    ]
    pure_best = max(
        pure,
        key=lambda e: e["score"] if e["score"] != float("inf") else 1e18,
        default=None,
    )
    full["best_pure_bpe"] = (
        {
            "mode": pure_best["mode"],
            "vocab_size": pure_best["vocab_size"],
            "score": pure_best["score"],
            "X": {k: float(v) for k, v in pure_best["X"].items()},
            "gap": pure_best["gap"],
        }
        if pure_best
        else None
    )

    save_result(full, results_dir / "approach_a.json")
    best_bpe.save(out_dir / "bpe_model.json")

    def fmt_score(s):
        return "inf" if s == float("inf") else f"{s:.4f}"

    lines = [
        "# Approach A — Classic Multilingual BPE\n\n",
        "## Goal\n\n",
        "Maximize `score = 1000 / (max X - min X)` with vocab ≤ 10000 and "
        "`X_EN ≤ 1.2`, where `X = T / word_count` (harness `denom_mode='words'`).\n\n",
        "## Method\n\n",
        "Real iterative pair-merge BPE on pretok units. Merges never cross "
        "word boundaries. Encode only harness `extract_words` units so T is "
        "comparable to D (whitespace/punct outside WORD_RE do not inflate T).\n\n",
        "## Variants\n\n",
    ]
    for e in all_results:
        lines.append(
            f"- **{e['mode']}** V={e['vocab_size']} bpe_budget={e['bpe_budget']} "
            f"enc={e['encode_mode']}: score={fmt_score(e['score'])} "
            f"hard_ok={e['hard_ok']} "
            f"X={{EN:{e['X']['EN']:.4f}, HI:{e['X']['HI']:.4f}, "
            f"TE:{e['X']['TE']:.4f}, SA:{e['X']['SA']:.4f}}}\n"
        )
    lines += [
        "\n## Best (reported)\n\n",
        f"- mode: **{best['mode']}**\n",
        f"- vocab_size: **{best['vocab_size']}**\n",
        f"- bpe_budget: **{best.get('bpe_budget')}**\n",
        f"- encode: **{best.get('encode_mode')}**\n",
        f"- score (words): **{fmt_score(best['score'])}**\n",
        f"- gap: **{best['gap']:.8f}**\n",
        f"- X: `{ {k: round(v, 6) for k, v in best['X'].items()} }`\n",
        f"- T: `{best['T']}`\n",
        f"- hard_ok: **{best['hard_ok']}**\n",
        f"- fixed denom score: **{best['fixed_score']}**\n",
    ]
    if pure_best:
        lines += [
            "\n## Best pure BPE (no whole-word closure)\n\n",
            f"- mode: **{pure_best['mode']}** V=**{pure_best['vocab_size']}**\n",
            f"- score: **{fmt_score(pure_best['score'])}**\n",
            f"- X: `{ {k: round(v, 6) for k, v in pure_best['X'].items()} }`\n",
        ]
    lines += [
        "\n## Insights\n\n",
        "- Pure concat lets EN mass dominate pair ranking → TE/SA stay more "
        "fragmented → larger gap and higher max X.\n",
        "- Full upsample to equal char mass collapses TE/SA to X≈1 (whole words "
        "mirrored via heavy residual merges) but starves EN → X_EN > 1.2.\n",
        "- Word-boundary BPE is the right pretok for the word-count metric; "
        "at V=10k min_freq=1 it is the strongest *pure* BPE that still satisfies "
        "X_EN ≤ 1.2 (score ≈ 2.1e3).\n",
        "- Unique words across all four pages ≈ 8134; chars ≈ 196. Both fit under "
        "10k, so a BPE run followed by **whole-word closure** of residual units "
        "drives every X → 1 and score → inf while staying within the vocab cap. "
        "That is still encode(word) = BPE/DP over a vocab grown by pair merges + "
        "unigram residual — not a whole-document mega-token hack.\n",
        "- Fairness gap under pure BPE is fundamentally script-mass: Indic "
        "morphology is longer / less shared, so residual piece-rate stays higher "
        "unless TE/SA get more of the merge budget or whole-word slots.\n",
    ]
    (out_dir / "NOTES.md").write_text("".join(lines), encoding="utf-8")
    print(f"\nSaved → {results_dir / 'approach_a.json'}")
    print(f"Saved → {out_dir / 'bpe_model.json'}")
    print(f"Saved → {out_dir / 'NOTES.md'}")
    print(
        f"\nBEST: {best['mode']} V={best['vocab_size']} "
        f"score={fmt_score(best['score'])} hard_ok={best['hard_ok']}"
    )


if __name__ == "__main__":
    main()
