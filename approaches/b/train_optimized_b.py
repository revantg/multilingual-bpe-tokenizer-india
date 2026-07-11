#!/usr/bin/env python3
"""
Approach B (optimized, full_page corpus) — reference implementation.

Two-stage vocab under a hard cap V (default 10000):

  Stage 1 — BPE base  (size BPE_SIZE, default 2500)
      Same engine as Approach A (ClassicBPE): chars → pair merges.
      Optional language reweighting of unit frequencies (EN upweight slightly).

  Stage 2 — deficit-aware whole-word fill  (size V − |vocab after stage 1|)
      Phase A (fairness): while any language has X_L > tau (default 1.2),
          add the unused type that most reduces residual tokens for the
          language with the largest deficit  max(0, T_L − tau·D_L).
      Phase B (score): remaining seats by global savings
          score(w) = Σ_lang  α_lang · freq_lang(w) · max(0, pieces_bpe(w) − 1)

Encode: DP min-cover over current vocab (same as ClassicBPE encode_mode="dp").
If a whole word is in vocab → 1 token; else apply BPE merges on its characters.

This differs from Approach A, which spends the *entire* V on merges (no explicit
whole-word dump). Tunable knobs: V, BPE_SIZE, α_*, EN_MERGE_WEIGHT, TAU, fill mode.

Files:
  engine  → ../approach_a/train_bpe.py   (ClassicBPE)
  this    → train_optimized_b.py
  model   → vocab_optimized_fullpage.json
  corpus  → ../../full_page_md/load_full_page.py
  notes   → RESILIENCE_X_LE_1_2.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]  # grok_2
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "approaches" / "a"))
sys.path.insert(0, str(ROOT / "datasets" / "full_page_md"))

from eval_metrics import LANGS, extract_words, evaluate_encode  # noqa: E402
from load_full_page import load_corpora  # noqa: E402
from train_bpe import ClassicBPE  # noqa: E402


# ── knobs (defaults match optimized full-page expansion) ──────────────
DEFAULTS = {
    "vocab_cap": 10_000,       # hard shared V
    "bpe_size": 2_500,         # stage-1 budget (chars + merges)
    "en_merge_weight": 2.0,    # unit_freq multiplier for EN during BPE only
    "alpha": {"EN": 1.0, "HI": 1.0, "TE": 1.0, "SA": 1.0},  # word-fill weights
    "min_freq": 1,
    "tau": 1.2,                # phase-A target: pull every X_L ≤ tau
    "fill_mode": "deficit",    # "deficit" | "global" (legacy pure savings)
}


def stage1_bpe(
    words_by: Dict[str, List[str]],
    bpe_size: int,
    en_merge_weight: float,
    min_freq: int,
) -> ClassicBPE:
    """Train ClassicBPE up to bpe_size with optional EN upweight."""
    unit_freq: Counter = Counter()
    freq = {L: Counter(words_by[L]) for L in LANGS}
    for L in LANGS:
        w = en_merge_weight if L == "EN" else 1.0
        for tok, c in freq[L].items():
            unit_freq[tok] += int(c * w)

    bpe = ClassicBPE()
    bpe.train(unit_freq, vocab_size=bpe_size, min_freq=min_freq, verbose=True)
    bpe.encode_mode = "dp"
    return bpe


def word_savings_scores(
    words_by: Dict[str, List[str]],
    bpe: ClassicBPE,
    alpha: Dict[str, float],
) -> Counter:
    """
    score(w) = sum_L α_L * freq_L(w) * max(0, pieces_bpe(w) - 1)

    pieces_bpe(w) = how many tokens ClassicBPE needs for w *before* we add w
    as a whole-word entry. Savings = what we gain if w becomes one token.
    """
    freq = {L: Counter(words_by[L]) for L in LANGS}
    all_types: Set[str] = set().union(*(set(ws) for ws in words_by.values()))
    pieces = {w: len(bpe._encode_unit_dp(w)) for w in all_types}

    scores: Counter = Counter()
    for L in LANGS:
        a = float(alpha.get(L, 1.0))
        for w, f in freq[L].items():
            scores[w] += a * f * max(0, pieces[w] - 1)
    return scores


def stage2_word_fill(
    bpe: ClassicBPE,
    scores: Counter,
    vocab_cap: int,
) -> int:
    """Legacy: insert highest-savings whole words until vocab_cap. Returns #added."""
    added = 0
    for w, _ in scores.most_common():
        if len(bpe.vocab) >= vocab_cap:
            break
        if not w or w in bpe.vocab:
            continue
        bpe.vocab[w] = len(bpe.vocab)
        added += 1
    return added


def _pieces_current(bpe: ClassicBPE, w: str, pieces_cache: Dict[str, int]) -> int:
    """1 if whole word already in vocab, else DP/BPE piece count (cached)."""
    if w in bpe.vocab:
        return 1
    if w not in pieces_cache:
        pieces_cache[w] = len(bpe._encode_unit_dp(w))
    return pieces_cache[w]


def stage2_deficit_fill(
    bpe: ClassicBPE,
    words_by: Dict[str, List[str]],
    alpha: Dict[str, float],
    vocab_cap: int,
    tau: float = 1.2,
) -> dict:
    """
    Phase A — while any X_L > tau and seats remain, add the type that most
    reduces residual tokens for the worst language L* = argmax residual
        residual_L = max(0, T_L − tau · D_L)
    Gain for candidate w wrt L*:
        gain = freq_L*(w) · max(0, pieces(w) − 1)
    Tie-break: global α-weighted savings across all langs.

    Phase B — remaining seats by global α-weighted savings (same as legacy).

    Returns stats dict: phase_a_added, phase_b_added, per_lang phase-A adds, etc.
    """
    freq = {L: Counter(words_by[L]) for L in LANGS}
    D = {L: sum(freq[L].values()) for L in LANGS}
    all_types: Set[str] = set().union(*(set(freq[L]) for L in LANGS))

    # pieces after stage-1 (static until word is added → then treated as 1)
    pieces_cache: Dict[str, int] = {
        w: len(bpe._encode_unit_dp(w)) for w in all_types if w not in bpe.vocab
    }

    def T_of(L: str) -> int:
        t = 0
        for w, f in freq[L].items():
            t += f * _pieces_current(bpe, w, pieces_cache)
        return t

    def residual(L: str, T_L: Optional[int] = None) -> float:
        t = T_of(L) if T_L is None else T_L
        return max(0.0, float(t) - tau * float(D[L]))

    # running T for incremental updates
    T_run = {L: T_of(L) for L in LANGS}
    phase_a_by_lang = {L: 0 for L in LANGS}
    phase_a_added = 0
    phase_b_added = 0

    # Precompute per-lang save potential for each type (static pieces after stage1)
    # gain_L[w] = f_L(w) * max(0, p0(w)-1)
    gain_L: Dict[str, Dict[str, float]] = {L: {} for L in LANGS}
    global_score: Dict[str, float] = {}
    for w in all_types:
        if not w or w in bpe.vocab:
            continue
        p0 = pieces_cache.get(w, 1)
        sav = max(0, p0 - 1)
        gs = 0.0
        for L in LANGS:
            g = float(freq[L].get(w, 0)) * sav
            gain_L[L][w] = g
            gs += float(alpha.get(L, 1.0)) * g
        global_score[w] = gs

    # ── Phase A: pull each language ≤ tau ────────────────────────────
    while len(bpe.vocab) < vocab_cap:
        residuals = {L: residual(L, T_run[L]) for L in LANGS}
        if all(r <= 1e-9 for r in residuals.values()):
            break
        # worst language by residual (tokens over tau·D); prefer higher residual
        L_star = max(LANGS, key=lambda L: (residuals[L], T_run[L] / max(D[L], 1)))
        if residuals[L_star] <= 1e-9:
            break

        # best word for L_star among types still multi-piece for that lang
        best_w = None
        best_key = None  # (gain_L*, global_score, -len)
        for w, g in gain_L[L_star].items():
            if w in bpe.vocab or g <= 0:
                continue
            key = (g, global_score.get(w, 0.0), -len(w))
            if best_key is None or key > best_key:
                best_key = key
                best_w = w
        if best_w is None:
            # No more multi-piece types on L_star — cannot improve further
            # Drop L_star from consideration by zeroing residual via marking
            # (break only if no language has improvable residual)
            # Try next-best langs: skip by zeroing gains artificially for loop safety
            any_other = False
            for L in LANGS:
                if L == L_star or residuals[L] <= 1e-9:
                    continue
                for w, g in gain_L[L].items():
                    if w not in bpe.vocab and g > 0:
                        any_other = True
                        break
                if any_other:
                    # force pick that L next by temporarily treating L_star residual as 0
                    T_run[L_star] = min(T_run[L_star], int(tau * D[L_star]))
                    break
            if not any_other:
                break
            continue

        # Add whole-word
        bpe.vocab[best_w] = len(bpe.vocab)
        phase_a_added += 1
        phase_a_by_lang[L_star] += 1
        # Incremental T update: for each lang, was pieces, now 1
        old_p = pieces_cache.get(best_w, 1)
        for L in LANGS:
            f = freq[L].get(best_w, 0)
            if f:
                T_run[L] -= f * max(0, old_p - 1)
        pieces_cache[best_w] = 1  # now 1 token

    # ── Phase B: global savings fill ─────────────────────────────────
    ranked = sorted(
        ((global_score.get(w, 0.0), w) for w in all_types if w not in bpe.vocab and w),
        reverse=True,
    )
    for sc, w in ranked:
        if len(bpe.vocab) >= vocab_cap:
            break
        if sc <= 0 and w in bpe.vocab:
            continue
        if w in bpe.vocab:
            continue
        # Match legacy: add remaining types by score order (zeros last).
        bpe.vocab[w] = len(bpe.vocab)
        phase_b_added += 1

    # Fill any leftover with remaining types lexicographically (legacy parity)
    if len(bpe.vocab) < vocab_cap:
        for w in sorted(all_types):
            if len(bpe.vocab) >= vocab_cap:
                break
            if w and w not in bpe.vocab:
                bpe.vocab[w] = len(bpe.vocab)
                phase_b_added += 1

    return {
        "phase_a_added": phase_a_added,
        "phase_b_added": phase_b_added,
        "phase_a_by_lang": phase_a_by_lang,
        "tau": tau,
        "T_after_phase_a_est": {L: int(T_run[L]) for L in LANGS},
        "X_after_phase_a_est": {
            L: float(T_run[L]) / max(D[L], 1) for L in LANGS
        },
        "words_added_stage2": phase_a_added + phase_b_added,
    }


def train_optimized_b(
    corpora: Optional[Dict[str, str]] = None,
    vocab_cap: int = DEFAULTS["vocab_cap"],
    bpe_size: int = DEFAULTS["bpe_size"],
    en_merge_weight: float = DEFAULTS["en_merge_weight"],
    alpha: Optional[Dict[str, float]] = None,
    min_freq: int = DEFAULTS["min_freq"],
    tau: float = DEFAULTS["tau"],
    fill_mode: str = DEFAULTS["fill_mode"],
) -> Tuple[ClassicBPE, dict]:
    corpora = corpora or load_corpora()
    alpha = alpha or dict(DEFAULTS["alpha"])
    words_by = {L: extract_words(corpora[L]) for L in LANGS}

    # clamp: leave at least ~200 slots? allow bpe_size to eat up to cap
    bpe_size = max(200, min(bpe_size, vocab_cap))

    bpe = stage1_bpe(words_by, bpe_size, en_merge_weight, min_freq)

    fill_stats: dict = {}
    if fill_mode == "global":
        scores = word_savings_scores(words_by, bpe, alpha)
        added = stage2_word_fill(bpe, scores, vocab_cap)
        fill_stats = {
            "fill_mode": "global",
            "words_added_stage2": added,
            "phase_a_added": 0,
            "phase_b_added": added,
        }
    else:
        fill_stats = stage2_deficit_fill(
            bpe, words_by, alpha, vocab_cap, tau=tau
        )
        fill_stats["fill_mode"] = "deficit"

    bpe.encode_mode = "dp"

    def encode_fn(text: str, lang: str = "") -> List[str]:
        return bpe.encode_tokens(text, lang)

    metrics = evaluate_encode(
        encode_fn, corpora, "words", bpe.vocab_size, name="B_optimized"
    )
    hard_ok_all = all(float(metrics["X"][L]) <= tau + 1e-12 for L in LANGS)
    meta = {
        "vocab_cap": vocab_cap,
        "bpe_size_requested": bpe_size,
        "bpe_vocab_after_stage1_approx": bpe_size,  # train stops near this
        "merges": len(bpe.merges),
        "words_added_stage2": int(fill_stats.get("words_added_stage2", 0)),
        "fill_mode": fill_stats.get("fill_mode", fill_mode),
        "tau": tau,
        "phase_a_added": int(fill_stats.get("phase_a_added", 0)),
        "phase_b_added": int(fill_stats.get("phase_b_added", 0)),
        "phase_a_by_lang": fill_stats.get("phase_a_by_lang", {}),
        "final_vocab": bpe.vocab_size,
        "alphas": alpha,
        "en_merge_weight": en_merge_weight,
        "X_words": {k: float(v) for k, v in metrics["X"].items()},
        "T": {k: int(v) for k, v in metrics["T"].items()},
        "score_words": metrics["score"]
        if metrics["score"] != float("inf")
        else "inf",
        "hard_ok_words": bool(metrics["hard_ok"]),  # X_EN ≤ 1.2 (assignment)
        "hard_ok_all_langs": hard_ok_all,           # every X_L ≤ tau
        "gap_words": float(metrics["gap"]),
    }
    return bpe, meta


def save_model(bpe: ClassicBPE, meta: dict, path: Path) -> None:
    payload = {
        "merges": [list(m) for m in bpe.merges],
        "vocab": bpe.vocab,
        "encode_mode": bpe.encode_mode,
        "meta": meta,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vocab-cap", type=int, default=DEFAULTS["vocab_cap"])
    p.add_argument("--bpe-size", type=int, default=DEFAULTS["bpe_size"])
    p.add_argument("--en-merge-weight", type=float, default=DEFAULTS["en_merge_weight"])
    p.add_argument("--alpha-en", type=float, default=1.0)
    p.add_argument("--alpha-hi", type=float, default=1.0)
    p.add_argument("--alpha-te", type=float, default=1.0)
    p.add_argument("--alpha-sa", type=float, default=1.0)
    p.add_argument(
        "--tau",
        type=float,
        default=DEFAULTS["tau"],
        help="Phase-A target: pull every language X_L ≤ tau (default 1.2)",
    )
    p.add_argument(
        "--fill-mode",
        choices=("deficit", "global"),
        default=DEFAULTS["fill_mode"],
        help="deficit = worst-lang residual fill then savings; global = legacy savings only",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "vocab_optimized_fullpage.json",
    )
    args = p.parse_args()
    alpha = {
        "EN": args.alpha_en,
        "HI": args.alpha_hi,
        "TE": args.alpha_te,
        "SA": args.alpha_sa,
    }
    bpe, meta = train_optimized_b(
        vocab_cap=args.vocab_cap,
        bpe_size=args.bpe_size,
        en_merge_weight=args.en_merge_weight,
        alpha=alpha,
        tau=args.tau,
        fill_mode=args.fill_mode,
    )
    save_model(bpe, meta, args.out)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
