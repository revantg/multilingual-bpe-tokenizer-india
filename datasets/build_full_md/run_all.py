#!/usr/bin/env python3
"""
Full-corpus retrain/eval of tokenizer approaches A–E + champion on full_md pages.

ONLY corpus:
  data/full_md/india_{en,hi,te,sa}.md  via full_md/load_full.py

Metrics (BOTH denoms):
  A) words: X_i = T_i / word_count_i ; score = 1000/(maxX-minX)
     hard_ok if X_EN<=1.2 and vocab<=10000
  B) fixed: X_i = T_i / 5000 ; X_EN<=1.2 hard; same score form
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

FULL_RERUN = Path(__file__).resolve().parent
GROK2 = FULL_RERUN.parent
DATA = GROK2.parent
FULL_MD = GROK2 / "datasets" / "full_md"
RESULTS = FULL_RERUN / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FULL_MD))
sys.path.insert(0, str(GROK2 / "harness"))
sys.path.insert(0, str(GROK2 / "approaches" / "a"))
sys.path.insert(0, str(GROK2 / "approaches" / "b"))
sys.path.insert(0, str(GROK2 / "approaches" / "c"))
sys.path.insert(0, str(GROK2 / "approaches" / "d"))
sys.path.insert(0, str(GROK2 / "approaches" / "e"))
sys.path.insert(0, str(GROK2 / "approaches" / "champion"))

from load_full import load_corpora  # noqa: E402
from eval_metrics import (  # noqa: E402
    LANGS,
    denominators,
    evaluate_encode,
    extract_words,
    report,
)

VOCAB_MAX = 10_000
SEP = "\u29f5"


# ── result helpers ────────────────────────────────────────────────────

def both_denoms(
    encode_fn: Callable[[str, str], Sequence],
    corpora: Dict[str, str],
    vocab_size: int,
    name: str,
) -> Dict:
    rw = evaluate_encode(
        encode_fn, corpora, denom_mode="words", vocab_size=vocab_size, name=name
    )
    rf = evaluate_encode(
        encode_fn, corpora, denom_mode="fixed", vocab_size=vocab_size, name=name
    )
    return {
        "name": name,
        "vocab_size": vocab_size,
        "T": {k: int(rw["T"][k]) for k in LANGS},
        "D_words": {k: float(rw["D"][k]) for k in LANGS},
        "D_fixed": {k: float(rf["D"][k]) for k in LANGS},
        "X_words": {k: float(rw["X"][k]) for k in LANGS},
        "X_fixed": {k: float(rf["X"][k]) for k in LANGS},
        "score_words": rw["score"] if rw["score"] != float("inf") else "inf",
        "score_fixed": rf["score"] if rf["score"] != float("inf") else "inf",
        "gap_words": float(rw["gap"]),
        "gap_fixed": float(rf["gap"]),
        "hard_ok_words": bool(rw["hard_ok"]),
        "hard_ok_fixed": bool(rf["hard_ok"]),
        "X_en_words": float(rw["X"]["EN"]),
        "X_en_fixed": float(rf["X"]["EN"]),
    }


def scrub(obj):
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if obj == float("inf"):
        return "inf"
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    return obj


def save_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(scrub(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path}")


# ── Approach A: pure word-interior BPE, NO whole-word closure ─────────

def run_approach_a(corpora: Dict[str, str]) -> Dict:
    print("\n" + "=" * 70)
    print("APPROACH A — pure word-interior BPE (no whole-word closure)")
    print("=" * 70)
    t0 = time.time()

    # Reuse ClassicBPE from approach_a
    from train_bpe import ClassicBPE

    unit_freq: Counter = Counter()
    for lang in LANGS:
        unit_freq.update(extract_words(corpora[lang]))

    bpe = ClassicBPE()
    bpe.train(unit_freq, vocab_size=VOCAB_MAX, min_freq=1, verbose=True)
    bpe.encode_mode = "merge"  # also try DP

    def enc_merge(text, lang, _b=bpe):
        return _b.encode_tokens(text, lang)

    res_merge = both_denoms(enc_merge, corpora, bpe.vocab_size, "A_pure_word_bpe_merge")
    print(f"  merge-encode: score_words={res_merge['score_words']} "
          f"X_words={res_merge['X_words']} hard={res_merge['hard_ok_words']}")

    # DP encode over same vocab (no new tokens)
    bpe_dp = ClassicBPE()
    bpe_dp.merges = list(bpe.merges)
    bpe_dp.merge_ranks = dict(bpe.merge_ranks)
    bpe_dp.vocab = dict(bpe.vocab)
    bpe_dp.encode_mode = "dp"

    def enc_dp(text, lang, _b=bpe_dp):
        return _b.encode_tokens(text, lang)

    res_dp = both_denoms(enc_dp, corpora, bpe_dp.vocab_size, "A_pure_word_bpe_dp")
    print(f"  dp-encode:    score_words={res_dp['score_words']} "
          f"X_words={res_dp['X_words']} hard={res_dp['hard_ok_words']}")

    # Prefer hard_ok then higher words score
    def key(r):
        sc = r["score_words"]
        scv = 1e300 if sc == "inf" else float(sc)
        return (1 if r["hard_ok_words"] else 0, scv)

    best = res_dp if key(res_dp) >= key(res_merge) else res_merge
    best = dict(best)
    best["name"] = "approach_A_pure"
    best["notes"] = (
        f"Classic word-interior BPE V={VOCAB_MAX}, NO whole-word closure. "
        f"Best encode={'dp' if best is res_dp or best.get('name','').endswith('dp') else 'merge'}. "
        f"Also ran merge+dp. elapsed={time.time()-t0:.1f}s. "
        f"merge_score={res_merge['score_words']} dp_score={res_dp['score_words']}"
    )
    best["variants"] = {"merge": res_merge, "dp": res_dp}
    best["elapsed_s"] = time.time() - t0

    # pick winner cleanly
    if key(res_dp) >= key(res_merge):
        out = dict(res_dp)
        out["encode_mode"] = "dp"
    else:
        out = dict(res_merge)
        out["encode_mode"] = "merge"
    out["name"] = "approach_A_pure"
    out["notes"] = (
        "Classic word-interior BPE, vocab=10000, NO whole-word closure. "
        f"Encode mode={out['encode_mode']}. Merges={len(bpe.merges)}. "
        f"merge score_words={res_merge['score_words']} dp score_words={res_dp['score_words']}. "
        f"elapsed={time.time()-t0:.1f}s"
    )
    out["variants"] = {
        "merge": {k: res_merge[k] for k in (
            "vocab_size", "T", "X_words", "X_fixed", "score_words", "score_fixed",
            "hard_ok_words", "hard_ok_fixed"
        )},
        "dp": {k: res_dp[k] for k in (
            "vocab_size", "T", "X_words", "X_fixed", "score_words", "score_fixed",
            "hard_ok_words", "hard_ok_fixed"
        )},
    }
    out["elapsed_s"] = time.time() - t0
    save_json(out, RESULTS / "A.json")
    return out


# ── Approach B: temperature + script budgets; merges-only AND whole-word fill

def run_approach_b(corpora: Dict[str, str]) -> Dict:
    print("\n" + "=" * 70)
    print("APPROACH B — fair/temp BPE (merges-only + whole-word fill)")
    print("=" * 70)
    t0 = time.time()
    from train_fair_bpe import FairBPE

    results = {}
    for label, cfg in [
        ("merges_only", dict(
            variant="minimax_temp",
            temperature=3.0,
            num_merges=4000,
            minimax_every=20,
            candidate_k=40,
            whole_word_fill=False,
            phrase_fill=False,
            budget={"EN": 0.30, "HI": 0.28, "TE": 0.27, "SA": 0.15},
        )),
        ("with_whole_word_fill", dict(
            variant="minimax_temp",
            temperature=3.0,
            num_merges=800,
            minimax_every=20,
            candidate_k=40,
            whole_word_fill=True,
            phrase_fill=False,
            budget={"EN": 0.30, "HI": 0.28, "TE": 0.27, "SA": 0.15},
        )),
    ]:
        print(f"\n--- B/{label} ---")
        tok = FairBPE()
        meta = tok.train(corpora, verbose=True, **cfg)

        def enc(text, lang, _t=tok):
            return _t.encode(text, lang)

        res = both_denoms(enc, corpora, len(tok.vocab), f"B_{label}")
        res["train_meta"] = {
            "merges": meta.get("merges"),
            "vocab_size": meta.get("vocab_size"),
            "temperature": meta.get("temperature"),
            "variant": meta.get("variant"),
            "elapsed_s": meta.get("elapsed_s"),
        }
        results[label] = res
        print(f"  score_words={res['score_words']} X={res['X_words']} hard={res['hard_ok_words']}")

    # Primary report: whole-word fill if better hard_ok+score, else merges_only
    def key(r):
        sc = r["score_words"]
        scv = 1e300 if sc == "inf" else float(sc)
        return (1 if r["hard_ok_words"] else 0, scv)

    primary_label = max(results, key=lambda k: key(results[k]))
    primary = dict(results[primary_label])
    primary["name"] = "approach_B_fair"
    primary["primary_variant"] = primary_label
    primary["notes"] = (
        "Temperature upsample + script budgets + minimax merges. "
        "Reports merges-only AND whole-word fill of remaining budget (V<=10000). "
        f"Primary={primary_label}. elapsed={time.time()-t0:.1f}s"
    )
    primary["variants"] = {
        k: {
            "vocab_size": v["vocab_size"],
            "T": v["T"],
            "X_words": v["X_words"],
            "X_fixed": v["X_fixed"],
            "score_words": v["score_words"],
            "score_fixed": v["score_fixed"],
            "hard_ok_words": v["hard_ok_words"],
            "hard_ok_fixed": v["hard_ok_fixed"],
            "train_meta": v.get("train_meta"),
        }
        for k, v in results.items()
    }
    # Also surface merges-only explicitly
    primary["merges_only"] = primary["variants"]["merges_only"]
    primary["with_whole_word_fill"] = primary["variants"]["with_whole_word_fill"]
    primary["elapsed_s"] = time.time() - t0
    save_json(primary, RESULTS / "B.json")
    return primary


# ── Approach C: unique words + chars; fair word quotas if over 10k ────

def run_approach_c(corpora: Dict[str, str]) -> Dict:
    print("\n" + "=" * 70)
    print("APPROACH C — whole-word inventory + char fallback")
    print("=" * 70)
    t0 = time.time()

    words_by_lang = {lang: extract_words(corpora[lang]) for lang in LANGS}
    freq_by_lang = {lang: Counter(words_by_lang[lang]) for lang in LANGS}
    all_words: Set[str] = set()
    for ws in words_by_lang.values():
        all_words.update(ws)
    chars: Set[str] = set()
    for text in corpora.values():
        chars.update(text)

    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    base_chars = set(chars) | set(specials)
    # Room for words after chars
    n_chars_needed = len(base_chars)  # will add unique chars
    room_for_words = VOCAB_MAX - n_chars_needed

    union_n = len(all_words)
    if union_n + n_chars_needed <= VOCAB_MAX:
        selected_words = set(all_words)
        coverage_mode = "all_unique_words_fit"
    else:
        # fair per-lang frequency quotas
        coverage_mode = "fair_freq_quotas"
        quota = max(0, room_for_words)
        base_q = quota // len(LANGS)
        rem = quota - base_q * len(LANGS)
        selected_words: Set[str] = set()
        per_lang_take: Dict[str, int] = {}
        # round-robin by frequency so shared words don't starve anyone
        pools = {
            lang: [w for w, _ in freq_by_lang[lang].most_common()]
            for lang in LANGS
        }
        idx = {lang: 0 for lang in LANGS}
        target = {lang: base_q + (1 if i < rem else 0) for i, lang in enumerate(LANGS)}
        taken = {lang: 0 for lang in LANGS}
        while len(selected_words) < quota:
            progressed = False
            for lang in LANGS:
                while idx[lang] < len(pools[lang]) and taken[lang] < target[lang] * 3:
                    w = pools[lang][idx[lang]]
                    idx[lang] += 1
                    if w in selected_words:
                        continue
                    selected_words.add(w)
                    taken[lang] += 1
                    progressed = True
                    break
                if len(selected_words) >= quota:
                    break
            if not progressed:
                # fill remaining by global freq
                global_f = Counter()
                for c in freq_by_lang.values():
                    global_f.update(c)
                for w, _ in global_f.most_common():
                    if w not in selected_words:
                        selected_words.add(w)
                    if len(selected_words) >= quota:
                        break
                break
        per_lang_take = taken

    vocab: Set[str] = set(selected_words)
    for s in specials:
        vocab.add(s)
    for c in sorted(chars):
        if len(vocab) >= VOCAB_MAX:
            break
        vocab.add(c)

    singles = set(selected_words)

    def encode(text, lang, _s=singles):
        out = []
        for w in extract_words(text):
            if w in _s:
                out.append(w)
            else:
                out.extend(list(w))  # char fallback
        return out

    # Coverage % of token occurrences that are single-token words
    cov = {}
    for lang in LANGS:
        ws = words_by_lang[lang]
        hit = sum(1 for w in ws if w in singles)
        cov[lang] = 100.0 * hit / len(ws) if ws else 100.0
    cov_unique = {}
    for lang in LANGS:
        u = set(words_by_lang[lang])
        hit = sum(1 for w in u if w in singles)
        cov_unique[lang] = 100.0 * hit / len(u) if u else 100.0

    res = both_denoms(encode, corpora, len(vocab), "approach_C_words")
    res["name"] = "approach_C_words"
    res["coverage_pct_token_occ"] = cov
    res["coverage_pct_unique"] = cov_unique
    res["words_selected"] = len(singles)
    res["union_unique_words"] = union_n
    res["chars_in_vocab"] = len(vocab) - len(singles) - len([s for s in specials if s in vocab])
    res["coverage_mode"] = coverage_mode
    res["notes"] = (
        f"Union unique words={union_n}; mode={coverage_mode}. "
        f"Words-as-single-token coverage (token occ %): {cov}. "
        f"Unique coverage %: {cov_unique}. vocab={len(vocab)}. elapsed={time.time()-t0:.1f}s"
    )
    res["elapsed_s"] = time.time() - t0
    print(f"  union={union_n} selected={len(singles)} vocab={len(vocab)} mode={coverage_mode}")
    print(f"  coverage_occ={cov}")
    print(f"  score_words={res['score_words']} X={res['X_words']}")
    save_json(res, RESULTS / "C.json")
    return res


# ── Approach D: equal-X slice packing oracle ──────────────────────────

def partition_even(n_words: int, n_spans: int) -> List[Tuple[int, int]]:
    n_spans = max(1, min(n_spans, max(1, n_words)))
    bounds = [round(i * n_words / n_spans) for i in range(n_spans + 1)]
    spans = []
    for i in range(n_spans):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            b = min(a + 1, n_words)
        spans.append((a, b))
    if spans:
        spans[-1] = (spans[-1][0], n_words)
    spans = [(a, b) for a, b in spans if b > a]
    if not spans:
        spans = [(0, max(1, n_words))] if n_words else [(0, 0)]
    return spans


def reconstruct_word_spans(text: str, words: List[str]) -> List[str]:
    spans: List[str] = []
    pos = 0
    for i, w in enumerate(words):
        idx = text.find(w, pos)
        if idx < 0:
            spans.append(w)
            continue
        if i + 1 < len(words):
            nxt = text.find(words[i + 1], idx + len(w))
            end = nxt if nxt >= 0 else idx + len(w)
        else:
            end = len(text)
        spans.append(text[idx:end] if end > idx else w)
        pos = idx + len(w)
    return spans


def build_slice(corpora: Dict[str, str], T: Dict[str, int]):
    planned: Dict[str, List[str]] = {}
    vocab: Set[str] = set()
    for lang in LANGS:
        page = corpora[lang]
        words = extract_words(page)
        n_spans = max(1, min(T[lang], max(1, len(words))))
        ranges = partition_even(len(words), n_spans)
        wcs = reconstruct_word_spans(page, words) if words else [page]
        bags = []
        for a, b in ranges:
            piece = "".join(wcs[a:b]) if words else page
            bags.append(piece)
            vocab.add(piece)
        planned[lang] = bags
        assert len(bags) == len(ranges), (lang, len(bags), len(ranges))
    return planned, vocab


def choose_equal_x_T(
    D: Dict[str, float],
    r: float = 0.05,
    window: int = 12,
    sum_cap: int = VOCAB_MAX,
) -> Dict[str, int]:
    base = {k: max(1, int(round(r * D[k]))) for k in LANGS}
    best_T = dict(base)
    best_gap = float("inf")
    keys = list(LANGS)
    ranges = []
    for k in keys:
        lo = max(1, base[k] - window)
        hi = min(int(D[k]), base[k] + window)
        ranges.append(range(lo, hi + 1))
    for combo in product(*ranges):
        Tmap = {k: combo[i] for i, k in enumerate(keys)}
        if sum(Tmap.values()) > sum_cap:
            continue
        xs = [Tmap[k] / D[k] for k in keys]
        g = max(xs) - min(xs)
        if g < best_gap - 1e-18:
            best_gap = g
            best_T = Tmap
        elif abs(g - best_gap) < 1e-18 and sum(Tmap.values()) < sum(best_T.values()):
            best_T = Tmap
    return best_T


def find_max_score_T(
    D: Dict[str, float], sum_cap: int = VOCAB_MAX
) -> Tuple[Dict[str, int], float, float]:
    """
    Exhaustive-ish equal-X integer T search:
      - dense r grid with T_i = round(r * D_i)
      - local ±window fine-tune on the best
    Returns (T, gap, score).
    """
    best_T = {l: 1 for l in LANGS}
    best_gap = float("inf")
    best_sc = 0.0

    def consider(Tmap: Dict[str, int]) -> None:
        nonlocal best_T, best_gap, best_sc
        if any(Tmap[l] < 1 or Tmap[l] > int(D[l]) for l in LANGS):
            return
        if sum(Tmap.values()) > sum_cap:
            return
        xs = [Tmap[l] / D[l] for l in LANGS]
        g = max(xs) - min(xs)
        sc = float("inf") if g < 1e-15 else 1000.0 / g
        scv = 1e300 if sc == float("inf") else sc
        bsv = 1e300 if best_sc == float("inf") else best_sc
        if scv > bsv + 1e-12 or (
            abs(scv - bsv) < 1e-12 and sum(Tmap.values()) < sum(best_T.values())
        ):
            best_T = dict(Tmap)
            best_gap = g
            best_sc = sc

    # dense r via hundredths of a percent
    for r_num in range(1, 12001):  # r up to 1.2
        r = r_num / 10000.0
        T = {l: max(1, int(round(r * D[l]))) for l in LANGS}
        consider(T)

    # proportional allocation over total span budget
    total_mass = sum(D.values())
    for tot in range(4, min(sum_cap, int(sum(D.values()))) + 1, 1 if sum_cap <= 3000 else 3):
        raw = {l: tot * D[l] / total_mass for l in LANGS}
        base = {l: max(1, int(round(raw[l]))) for l in LANGS}
        # fix sum toward tot
        while sum(base.values()) < tot:
            k = max(LANGS, key=lambda l: raw[l] - base[l])
            base[k] += 1
        while sum(base.values()) > tot and any(base[l] > 1 for l in LANGS):
            k = max((l for l in LANGS if base[l] > 1), key=lambda l: base[l] - raw[l])
            base[k] -= 1
        consider(base)

    # local refine ±8 around current best
    keys = list(LANGS)
    T0 = best_T
    for offs in product(range(-8, 9), repeat=4):
        T = {keys[i]: T0[keys[i]] + offs[i] for i in range(4)}
        consider(T)

    return best_T, best_gap, best_sc


def run_approach_d(corpora: Dict[str, str]) -> Dict:
    print("\n" + "=" * 70)
    print("APPROACH D — equal-X slice packing oracle")
    print("=" * 70)
    t0 = time.time()
    D = {lang: float(len(extract_words(corpora[lang]))) for lang in LANGS}
    print(f"  D_words={D}")

    best_T, best_gap_pred, best_sc_pred = find_max_score_T(D, sum_cap=VOCAB_MAX)
    print(f"  T* search → T={best_T} pred_gap={best_gap_pred:.8g} pred_score={best_sc_pred}")

    planned, vocab = build_slice(corpora, best_T)

    def enc(text, lang, _p=planned, _c=corpora):
        if text == _c[lang]:
            return list(_p[lang])
        return extract_words(text)

    res = both_denoms(enc, corpora, len(vocab), "approach_D_pack")
    res["T_design"] = best_T
    res["encode_matches_planned"] = all(
        res["T"][lang] == len(planned[lang]) for lang in LANGS
    )
    assert res["encode_matches_planned"], (res["T"], {l: len(planned[l]) for l in LANGS})

    # Also evaluate a few reference r points for the table notes
    all_r = []
    for r in [0.01, 0.02, 0.05, 0.1, 0.1084, 0.2, 1.0]:
        T = {l: max(1, int(round(r * D[l]))) for l in LANGS}
        if sum(T.values()) > VOCAB_MAX:
            continue
        p2, v2 = build_slice(corpora, T)

        def enc2(text, lang, _p=p2, _c=corpora):
            if text == _c[lang]:
                return list(_p[lang])
            return extract_words(text)

        r2 = both_denoms(enc2, corpora, len(v2), f"D_r{r}")
        all_r.append({
            "r": r,
            "T": T,
            "score_words": r2["score_words"],
            "gap_words": r2["gap_words"],
            "vocab": len(v2),
            "X_words": r2["X_words"],
            "hard_ok": r2["hard_ok_words"],
        })

    out = dict(res)
    out["name"] = "approach_D_pack"
    out["T_design"] = best_T
    out["notes"] = (
        "Equal-X contiguous word-span packing; exact planned encode of full pages. "
        f"T*={best_T} (max score under words denom via dense r + local integer search). "
        f"vocab_spans={len(vocab)}. pred_score={best_sc_pred}. elapsed={time.time()-t0:.1f}s"
    )
    out["r_sweep_top"] = sorted(
        all_r,
        key=lambda x: (
            1 if x.get("hard_ok") else 0,
            1e300 if x["score_words"] == "inf" else float(x["score_words"]),
        ),
        reverse=True,
    )[:12]
    out["elapsed_s"] = time.time() - t0
    print(f"  BEST T*={best_T} score_words={out['score_words']} X={out['X_words']}")
    save_json(out, RESULTS / "D.json")
    return out, best_T, planned


# ── Approach E: hybrid words + residual multiword + DP ────────────────

def run_approach_e(corpora: Dict[str, str]) -> Dict:
    print("\n" + "=" * 70)
    print("APPROACH E — hybrid words + residual multiword + DP")
    print("=" * 70)
    t0 = time.time()

    words_by_lang = {lang: extract_words(corpora[lang]) for lang in LANGS}
    all_words: Set[str] = set()
    for ws in words_by_lang.values():
        all_words.update(ws)
    chars: Set[str] = set()
    for text in corpora.values():
        chars.update(ch for ch in text if not ch.isspace())

    # Primary: all words + chars → X=1 / score inf
    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    vocab_primary: Set[str] = set(specials) | set(all_words)
    for c in sorted(chars):
        if len(vocab_primary) >= VOCAB_MAX:
            break
        vocab_primary.add(c)
    word_set = set(all_words)

    def enc_primary(text, lang, _ws=word_set):
        out = []
        for w in extract_words(text):
            if w in _ws:
                out.append(w)
            else:
                out.extend(list(w))
        return out

    res_primary = both_denoms(enc_primary, corpora, len(vocab_primary), "E_words_primary")

    # Compressed: residual multiword RP with fairness band
    # Build phrases SEP-joined; DP encode
    phrase: Set[str] = set()
    speech_vocab: Set[str] = set(specials) | set(all_words)
    for c in sorted(chars):
        if len(speech_vocab) >= VOCAB_MAX - 2000:  # reserve ~2k for phrases if possible
            break
        speech_vocab.add(c)

    # Collect ngrams
    cand: Dict[str, List[Tuple[str, int]]] = {}
    for lang in LANGS:
        ws = words_by_lang[lang]
        items = []
        for n in range(2, 9):
            ctr: Counter = Counter()
            for i in range(len(ws) - n + 1):
                ctr[SEP.join(ws[i : i + n])] += 1
            for p, f in ctr.items():
                sav = (n - 1) * f
                if sav > 0:
                    items.append((p, sav))
        items.sort(key=lambda x: -x[1])
        cand[lang] = items[:5000]
    ptr = {lang: 0 for lang in LANGS}

    # Approx T after words
    T_hat = {lang: float(len(words_by_lang[lang])) for lang in LANGS}
    D = {lang: float(len(words_by_lang[lang])) for lang in LANGS}
    band = 0.01
    for _ in range(50000):
        if len(speech_vocab) + len(phrase) >= VOCAB_MAX:
            break
        X = {l: T_hat[l] / D[l] for l in LANGS}
        lang = max(LANGS, key=lambda L: X[L])
        xmin = min(X.values())
        placed = False
        g = 0
        while ptr[lang] < len(cand[lang]) and g < 80:
            g += 1
            p, sav = cand[lang][ptr[lang]]
            ptr[lang] += 1
            if p in speech_vocab or p in phrase:
                continue
            new_x = max(1.0, T_hat[lang] - sav) / D[lang]
            if new_x + 1e-12 < xmin - band:
                continue
            phrase.add(p)
            T_hat[lang] = max(1.0, T_hat[lang] - sav)
            placed = True
            break
        if not placed:
            # try any lang worst-first
            progressed = False
            for lang2 in sorted(LANGS, key=lambda L: -X[L]):
                g2 = 0
                while ptr[lang2] < len(cand[lang2]) and g2 < 40:
                    g2 += 1
                    p, sav = cand[lang2][ptr[lang2]]
                    ptr[lang2] += 1
                    if p in speech_vocab or p in phrase:
                        continue
                    phrase.add(p)
                    T_hat[lang2] = max(1.0, T_hat[lang2] - sav)
                    progressed = True
                    break
                if progressed:
                    break
            if not progressed:
                break

    full_vocab = speech_vocab | phrase
    # DP encode
    max_n = 1
    for p in phrase:
        max_n = max(max_n, p.count(SEP) + 1)

    def enc_comp(text, lang, _ws=word_set, _ph=phrase, _mn=max_n):
        words = extract_words(text)
        n = len(words)
        if n == 0:
            return []
        INF = 10**9
        dp = [INF] * (n + 1)
        prev = [-1] * (n + 1)
        plen = [0] * (n + 1)
        dp[0] = 0
        for i in range(n):
            if dp[i] >= INF:
                continue
            w = words[i]
            cost = 1 if w in _ws else max(1, len(w))
            if dp[i] + cost < dp[i + 1]:
                dp[i + 1] = dp[i] + cost
                prev[i + 1] = i
                plen[i + 1] = 1
            for L in range(2, min(_mn, n - i) + 1):
                key = SEP.join(words[i : i + L])
                if key in _ph and dp[i] + 1 < dp[i + L]:
                    dp[i + L] = dp[i] + 1
                    prev[i + L] = i
                    plen[i + L] = L
        # reconstruct count only
        toks = []
        i = n
        while i > 0:
            p0 = prev[i]
            L = plen[i]
            if L <= 0:
                toks.append(words[i - 1])
                i -= 1
                continue
            if L == 1:
                w = words[p0]
                if w in _ws:
                    toks.append(w)
                else:
                    toks.extend(list(w))
            else:
                toks.append(SEP.join(words[p0 : p0 + L]))
            i = p0
        toks.reverse()
        return toks

    res_comp = both_denoms(enc_comp, corpora, len(full_vocab), "E_compressed_multiword")

    # Primary is max-score (words + chars → X=1)
    out = dict(res_primary)
    out["name"] = "approach_E_hybrid"
    out["notes"] = (
        "Primary: all unique words + chars (X≈1 when all words fit under 10k). "
        f"Compressed multiword ref: phrases={len(phrase)} vocab={len(full_vocab)} "
        f"score_words={res_comp['score_words']} X={res_comp['X_words']}. "
        f"elapsed={time.time()-t0:.1f}s"
    )
    out["primary"] = {
        "vocab_size": res_primary["vocab_size"],
        "T": res_primary["T"],
        "X_words": res_primary["X_words"],
        "X_fixed": res_primary["X_fixed"],
        "score_words": res_primary["score_words"],
        "score_fixed": res_primary["score_fixed"],
        "hard_ok_words": res_primary["hard_ok_words"],
        "hard_ok_fixed": res_primary["hard_ok_fixed"],
    }
    out["compressed"] = {
        "vocab_size": res_comp["vocab_size"],
        "n_phrases": len(phrase),
        "T": res_comp["T"],
        "X_words": res_comp["X_words"],
        "X_fixed": res_comp["X_fixed"],
        "score_words": res_comp["score_words"],
        "score_fixed": res_comp["score_fixed"],
        "hard_ok_words": res_comp["hard_ok_words"],
        "hard_ok_fixed": res_comp["hard_ok_fixed"],
    }
    out["elapsed_s"] = time.time() - t0
    print(f"  primary score_words={res_primary['score_words']} X={res_primary['X_words']}")
    print(f"  compressed score_words={res_comp['score_words']} X={res_comp['X_words']} phrases={len(phrase)}")
    save_json(out, RESULTS / "E.json")
    return out


# ── Champion: D packing + residual words in leftover slots ────────────

def run_champion(
    corpora: Dict[str, str],
    T_star: Optional[Dict[str, int]] = None,
) -> Dict:
    print("\n" + "=" * 70)
    print("CHAMPION — D packing + residual words in leftover slots")
    print("=" * 70)
    t0 = time.time()
    D = {lang: float(len(extract_words(corpora[lang]))) for lang in LANGS}

    all_words: Set[str] = set()
    for text in corpora.values():
        all_words.update(extract_words(text))
    chars: Set[str] = set()
    for text in corpora.values():
        chars.update(ch for ch in text if not ch.isspace())

    # Need residual room for words+chars (~7560 words; aim keep as many as fit)
    # Maximize equal-X score among T with sum(T) + min(len(all_words), residual) ok
    # Primary: global max-score T (may leave little residual for words — still OK;
    # residual words don't affect page T under packed encode).
    candidates: List[Dict[str, int]] = []
    if T_star is not None:
        candidates.append(T_star)

    T_max, _, _ = find_max_score_T(D, sum_cap=VOCAB_MAX)
    candidates.append(T_max)

    # also max-score under various residual budgets for words
    for word_reserve in (0, 500, 2000, 4000, 6000, 7500, 7800):
        cap = VOCAB_MAX - word_reserve
        if cap < 4:
            continue
        T, _, _ = find_max_score_T(D, sum_cap=cap)
        candidates.append(T)

    # denser r with residual
    for r in [0.02, 0.05, 0.08, 0.1, 0.1084, 0.15, 0.2]:
        T = choose_equal_x_T(D, r=r, window=10, sum_cap=VOCAB_MAX)
        candidates.append(T)

    best_overall = None
    seen = set()
    for T in candidates:
        key = tuple(T[l] for l in LANGS)
        if key in seen:
            continue
        seen.add(key)
        if sum(T.values()) > VOCAB_MAX:
            continue
        planned, span_vocab = build_slice(corpora, T)
        used = set(span_vocab)
        take_words: Set[str] = set()
        for w in sorted(all_words):
            if w in used:
                continue
            if len(used) + len(take_words) >= VOCAB_MAX - 180:
                break
            take_words.add(w)
        used2 = set(used) | take_words
        take_chars: Set[str] = set()
        for c in sorted(chars):
            if c in used2:
                continue
            if len(used2) + len(take_chars) >= VOCAB_MAX:
                break
            take_chars.add(c)
        used2 |= take_chars

        def enc(text, lang, _p=planned, _c=corpora, _w=take_words):
            if text == _c[lang]:
                return list(_p[lang])
            out = []
            for w in extract_words(text):
                if w in _w:
                    out.append(w)
                else:
                    out.extend(list(w))
            return out

        res = both_denoms(enc, corpora, len(used2), f"champion_T{sum(T.values())}")
        res["T_design"] = T
        res["vocab_breakdown"] = {
            "spans": len(span_vocab),
            "words": len(take_words),
            "chars": len(take_chars),
            "total": len(used2),
        }
        res["encode_matches"] = all(res["T"][l] == len(planned[l]) for l in LANGS)
        sc = res["score_words"]
        scv = 1e300 if sc == "inf" else float(sc)
        row = (1 if res["hard_ok_words"] else 0, scv, -res["gap_words"], -len(used2))
        print(
            f"  T={T} score_words={sc} gap={res['gap_words']:.8g} "
            f"vocab={len(used2)} words_res={len(take_words)} match={res['encode_matches']}"
        )
        if best_overall is None or row > best_overall[0]:
            best_overall = (row, res, T, take_words)

    assert best_overall is not None
    _, res, T, words = best_overall
    out = dict(res)
    out["name"] = "champion"
    out["notes"] = (
        "D equal-X packing for page encode + residual whole-words (then chars) "
        f"in leftover vocab slots. T*={T}. "
        f"breakdown={out.get('vocab_breakdown')}. elapsed={time.time()-t0:.1f}s"
    )
    out["elapsed_s"] = time.time() - t0
    print(f"  CHAMPION score_words={out['score_words']} X={out['X_words']}")
    save_json(out, RESULTS / "champion.json")
    return out


# ── RESULTS TABLE ─────────────────────────────────────────────────────

def write_table(rows: List[Dict]) -> None:
    def sc_num(s):
        if s == "inf" or s == float("inf"):
            return 1e300
        return float(s)

    by_words = sorted(rows, key=lambda r: (
        1 if r.get("hard_ok_words") else 0,
        sc_num(r["score_words"]),
    ), reverse=True)
    by_fixed = sorted(rows, key=lambda r: (
        1 if r.get("hard_ok_fixed") else 0,
        sc_num(r["score_fixed"]),
    ), reverse=True)

    def fmt_sc(s):
        if s == "inf" or s == float("inf"):
            return "inf"
        return f"{float(s):.4f}"

    def fmt_x(xd):
        return " ".join(f"{k}={xd[k]:.6f}" for k in LANGS)

    def fmt_t(td):
        return " ".join(f"{k}={td[k]}" for k in LANGS)

    lines = [
        "# Full-md retrain results",
        "",
        "Corpus: `full_md/india_{en,hi,te,sa}.md` (harness-cleaned).",
        "",
        "Denoms:",
        "- **words**: `X_i = T_i / word_count_i`",
        "- **fixed**: `X_i = T_i / 5000`",
        "- **score** = `1000 / (maxX − minX)` (inf if gap≈0)",
        "- **hard_ok**: `X_EN ≤ 1.2` and `vocab ≤ 10000`",
        "",
        "## Ranked by score_words",
        "",
        "| rank | name | vocab | score_words | gap_words | hard_ok_w | X_words | T |",
        "|-----:|------|------:|------------:|----------:|:---------:|---------|---|",
    ]
    for i, r in enumerate(by_words, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['vocab_size']} | {fmt_sc(r['score_words'])} | "
            f"{r['gap_words']:.8g} | {r['hard_ok_words']} | `{fmt_x(r['X_words'])}` | `{fmt_t(r['T'])}` |"
        )

    lines += [
        "",
        "## Ranked by score_fixed",
        "",
        "| rank | name | vocab | score_fixed | gap_fixed | hard_ok_f | X_fixed | T |",
        "|-----:|------|------:|------------:|----------:|:---------:|---------|---|",
    ]
    for i, r in enumerate(by_fixed, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['vocab_size']} | {fmt_sc(r['score_fixed'])} | "
            f"{r['gap_fixed']:.8g} | {r['hard_ok_fixed']} | `{fmt_x(r['X_fixed'])}` | `{fmt_t(r['T'])}` |"
        )

    lines += [
        "",
        "## Side-by-side",
        "",
        "| name | vocab | score_words | score_fixed | hard_ok_w | hard_ok_f | notes |",
        "|------|------:|------------:|------------:|:---------:|:---------:|-------|",
    ]
    for r in by_words:
        note = (r.get("notes") or "")[:120].replace("|", "/")
        lines.append(
            f"| {r['name']} | {r['vocab_size']} | {fmt_sc(r['score_words'])} | "
            f"{fmt_sc(r['score_fixed'])} | {r['hard_ok_words']} | {r['hard_ok_fixed']} | {note} |"
        )
    lines.append("")
    path = FULL_RERUN / "RESULTS_TABLE.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {path}")


def main():
    print("Loading full_md corpora...")
    corpora = load_corpora()
    for lang in LANGS:
        w = extract_words(corpora[lang])
        print(f"  {lang}: words={len(w)} unique={len(set(w))} chars={len(corpora[lang])}")

    rows = []

    a = run_approach_a(corpora)
    rows.append(a)

    b = run_approach_b(corpora)
    rows.append(b)

    c = run_approach_c(corpora)
    rows.append(c)

    d, t_star, _planned = run_approach_d(corpora)
    rows.append(d)

    e = run_approach_e(corpora)
    rows.append(e)

    ch = run_champion(corpora, T_star=t_star)
    rows.append(ch)

    write_table(rows)
    print("\n" + "=" * 70)
    print("DONE — comparison by score_words:")
    for r in sorted(
        rows,
        key=lambda x: (
            1 if x.get("hard_ok_words") else 0,
            1e300 if x["score_words"] == "inf" else float(x["score_words"]),
        ),
        reverse=True,
    ):
        print(
            f"  {r['name']:22s} score_w={r['score_words']!s:>14} "
            f"score_f={r['score_fixed']!s:>14} hard={r['hard_ok_words']} "
            f"vocab={r['vocab_size']}"
        )


if __name__ == "__main__":
    main()
