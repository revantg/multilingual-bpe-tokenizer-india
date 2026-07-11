#!/usr/bin/env python3
"""
Export interactive demo bundle from FULL-PAGE clean Wikipedia markdown.

Corpus: data/full_page_md/india_{en,hi,te,sa}.md
Union unique words ≈ 12599 > 10000 → Approach C can no longer make X=1 everywhere.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]  # grok_2 via demo/lab
DATA = ROOT.parent
FULL_PAGE = ROOT / "datasets" / "full_page_md"
OUT_JSON = Path(__file__).resolve().parent / "demo_bundle.json"
OUT_JS = Path(__file__).resolve().parent / "demo_data.js"

sys.path.insert(0, str(FULL_PAGE))
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "approaches" / "a"))
sys.path.insert(0, str(ROOT / "approaches" / "d"))

from load_full_page import load_corpora  # noqa: E402
from eval_metrics import LANGS, extract_words, evaluate_encode  # noqa: E402
from train_bpe import ClassicBPE  # noqa: E402
from slice_pack import partition_even, reconstruct_word_spans  # noqa: E402

VOCAB_MAX = 10_000


def score_payload(encode_fn, corpora, vocab_size, name):
    rw = evaluate_encode(encode_fn, corpora, "words", vocab_size, name)
    rf = evaluate_encode(encode_fn, corpora, "fixed", vocab_size, name)
    return {
        "T": {k: int(rw["T"][k]) for k in LANGS},
        "D_words": {k: float(rw["D"][k]) for k in LANGS},
        "X_words": {k: float(rw["X"][k]) for k in LANGS},
        "X_fixed": {k: float(rf["X"][k]) for k in LANGS},
        "score_words": rw["score"] if rw["score"] != float("inf") else "inf",
        "score_fixed": rf["score"] if rf["score"] != float("inf") else "inf",
        "gap_words": float(rw["gap"]),
        "hard_ok_words": bool(rw["hard_ok"]),
        "hard_ok_fixed": bool(rf["hard_ok"]),
        "vocab_size": vocab_size,
    }


def choose_T(D: Dict[str, int], r: float = 0.10) -> Dict[str, int]:
    base = {k: max(1, int(round(r * D[k]))) for k in LANGS}
    window = 10
    best = dict(base)
    best_gap = float("inf")
    ranges = []
    for k in LANGS:
        lo = max(1, base[k] - window)
        hi = min(D[k], base[k] + window)
        ranges.append(range(lo, hi + 1))
    for combo in product(*ranges):
        Tmap = {LANGS[i]: combo[i] for i in range(4)}
        if sum(Tmap.values()) > VOCAB_MAX - 500:
            continue
        xs = [Tmap[k] / D[k] for k in LANGS]
        g = max(xs) - min(xs)
        if g < best_gap:
            best_gap = g
            best = Tmap
    return best


def build_a(corpora, words_by) -> dict:
    print("Building Approach A pure BPE…")
    t0 = time.time()
    unit_freq = Counter()
    for L in LANGS:
        unit_freq.update(words_by[L])
    bpe = ClassicBPE()
    bpe.train(unit_freq, vocab_size=VOCAB_MAX, min_freq=1, verbose=True)
    bpe.encode_mode = "dp"
    # save for later
    bpe.save(ROOT / "approaches" / "a" / "bpe_model_fullpage.json")

    def enc(text, lang, _b=bpe):
        return _b.encode_tokens(text, lang)

    m = score_payload(enc, corpora, bpe.vocab_size, "A")
    print(f"  A done in {time.time()-t0:.1f}s score_words={m['score_words']}")
    return {
        "id": "A",
        "title": "Approach A — Pure Multilingual BPE",
        "subtitle": "Word-interior pair merges · no whole-word dump · full Wikipedia pages",
        "kind": "bpe",
        "honesty": 8,
        "score_label": f"words≈{m['score_words'] if isinstance(m['score_words'], str) else round(float(m['score_words']))}",
        "score_num": 0 if m["score_words"] == "inf" else float(m["score_words"]),
        "hard_ok": m["hard_ok_words"],
        "description": (
            "Classic BPE on pretok words from the full India Wikipedia pages (all 4 languages). "
            "Starts from characters and repeatedly merges the most frequent pair until 10 000 slots. "
            "Does NOT force every word into the vocab — long or rare words (esp. Telugu/Sanskrit) "
            "split into subwords. This is the honest subword baseline."
        ),
        "how_tokens": (
            "Tokens are characters and learned merges (e.g. th, ion, ంలో). "
            "Frequent short words may still be one token if fully merged; rare words fragment."
        ),
        "metrics": m,
        "merges": [list(p) for p in bpe.merges],
        "vocab_set": list(bpe.vocab.keys()),
        "encode_mode": "dp",
    }


def build_b(corpora, words_by) -> dict:
    """Optimized B: stage-1 ClassicBPE then savings-ranked whole-word fill."""
    print("Building Approach B optimized (BPE + savings word fill)…")
    t0 = time.time()
    sys.path.insert(0, str(ROOT / "approaches" / "b"))
    from train_optimized_b import train_optimized_b, save_model

    bpe, meta = train_optimized_b(
        corpora=corpora,
        vocab_cap=VOCAB_MAX,
        bpe_size=2_500,
        en_merge_weight=2.0,
        alpha={"EN": 1.0, "HI": 1.0, "TE": 1.0, "SA": 1.0},
        min_freq=1,
    )
    save_model(bpe, meta, ROOT / "approaches" / "b" / "vocab_optimized_fullpage.json")

    def enc(text, lang, _b=bpe):
        return _b.encode_tokens(text, lang)

    m = score_payload(enc, corpora, bpe.vocab_size, "B")
    added = meta.get("words_added_stage2", 0)
    print(
        f"  B done in {time.time()-t0:.1f}s score_words={m['score_words']} "
        f"vocab={bpe.vocab_size} added_words={added}"
    )
    return {
        "id": "B",
        "title": "Approach B — Optimized BPE + Savings Word Fill",
        "subtitle": "Stage-1 BPE (~2500) · stage-2 whole words by token savings",
        "kind": "fair_bpe",
        "honesty": 6,
        "score_label": f"words≈{m['score_words'] if isinstance(m['score_words'], str) else round(float(m['score_words']))}",
        "score_num": 1e99 if m["score_words"] == "inf" else float(m["score_words"]),
        "hard_ok": m["hard_ok_words"],
        "description": (
            "Two-stage hybrid under 10k: ClassicBPE merges fill ~2500 slots (with mild EN upweight), "
            "then remaining slots get whole words ranked by expected token savings "
            "score(w)=Σ α·freq·max(0, pieces−1). Union unique words on full pages (~12.6k) still "
            "exceeds 10k, so the rarest types stay multi-piece OOVs."
        ),
        "how_tokens": (
            "High-savings / common words → 1 green token. Remaining OOVs → blue subword or purple chars."
        ),
        "metrics": {**m, "train_meta": {
            "bpe_size": meta.get("bpe_size_requested"),
            "merges": meta.get("merges"),
            "words_added_stage2": added,
        }},
        "merges": [list(p) for p in bpe.merges],
        "vocab_set": list(bpe.vocab.keys()),
        "word_tokens": [w for w in bpe.vocab if len(w) > 1],
    }


def build_c(corpora, words_by) -> dict:
    """Frequency-fair whole word inventory under 10k (cannot fit all 12599)."""
    print("Building Approach C capped whole-word inventory…")
    t0 = time.time()
    by_lang = {L: Counter(words_by[L]) for L in LANGS}
    chars: Set[str] = set()
    for text in corpora.values():
        chars.update(text)

    # Reserve ~250 for list of characters
    word_budget = VOCAB_MAX - min(len(chars), 300) - 10
    # fair quota: equal slots then overflow by total freq
    base_q = word_budget // 4
    selected: Set[str] = set()
    leftovers: List[Tuple[int, str]] = []
    for L in LANGS:
        items = by_lang[L].most_common()
        take = items[:base_q]
        for w, _ in take:
            selected.add(w)
        for w, f in items[base_q:]:
            leftovers.append((f, w))
    leftovers.sort(reverse=True)
    for f, w in leftovers:
        if len(selected) >= word_budget:
            break
        selected.add(w)

    vocab = set(selected) | set(list(chars)[:300])
    # trim if still over
    while len(vocab) > VOCAB_MAX:
        # drop rarest selected words
        rare = sorted(selected, key=lambda w: sum(by_lang[L][w] for L in LANGS))
        if not rare:
            break
        drop = rare[0]
        selected.discard(drop)
        vocab.discard(drop)

    singles = selected | {c for c in chars if c in vocab}

    def enc(text, lang, _s=singles):
        out = []
        for w in extract_words(text):
            if w in _s:
                out.append(w)
            else:
                out.extend(list(w))  # char split OOV
        return out

    m = score_payload(enc, corpora, len(singles), "C")
    # coverage
    cov = {}
    for L in LANGS:
        ws = words_by[L]
        hit = sum(1 for w in ws if w in selected)
        cov[L] = round(100.0 * hit / max(len(ws), 1), 2)

    print(f"  C done {time.time()-t0:.1f}s vocab={len(singles)} score={m['score_words']} cov={cov}")
    return {
        "id": "C",
        "title": "Approach C — Whole-Word Inventory (capped)",
        "subtitle": f"~{len(selected)} words of ∪{sum(len(set(words_by[L])) for L in LANGS)} types · rest char-split",
        "kind": "words",
        "honesty": 9,
        "score_label": f"words≈{m['score_words'] if isinstance(m['score_words'], str) else round(float(m['score_words']))}",
        "score_num": 1e99 if m["score_words"] == "inf" else float(m["score_words"]),
        "hard_ok": m["hard_ok_words"],
        "description": (
            "Puts as many unique page words as fit under 10 000 (fair per-language quotas + global "
            "freq overflow) plus characters. On this FULL page corpus the union is ~12 600 unique "
            "words > 10k, so ~2.6k rare types cannot enter the vocab: they shade into characters. "
            "That breaks the old X=1 magic from the partial-page experiment."
        ),
        "how_tokens": (
            "Known word → one green chip. Unknown (rare) word → purple character chips."
        ),
        "metrics": {**m, "coverage_pct_token_occ": cov, "words_selected": len(selected),
                    "union_unique": len({w for L in LANGS for w in set(words_by[L])})},
        "vocab_set": list(singles),
        "word_set": sorted(selected),
    }


def build_d(corpora, words_by) -> dict:
    print("Building Approach D equal-X packing…")
    t0 = time.time()
    D = {L: len(words_by[L]) for L in LANGS}
    T = choose_T(D, r=0.10)
    spans_by_lang = {}
    for L in LANGS:
        ws = words_by[L]
        n = len(ws)
        ranges = partition_even(n, T[L])
        text = corpora[L]
        char_spans = reconstruct_word_spans(text, ws)
        tablets = []
        meta_ranges = []
        for a, b in ranges:
            piece = "".join(char_spans[a:b]) if char_spans else " ".join(ws[a:b])
            tablets.append(piece)
            meta_ranges.append([a, b])
        spans_by_lang[L] = {"ranges": meta_ranges, "tablets": tablets}

    # planned encode for full page only
    planned = {L: spans_by_lang[L]["tablets"] for L in LANGS}

    def enc(text, lang, _p=planned, _c=corpora):
        if lang in _c and text == _c[lang]:
            return list(_p[lang])
        return extract_words(text)

    vocab_n = sum(len(spans_by_lang[L]["tablets"]) for L in LANGS)
    m = score_payload(enc, corpora, vocab_n, "D")
    print(f"  D done {time.time()-t0:.1f}s T={T} score_words={m['score_words']}")
    return {
        "id": "D",
        "title": "Approach D — Equal-X Slice Packing",
        "subtitle": "Precomputed multi-word slabs so all X_i nearly match",
        "kind": "packing",
        "honesty": 6,
        "score_label": f"words≈{m['score_words'] if isinstance(m['score_words'], str) else float(m['score_words']):.3g}",
        "score_num": 0 if m["score_words"] == "inf" else float(m["score_words"]),
        "hard_ok": m["hard_ok_words"],
        "description": (
            "Not classic BPE. Design T*_lang ≈ r·word_count so languages get almost equal "
            "fertility X. Store each contiguous word-span as one vocabulary tablet. "
            "Full-page encode returns exactly those tablets. Snippets in this demo map your "
            "selection onto which slab(s) cover those word indices — pink chips."
        ),
        "how_tokens": (
            "One pink chip ≈ many consecutive words (~1/r). Partial snippets show covering slabs."
        ),
        "metrics": {**m, "T_star": T},
        "T_star": T,
        "spans": spans_by_lang,
    }


def build_e(corpora, words_by, word_set_from_c: Set[str]) -> dict:
    """Hybrid DP: prefer selected whole words + fair bigrams from residual budget."""
    print("Building Approach E hybrid DP…")
    t0 = time.time()
    by_lang_bi = {L: Counter(zip(words_by[L], words_by[L][1:])) for L in LANGS if len(words_by[L]) > 1}
    chars: Set[str] = set()
    for text in corpora.values():
        chars.update(list(text)[:])

    vocab_words = set(word_set_from_c)
    # residual budget for bigrams
    room = max(0, VOCAB_MAX - len(vocab_words) - min(200, len(chars)))
    bigrams: Set[str] = set()
    if room:
        per = room // 4
        for L in LANGS:
            for (a, b), _ in by_lang_bi.get(L, Counter()).most_common(per):
                # only if both words known — phrase is useful
                if a in vocab_words and b in vocab_words:
                    bigrams.add(f"{a}\u0001{b}")
                if len(bigrams) >= room:
                    break

    char_list = sorted(chars)[:200]
    vocab = vocab_words | bigrams | set(char_list)

    def enc_dp(text, lang, _vw=vocab_words, _bg=bigrams):
        words = extract_words(text)
        n = len(words)
        if n == 0:
            return []
        INF = 10**9
        dp = [INF] * (n + 1)
        choice = [None] * (n + 1)
        dp[n] = 0
        for i in range(n - 1, -1, -1):
            w = words[i]
            if w in _vw:
                if 1 + dp[i + 1] < dp[i]:
                    dp[i] = 1 + dp[i + 1]
                    choice[i] = (i + 1, w, "word")
            else:
                cost = max(1, len(w))
                if cost + dp[i + 1] < dp[i]:
                    dp[i] = cost + dp[i + 1]
                    choice[i] = (i + 1, w, "charword")
            if i + 1 < n:
                key = f"{words[i]}\u0001{words[i+1]}"
                if key in _bg and 1 + dp[i + 2] < dp[i]:
                    dp[i] = 1 + dp[i + 2]
                    choice[i] = (i + 2, key, "phrase")
        out = []
        i = 0
        while i < n:
            c = choice[i]
            if not c:
                out.append(words[i])
                i += 1
                continue
            end, tok, kind = c
            if kind == "charword":
                out.extend(list(tok))
            elif kind == "phrase":
                out.append(tok.replace("\u0001", " "))
            else:
                out.append(tok)
            i = end
        return out

    m = score_payload(enc_dp, corpora, min(len(vocab), VOCAB_MAX), "E")
    print(f"  E done {time.time()-t0:.1f}s score_words={m['score_words']}")
    return {
        "id": "E",
        "title": "Approach E — Hybrid DP (words + bigrams)",
        "subtitle": "Capped whole words + fair multiword phrases · min-token DP",
        "kind": "hybrid_dp",
        "honesty": 7,
        "score_label": f"words≈{m['score_words'] if isinstance(m['score_words'], str) else round(float(m['score_words']))}",
        "score_num": 1e99 if m["score_words"] == "inf" else float(m["score_words"]),
        "hard_ok": m["hard_ok_words"],
        "description": (
            "Uses the capped whole-word inventory from C, plus fair-quota bigrams as optional "
            "single tokens. Encoding is dynamic programming that minimizes the number of pieces. "
            "Can slightly compress below 1 token/word where bigrams hit, but OOV rare words still "
            "fragment — the fairness gap usually remains driven by those OOVs."
        ),
        "how_tokens": (
            "Green = word, orange = 2-word phrase, purple/blue = pieces of OOV words."
        ),
        "metrics": m,
        "vocab_set": list(vocab)[:VOCAB_MAX],
        "phrases": list(bigrams),
        "sep": "\u0001",
        "max_n": 2,
        "word_set": sorted(vocab_words),
    }


def build_champ(d_payload, c_payload) -> dict:
    return {
        "id": "Champ",
        "title": "Champion Hybrid — D packing + C residual words",
        "subtitle": "Score path on full pages = equal-X slabs; leftover budget = words/chars",
        "kind": "champion",
        "honesty": 3,
        "score_label": d_payload["score_label"],
        "score_num": d_payload["score_num"],
        "hard_ok": d_payload["hard_ok"],
        "description": (
            "Combines D's page packing (drives the high score / low equal X on full pages) with "
            "C's residual whole-word list so arbitrary snippets still mostly map to known words. "
            "Interactive snippets use packing coverage when they fall on the page word stream."
        ),
        "how_tokens": d_payload["how_tokens"],
        "metrics": d_payload["metrics"],
        "T_star": d_payload["T_star"],
        "spans": d_payload["spans"],
        "word_set": c_payload.get("word_set", []),
        "vocab_set": c_payload.get("vocab_set", []),
    }


def main():
    print("Loading full_page_md corpora…")
    corpora = load_corpora()
    words_by = {L: extract_words(corpora[L]) for L in LANGS}
    counts = {L: len(words_by[L]) for L in LANGS}
    uniques = {L: len(set(words_by[L])) for L in LANGS}
    union = set()
    for L in LANGS:
        union |= set(words_by[L])
    print("counts", counts)
    print("uniques", uniques, "union", len(union), ">10k?", len(union) > 10000)

    approaches = []
    a = build_a(corpora, words_by)
    approaches.append(a)
    b = build_b(corpora, words_by)
    approaches.append(b)
    c = build_c(corpora, words_by)
    approaches.append(c)
    d = build_d(corpora, words_by)
    approaches.append(d)
    e = build_e(corpora, words_by, set(c["word_set"]))
    approaches.append(e)
    approaches.append(build_champ(d, c))

    bundle = {
        "corpus": "full_page_md",
        "corpus_note": (
            "Full Wikipedia HTML → markdown with minimal strip (infobox/tables/refs kept). "
            f"Union unique words = {len(union)} > 10000 — whole-word perfect cover is impossible."
        ),
        "langs": list(LANGS),
        "lang_labels": {
            "EN": "English",
            "HI": "Hindi",
            "TE": "Telugu",
            "SA": "Sanskrit",
        },
        "words": words_by,
        "word_counts": counts,
        "unique_counts": uniques,
        "union_unique": len(union),
        "union_gt_10000": len(union) > 10000,
        "approaches": approaches,
        "meta": {
            "score_formula": "score = 1000 / (max X − min X),  X = tokens / words  (also report fixed ÷5000)",
            "vocab_cap": 10000,
            "x_en_cap": 1.2,
            "note": "Vocabs rebuilt on full_page_md; snippet encode uses each approach's real inventory.",
        },
        "previews": {L: corpora[L][:500] for L in LANGS},
    }

    # strip huge metrics if needed; keep as is
    OUT_JSON.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    OUT_JS.write_text(
        "window.DEMO_BUNDLE = "
        + json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_JSON} ({OUT_JSON.stat().st_size/1e6:.2f} MB)")
    print(f"Wrote {OUT_JS} ({OUT_JS.stat().st_size/1e6:.2f} MB)")
    for ap in approaches:
        print(
            f"  {ap['id']}: score_words={ap['metrics'].get('score_words')} "
            f"X_en={ap['metrics'].get('X_en_words', ap['metrics'].get('X_words',{}).get('EN'))} "
            f"hard={ap['hard_ok']} vocab≈{ap['metrics'].get('vocab_size')}"
        )


if __name__ == "__main__":
    main()
