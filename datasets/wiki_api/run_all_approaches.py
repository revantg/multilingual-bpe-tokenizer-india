#!/usr/bin/env python3
"""
Retrain / eval A–E + optimized B + champion on Wikipedia *API* India extracts.

Corpus: data/wiki_api/{en,hi,te,sa}.txt  (MediaWiki prop=extracts&explaintext=1)
Results: data/wiki_api/results/
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

WIKI_API = Path(__file__).resolve().parent
DATA = WIKI_API.parent
GROK2 = DATA / "grok_2"
FULL_PAGE_RERUN = GROK2 / "full_page_rerun"
RESULTS = WIKI_API / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WIKI_API))
sys.path.insert(0, str(GROK2 / "harness"))
sys.path.insert(0, str(GROK2 / "approach_a"))
sys.path.insert(0, str(GROK2 / "approach_b"))
sys.path.insert(0, str(FULL_PAGE_RERUN))

from load_wiki_api import load_corpora  # noqa: E402
from eval_metrics import (  # noqa: E402
    LANGS,
    evaluate_encode,
    extract_words,
)
from train_bpe import ClassicBPE  # noqa: E402
from train_optimized_b import train_optimized_b  # noqa: E402

# Reuse implementations from full_page_rerun for C/D/E/champion/fair-B
import run_all as fpr  # noqa: E402

fpr.RESULTS = RESULTS
fpr.VOCAB_MAX = 10_000


def both_denoms(encode_fn, corpora, vocab_size, name):
    return fpr.both_denoms(encode_fn, corpora, vocab_size, name)


def save_json(payload: dict, path: Path) -> None:
    fpr.save_json(payload, path)


def corpus_stats(corpora: Dict[str, str]) -> dict:
    stats = {}
    union = set()
    for L in LANGS:
        ws = extract_words(corpora[L])
        us = set(ws)
        union |= us
        stats[L] = {
            "chars": len(corpora[L]),
            "words": len(ws),
            "unique": len(us),
        }
    stats["union_unique_words"] = len(union)
    return stats


def run_a(corpora):
    # Patch RESULTS write dir already set; reuse runner
    return fpr.run_approach_a(corpora)


def run_b_fair(corpora):
    return fpr.run_approach_b(corpora)


def run_b_optimized(corpora) -> dict:
    print("\n" + "=" * 70)
    print("APPROACH B_optimized — stage1 BPE + savings word fill")
    print("=" * 70)
    t0 = time.time()
    bpe, meta = train_optimized_b(
        corpora=corpora,
        vocab_cap=10_000,
        bpe_size=2_500,
        en_merge_weight=2.0,
        alpha={"EN": 1.0, "HI": 1.0, "TE": 1.0, "SA": 1.0},
        min_freq=1,
    )

    def enc(text, lang, _b=bpe):
        return _b.encode_tokens(text, lang)

    res = both_denoms(enc, corpora, bpe.vocab_size, "approach_B_optimized")
    res["name"] = "approach_B_optimized"
    res["train_meta"] = meta
    res["notes"] = (
        f"stage1 bpe_size≈{meta.get('bpe_size_requested')} merges={meta.get('merges')} "
        f"words_added={meta.get('words_added_stage2')} final_vocab={meta.get('final_vocab')} "
        f"α={meta.get('alphas')} en_merge_weight={meta.get('en_merge_weight')} "
        f"elapsed={time.time()-t0:.1f}s"
    )
    res["elapsed_s"] = time.time() - t0
    print(
        f"  score_words={res['score_words']} X={res['X_words']} hard={res['hard_ok_words']}"
    )
    save_json(res, RESULTS / "B_optimized.json")
    # also persist model
    from train_optimized_b import save_model

    save_model(bpe, meta, WIKI_API / "vocab_B_optimized.json")
    return res


def run_c(corpora):
    return fpr.run_approach_c(corpora)


def run_d(corpora):
    return fpr.run_approach_d(corpora)


def run_e(corpora):
    return fpr.run_approach_e(corpora)


def run_champion(corpora, T_star=None):
    return fpr.run_champion(corpora, T_star=T_star)


def fmt_sc(sc):
    if sc == "inf" or sc == float("inf"):
        return "inf"
    return f"{float(sc):.4f}"


def write_table(rows, stats):
    lines = [
        "# Wikipedia API corpus — approach results",
        "",
        "Source: MediaWiki `action=query&prop=extracts&explaintext=1&titles=India` "
        "on en/hi/te/sa.wikipedia.org (redirects=1).",
        "",
        "## Corpus stats",
        "",
        "| lang | chars | words D | unique |",
        "|------|------:|--------:|-------:|",
    ]
    for L in LANGS:
        s = stats[L]
        lines.append(f"| {L} | {s['chars']} | {s['words']} | {s['unique']} |")
    lines.append(f"| **union unique** |  |  | **{stats['union_unique_words']}** |")
    lines += [
        "",
        "## Ranked by score_words (hard_ok first)",
        "",
        "| rank | name | vocab | score_words | score_fixed | hard_ok_w | hard_ok_f | X_EN | X_HI | X_TE | X_SA | T |",
        "|-----:|------|------:|------------:|------------:|:---------:|:---------:|-----:|-----:|-----:|-----:|---|",
    ]

    def sort_key(r):
        sc = r["score_words"]
        scv = 1e300 if sc == "inf" else float(sc)
        return (1 if r.get("hard_ok_words") else 0, scv)

    ranked = sorted(rows, key=sort_key, reverse=True)
    for i, r in enumerate(ranked, 1):
        X = r["X_words"]
        T = r["T"]
        lines.append(
            f"| {i} | {r['name']} | {r['vocab_size']} | {fmt_sc(r['score_words'])} | "
            f"{fmt_sc(r['score_fixed'])} | {r['hard_ok_words']} | {r['hard_ok_fixed']} | "
            f"{X['EN']:.4f} | {X['HI']:.4f} | {X['TE']:.4f} | {X['SA']:.4f} | "
            f"`EN={T['EN']} HI={T['HI']} TE={T['TE']} SA={T['SA']}` |"
        )

    lines += [
        "",
        "## Notes",
        "",
    ]
    for r in ranked:
        lines.append(f"- **{r['name']}**: {(r.get('notes') or '')[:300]}")
        if r["name"] == "approach_B_fair" and "variants" in r:
            for vk, vv in r["variants"].items():
                lines.append(
                    f"  - variant `{vk}`: score_w={vv['score_words']} "
                    f"X={vv['X_words']} hard={vv['hard_ok_words']}"
                )
        if r["name"] == "approach_E_hybrid" and "compressed" in r:
            c = r["compressed"]
            lines.append(
                f"  - compressed: score_w={c['score_words']} X={c['X_words']} "
                f"phrases={c.get('n_phrases')}"
            )
        if r["name"] == "approach_C_words":
            lines.append(
                f"  - coverage_occ={r.get('coverage_pct_token_occ')} mode={r.get('coverage_mode')}"
            )

    lines.append("")
    path = WIKI_API / "RESULTS_TABLE.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {path}")
    return ranked


def main():
    print("Loading Wikipedia API extracts...")
    corpora = load_corpora()
    stats = corpus_stats(corpora)
    for L in LANGS:
        s = stats[L]
        print(f"  {L}: chars={s['chars']} words={s['words']} unique={s['unique']}")
    print(f"  union unique words={stats['union_unique_words']}")
    save_json({"stats": stats, "source": "wikipedia_api_extracts"}, RESULTS / "_corpus.json")

    rows: List[dict] = []

    a = run_a(corpora)
    rows.append(a)

    b_opt = run_b_optimized(corpora)
    rows.append(b_opt)

    b_fair = run_b_fair(corpora)
    rows.append(b_fair)

    c = run_c(corpora)
    rows.append(c)

    d, t_star, _planned = run_d(corpora)
    rows.append(d)

    e = run_e(corpora)
    rows.append(e)

    ch = run_champion(corpora, T_star=t_star)
    rows.append(ch)

    ranked = write_table(rows, stats)

    print("\n" + "=" * 70)
    print("DONE — Wikipedia API corpus ranking (score_words, hard first):")
    for r in ranked:
        print(
            f"  {r['name']:24s} score_w={fmt_sc(r['score_words']):>12} "
            f"hard={r['hard_ok_words']} vocab={r['vocab_size']} "
            f"X=({r['X_words']['EN']:.3f},{r['X_words']['HI']:.3f},"
            f"{r['X_words']['TE']:.3f},{r['X_words']['SA']:.3f})"
        )


if __name__ == "__main__":
    main()
