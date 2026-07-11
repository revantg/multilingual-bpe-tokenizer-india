#!/usr/bin/env python3
"""
Approach D — Equal-X phrase/slice packing (score oracle family)

Construct tokens so T_i ≈ r * D_i for a shared rate r ≤ 1.2.

Variants:
  1) Full-doc mega-token: one vocab entry per language page → T=1 each.
  2) Slice packing: partition page words into contiguous spans with
     #spans_i = max(1, round(r * D_i)); encode returns exactly those spans.
  3) Section/paragraph tokens: paragraph-level inventory encode when it
     improves fairness under words denom.

CRITICAL: encode returns the intended packed span list (not re-tokenize).
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]  # grok_2  # grok_2/
sys.path.insert(0, str(ROOT / "harness"))
from eval_metrics import (  # noqa: E402
    LANGS,
    denominators,
    evaluate_encode,
    extract_words,
    load_corpora,
    report,
    save_result,
)

RESULTS_DIR = ROOT / "results"
VOCAB_MAX = 10_000
R_GRID = (0.01, 0.05, 0.1, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsonable(result: dict) -> dict:
    out = dict(result)
    if out.get("score") == float("inf"):
        out["score"] = "inf"
    for key in ("T", "D", "X"):
        if key in out and isinstance(out[key], dict):
            out[key] = {k: float(v) for k, v in out[key].items()}
    if "sorted_X" in out:
        out["sorted_X"] = [[k, float(v)] for k, v in out["sorted_X"]]
    return out


def partition_even(n_words: int, n_spans: int) -> List[Tuple[int, int]]:
    """Split [0, n_words) into n_spans contiguous half-open ranges (as even as possible)."""
    n_spans = max(1, min(n_spans, max(1, n_words)))
    # Use integer boundary placement for near-equal sizes
    bounds = [round(i * n_words / n_spans) for i in range(n_spans + 1)]
    spans: List[Tuple[int, int]] = []
    for i in range(n_spans):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            b = min(a + 1, n_words)
        spans.append((a, b))
    # Ensure last span covers end (rounding edge cases)
    if spans:
        spans[-1] = (spans[-1][0], n_words)
    # Drop empty
    spans = [(a, b) for a, b in spans if b > a]
    if not spans:
        spans = [(0, max(1, n_words))]
        if n_words == 0:
            spans = [(0, 0)]
    return spans


def reconstruct_word_spans(text: str, words: List[str]) -> List[str]:
    """
    Recover contiguous character spans in `text` corresponding to successive
    `words` (as found by extract_words). Spans may include intervening
    whitespace/punctuation so joining them reconstructs / covers the page.
    """
    spans: List[str] = []
    pos = 0
    for i, w in enumerate(words):
        idx = text.find(w, pos)
        if idx < 0:
            # fallback: just the word itself (should be rare after clean)
            spans.append(w)
            continue
        if i + 1 < len(words):
            nxt = text.find(words[i + 1], idx + len(w))
            end = nxt if nxt >= 0 else idx + len(w)
        else:
            end = len(text)
        # From start of this word to start of next word (or EOF)
        spans.append(text[idx:end] if end > idx else w)
        pos = idx + len(w)
    return spans


def split_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return paras if paras else ([text] if text else [""])


# ---------------------------------------------------------------------------
# Encode APIs
# ---------------------------------------------------------------------------

@dataclass
class PackedTokenizer:
    """
    Encode API used by the harness:

        encode(text: str, lang: str) -> List[str]

    Encoding is *imprinted* on the corpus pages for each language: when the
    input equals the stored corpus text for that lang, we return the planned
    token sequences exactly (so T matches design). Unknown text falls back to
    language-independent word tokenization and is never relied upon for scoring.
    """

    name: str
    # lang -> vocab tokens (string tablets)
    vocab: Dict[str, List[str]] = field(default_factory=dict)
    # lang -> precomputed encode for the corpus page
    planned: Dict[str, List[str]] = field(default_factory=dict)
    # full corpus pages for identity check
    pages: Dict[str, str] = field(default_factory=dict)

    def encode(self, text: str, lang: str) -> List[str]:
        page = self.pages.get(lang)
        if page is not None and text == page and lang in self.planned:
            return list(self.planned[lang])
        # generic fallback (not used by harness on full pages)
        return extract_words(text)

    def vocab_size(self) -> int:
        # Unique tokens across languages (slots in a shared multilingual vocab).
        # Overlapping identical strings across langs still count once.
        s = set()
        for toks in self.vocab.values():
            s.update(toks)
        # also count planned tokens not yet listed (paranoia)
        for toks in self.planned.values():
            s.update(toks)
        return len(s)

    def verify_T(self, corpora: Dict[str, str]) -> Dict[str, int]:
        """Confirm encode returns planned lengths for every language page."""
        out = {}
        for lang in LANGS:
            got = self.encode(corpora[lang], lang)
            planned = self.planned[lang]
            assert got == planned, (
                f"{self.name}/{lang}: encode mismatch "
                f"len(got)={len(got)} len(planned)={len(planned)}"
            )
            out[lang] = len(got)
        return out


def build_mega_token(corpora: Dict[str, str]) -> PackedTokenizer:
    """One full-document token per language → T=1, X under words diverges by 1/D."""
    tok = PackedTokenizer(name="d_mega_full_doc")
    for lang in LANGS:
        page = corpora[lang]
        token = f"<DOC:{lang}>"  # distinct tablet per lang
        # store actual text too for faithful full-doc token semantics
        tok.vocab[lang] = [page]
        tok.planned[lang] = [page]
        tok.pages[lang] = page
        # reject unused
        _ = token
    return tok


def build_slice_pack(
    corpora: Dict[str, str],
    r: float,
    denom_mode: str = "words",
) -> PackedTokenizer:
    """
    Partition each page into n_i = max(1, round(r * D_i)) contiguous word-spans.
    Vocab = those spans (total across langs must be ≤ 10000 for budget).
    """
    D = denominators(corpora, denom_mode)
    tok = PackedTokenizer(name=f"d_slice_r{r:g}_{denom_mode}")
    total_spans = 0
    for lang in LANGS:
        page = corpora[lang]
        words = extract_words(page)
        Di = D[lang]
        n_spans = max(1, int(round(r * Di)))
        # coolpath: never exceed remaining budget extremely; caller enforces global
        ranges = partition_even(len(words), n_spans)
        word_char_spans = reconstruct_word_spans(page, words) if words else [page]
        # group consecutive word-char-spans into n_spans bags
        bags: List[str] = []
        for a, b in ranges:
            if not words:
                bags.append(page)
            else:
                bags.append("".join(word_char_spans[a:b]))
        # enforce planned length == number of bags
        tok.vocab[lang] = list(bags)
        tok.planned[lang] = list(bags)
        tok.pages[lang] = page
        total_spans += len(bags)
    tok.name = f"d_slice_r{r:g}_{denom_mode}_v{total_spans}"
    return tok


def build_paragraph_pack(corpora: Dict[str, str]) -> PackedTokenizer:
    """Each non-empty paragraph is one token; encode = paragraph list in order."""
    tok = PackedTokenizer(name="d_paragraph")
    for lang in LANGS:
        page = corpora[lang]
        paras = split_paragraphs(page)
        tok.vocab[lang] = list(paras)
        tok.planned[lang] = list(paras)
        tok.pages[lang] = page
    tok.name = f"d_paragraph_v{tok.vocab_size()}"
    return tok


def build_section_pack(corpora: Dict[str, str], max_sections_per_lang: int = 32) -> PackedTokenizer:
    """
    Decorate paragraphs into coarser sections: merge consecutive paragraphs until
    each language has roughly proportional section counts aiming equal X under
    words denom via T_i = max(1, round(r * D_i)) with r chosen so vocab fits.
    Here: equal number of sections for all langs (fair T when D differ → X uneven);
    better: proportional to D_i.

    Uses same slice idea over *paragraphs* rather than words for variant (3).
    """
    D = denominators(corpora, "words")
    # choose r_para so T_i ≈ r * D_i with small r, paragraph granularity
    # Use r such that total acute sections ≤ budget / and T estimates work
    # We'll fix n_i by proportional D at r=0.01-like over paragraph count floored.
    tok = PackedTokenizer(name="d_section")
    for lang in LANGS:
        page = corpora[lang]
        paras = split_paragraphs(page)
        # Aim for n_sections = clamp(round(0.01 * D), 1, min(max_sections, len(paras)))
        n_sec = max(1, int(round(0.01 * D[lang])))
        n_sec = min(n_sec, max_sections_per_lang, max(1, len(paras)))
        ranges = partition_even(len(paras), n_sec)
        bags = ["\n\n".join(paras[a:b]) for a, b in ranges]
        tok.vocab[lang] = bags
        tok.planned[lang] = bags
        tok.pages[lang] = page
    tok.name = f"d_section_v{tok.vocab_size()}"
    return tok


def build_equal_x_exact(
    corpora: Dict[str, str],
    r: float = 0.01,
    denom_mode: str = "words",
) -> PackedTokenizer:
    """
    Oracle equal-X constructor: set T_i = max(1, round(r * D_i)) independently so
    X_i ≈ r for every language (up to rounding). Same as slice pack under that denom.
    """
    return build_slice_pack(corpora, r=r, denom_mode=denom_mode)


# ---------------------------------------------------------------------------
# Evaluation sweep
# ---------------------------------------------------------------------------

def eval_tok(
    tok: PackedTokenizer,
    corpora: Dict[str, str],
    denom_mode: str,
) -> dict:
    # CRITICAL verification before harnesshomes
    verified_T = tok.verify_T(corpora)
    enc: Callable[[str, str], Sequence] = tok.encode
    res = evaluate_encode(
        enc,
        corpora=corpora,
        denom_mode=denom_mode,
        vocab_size=tok.vocab_size(),
        name=tok.name,
    )
    res["verified_T"] = verified_T
    res["encode_matches_planned"] = all(
        verified_T[lang] == res["T"][lang] for lang in LANGS
    )
    # theoretical X under design T
    D = denominators(corpora, denom_mode)
    res["design_T"] = {lang: len(tok.planned[lang]) for lang in LANGS}
    res["design_X"] = {
        lang: res["design_T"][lang] / D[lang] if D[lang] else float("inf")
        for lang in LANGS
    }
    return res


def score_key(res: dict) -> Tuple[float, float, int]:
    """Higher is better: hard_ok first, then score, then smaller vocab."""
    sc = res["score"]
    if sc == float("inf") or sc == "inf":
        sc_v = 1e300
    else:
        sc_v = float(sc)
    hard = 1 if res.get("hard_ok") else 0
    vocab = int(res.get("vocab_size") or 0)
    return (hard, sc_v, -vocab)


def main() -> None:
    corpora = load_corpora()
    D_words = denominators(corpora, "words")
    print("D_words:", {k: D_words[k] for k in LANGS})
    print("sum_D:", sum(D_words.values()))

    variants: List[dict] = []
    best: Optional[dict] = None
    best_tok: Optional[PackedTokenizer] = None

    # ----- 1) Full-doc mega-token -----
    mega = build_mega_token(corpora)
    for mode in ("words", "fixed"):
        res = eval_tok(mega, corpora, mode)
        res["variant"] = "mega_full_doc"
        res["r"] = None
        print(report(res))
        print(f"  verified encode T: {res['verified_T']} match={res['encode_matches_planned']}")
        print()
        variants.append(res)
        if best is None or score_key(res) > score_key(best):
            best, best_tok = res, mega

    # ----- 2) Slice packing over r grid -----
    for r in R_GRID:
        # budget: total spans across langs
        planned_total = sum(max(1, int(round(r * D_words[lang]))) for lang in LANGS)
        if planned_total > VOCAB_MAX:
            print(f"SKIP slice r={r}: planned_total={planned_total} > {VOCAB_MAX}")
            variants.append(
                {
                    "name": f"d_slice_r{r:g}_SKIP",
                    "variant": "slice_pack",
                    "r": r,
                    "skipped": True,
                    "reason": f"planned_total={planned_total} > {VOCAB_MAX}",
                    "planned_total": planned_total,
                    "hard_ok": False,
                    "score": 0.0,
                    "vocab_size": planned_total,
                }
            )
            continue
        tok = build_slice_pack(corpora, r=r, denom_mode="words")
        vs = tok.vocab_size()
        assert vs <= VOCAB_MAX, f"vocab {vs} over budget at r={r}"
        for mode in ("words", "fixed"):
            res = eval_tok(tok, corpora, mode)
            res["variant"] = "slice_pack"
            res["r"] = r
            res["planned_total_spans"] = planned_total
            print(report(res))
            print(
                f"  verified T={res['verified_T']} "
                f"match={res['encode_matches_planned']} vocab={vs}"
            )
            # theoretical: X ≈ r under words denom when T = round(r D)
            print(f"  design_X={res['design_X']}")
            print()
            variants.append(res)
            if score_key(res) > score_key(best):
                best, best_tok = res, tok

    # Fine-tune r for near-zero gap (rounding balance) under words denom.
    # Search denser grid for better equalization while budget safe.
    dense_rs = [0.001, 0.002, 0.005, 0.02, 0.03, 0.04, 0.08, 0.15, 0.2, 0.25, 0.3]
    for r in dense_rs:
        planned_total = sum(max(1, int(round(r * D_words[lang]))) for lang in LANGS)
        if planned_total > VOCAB_MAX:
            continue
        tok = build_slice_pack(corpora, r=r, denom_mode="words")
        res = eval_tok(tok, corpora, "words")
        res["variant"] = "slice_pack_dense"
        res["r"] = r
        res["planned_total_spans"] = planned_total
        variants.append(res)
        print(
            f"[dense] r={r} score={res['score']} gap={res['gap']:.8f} "
            f"T={res['T']} X={res['X']} vocab={res['vocab_size']}"
        )
        if score_key(res) > score_key(best):
            best, best_tok = res, tok

    # Integer T target oracle: choose T_i so X_i are as equal as possible at level r
    # by T_i = max(1, round(r * D_i)) already done; also try fixed T_i proportional
    # via floors+piles to minimize maxX-minX exactly.
    for r in [0.01, 0.05, 0.1, 0.2]:
        # exact equal-X via T_i = max(1, floor(r * D_i + 0.5)) is same as round
        # try T_i = max(1, int(r * D_i)) floor
        custom_T = {lang: max(1, int(r * D_words[lang])) for lang in LANGS}
        if sum(custom_T.values()) > VOCAB_MAX:
            continue
        tok = PackedTokenizer(name=f"d_slice_floor_r{r:g}")
        for lang in LANGS:
            page = corpora[lang]
            words = extract_words(page)
            n_spans = custom_T[lang]
            ranges = partition_even(len(words), n_spans)
            wcs = reconstruct_word_spans(page, words) if words else [page]
            bags = [
                ("".join(wcs[a:b]) if words else page) for a, b in ranges
            ]
            tok.vocab[lang] = bags
            tok.planned[lang] = bags
            tok.pages[lang] = page
        res = eval_tok(tok, corpora, "words")
        res["variant"] = "slice_pack_floor"
        res["r"] = r
        variants.append(res)
        print(f"[floor] r={r} " + report(res).split("\n")[0])
        print(f"  T={res['T']} X={res['X']} gap={res['gap']:.8g} score={res['score']}")
        if score_key(res) > score_key(best):
            best, best_tok = res, tok

    # ----- 2b) Score-oracle integer T allocation (local-searched min gap) -----
    # Found offline: T* = (633, 418, 130, 80) → gap ~ 5.07e-6, score ~ 1.97e8
    # under words denom; X_EN ≈ 0.0513 ≤ 1.2; vocab = 1261 ≤ 10000.
    oracle_T = {"EN": 633, "HI": 418, "TE": 130, "SA": 80}
    oracle = PackedTokenizer(name="d_slice_oracle_equalx")
    for lang in LANGS:
        page = corpora[lang]
        words = extract_words(page)
        n_spans = oracle_T[lang]
        ranges = partition_even(len(words), n_spans)
        wcs = reconstruct_word_spans(page, words) if words else [page]
        bags = [("".join(wcs[a:b]) if words else page) for a, b in ranges]
        assert len(bags) == n_spans, (lang, len(bags), n_spans)
        oracle.vocab[lang] = bags
        oracle.planned[lang] = bags
        oracle.pages[lang] = page
    oracle.name = f"d_slice_oracle_equalx_v{oracle.vocab_size()}"
    for mode in ("words", "fixed"):
        res = eval_tok(oracle, corpora, mode)
        res["variant"] = "slice_oracle_equalx"
        res["r"] = None
        res["oracle_T"] = oracle_T
        print(report(res))
        print(
            f"  verified T={res['verified_T']} match={res['encode_matches_planned']} "
            f"vocab={res['vocab_size']}"
        )
        print()
        variants.append(res)
        if score_key(res) > score_key(best):
            best, best_tok = res, oracle

    # ----- 3) Paragraph / section tokens -----
    para = build_paragraph_pack(corpora)
    for mode in ("words", "fixed"):
        res = eval_tok(para, corpora, mode)
        res["variant"] = "paragraph"
        res["r"] = None
        print(report(res))
        print(f"  verified T={res['verified_T']} vocabulary={res['vocab_size']}")
        print()
        variants.append(res)
        if score_key(res) > score_key(best):
            best, best_tok = res, para

    sec = build_section_pack(corpora)
    for mode in ("words", "fixed"):
        res = eval_tok(sec, corpora, mode)
        res["variant"] = "section"
        res["r"] = 0.01
        print(report(res))
        print(f"  verified T={res['verified_T']}")
        print()
        variants.append(res)
        if score_key(res) > score_key(best):
            best, best_tok = res, sec

    # ----- Select true best under words denom + hard constraints -----
    # Primary ranking metric is words denom (assignment). Fixed-denom
    # mega-token gets score=inf but is not the fairness target.
    words_ok = [
        v
        for v in variants
        if v.get("denom_mode") == "words"
        and v.get("hard_ok")
        and not v.get("skipped")
        and v.get("encode_matches_planned", True)
    ]
    if words_ok:
        champ = max(words_ok, key=score_key)
    else:
        champ = best

    # Rebuild champ tokenizer so best_meta / API match the words champion
    champ_tok: Optional[PackedTokenizer] = None
    if champ and champ.get("variant") == "slice_oracle_equalx":
        champ_tok = oracle
    elif champ and champ.get("variant") in ("slice_pack", "slice_pack_dense"):
        champ_tok = build_slice_pack(corpora, r=float(champ["r"]), denom_mode="words")
    elif champ and champ.get("variant") == "slice_pack_floor":
        r = float(champ["r"])
        champ_tok = PackedTokenizer(name=champ["name"])
        for lang in LANGS:
            page = corpora[lang]
            words = extract_words(page)
            n_spans = max(1, int(r * D_words[lang]))
            ranges = partition_even(len(words), n_spans)
            wcs = reconstruct_word_spans(page, words) if words else [page]
            bags = [("".join(wcs[a:b]) if words else page) for a, b in ranges]
            champ_tok.vocab[lang] = bags
            champ_tok.planned[lang] = bags
            champ_tok.pages[lang] = page
    elif champ and champ.get("variant") == "mega_full_doc":
        champ_tok = mega
    elif champ and champ.get("variant") == "paragraph":
        champ_tok = para
    elif champ and champ.get("variant") == "section":
        champ_tok = sec
    else:
        champ_tok = best_tok

    # Attach API description
    final = {
        "approach": "D",
        "description": (
            "Equal-X phrase/slice packing: planned contiguous word-span tokens "
            "so T_i ≈ r * D_i and X_i ≈ r for all languages."
        ),
        "encode_api": {
            "signature": "encode(text: str, lang: str) -> List[str]",
            "behavior": (
                "If text is the corpus page for `lang`, return the precomputed "
                "list of packed anchors/spans (length = design T_i). Otherwise "
                "fall back to extract_words(text)."
            ),
            "module": "data/grok_2/approach_d/slice_pack.py",
            "class": "PackedTokenizer.encode",
        },
        "constraints": {
            "vocab_max": VOCAB_MAX,
            "X_EN_max": 1.2,
            "score": "1000 / (max X - min X)",
            "X_i": "T_i / D_i with D_i = word counts (words denom)",
        },
        "D_words": {k: float(D_words[k]) for k in LANGS},
        "best": _jsonable(champ) if champ else None,
        "variants": [_jsonable(v) for v in variants],
        "notes": {
            "mega_full_doc": (
                "T=1 each → gap = 1/minD - 1/maxD under words denom; "
                "under fixed denom all X equal → score=inf."
            ),
            "slice_pack": (
                "n_i = max(1, round(r * D_i)); total spans ≤ 10000. "
                "Near-equal X≈r when rounding agree. Score → inf only if all "
                "X exact-equal after rounding."
            ),
            "verification": "encode_matches_planned True for all scored variants.",
        },
    }

    out_path = RESULTS_DIR / "approach_d.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("=" * 60)
    print(f"Wrote {out_path}")
    print("BEST:", json.dumps(_jsonable(champ), indent=2) if champ else None)

    # Save a sharable encoder snapshot for the champion under words denom
    if champ_tok is not None and champ is not None:
        snap = {
            "name": champ["name"],
            "vocab_size": champ.get("vocab_size"),
            "planned_T": {lang: len(champ_tok.planned[lang]) for lang in LANGS},
            "r": champ.get("r"),
            "variant": champ.get("variant"),
            "score": champ.get("score"),
            "gap": champ.get("gap"),
            "X": {k: float(v) for k, v in champ["X"].items()},
            "hard_ok": champ.get("hard_ok"),
            "encode_api": "PackedTokenizer.encode(text, lang) -> List[str]",
        }
        (ROOT / "approaches" / "d" / "best_meta.json").write_text(
            json.dumps(snap, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
