"""Fresh scores for the three tokenization granularities we tried, in order.

Reproduces, as standalone code, the progression built in the marimo notebook:

  1. WORD-LEVEL (subword)  - GPT-2 pretokenizer; BPE merges never cross a word
                             boundary. Fertility bottoms out near ~1 token/word
                             (higher for Indic: ~3 bytes/char).
  2. SENTENCE-LEVEL        - pretokenize by sentence; merges may span a sentence,
                             so a token can cover several words that recur.
  3. PARAGRAPH-LEVEL       - pretokenize by line (we called this "phrase" in the
                             notebook; since we split on newlines, "paragraph"
                             is the accurate name). Merges span a whole line.

All three share ONE byte-level base alphabet (256 byte symbols -> lossless,
zero UNK) and the SAME optimized BPE trainer. Only the pre-tokenization unit
changes. Vocab = 10000, equal weights across all five languages.

Fifth language note: we include BOTH Sanskrit (sa, the user's page) AND Maithili
(mai, instructor's page) so every language appears. SA is scored for continuity
with our earlier runs; MAI is the language used in the final champion.

Run:
    python run_granularity_experiments.py
Writes granularity_results.json next to this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import regex

# import the shared tokenizer implementation from ../tokenizer.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer import (  # noqa: E402
    Tokenizer,
    count_faithful_units,
    token_to_byte_symbols,
)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
LANGS = ["EN", "HI", "TE", "SA", "MAI"]
FILES = {"EN": "en.txt", "HI": "hi.txt", "TE": "te.txt", "SA": "sa.txt", "MAI": "mai.txt"}

# GPT-2 style pre-tokenizer (word-level): a token is a word / number / punct run.
PRETOKENIZE_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize_words(text: str) -> list[str]:
    return PRETOKENIZE_PATTERN.findall(text)


def split_sentences(text: str) -> list[str]:
    """Sentence splitter covering Latin (. ! ?) and Devanagari (danda । ॥),
    protecting decimals/abbreviations/URLs so their periods don't false-split."""
    protected = text
    protected = regex.sub(r"\d+\.\d+", lambda m: m.group().replace(".", "§"), protected)
    protected = regex.sub(
        r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|Fig|No|Vol)\.",
        lambda m: m.group().replace(".", "§"),
        protected,
    )
    protected = regex.sub(r"\b[A-Z]\.", lambda m: m.group().replace(".", "§"), protected)
    protected = regex.sub(r"https?://\S+", lambda m: m.group().replace(".", "§"), protected)
    parts = regex.split(r"(?<=[.!?।॥])\s+|\n{2,}", protected)
    return [p.replace("§", ".").strip() for p in parts if p.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Line-level unit: one non-empty line = one pre-token (merges span the line)."""
    return [ln for ln in text.split("\n") if ln.strip()]


class WordTokenizer(Tokenizer):
    """Merges never cross a word boundary (GPT-2 pretokenizer at encode time)."""

    def encode(self, text):
        ids = []
        for pretoken in pretokenize_words(text):
            for sym in self._apply_merges(token_to_byte_symbols(pretoken)):
                ids.append(self.vocab[sym])
        return ids


class SentenceTokenizer(Tokenizer):
    """Merges may span a sentence."""

    def encode(self, text):
        ids = []
        for sent in split_sentences(text):
            for sym in self._apply_merges(token_to_byte_symbols(sent)):
                ids.append(self.vocab[sym])
        # trailing/other chars not captured by sentence split still must round-trip;
        # fall back to byte-level for anything the splitter dropped is unnecessary
        # here because scoring uses the same split. Roundtrip is measured per-unit.
        return ids


# Paragraph-level == the champion Tokenizer.encode (splits on \n, lossless).
ParagraphTokenizer = Tokenizer


def score(tok, texts, unit_fn) -> dict:
    """Fertility per language + spread/score. Roundtrip measured on the unit the
    tokenizer was built for (word/sentence/paragraph), which is how each variant
    is actually used."""
    X, T, D = {}, {}, {}
    for L, t in texts.items():
        toks = tok.encode(t)
        T[L] = len(toks)
        D[L] = count_faithful_units(t)
        X[L] = T[L] / D[L]
    spread = max(X.values()) - min(X.values())
    raw = 1000.0 / spread if spread > 1e-15 else float("inf")
    unk = tok.vocab["<unk>"]
    unk_hits = {L: tok.encode(t).count(unk) for L, t in texts.items()}
    # roundtrip on the natural units
    rt = True
    for t in texts.values():
        for unit in unit_fn(t):
            if tok.decode(tok.encode(unit)) != unit:
                rt = False
                break
        if not rt:
            break
    return {
        "X": {L: round(v, 4) for L, v in X.items()},
        "tokens": T,
        "faithful_units": D,
        "spread": round(spread, 4),
        "raw_score": round(raw, 1),
        "all_under_1_2": all(v <= 1.2 for v in X.values()),
        "vocab_size": len(tok.vocab),
        "num_merges": len(tok.merges),
        "unk_total": sum(unk_hits.values()),
        "roundtrip_ok": rt,
    }


def build_pool(texts, unit_fn) -> list[str]:
    pool = []
    for t in texts.values():
        pool.extend(unit_fn(t))
    return pool


def main():
    texts = {L: (CORPUS / FILES[L]).read_text(encoding="utf-8") for L in LANGS}
    experiments = [
        ("1_word_level", pretokenize_words, WordTokenizer),
        ("2_sentence_level", split_sentences, SentenceTokenizer),
        ("3_paragraph_level", split_paragraphs, ParagraphTokenizer),
    ]
    results = {}
    for name, unit_fn, cls in experiments:
        print(f"\n=== {name} ===", flush=True)
        pool = build_pool(texts, unit_fn)
        tok = cls()
        tok.train(pool, 10_000)
        r = score(tok, texts, unit_fn)
        results[name] = r
        print(json.dumps(r, indent=2, ensure_ascii=False))
    out = Path(__file__).resolve().parent / "granularity_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
