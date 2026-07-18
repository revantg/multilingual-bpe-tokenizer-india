# Validation task — cross-word BPE sweep runs

## Background
A parallel sweep (`sweep_vocab.py`) uses a STANDALONE reimplementation of a
cross-word BPE tokenizer to score per-language vocab-size distributions. We do
NOT trust that reimplementation. Your job: independently reproduce ONE run
using the **notebook's own tokenizer classes** (the source of truth) and check
its claimed numbers + three hard invariants.

## Source of truth
The real tokenizer lives in `/home/rev/timepass/tokenizer-assignment/poc_gpt2.py`
(a marimo notebook = plain python). The relevant definitions:
- `bytes_to_unicode`, `BYTE_ENCODER`, `BYTE_DECODER`, `SPECIAL_TOKENS`,
  `pretokenize`, `token_to_byte_symbols`  (a `@app.cell`)
- `merge_optimized`  (`@app.function`)
- `MergeResult`, `TokenizerStatsV2`  (`@app.class_definition`)
- `BPETokenizer`  (defined inside a `@app.cell` — signature:
  `__init__(self, vocab=None, merges=None)` and
  `train_bpe_optimized(self, corpus, vocab_size)`)
- `PhraseBPETokenizer(BPETokenizer)`  (overrides `encode`: splits text on
  `regex.split(r"(\n+)", text)`, keeps newline chunks; pretokenizes by LINES)
- `count_total_vocab(text)` = `len(FAITHFUL_UNIT_RE.findall(text))` where
  `FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")`

You may either import these from `poc_gpt2.py` (it is a marimo app; extracting
functions may be fiddly) OR — cleaner — **copy the class/function source
verbatim** into your own standalone script. If you copy, copy EXACTLY from
`poc_gpt2.py`; do not paraphrase. Do NOT reuse `sweep_vocab.py` — that is the
thing under test.

## Corpora
```
base = /home/rev/timepass/tokenizer-assignment/datasets/full_page_md
EN = india_en.md   HI = india_hi.md   TE = india_te.md   SA = india_sa.md
```
Read each as UTF-8 text.

## How a "run" is defined
A run = a cumulative per-language vocab schedule, trained on ONE tokenizer in
this fixed order: **EN, then HI, then SA, then TE**. For each language:
```
lines = [l for l in TEXT.split("\n") if l.strip()]
tok.train_bpe_optimized(lines, cumulative_vocab_target_for_that_language)
```
So after the EN phase `len(tok.vocab)==EN_target`, after HI phase
`==HI_target`, etc. TE target is 10000 (fills the budget).

Then score EACH language on the FINAL tokenizer:
```
X[lang] = len(tok.encode(full_text_of_lang)) / count_total_vocab(full_text_of_lang)
spread  = max(X) - min(X)
```

## Checks to perform and REPORT
1. **Reproduce X**: report X_EN, X_HI, X_TE, X_SA and spread. State whether they
   match the claimed values (tolerance ±0.01).
2. **Round-trip lossless**: for every language, assert
   `tok.decode(tok.encode(full_text)) == full_text`. Report True/False per lang.
   (PhraseBPETokenizer preserves newlines, so FULL-TEXT round-trip must hold —
   not just per-line.)
3. **No UNK after encode**: `<unk>` is `tok.vocab["<unk>"]`. Assert that id never
   appears in `tok.encode(full_text)` for any language. Report count of UNK ids.
4. **Vocab size == 10000**: assert `len(tok.vocab) == 10000` on the final
   tokenizer. Report the actual number.

## Output format
Report a compact table: claimed vs reproduced X per language, spread, and
PASS/FAIL for each of the 3 invariants, plus any discrepancy you find and your
best explanation. Keep it tight. Do not edit any repo files except your own
scratch script under /tmp.
