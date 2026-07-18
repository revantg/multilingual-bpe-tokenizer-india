# RED-PILL FINDINGS: `2/2/3/2` byte-level BPE tokenizer vs the Axiom grading gate

Adversarial validation executed end-to-end per `REDPILL_VALIDATION_BRIEF.md`.
Everything below was recomputed independently on a fresh, from-scratch retrain
(single process, ~139 s, CORPUS=instructor_mai, weights EN:2/HI:2/TE:3/MAI:2).
Nothing was taken from the script's self-reported numbers.

## Gate-by-gate verdicts

| Gate | Verdict | Evidence |
|---|---|---|
| FAITHFUL ROUNDTRIP (visible non-ws chars preserved) | **PASS** | Byte-**exact** (not merely visible-char) roundtrip on all 4 full corpus files AND all 54 adversarial samples AND raw-byte torture tests. 0 mismatches. |
| NO UNK | **PASS** | 0 UNK on full corpus, 0 UNK on every adversarial input, plus a structural proof (below) that UNK is unreachable. |
| All four fertilities ≤ 1.2 | **PASS** | max fertility = 0.6674 (MAI), far under 1.2. |
| vocab == 10000 | **PASS** | Exactly 10000 = 4 specials + 256 base byte symbols + 9740 merges. |
| English penalty | **PASS** (factor = 1.0) | X_EN = 0.5669 ≤ 1.2 → exp(max(0, 0.5669/1.2 − 1)) = 1.0. Production score = raw score. |
| Deterministic encode | **PASS** | Full EN corpus encoded twice → identical id sequences; all 54 samples re-encoded → identical. Merge ranks are unique so `_apply_merges` tie-breaking is unambiguous. |
| Reproducibility | **PASS** | Fresh retrain produced a vocab AND merge list **bit-identical** to `/tmp/tok_2232.json` (10000 vocab, 9740 merges). |
| Fourth language = Maithili (instructor's), not Sanskrit | **CONFIRMED** | Loader mode `instructor_mai` reads `datasets/instructor/mai.txt`; the zero was recorded against Sanskrit. This is the intended fix. |

## Independently recomputed numbers (my regex, my counter — not the script's)

Faithful-unit regex used: `regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")`
(written and applied independently of `count_total_vocab`).

| Lang | tokens | faithful units | fertility X |
|---|---|---|---|
| EN  | 105,649 | 186,367 | **0.5669** |
| HI  | 51,019  | 88,359  | **0.5774** |
| TE  | 21,817  | 36,292  | **0.6012** |
| MAI | 3,876   | 5,808   | **0.6674** |

- spread (X4 − X1) = **0.1005**
- raw score = 1000 / 0.1005 = **9953.4**
- penalty factor = **1.0** → adjusted/production score = **9953.4**

These match the candidate's self-reported X 0.567/0.577/0.601/0.667, spread
0.1005, adj 9953 to all reported figures. **No discrepancy** between my
independent implementation and the script's.

Note: the brief defines the penalty on `en_ratio` but `sweep_weighted.py:216`
computes it on `X["HI"]`. Immaterial here — both are ≤ 1.2 so both give
factor 1.0 — but if fertilities ever rose the script would under-report the
penalty. Flagged for hygiene, not a grading risk with these numbers.

## Full-corpus roundtrip

For each of en/hi/te/mai: `decode(encode(text)) == text` **byte-exactly**
(a strictly stronger condition than the visible-char gate). 0 UNK, 0 special
token ids ever emitted by `encode`. PASS on all four.

## Adversarial battery (54 held-out samples) — 0 failures

Coverage: the exact grader sample `"India's population is 1,428,627,663."`
(decodes exactly, 13 tokens, 0 UNK — the previous zero is fixed), large
comma-grouped numbers (US and Indian lakh/crore grouping), decimals, percent,
scientific notation, currency ₹/$/€/¥/£, negatives with °C/°F, phone numbers,
mixed en+hi+te+mai single lines, emoji incl. ZWJ families/skin tones/flag
sequences, rare CJK ext-B, ZWJ/ZWNJ inside Devanagari, stacked combining
marks, Devanagari and Telugu conjunct clusters, Vedic accents, Arabic/Hebrew
RTL + bidi control marks, Thai/Lao, tabs/newlines/CRLF, URLs (incl.
percent-encoded), emails, markdown tables and mixed markdown, pure
punctuation/symbol runs, brackets/quotes of many flavors, math/arrow/box
symbols, non-BMP mathematical alphanumerics, Gothic/Ugaritic/hieroglyphs,
C0 controls (BEL/BS/VT/FF/ESC/DEL), NBSP/figure-space/thin-space/ideographic
space/BOM, soft hyphens, JSON blobs, the faithful-unit regex source itself,
a 3.4 KB repetition-stress string, mixed digit systems
(Arabic-Indic/Devanagari/Telugu/Thai), mojibake-style Krutidev text, empty
string, whitespace-only, lone newlines, single characters.

Every sample: **byte-exact roundtrip, 0 UNK, deterministic**.
**No roundtrip or UNK failure found after 54 adversarial samples.**

## UNK production attempts (raw-byte torture) — could not produce one

- All 256 bytes 0–255 as one string (latin-1 decoded): 0 UNK, exact roundtrip.
- Invalid UTF-8 byte sequences decoded with `errors="replace"` (U+FFFD-laden
  text): 0 UNK, exact roundtrip.
- 5,000 uniformly random codepoints across all planes (surrogates excluded):
  0 UNK, exact roundtrip.
- Sample from every plane incl. U+10FFFF, PUA, U+2028/2029: 0 UNK, exact.

**Structural proof that UNK is unreachable:** `encode` maps text →
UTF-8 bytes → the 256 base byte symbols, all of which are verified present in
the vocab; `_apply_merges` only ever produces concatenations of merge pairs,
and every one of the 9740 merge products is verified present in the vocab.
Therefore `self.vocab[sym]` can never miss and `<unk>` (id 3) can never be
emitted. No hole found.

**One non-gate edge case:** a Python string containing a *lone surrogate*
(e.g. `chr(0xD800)`, only constructible programmatically, impossible in any
UTF-8 file the grader reads) makes `encode` raise `UnicodeEncodeError` rather
than emit UNK or corrupt output. A raise is fail-safe (no silent corruption)
and cannot occur on file-based grading input. Not a grading risk.

## Residual risks (named exactly)

1. **Grader corpus mismatch** — validation used
   `datasets/instructor/{en,hi,te,mai}.txt`. If the grader re-runs on a
   *different* Maithili file than `mai.txt`, fertilities shift (the tokenizer
   itself will still roundtrip anything, so only the score, not the gate, is
   exposed). Fix: confirm `mai.txt` is byte-identical to the instructor
   bundle's Maithili file before submission.
2. **Whitespace normalization** — none needed: roundtrip is byte-exact, which
   dominates the visible-char rule regardless of how the grader strips
   whitespace.
3. **Penalty variable** — `sweep_weighted.py` uses X_HI where the contract says
   en_ratio; harmless at these values, worth a one-line fix if the script is
   ever reused near the 1.2 boundary.

## Final judgment

**This would NOT get a zero again.** The old failure mode is structurally
eliminated: the byte-level base vocabulary makes UNK impossible and roundtrip
byte-exact on *arbitrary* input, proven by construction and by 54 adversarial
samples plus exhaustive byte/codepoint torture with zero failures. The
independently recomputed gate numbers — X 0.5669/0.5774/0.6012/0.6674, all
≤ 1.2, vocab exactly 10000, penalty factor 1.0, production score **9953.4** —
match the self-report and were reproduced bit-identically from scratch. Every
grading gate: **PASS**.
