# Verified numbers (single source of truth) — DO NOT INVENT, copy exactly

All numbers below are reproduced by `solution/train.py` and
`solution/experiments/run_granularity_experiments.py`. The grader RE-RUNS the
tokenizer, so every claim on the frontend must match these exactly.

## Assignment
- Score = 1000 / (X4 − X1); X_L = tokens(L) / faithful_units(L); X4=max, X1=min.
- faithful_unit = one contiguous \p{L}\p{M}\p{N} run OR one visible non-space punct/symbol char.
- Gates: vocab = 10000; every X_L ≤ 1.2; decode(encode(text)) preserves visible chars; 0 UNK.
- English penalty: score / exp(max(0, X_EN/1.2 − 1)). Here X_EN=0.567 ≤ 1.2 → factor 1.0.

## CHAMPION (what we submit) — weights EN:2 HI:2 TE:3 MAI:2, clean corpus
| Lang | tokens | faithful units | fertility X |
|---|---:|---:|---:|
| English  | 105,649 | 186,367 | 0.5669 |
| Hindi    | 51,019  | 88,359  | 0.5774 |
| Telugu   | 21,817  | 36,292  | 0.6012 |
| Maithili | 3,876   | 5,808   | 0.6674 |

- spread = 0.1005, score = **9953**, penalty factor 1.0
- vocab = 10000 (4 specials + 256 byte symbols + 9740 merges)
- roundtrip lossless, 0 UNK, all X ≤ 1.2

## Approaches (fresh scores, all 5 langs en/hi/te/sa/mai; vocab 10000, 0 UNK, lossless)
### 1. Word-level (subword) — FAILS gate
X: EN 0.9841, HI 1.5637, TE 1.6712, SA 1.5139, MAI 1.7119 · spread 0.7278 · score 1374 · all≤1.2 NO
### 2. Sentence-level — passes
X: EN 0.5584, HI 0.5593, TE 0.6590, SA 0.2814, MAI 0.6176 · spread 0.3777 · score 2648 · all≤1.2 YES
### 3. Paragraph-level (split on newline) — passes
X: EN 0.5637, HI 0.5690, TE 0.6720, SA 0.2881, MAI 0.6391 · spread 0.3839 · score 2605 · all≤1.2 YES

## Instructor reference (their shipped metrics.json) — for comparison
weights en:3 hi:4 te:4 mai:2, Metaspace base, HF tokenizers
X: en 0.5977, hi 0.5793, te 0.6731, mai 0.7331 · spread 0.1538 · score 6503
→ We beat it by +3451 (9953 vs 6503) via tighter spread (near-equal weights + clean corpus).

## Clean-corpus weight sweep (mai_sweep_results.json), top rows — all pass gates
| weights EN/HI/TE/MAI | X_EN | X_HI | X_TE | X_MAI | spread | score |
|---|---|---|---|---|---|---|
| 2/2/3/2 | 0.567 | 0.577 | 0.601 | 0.667 | 0.1005 | 9953 |
| 1/1/1/1 | 0.558 | 0.567 | 0.663 | 0.657 | 0.1048 | 9546 |
| 1/1/2/1 | 0.575 | 0.588 | 0.559 | 0.681 | 0.1218 | 8212 |
| 2/3/3/2 | 0.580 | 0.541 | 0.629 | 0.671 | 0.1298 | 7707 |
| 3/3/3/2 | 0.558 | 0.567 | 0.655 | 0.715 | 0.1574 | 6353 |

## Downloadable artifacts (in solution/)
- vocab.json  (10000 tokens + 9740 merges + metadata) — THE downloadable vocab
- tokens.txt  (id<TAB>token, one per line)
- tokenizer.json (vocab + merges, minimal)

## Key story points
- First submission scored 0 on the faithfulness gate (word-level emitted <UNK> on
  "India's population is 1,428,627,663."). Fixed by a byte-level base (256 byte symbols)
  → any UTF-8 decodes byte-exact, UNK structurally impossible.
- Hand-written BPE (no libraries), max-heap for production-grade training speed.
- Progression: word → sentence → paragraph granularity, then clean corpus + near-equal
  weights converged all four fertilities into a 0.567–0.667 band.
