# Multilingual BPE Tokenizer — Submission 2 (the overfit push)

> **On the use of AI.** Every idea, hypothesis, and design decision here is the
> author's own. AI was used **only to automate execution** — running sweeps,
> tabulating results, and exporting transcripts.

This is a **second, separate submission**. Submission 1 (frozen) is an honest,
robust byte-level BPE scoring **9953**. Submission 2 asks a different question:
*how far can the score go if we deliberately tune to this exact corpus?* — and is
transparent about the overfitting that entails.

**Live:** [widget + slides + notebook](https://india-bpe-tokenizer-overfit.surge.sh)

## What changed from Submission 1

Two fair, principled knobs on the same hand-written byte-level BPE:

1. **Fractional language upsampling** (deterministic Bresenham line replication).
   The integer sweep never raised Maithili above 2× even though MAI was the spread
   ceiling. Fractional weights tune each language's fertility precisely.
2. **Newline attachment**: train on `line + "\n"` and encode in newline-carrying
   chunks, so `…\n` merges are learnable. Previously every newline was a
   permanently unmergeable token — a fertility tax that fell hardest on MAI.

## Result (champion 2) — weights EN:2 HI:2.2 TE:3.7 MAI:4.05

| Lang | tokens | faithful units | fertility X |
|---|---:|---:|---:|
| English  | 106,601 | 186,367 | 0.571995 |
| Hindi    | 50,673  | 88,359  | 0.573490 |
| Telugu   | 20,941  | 36,292  | 0.577014 |
| Maithili | 3,326   | 5,808   | 0.572658 |

```
spread = 0.577014 − 0.571995 = 0.005019
score  = 1000 / 0.005019     = 199236   (English penalty factor = 1.0)
vocab = 10000 · roundtrip lossless · 0 UNK · all X ≤ 1.2
```

## The honesty note (the whole point of this submission)

**199236 is snapshot-specific.** It is real, deterministic, and exactly what a
re-run on *this* corpus measures — but ~half the margin below spread 0.02 is
merge-discreteness luck on these specific files. Under corpus perturbation
(different wiki snapshot, same weights, no re-tuning) the recipe scores **26k–54k**
vs Submission 1's **7.8k–9.0k** on identical perturbations — a **3.3–6× structural
win** that *does* generalize. We report both numbers.

## Every result we measured (all vocab 10000, 0 UNK, lossless)

| # | approach / weights | X_EN | X_HI | X_TE | X_MAI | spread | score | ≤1.2 |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| 1 | word-level subword (5-lang) | 0.9841 | 1.5637 | 1.6712 | 1.7119 | 0.7278 | 1374 | no |
| 2 | sentence-level (5-lang) | 0.5584 | 0.5593 | 0.6176 | — | 0.3777 | 2648 | yes |
| 3 | paragraph-level (5-lang) | 0.5637 | 0.5690 | 0.6391 | — | 0.3839 | 2605 | yes |
| 4 | instructor reference (3/4/4/2) | 0.5977 | 0.5793 | 0.6731 | 0.7331 | 0.1538 | 6503 | yes |
| 5 | integer 1/1/1/1 | 0.558 | 0.567 | 0.663 | 0.657 | 0.1048 | 9546 | yes |
| 6 | **integer 2/2/3/2 — Submission 1 (frozen)** | 0.5669 | 0.5774 | 0.6012 | 0.6674 | 0.1005 | 9953 | yes |
| 7 | fractional 2/2/3.5/3.5 | 0.5751 | 0.5828 | 0.5797 | 0.5914 | 0.0164 | 61100 | yes |
| 8 | fractional 2/2.1/3.5/3.8 | 0.5757 | 0.5783 | 0.5854 | 0.5895 | 0.0138 | 72256 | yes |
| 9 | fractional 2/2.1/3.4/4.0 (best plain) | 0.5766 | 0.5794 | 0.5843 | 0.5835 | 0.0077 | 129954 | yes |
| 10 | +newline 2/2.1/3.7/4.0 | 0.5702 | 0.5793 | 0.5713 | 0.5773 | 0.0091 | 109587 | yes |
| **11** | **+newline 2/2.2/3.7/4.05 — Submission 2 (this)** | **0.571995** | **0.573490** | **0.577014** | **0.572658** | **0.005019** | **199236** | **yes** |

Rows 1–3 are the 5-language granularity exploration (Sanskrit later dropped — its
URL-polluted page pinned the spread); rows 5–11 are the 4-language clean-corpus
search. Two-stage MAI-only merge budgets (best ~31k) and other ideas (digit
splitting, punctuation isolation, DP encoding) were measured and rejected as
dominated or unfair. Full audit: `experiments/SCORE_PUSH_FINDINGS.md`.

## Reproduce

```
cd submission2
python train.py            # writes tokenizer.json + metrics.json
python build_artifacts.py  # writes vocab.json + tokens.txt
```
