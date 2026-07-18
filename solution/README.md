# Multilingual BPE Tokenizer — ERA V5 Assignment 2

> **On the use of AI.** Every idea, hypothesis, and design decision in this
> solution is the author's own. AI was used **only to automate execution** —
> writing boilerplate, running sweeps, tabulating results, and exporting
> transcripts. It did **not** generate the approach, the diagnosis, or the
> strategy.

## What this is

One shared **10,000-token** byte-level BPE tokenizer for the *India* Wikipedia
page in four languages. The assignment score is `1000 / (X4 − X1)` where each
`X_L = tokens(L) / faithful_units(L)` is a language's fertility, and X4/X1 are
the largest/smallest of the four. Lower spread ⇒ higher score. Hard gates:
vocab = 10000, every `X_L ≤ 1.2`, `decode(encode(text))` preserves all visible
characters, and **zero UNK**.

## Final result (champion)

Trained on the clean wiki-faithful corpus, weights **EN:2 HI:2 TE:3 MAI:2**:

| Lang | tokens | faithful units | fertility X |
|---|---:|---:|---:|
| English  | 105,649 | 186,367 | 0.5669 |
| Hindi    | 51,019  | 88,359  | 0.5774 |
| Telugu   | 21,817  | 36,292  | 0.6012 |
| Maithili | 3,876   | 5,808   | 0.6674 |

```
spread = 0.6674 − 0.5669 = 0.1005
score  = 1000 / 0.1005   = 9953   (English penalty factor = 1.0)
vocab = 10000 · roundtrip lossless · 0 UNK · all X ≤ 1.2
```

Reproduce: `python train.py` (writes `tokenizer.json` + `metrics.json`).
Compare to the instructor: `python compare_to_instructor.py`.

## Why a byte-level base — the gate that mattered

The **first submission scored 0**, not on the metric but on the faithfulness
gate: its whitespace/word-level scheme emitted `<UNK>` on unseen text
(`"India's population is 1,428,627,663."` → `"… 1 <UNK> 4 28 <UNK> …"`).

The fix is a **GPT-2-style byte-level base alphabet**: all 256 byte values get a
symbol, so *any* UTF-8 input decodes back byte-exactly and UNK is structurally
impossible. This is verified adversarially (54 held-out samples + full byte
torture, 0 failures) in [`experiments/REDPILL_FINDINGS.md`](experiments/REDPILL_FINDINGS.md).

---

# Approaches tried (in order), with fresh scores

All four use the same byte-level base and the same optimized BPE trainer
(dedup pre-tokens via a `Counter`, then a max-heap that pops the most frequent
pair in O(log n) and updates only touched pairs). Only the **pre-tokenization
unit** — how far a single merge is allowed to reach — changes between 1→3.
Approaches 1–3 were re-run fresh for this writeup on all five languages
(en/hi/te + both **sa** and **mai**); raw JSON in
[`experiments/granularity_results.json`](experiments/granularity_results.json).

Every run below: vocab = 10000, 0 UNK, lossless roundtrip.

## 1. Word-level (subword) BPE

The classic scheme: a GPT-2 regex pre-tokenizer cuts text into word / number /
punctuation runs, and **merges never cross a word boundary**. Fertility bottoms
out near ~1 token per word for English, but Indic scripts (≈3 UTF-8 bytes per
character) stay high — so the spread is huge and the ≤1.2 gate fails.

| Lang | X (fertility) |
|---|---:|
| EN  | 0.9841 |
| HI  | 1.5637 |
| TE  | 1.6712 |
| SA  | 1.5139 |
| MAI | 1.7119 |

```
spread = 0.7278 · score = 1374 · all X ≤ 1.2? NO (four languages over 1.2)
```

**Verdict:** cannot pass the gate. Subword-only BPE has no way to push Indic
fertility down — the floor is "one token per character-ish," well above 1.2.

## 2. Sentence-level BPE

Pre-tokenize by **sentence** (Latin `. ! ?` and Devanagari danda `। ॥`, with
decimals/abbreviations/URLs protected). Now a merge can span several words that
recur inside a sentence, so a single token can cover multiple faithful units and
fertility drops **below 1**.

| Lang | X (fertility) |
|---|---:|
| EN  | 0.5584 |
| HI  | 0.5593 |
| TE  | 0.6590 |
| SA  | 0.2814 |
| MAI | 0.6176 |

```
spread = 0.3777 · score = 2648 · all X ≤ 1.2? YES
```

**Verdict:** clears the gate and jumps the score ~2×. But SA collapses to 0.28
(its page is dominated by repeated URL-encoded strings that compress trivially),
pinning the spread's minimum and holding the score down.

## 3. Paragraph-level BPE (called "phrase" in the notebook)

Pre-tokenize by **line** (`split("\n")`). Since we split on newlines, the unit
is really a paragraph/line, not a phrase — hence the rename. Merges may now span
an entire line, the most aggressive cross-word setting. Newlines are preserved
as their own chunks so roundtrip stays byte-exact.

| Lang | X (fertility) |
|---|---:|
| EN  | 0.5637 |
| HI  | 0.5690 |
| TE  | 0.6720 |
| SA  | 0.2881 |
| MAI | 0.6391 |

```
spread = 0.3839 · score = 2605 · all X ≤ 1.2? YES
```

**Verdict:** essentially tied with sentence-level here. Reaching further per
merge lowers each fertility slightly but does **not** shrink the spread, because
the spread is dominated by one outlier language (SA at 0.29), not by the
granularity. This proved the granularity knob was exhausted — the lever left was
**which languages and how the vocab is split among them.**

### The corpus insight behind approaches 2–3

SA (the user's Sanskrit page) sits at ~0.28 in every cross-word run and single-
handedly sets X1. Investigation ([`experiments/DEEPDIVE_FINDINGS.md`](experiments/DEEPDIVE_FINDINGS.md))
showed the Indic pages were ~30–70% **percent-encoded URL hex** (`%E0%A4%AD…`),
which BPE compresses into giant tokens and distorts fertility. Switching the 4th
language to the instructor's clean **Maithili** page (and using clean en/hi/te)
removed the outlier — that clean corpus is what the champion trains on.

## 4. Vocab-distribution sweep (reducing the spread)

With granularity exhausted, the remaining lever is **how the 10,000 slots are
split across languages** — done by upsampling each language's lines before
pooling (a language with more weight gets more merges, lowering its fertility).
We swept many EN/HI/TE/SA vocab-budget schedules; the table below is the full
run captured in [`experiments/vocab_schedule_sweep_results.json`](experiments/vocab_schedule_sweep_results.json)
(35 candidates, sorted by score). None clears `≤1.2` because these ran on the
**polluted** corpus — this is the sweep that exposed the corpus problem, since
no budget split could rescue EN from >1.2 while SA sat near 0.5.

| EN/HI/TE/SA (budget) | X_EN | X_HI | X_TE | X_SA | spread | score | all ≤1.2 |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 4500/6500/10000/7000 | 1.460 | 0.881 | 1.037 | 0.817 | 0.6432 | 1555 | no |
| 4500/7500/10000/8000 | 1.464 | 0.868 | 1.066 | 0.816 | 0.6481 | 1543 | no |
| 3500/6500/10000/7000 | 1.468 | 0.864 | 1.032 | 0.809 | 0.6590 | 1518 | no |
| 3500/7500/10000/8000 | 1.474 | 0.862 | 1.063 | 0.808 | 0.6655 | 1503 | no |
| 3500/5500/10000/6500 | 1.470 | 0.878 | 1.040 | 0.803 | 0.6667 | 1500 | no |
| 4500/6500/10000/7500 | 1.467 | 0.874 | 1.052 | 0.800 | 0.6668 | 1500 | no |
| 2500/6500/10000/7000 | 1.480 | 0.862 | 1.023 | 0.809 | 0.6711 | 1490 | no |
| 2500/5500/10000/6500 | 1.479 | 0.865 | 1.030 | 0.808 | 0.6715 | 1489 | no |
| 4500/6500/10000/8000 | 1.468 | 0.874 | 1.066 | 0.795 | 0.6727 | 1487 | no |
| 3500/6500/10000/7500 | 1.475 | 0.857 | 1.047 | 0.796 | 0.6787 | 1474 | no |
| 3500/5500/10000/7000 | 1.472 | 0.869 | 1.032 | 0.788 | 0.6847 | 1460 | no |
| 2500/6500/10000/7500 | 1.486 | 0.854 | 1.038 | 0.797 | 0.6892 | 1451 | no |
| 4500/7500/10000/8500 | 1.505 | 0.872 | 1.095 | 0.814 | 0.6912 | 1447 | no |
| 3500/6500/10000/8000 | 1.480 | 0.856 | 1.063 | 0.788 | 0.6921 | 1445 | no |
| 2500/5500/10000/7000 | 1.487 | 0.858 | 1.023 | 0.790 | 0.6975 | 1434 | no |
| 4500/8500/10000/9000 | 1.526 | 0.880 | 1.124 | 0.829 | 0.6976 | 1434 | no |
| 2500/6500/10000/8000 | 1.491 | 0.853 | 1.055 | 0.789 | 0.7024 | 1424 | no |
| 3500/7500/10000/8500 | 1.512 | 0.864 | 1.092 | 0.806 | 0.7059 | 1417 | no |
| 4500/7500/10000/9000 | 1.528 | 0.875 | 1.124 | 0.809 | 0.7189 | 1391 | no |
| 3500/7500/10000/9000 | 1.538 | 0.867 | 1.127 | 0.806 | 0.7316 | 1367 | no |
| 4500/8500/10000/9500 | 1.557 | 0.873 | 1.185 | 0.808 | 0.7486 | 1336 | no |
| 2500/4500/10000/5000 | 1.441 | 0.618 | 1.022 | 0.513 | 0.9280 | 1078 | no |
| 3500/5500/10000/6000 | 1.462 | 0.624 | 1.036 | 0.520 | 0.9425 | 1061 | no |
| 2500/4500/10000/5500 | 1.448 | 0.609 | 1.025 | 0.490 | 0.9572 | 1045 | no |
| 2500/4500/10000/6000 | 1.448 | 0.608 | 1.025 | 0.485 | 0.9626 | 1039 | no |
| 2500/5500/10000/6000 | 1.476 | 0.609 | 1.025 | 0.513 | 0.9629 | 1038 | no |
| 1500/3500/10000/4000 | 1.469 | 0.611 | 1.005 | 0.497 | 0.9717 | 1029 | no |
| 1500/5500/10000/6000 | 1.500 | 0.594 | 1.009 | 0.504 | 0.9964 | 1004 | no |
| 1500/4500/10000/5000 | 1.498 | 0.599 | 1.006 | 0.496 | 1.0017 | 998  | no |
| 1500/3500/10000/4500 | 1.485 | 0.602 | 1.006 | 0.473 | 1.0120 | 988  | no |
| 1500/3500/10000/5000 | 1.485 | 0.601 | 1.006 | 0.467 | 1.0175 | 983  | no |
| 1500/5500/10000/6500 | 1.514 | 0.585 | 1.024 | 0.486 | 1.0275 | 973  | no |
| 1500/4500/10000/5500 | 1.514 | 0.591 | 1.009 | 0.477 | 1.0370 | 964  | no |
| 1500/4500/10000/6000 | 1.518 | 0.591 | 1.009 | 0.469 | 1.0490 | 953  | no |
| 1500/5500/10000/7000 | 1.522 | 0.578 | 1.020 | 0.469 | 1.0531 | 950  | no |

**Verdict:** on the polluted corpus every split fails — EN never drops under 1.2
while SA is stuck near 0.5, so the spread floor is ~0.64. This is the run that
proved *the corpus, not the vocab split, was the problem.* After cleaning the
corpus and switching SA→MAI, the same weighting lever (now near-equal,
`{2,2,3,2}`) converged all four fertilities into a 0.567–0.667 band → spread
0.10 → **9953** (see [`experiments/mai_sweep_results.json`](experiments/mai_sweep_results.json)
for the 10-candidate clean sweep that picked the champion).

---

# How we compare to the instructor

Same metric, same four languages:

| field | OURS | INSTRUCTOR |
|---|---:|---:|
| base alphabet | byte-level (256) | Metaspace (char) |
| library | hand-rolled BPE | HF `tokenizers` |
| weights EN/HI/TE/MAI | 2/2/3/2 | 3/4/4/2 |
| X_en | 0.5669 | 0.5977 |
| X_hi | 0.5774 | 0.5793 |
| X_te | 0.6012 | 0.6731 |
| X_mai | 0.6674 | 0.7331 |
| **spread** | **0.1005** | 0.1538 |
| **score** | **9953** | 6503 |
| vocab | 10000 | 10000 |
| roundtrip / UNK | lossless / 0 | lossless / 0 |

We beat the reference by **+3451** points purely on a tighter spread: near-equal
weights keep all four fertilities clustered, where the instructor's Indic-heavy
`{3,4,4,2}` strands Maithili at 0.733. Details and the full investigation of
"how did the instructor get such a good score" are in
[`experiments/DEEPDIVE_FINDINGS.md`](experiments/DEEPDIVE_FINDINGS.md).

---

# Directory layout

```
solution/
├── README.md                     ← this file
├── tokenizer.py                  ← byte-level BPE trainer + encode/decode (lossless)
├── train.py                      ← reproduce champion; writes tokenizer.json + metrics.json
├── compare_to_instructor.py      ← side-by-side vs the reference
├── tokenizer.json                ← the champion tokenizer (vocab + merges), downloadable
├── metrics.json                  ← champion scores/gates
├── corpus/                       ← clean wiki-faithful text: en, hi, te, mai (+ sa for exp.)
├── experiments/
│   ├── run_granularity_experiments.py   ← reruns approaches 1–3 fresh
│   ├── granularity_results.json         ← their fresh scores
│   ├── vocab_schedule_sweep_results.json← approach 4 table (35 candidates)
│   ├── mai_sweep_results.json           ← clean-corpus weight sweep (picks champion)
│   ├── DEEPDIVE_FINDINGS.md             ← why the instructor scored well; corpus root-cause
│   └── REDPILL_FINDINGS.md              ← adversarial validation of the champion vs the gate
└── instructor_reference/         ← the instructor's shipped solution (for comparison)
```

## Reproduce everything

```bash
python train.py                              # champion → tokenizer.json + metrics.json
python compare_to_instructor.py              # table vs reference
python experiments/run_granularity_experiments.py   # fresh scores for approaches 1–3
```
