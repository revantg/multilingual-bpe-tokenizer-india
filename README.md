# Multilingual BPE Tokenizer — India Wikipedia

> **On the use of AI.** Every idea, hypothesis, and design decision in this repo
> is the author's own. The proof-of-concept BPE notebooks (`poc.py`,
> `poc_gpt2.py`, `naive_bpe.py`) and the core `BPETokenizer` class are
> hand-written. AI was used **only to automate execution** — running sweeps,
> tabulating results, orchestrating the fractional-weight experiments, and
> exporting notebooks. It did not generate the approach, the diagnosis, or the
> strategy.

One shared **10,000-token** byte-level BPE tokenizer for the *India* Wikipedia
page in four languages (English, Hindi, Telugu, and Maithili as the fourth).

**Score** `= 1000 / (X4 − X1)`, where each `X_L = tokens(L) / faithful_units(L)`
is a language's fertility and X4 / X1 are the largest / smallest of the four.
Lower spread ⇒ higher score. Hard gates: vocab = 10000, every `X_L ≤ 1.2`,
`decode(encode(text))` preserves all visible characters, and **zero UNK**.

---

## The story of this repo

This repo documents everything that happened **after a first submission that
scored 0**. Each stage is preserved and reproducible.

### Stage 0 — the first submission (scored 0)

The initial tokenizer used a **whitespace / word-level** scheme. It failed the
**faithfulness gate**, not the fertility metric: on unseen text it emitted
`<UNK>`. For example `"India's population is 1,428,627,663."` tokenized to
`"… 1 <UNK> 4 28 <UNK> …"`. Any `<UNK>` means `decode(encode(text))` loses
characters, so the submission scored **0** regardless of its fertility numbers.

**Root cause:** a word/subword vocabulary can never cover arbitrary UTF-8 — there
will always be an unseen byte sequence.

**Fix:** a **GPT-2-style byte-level base alphabet**. All 256 byte values get a
symbol, so *any* input round-trips byte-exactly and UNK becomes structurally
impossible. This byte-level base underpins every later stage.

### Stage 1 — the honest solution (score 9953) → [`solution/`](solution/)

With the byte-level base fixed, the search moved through pre-tokenization
granularity and then vocab distribution:

| approach | what changed | X_EN | X_HI | X_TE | X_MAI | spread | score | gate |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| word / subword | merges never cross a word | 0.9841 | 1.5637 | 1.6712 | 1.7119 | 0.7278 | 1374 | ✗ (Indic > 1.2) |
| sentence-level | merges span a sentence | 0.5584 | 0.5593 | 0.6590 | 0.6176 | 0.3777 | 2648 | ✓ |
| paragraph-level | merges span a line | 0.5637 | 0.5690 | 0.6720 | 0.6391 | 0.3839 | 2605 | ✓ |
| **weight sweep on clean corpus** | upsample per language `{2,2,3,2}` | **0.5669** | **0.5774** | **0.6012** | **0.6674** | **0.1005** | **9953** | ✓ |

Two insights unlocked 9953: (1) the user's Sanskrit page was ~30–70%
percent-encoded URL hex that distorted fertility — swapping to the instructor's
clean **Maithili** page removed the outlier; (2) near-equal integer weights
`{2,2,3,2}` converged all four fertilities into a `0.567–0.667` band. This beats
the instructor's own reference (6503) by +3451. **This is the honest, robust
submission** — full writeup in [`solution/README.md`](solution/README.md).

### Stage 2 — the overfit push (score 199236) → [`submission2/`](submission2/)

A separate, second submission asking a different question: *how far can the score
go if we deliberately tune to this exact corpus?* — while being transparent about
the overfitting. Two fair, principled knobs on the **same hand-written
`BPETokenizer`**; only the orchestration around it changed:

1. **Fractional language upsampling** (deterministic Bresenham line replication).
   The integer grid never raised Maithili past 2× even though MAI was the spread
   ceiling; fractional weights tune each language's fertility precisely.
2. **Newline attachment** — train on `line + "\n"` and encode in newline-carrying
   chunks so `…\n` merges are learnable, instead of every newline being a
   permanently unmergeable token (a tax that fell hardest on MAI).

Result with weights **EN:2 HI:2.2 TE:3.7 MAI:4.05**: spread **0.005019**, score
**199236**, all gates pass.

**Stage 1 — clean-corpus integer weight sweep** (all pass gates; sorted by score):

| weights EN/HI/TE/MAI | X_EN | X_HI | X_TE | X_MAI | spread | score |
|---|---:|---:|---:|---:|---:|---:|
| **2/2/3/2 — champion** | **0.5669** | **0.5774** | **0.6012** | **0.6674** | **0.1005** | **9953** |
| 1/1/1/1 | 0.5580 | 0.5665 | 0.6627 | 0.6565 | 0.1048 | 9546 |
| 1/1/2/1 | 0.5748 | 0.5877 | 0.5592 | 0.6810 | 0.1218 | 8212 |
| 2/3/3/2 | 0.5802 | 0.5410 | 0.6288 | 0.6708 | 0.1298 | 7707 |
| 3/3/3/2 | 0.5578 | 0.5675 | 0.6550 | 0.7152 | 0.1574 | 6353 |
| 3/3/4/2 | 0.5591 | 0.5767 | 0.6256 | 0.7347 | 0.1755 | 5696 |
| 2/2/2/1 | 0.5578 | 0.5686 | 0.6502 | 0.7441 | 0.1863 | 5367 |
| 1/2/2/1 | 0.6089 | 0.5166 | 0.6024 | 0.7030 | 0.1864 | 5366 |
| 2/2/3/1 | 0.5643 | 0.5791 | 0.6007 | 0.7550 | 0.1907 | 5243 |
| 3/2/2/1 | 0.5330 | 0.6077 | 0.7026 | 0.7862 | 0.2532 | 3950 |

**Stage 2 — fractional + newline search** (coordinate descent from 9953 → 199236;
all vocab 10000, 0 UNK, lossless):

| # | approach / weights | X_EN | X_HI | X_TE | X_MAI | spread | score |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | word-level subword | 0.9841 | 1.5637 | 1.6712 | 1.7119 | 0.7278 | 1374 |
| 2 | sentence-level | 0.5584 | 0.5593 | 0.6590 | 0.6176 | 0.3777 | 2648 |
| 3 | paragraph-level | 0.5637 | 0.5690 | 0.6720 | 0.6391 | 0.3839 | 2605 |
| 4 | instructor reference 3/4/4/2 | 0.5977 | 0.5793 | 0.6731 | 0.7331 | 0.1538 | 6503 |
| 5 | integer 1/1/1/1 | 0.5580 | 0.5665 | 0.6627 | 0.6565 | 0.1048 | 9546 |
| 6 | **integer 2/2/3/2 — Stage 1 champion** | 0.5669 | 0.5774 | 0.6012 | 0.6674 | 0.1005 | 9953 |
| 7 | fractional 2/2/3.5/3.5 | 0.5751 | 0.5828 | 0.5797 | 0.5914 | 0.0164 | 61100 |
| 8 | fractional 2/2.1/3.5/3.8 | 0.5757 | 0.5783 | 0.5854 | 0.5895 | 0.0138 | 72256 |
| 9 | fractional 2/2.1/3.4/4.0 (best plain) | 0.5766 | 0.5794 | 0.5843 | 0.5835 | 0.0077 | 129954 |
| 10 | +newline 2/2.1/3.7/4.0 | 0.5702 | 0.5793 | 0.5713 | 0.5773 | 0.0091 | 109587 |
| **11** | **+newline 2/2.2/3.7/4.05 — Stage 2 (this)** | **0.571995** | **0.573490** | **0.577014** | **0.572658** | **0.005019** | **199236** |

**Honesty caveat (the whole point of Stage 2):** the exact 199236 is
**snapshot-specific**. Under corpus perturbation (a simulated different wiki
snapshot, same weights, no re-tuning) the recipe scores **26k–54k** vs Stage 1's
**7.8k–9.0k** — a genuine **3.3–6× structural win** that *does* generalize, even
though the literal 199k does not. Both numbers are reported. Full audit in
[`submission2/README.md`](submission2/README.md) and
[`submission2/experiments/SCORE_PUSH_FINDINGS.md`](submission2/experiments/SCORE_PUSH_FINDINGS.md).

---

## Proof-of-concept notebooks (how the BPE was built by hand)

Marimo notebooks written from scratch, kept to show the progression. Static HTML
exports sit next to each `.py` (tracked via git LFS).

| notebook | what it shows |
|---|---|
| [`naive_bpe.py`](naive_bpe.py) | the simplest hand-written BPE, first principles |
| [`poc.py`](poc.py) | byte-level BPE proof-of-concept + fertility intuition |
| [`poc_gpt2.py`](poc_gpt2.py) | GPT-2-style byte map, max-heap trainer, vocab analysis |

Run any of them live: `marimo edit poc.py`.

---

## Repository layout

```
.
├── README.md                 ← this file (the full story)
├── solution/                 ← Stage 1: the honest champion (score 9953)
├── submission2/              ← Stage 2: the overfit push  (score 199236)
├── datasets/                 ← raw + derived Wikipedia corpora (all languages tried)
├── naive_bpe.py  poc.py  poc_gpt2.py   ← hand-written BPE proof-of-concept notebooks
└── *.html                    ← static exports of the notebooks (git LFS)
```

The full pre-9953 exploration history (early approaches, archives, demos, agent
transcripts) is preserved on the **`zero-score-submission`** branch.

## Reproduce

```bash
# Stage 1 — honest champion (9953)
cd solution && python train.py && python compare_to_instructor.py

# Stage 2 — overfit push (199236)
cd submission2 && python train.py && python build_artifacts.py
```

## Live sites

- Stage 1 (honest, 9953): [widget + story](https://india-bpe-tokenizer.surge.sh) · [slides](https://revant-tokenizer-slides.surge.sh)
- Stage 2 (overfit, 199236): https://india-bpe-tokenizer-overfit.surge.sh
