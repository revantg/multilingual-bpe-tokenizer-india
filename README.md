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

| approach | what changed | spread | score | gate |
|---|---|---:|---:|:--:|
| word / subword | merges never cross a word | 0.7278 | 1374 | ✗ (Indic > 1.2) |
| sentence-level | merges span a sentence | 0.3777 | 2648 | ✓ |
| paragraph-level | merges span a line | 0.3839 | 2605 | ✓ |
| **weight sweep on clean corpus** | upsample per language | **0.1005** | **9953** | ✓ |

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
