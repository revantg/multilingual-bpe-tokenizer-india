# Multilingual BPE tokenizer — India Wikipedia
**Note:** The BPE proof-of-concept files (`poc.py` and `naive_bpe.py`) are handwritten and were not written by agents.


This repository is the solution to the tokenizer assignment below (English, Hindi, Telugu, Sanskrit; shared 10 000-token vocab; score $1000 / (X_{\max}-X_{\min})$).

---

## Assignment

Verbatim from the course (Axiom, "Session 2 Assignment QnA", fetched live via
the `tsai-skills` skill — this is the actual assignment text, not a
paraphrase):

> Your assignment would be to pick India's page on Wikipedia in English,
> Hindi, Telugu, and one more language of your choice. Ask your AI Agent to
> design a BPE tokenizer in such a way that:
> - you have 10000 tokens (your vocab) overall for all languages,
> - (Total English tokens)/(Total English Vocab, say 5000 words) must be
>   **1.2** or less, let's call this X1
> - Similarly ratios for your Hindi, Telugu and another language is X2, X3,
>   X4
> - Sort X1, X2, X3, X4.. say its X4 (largest), X2, X3, X1 (least).
> - Your assignment score is going to be 1000/(X4 - X1).

### What you are submitting

1. A **widget** that shows these ratios, token statistics and calculations, and your self score
2. Allows the grader to **see and download** your tokenizer (list of all tokens)
3. A **URL** for this widget on Netlify (or anywhere else you want to host)
4. **Please remember** that graders will run your tokenizer themselves to confirm your scores — be careful when you present claims about your numbers

### Denominator ambiguity — read carefully

The brief names the denominator **"Total English Vocab, say 5000 words"** —
not "Total English words". That phrasing points at a **distinct/unique-word
count** for the page, not the raw running word count. This repo initially
built (and Approach B below still reports against) the raw running word
count. Both readings are tracked here so scores are comparable either way:

| Reading of $D_L$ | Definition | EN example |
|---|---|---|
| Raw word count (this repo's original mapping) | every pretokenized word occurrence, duplicates included | 30 789 |
| Unique-word "Vocab" (assignment's literal wording) | distinct words on the page, case-folded | 6 271–7 015 (cleaning-method dependent) |

Using the unique-word reading shrinks $D_L$ a lot, which inflates every
$X_L$ — a tokenizer built to satisfy $X_{\mathrm{EN}}\le1.2$ under the raw
reading is **not** guaranteed to satisfy it under the unique-word reading.
See `poc_gpt2.py` below for a worked example of both.

### How we mapped the brief

| Brief | This folder |
|-------|-------------|
| English, Hindi, Telugu | India Wikipedia pages (`en` / `hi` / `te`) |
| One more language | **Sanskrit** (`sa`) |
| “Vocab”/“words” in the ratio | Ambiguous — see above; Approach B numbers use raw pretokenized word count $D_L$ (scripts: Latin, Devanagari, Telugu + digits) |
| $X_L$ | $T_L / D_L$ after encoding language $L$ |
| Shared vocab | **10 000** tokens for all four languages |

---

## Submission (Approach B)

| | |
|--|--|
| **Method** | Short BPE (~2 500 slots) + whole-word fill; languages still above $X=1.2$ get seats first (`deficit`) |
| **Self score** | **9237.39** |
| **Vocab size** | **10 000** |
| **$X_{\mathrm{EN}} \le 1.2$** | **yes** (1.158) |
| **Slideshow** | https://revant-tokenizer-slides.surge.sh (mirror: https://tokenizer-india-slides-rev.netlify.app) |
| **Widget** | https://tokenizer-india-b-opt-rev-28915.netlify.app |
| **Local packs** | [`demo/slides/`](demo/slides/) · [`demo/widget/`](demo/widget/) |
| **Tokenizer file** | [`approaches/b/vocab_optimized_fullpage.json`](approaches/b/vocab_optimized_fullpage.json) |
| **Trainer** | [`approaches/b/train_optimized_b.py`](approaches/b/train_optimized_b.py) |
| **Downloadable token list** | `demo/slides/tokens.txt` / `tokenizer.json` (same on widget) |

### Per-language metrics (Approach B · primary corpus)

| Language | Words $D$ | Tokens $T$ | $X = T/D$ | ≤ 1.2? |
|----------|------------:|-------------:|------------:|:------:|
| English | 30 789 | 35 649 | **1.158** | yes |
| Hindi | 13 630 | 14 788 | **1.085** | yes |
| Telugu | 4 053 | 4 254 | **1.050** | yes |
| Sanskrit | 2 091 | 2 322 | **1.110** | yes |

$$
X_{\max} = 1.15785,\quad X_{\min} = 1.04959,\quad
\text{gap} = 0.10826,\quad
\text{score} = 1000 / 0.10826 \approx \mathbf{9237.39}
$$

All four $X \le 1.2$. UNK count on these training pages: **0**.

### Vocab distribution (Approach B)

| Component | Count |
|-----------|------:|
| Stage-1 BPE (characters + merges) | **2 500** (≈ **2 297** merge rules) |
| Stage-2 whole words | **7 500** |
| Priority fill while some $X > 1.2$ | EN **2 713** · HI **1 345** · TE **1 278** · SA **867** |
| Leftover fill after all $X \le 1.2$ | **1 297** |
| **Total** | **10 000** |

Types: stored as full words vs subword fallback

| Language | Unique on page | Full-word in vocab | Not listed (BPE pieces) |
|----------|---------------:|-------------------:|------------------------:|
| EN | 7 015 | 3 944 | 3 071 |
| HI | 3 589 | 2 790 | 799 |
| TE | 2 155 | 1 982 | 173 |
| SA | 1 275 | 1 110 | 165 |

Method notes: [`approaches/b/README.md`](approaches/b/README.md).

### `poc_gpt2.py` — byte-level BPE, both denominator readings

A separate from-scratch byte-level BPE tokenizer (GPT-2-style byte alphabet,
regex pre-tokenizer, real `encode()`/`decode()`, pooled multilingual
training — see [`poc_gpt2.py`](poc_gpt2.py)) trained at the same 10 000
shared-vocab cap, with **no whole-word stage**. Reported under both
denominator readings from the ambiguity note above:

| Language | $T$ | $D$ raw word count | $X$ (raw) | $D$ unique-word "Vocab" | $X$ (unique) |
|----------|------:|------:|------:|------:|------:|
| English | 48 413 | 30 789 | 1.572 | 6 271 | 7.720 |
| Hindi | 44 179 | 13 630 | 3.241 | 3 540 | 12.480 |
| Telugu | 19 093 | 4 053 | 4.711 | 2 131 | 8.960 |
| Sanskrit | 11 828 | 2 091 | 5.657 | 1 274 | 9.284 |

Raw-word reading: gap = 4.084, score = 1000/4.084 ≈ **244.85**, $X_{\mathrm{EN}}\le1.2$: **no**.
Unique-word reading: gap = 4.760, score = 1000/4.760 ≈ **210.09**, $X_{\mathrm{EN}}\le1.2$: **no**.

Both readings fail the $X_{\mathrm{EN}}\le1.2$ hard bar because this
tokenizer has no whole-word vocab stage — every word still costs multiple
BPE merges. It exists to demonstrate a **correct, lossless, byte-level**
tokenizer (any UTF-8 input round-trips exactly, no `<unk>` gaps), not to
compete on score. **Approach B remains the scored submission.**

---

## Training data

Primary pages: India (same topic) in four languages, under [`datasets/full_page_md/`](datasets/full_page_md/). Scoring applies a light clean (keep link labels, drop URLs).

| Language | File | Words $D$ | Unique words | MD links |
|----------|------|------------:|-------------:|---------:|
| English | `india_en.md` | **30 789** | **7 015** | 2 784 |
| Hindi | `india_hi.md` | **13 630** | **3 589** | 1 182 |
| Telugu | `india_te.md` | **4 053** | **2 155** | 613 |
| Sanskrit | `india_sa.md` | **2 091** | **1 275** | 388 |
| **Combined** | | **50 563** | **12 635** union unique | **4 967** |

Union unique words (**12 635**) exceed the 10k vocab cap, so not every type can be a single token. More: [`datasets/README.md`](datasets/README.md).

---

## Approaches explored

Writeups: [`approaches/`](approaches/README.md). **B** is the submission.

| ID | Idea | Vocab | score | $X_E$ | $X_H$ | $X_T$ | $X_S$ | Hard ok* |
|----|------|------:|------:|--------:|--------:|--------:|--------:|:--------:|
| **B** | Short BPE + deficit word fill | 10k | **9237** | 1.158 | 1.085 | 1.050 | 1.110 | **yes** |
| A | Pure BPE | 10k | ~1500 | 1.216 | 1.207 | 1.635 | 1.874 | no |
| C | Whole-word inventory | ~9914 | ~1804 | 1.554 | 1.000 | 1.000 | 1.000 | no |
| D | Equal-$X$ page packing | ~8076 | ~8.5e7 | 0.161 | 0.161 | 0.161 | 0.161 | yes |
| E | Words + phrases + DP | 12603† | ∞ | 1.0 | 1.0 | 1.0 | 1.0 | **no†** |
| Champion | D packing + residual words | 10k | ~8.5e7 | 0.161 | 0.161 | 0.161 | 0.161 | yes |
| **F** | OpenAI **tiktoken** (external) | 50k–200k | **~71–495** | ~1.6 | 1.9–7.0 | 2.9–15.7 | 3.6–13.2 | **no** |

\*Hard ok ≈ $X_{\mathrm{EN}}\le 1.2$ and vocab ≤ 10 000.  
†E as stored on full pages exceeds the vocab cap.

D / champion maximize the formula with page-specific multi-word slabs; comparison only — **not** the claimed product tokenizer.

**F** is a production-BPE reference on the same corpus/denom (best: `o200k_base` ≈ **495**). It fails the 10k vocab bar and English \(X\le 1.2\); writeup: [`approaches/f/README.md`](approaches/f/README.md).

---

## Layout

```
tokenizer-assignment/
├── README.md           assignment + submission numbers (this file)
├── AGENTS.md           repo working rules
├── approaches/         a · b★ · c · d · e · f · champion
├── datasets/           full_page_md★ and other corpora
├── demo/               slides · widget · lab
├── harness/            score = 1000/(maxX−minX), X = T/D
├── docs/               structure notes
├── tools/              helper scripts
├── archive/            dead vocabs
└── results/PRIMARY.md  B score of record
```

### Retrain (B)

```bash
python3 approaches/b/train_optimized_b.py \
  --fill-mode deficit --tau 1.2 \
  --out approaches/b/vocab_optimized_fullpage.json
```

### Preview submission UI locally

```bash
cd demo/slides && python3 -m http.server 8765
```
