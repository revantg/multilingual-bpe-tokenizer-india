# Multilingual BPE tokenizer — India Wikipedia

This folder is the solution to the tokenizer assignment below (English, Hindi, Telugu, Sanskrit; shared 10 000-token vocab; score \(1000 / (X_{\max}-X_{\min})\)).

---

## Assignment

Your assignment would be to pick India's page on Wikipedia in **English**, **Hindi**, **Telugu**, and **one more language of your choice**. Ask your AI Agent to design a BPE tokenizer in such a way that:

- you have **10 000 tokens** (your vocab) overall for all languages,
- \((\text{Total English tokens}) / (\text{Total English words})\) must be **1.2 or less** — call this \(X_1\)
- similarly ratios for Hindi, Telugu and another language are \(X_2\), \(X_3\), \(X_4\)
- sort \(X_1, X_2, X_3, X_4\) — say \(X_4\) is largest and \(X_1\) is least
- your **assignment score** is \(1000 / (X_4 - X_1)\)

### What you are submitting

1. A **widget** that shows these ratios, token statistics and calculations, and your self score
2. Allows the grader to **see and download** your tokenizer (list of all tokens)
3. A **URL** for this widget on Netlify (or anywhere else you want to host)
4. **Please remember** that graders will run your tokenizer themselves to confirm your scores — be careful when you present claims about your numbers

### How we mapped the brief

| Brief | This folder |
|-------|-------------|
| English, Hindi, Telugu | India Wikipedia pages (`en` / `hi` / `te`) |
| One more language | **Sanskrit** (`sa`) |
| “Words” in the ratio | Pretokenized word count \(D_L\) (scripts: Latin, Devanagari, Telugu + digits) |
| \(X_L\) | \(T_L / D_L\) after encoding language \(L\) |
| Shared vocab | **10 000** tokens for all four languages |

---

## Submission (Approach B)

| | |
|--|--|
| **Method** | Short BPE (~2 500 slots) + whole-word fill; languages still above \(X=1.2\) get seats first (`deficit`) |
| **Self score** | **9237.39** |
| **Vocab size** | **10 000** |
| **\(X_{\mathrm{EN}} \le 1.2\)** | **yes** (1.158) |
| **Slideshow** | https://tokenizer-india-slides-rev.netlify.app |
| **Widget** | https://tokenizer-india-b-opt-rev-28915.netlify.app |
| **Local packs** | [`demo/slides/`](demo/slides/) · [`demo/widget/`](demo/widget/) |
| **Tokenizer file** | [`approaches/b/vocab_optimized_fullpage.json`](approaches/b/vocab_optimized_fullpage.json) |
| **Trainer** | [`approaches/b/train_optimized_b.py`](approaches/b/train_optimized_b.py) |
| **Downloadable token list** | `demo/slides/tokens.txt` / `tokenizer.json` (same on widget) |

### Per-language metrics (Approach B · primary corpus)

| Language | Words \(D\) | Tokens \(T\) | \(X = T/D\) | ≤ 1.2? |
|----------|------------:|-------------:|------------:|:------:|
| English | 30 789 | 35 649 | **1.158** | yes |
| Hindi | 13 630 | 14 788 | **1.085** | yes |
| Telugu | 4 053 | 4 254 | **1.050** | yes |
| Sanskrit | 2 091 | 2 322 | **1.110** | yes |

\[
X_{\max} = 1.15785,\quad X_{\min} = 1.04959,\quad
\text{gap} = 0.10826,\quad
\text{score} = 1000 / 0.10826 \approx \mathbf{9237.39}
\]

All four \(X \le 1.2\). UNK count on these training pages: **0**.

### Vocab distribution (Approach B)

| Component | Count |
|-----------|------:|
| Stage-1 BPE (characters + merges) | **2 500** (≈ **2 297** merge rules) |
| Stage-2 whole words | **7 500** |
| Priority fill while some \(X > 1.2\) | EN **2 713** · HI **1 345** · TE **1 278** · SA **867** |
| Leftover fill after all \(X \le 1.2\) | **1 297** |
| **Total** | **10 000** |

Types: stored as full words vs subword fallback

| Language | Unique on page | Full-word in vocab | Not listed (BPE pieces) |
|----------|---------------:|-------------------:|------------------------:|
| EN | 7 015 | 3 944 | 3 071 |
| HI | 3 589 | 2 790 | 799 |
| TE | 2 155 | 1 982 | 173 |
| SA | 1 275 | 1 110 | 165 |

Method notes: [`approaches/b/README.md`](approaches/b/README.md).

---

## Training data

Primary pages: India (same topic) in four languages, under [`datasets/full_page_md/`](datasets/full_page_md/). Scoring applies a light clean (keep link labels, drop URLs).

| Language | File | Words \(D\) | Unique words | MD links |
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

| ID | Idea | Vocab | score | \(X_E\) | \(X_H\) | \(X_T\) | \(X_S\) | Hard ok* |
|----|------|------:|------:|--------:|--------:|--------:|--------:|:--------:|
| **B** | Short BPE + deficit word fill | 10k | **9237** | 1.158 | 1.085 | 1.050 | 1.110 | **yes** |
| A | Pure BPE | 10k | ~1500 | 1.216 | 1.207 | 1.635 | 1.874 | no |
| C | Whole-word inventory | ~9914 | ~1804 | 1.554 | 1.000 | 1.000 | 1.000 | no |
| D | Equal-\(X\) page packing | ~8076 | ~8.5e7 | 0.161 | 0.161 | 0.161 | 0.161 | yes |
| E | Words + phrases + DP | 12603† | ∞ | 1.0 | 1.0 | 1.0 | 1.0 | **no†** |
| Champion | D packing + residual words | 10k | ~8.5e7 | 0.161 | 0.161 | 0.161 | 0.161 | yes |

\*Hard ok ≈ \(X_{\mathrm{EN}}\le 1.2\) and vocab ≤ 10 000.  
†E as stored on full pages exceeds the vocab cap.

D / champion maximize the formula with page-specific multi-word slabs; comparison only — **not** the claimed product tokenizer.

---

## Layout

```
grok_2/
├── README.md           assignment + submission numbers (this file)
├── approaches/         a · b★ · c · d · e · champion
├── datasets/           full_page_md★ and other corpora
├── demo/               slides · widget · lab
├── harness/            score = 1000/(maxX−minX), X = T/D
└── results/PRIMARY.md  B score of record
```

### Retrain (B)

```bash
python3 data/grok_2/approaches/b/train_optimized_b.py \
  --fill-mode deficit --tau 1.2 \
  --out data/grok_2/approaches/b/vocab_optimized_fullpage.json
```

### Preview submission UI locally

```bash
cd data/grok_2/demo/slides && python3 -m http.server 8765
```
