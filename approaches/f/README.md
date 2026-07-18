# Approach F — tiktoken external baseline

**Status:** Comparison baseline only · **not** submitted  
**Shortcut:** `approaches/f` → this folder

## Idea (plain English)

Do **not** train a tokenizer. Take OpenAI’s public **tiktoken** encodings (the same family used by GPT‑3.5 / GPT‑4 / GPT‑4o) and run them on **our** four India Wikipedia pages with **our** word counts \(D\).

This answers: *“How fair is a strong production BPE on this assignment metric?”*

Answer in short: **English is fine-ish; Telugu/Sanskrit (and older Hindi) explode into many pieces → large \(X\) gap → low assignment score.** Even the best encoding here scores **~495** vs Approach B’s **~9237**, and all fail the **10k vocab** product bar.

## How it works

```
full_page_md pages
      │
      ▼
 harness clean_text + extract_words  →  D_L  (same as Approach B)
      │
      ▼
 tiktoken.encode(full cleaned page)  →  T_L
      │
      ▼
 X_L = T_L / D_L
 score = 1000 / (max X − min X)
```

| Piece | Definition |
|-------|------------|
| Corpus | `datasets/full_page_md/` India EN/HI/TE/SA |
| \(D\) | pretok word count (`harness/eval_metrics.extract_words`) |
| \(T\) | `len(encoding.encode(cleaned_page))` |
| Vocab | fixed by OpenAI (50k–200k) — **not** under our 10k cap |

Encodings measured:

| Encoding | Typical models | Vocab size |
|----------|----------------|----------:|
| `cl100k_base` | GPT‑4, GPT‑3.5-turbo | 100 277 |
| `o200k_base` | GPT‑4o family | 200 019 |
| `p50k_base` | Codex / some GPT‑3 variants | 50 281 |
| `r50k_base` | GPT‑2 / GPT‑3 davinci-era | 50 257 |

## Relevant files

| File | Role |
|------|------|
| **`run_tiktoken_baseline.py`** | Re-run metrics; write `results.json` |
| **`results.json`** | Frozen board from the measured run |
| `README.md` | This analysis |

| Shared | Role |
|--------|------|
| `../../harness/eval_metrics.py` | pretok words \(D\) |
| `../../datasets/full_page_md/` | primary corpus |
| `../b/` | reference submission scores |

## Run

Needs `tiktoken` (e.g. `pip install tiktoken` in a venv).

```bash
# from repo root
python3 approaches/f/run_tiktoken_baseline.py
python3 approaches/f/run_tiktoken_baseline.py --encodings cl100k_base o200k_base --per-word
```

---

## Metrics — full_page board

Same \(D\) as Approach B:

| Lang | Words \(D\) |
|------|------------:|
| EN | 30 789 |
| HI | 13 630 |
| TE | 4 053 |
| SA | 2 091 |

### Fertility \(X = T/D\) and assignment score

| Encoding | Vocab | \(X_{\mathrm{EN}}\) | \(X_{\mathrm{HI}}\) | \(X_{\mathrm{TE}}\) | \(X_{\mathrm{SA}}\) | gap | **score** | \(X_{\mathrm{EN}}\le 1.2\)? |
|----------|------:|--------------------:|--------------------:|--------------------:|--------------------:|----:|----------:|:------------------------:|
| **cl100k_base** | 100 277 | 1.626 | 4.719 | **10.272** | 8.462 | 8.646 | **115.66** | no |
| **o200k_base** | 200 019 | 1.600 | 1.859 | 2.920 | **3.621** | 2.021 | **494.82** | no |
| **p50k_base** | 50 281 | 1.586 | 6.999 | **15.668** | 13.246 | 14.082 | **71.01** | no |
| **r50k_base** | 50 257 | 1.586 | 6.999 | **15.668** | 13.246 | 14.082 | **71.01** | no |
| **Approach B** (ours) | **10 000** | **1.158** | **1.085** | **1.050** | **1.110** | **0.108** | **9237.39** | **yes** |

### Token counts \(T\) (full cleaned page)

| Encoding | EN | HI | TE | SA |
|----------|---:|---:|---:|---:|
| cl100k_base | 50 065 | 64 316 | 41 634 | 17 695 |
| o200k_base | 49 271 | 25 339 | 11 834 | 7 572 |
| p50k / r50k | 48 824 | 95 401 | 63 503 | 27 698 |
| Approach B | 35 649 | 14 788 | 4 254 | 2 322 |

### Optional: per-word sum fertility (`cl100k_base`)

Encode each pretok word alone and sum (ignores spaces/punctuation between words). Slightly lower \(X\), same story:

| Lang | \(X\) full page | \(X\) sum of words |
|------|----------------:|-------------------:|
| EN | 1.626 | 1.422 |
| HI | 4.719 | 4.344 |
| TE | 10.272 | 9.843 |
| SA | 8.462 | 7.936 |

---

## Analysis (why these numbers)

### 1. What “fertility” means here

\[
X_L = \frac{\text{tiktoken pieces on page } L}{\text{our pretok words on page } L}
\]

Higher \(X\) = more pieces per word = worse compression for that language **under this denom**.

### 2. Why tiktoken scores poorly on the *assignment* formula

The score rewards **close** \(X\) values, not “good English compression.”

```
cl100k:  EN ~1.6   HI ~4.7   TE ~10.3   SA ~8.5
         └─ cheap ─┘         └── Indic scripts shredded ──┘
         gap ≈ 8.65  →  score ≈ 116
```

English is near what you expect for GPT tokenizers. Telugu/Sanskrit fall into many short subwords/bytes → \(X_{\max}\) is huge → **score collapses**.

### 3. o200k is much fairer than cl100k on Indic — still not competitive

| | cl100k \(X_{\mathrm{TE}}\) | o200k \(X_{\mathrm{TE}}\) |
|--|---------------------------:|--------------------------:|
| Telugu | 10.27 | **2.92** |
| Hindi | 4.72 | **1.86** |
| score | 116 | **495** |

GPT‑4o’s vocab is larger and more multilingual. Gap shrinks, score jumps **~4×**, but:

- \(X_{\mathrm{EN}}\) still **> 1.2**
- Vocab is **200k**, not 10k
- Gap (~2.0) is still ~20× Approach B’s (~0.11)

### 4. p50k / r50k are the worst for this corpus

Older English-centric BPE: TE \(X \approx 15.7\), score **~71**. Same numbers for p50k and r50k on this text (same merge behaviour on these scripts for practical purposes).

### 5. Vocab cap: automatic hard fail

Assignment product bar used in this repo:

- \(X_{\mathrm{EN}} \le 1.2\)
- vocab \(\le 10\,000\)

tiktoken fails **both** (or at least vocab) on every encoding above. `hard_ok = false` for all F rows.

### 6. Why Approach B still “wins” this game

| | tiktoken (best: o200k) | Approach B |
|--|------------------------|------------|
| Goal | General web / chat models | Fair \(X\) on **these 4 pages** under 10k |
| Vocab | 50k–200k | **10k** |
| Whole-word seats for India pages | no | yes (deficit fill) |
| All \(X \le 1.2\) | no | **yes** |
| Assignment score | **~495** | **~9237** |

B is **page-specialized hybrid** (short BPE + whole words). tiktoken is a **general** tokenizer. Beating tiktoken on *this* score is not “we beat GPT”; it is “we optimized the fairness formula on a closed four-page set under a 10k cap.”

```
X by language (sketch)

          EN     HI      TE      SA
B      ████     ███     ███     ███      ~1.05–1.16   score ~9237
o200k  █████    █████   ████████ █████████  ~1.6–3.6  score ~495
cl100k █████    ████████████ ████████████████ ~1.6–10 score ~116
```

### 7. What this baseline is good for

| Use | Don’t use for |
|-----|----------------|
| Honest external reference | Claiming a 10k submission |
| Showing Indic fertility pain of production BPEs | Upper-bounding “true” multilingual quality worldwide |
| Motivating hybrid / deficit fill in B | Replacing A/C as a *trainable* baseline |

---

## Honesty

| | |
|--|--|
| **Score** | **9 / 10** as a labeled external baseline |
| **Why honest** | Public encodings; same corpus and \(D\) as B; numbers re-runnable via `run_tiktoken_baseline.py`; we state hard_ok fails. |
| **Shortcomings** | Not trained here; not ≤10k; full-page \(T\) includes punctuation/spaces that pretok \(D\) skips (same as any full-text encode vs word-only denom). Not an “approach” we ship. |

## Relation to submission

**Submitted story remains Approach B.**  
F is the “what does GPT’s tokenizer do on our pages?” appendix in the board. Best F score on record: **o200k_base ≈ 494.82**.
