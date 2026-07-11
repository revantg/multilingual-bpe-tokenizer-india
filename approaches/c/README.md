# Approach C — Whole-word inventory

**Status:** Experimental inventory baseline · **not** submitted  
**Shortcut:** `../approaches/c` → this folder

## Idea (plain English)

Fill the 10 000-token budget with **as many exact spellings of page words as possible**, plus characters for leftovers.  
No merge training. Encode = whole word if listed, else character pieces.

When unique words of all pages fit in 10k → \(X=1\) everywhere → score **∞**.  
On full_page, union unique ≈ **12.6k** > 10k → English OOVs make \(X_{\mathrm{EN}}\) bad.

## How it works

1. Collect unique words (+ characters).  
2. Keep under cap (all if they fit; else budget/race among languages).  
3. Optional bigram leftover seats.  
4. Encode word → 1 id or char breakdown.

## Relevant files

| File | Role |
|------|------|
| **`tokenizer.py`** | Build inventory + encode paths |
| **`run.py`** | Variants + scoring CLI |
| `NOTES.md` | Early-corpus ∞-score tables |

| Shared | Role |
|--------|------|
| `../harness/eval_metrics.py` | score |
| `../datasets/build_full_page/results/C.json` | full_page board |
| `../results/early/approach_c.json` | early board |

## Run

```bash
python3 data/grok_2/approaches/c/run.py
```

## Metrics

### Full_page board (`C.json`)

| | EN | HI | TE | SA |
|--|---:|---:|---:|---:|
| \(D\) | 30 820* | 13 635 | 4 053 | 2 092 |
| \(T\) | 47 902 | 13 635 | 4 053 | 2 092 |
| \(X\) | **1.554** | **1.000** | **1.000** | **1.000** |

| Aggregate | Value |
|-----------|------:|
| **Final score** | **≈ 1804.24** |
| Vocab size | **≈ 9 914** |
| hard_ok | **no** (\(X_{\mathrm{EN}}>1.2\)) |

\*Smaller langs get essentially full type coverage → \(X=1\); EN carries almost all residual OOV piece mass.

### Vocab distribution (typical C words-first)

| Component | Approx on full_page |
|-----------|---------------------|
| Character base | few hundred |
| Whole words (HI/TE/SA first or fair race) | majority of remaining ≤10k |
| EN types | incomplete (~half unique EN not full-listed) |
| BPE merges | **0** |
| Total | ≤ **10 000** |

### Early corpus (union unique ≤ 10k) — from NOTES

| Variant | vocab | \(X\) all | score |
|---------|------:|-----------|------:|
| C words-only | ~8311 | all 1.0 | **∞** |
| + fair bigrams | 10k | unequal | ~7k |

## Honesty

| | |
|--|--|
| **Score** | **9 / 10** |
| **Why honest** | Method is exactly what it claims: a **closed word list** + chars. No fake subword theory. Metrics are easy to recompute. |
| **Shortcomings** | Not BPE. When unique types exceed 10k, **equity depends on who gets listed** — typical fills protect small languages and leave EN with huge \(T\). No generalization beyond listed spellings (chars only). Bigrams can make score *worse* than pure \(X=1\). |

## Relation to submission

**C** is the honest inventory extreme. **B** steals the word-list idea but **keeps BPE** so missing words (and new words) do not collapse to pure spell-out of every character path always under low code quality — and keeps hard_ok on full_page.
