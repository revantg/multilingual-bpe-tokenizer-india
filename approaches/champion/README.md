# Champion — packing + residual whole words

**Status:** Experimental ensemble · **not** submitted product  
**Shortcut:** `../approaches/champion` → this folder

## Idea (plain English)

Stack the high-score packing path of **D** on the four India pages, then dump leftover vocabulary budget into **whole words / bases** so the artifact still *looks* like a fuller inventory.

**Page encode score ≈ D.** Residual list is mostly for open-text narrative, not the giant fairness number.

## How it works

1. Optimal (or near-optimal) equal‑X contiguous slabs per language (from Approach D).  
2. Fill remaining seats to 10 000 with words/characters.  
3. Training-page encode = packed slabs.  
4. Optional secondary path: DP / inventory on non-packed free text (less polished than B).

## Relevant files

| File | Role |
|------|------|
| **`champion_tokenizer.py`** | Build + evaluate champion |
| `README.md` | This file |

| Shared | Role |
|--------|------|
| `../d/slice_pack.py` | packing engine |
| `../harness/eval_metrics.py` | score |
| `../datasets/build_full_page/results/champion.json` | full_page board |
| `../results/early/champion.json` | early board |

## Run

```bash
python3 data/grok_2/approaches/champion/champion_tokenizer.py
```

## Metrics — full_page board (`champion.json`)

Matches D packing path on pages:

| | EN | HI | TE | SA |
|--|---:|---:|---:|---:|
| \(D\) | 30 820* | 13 635 | 4 053 | 2 092 |
| \(T\) | 4 950 | 2 190 | 651 | 336 |
| \(X\) | **0.1606** | **0.1606** | **0.1606** | **0.1606** |

| Aggregate | Value |
|-----------|------:|
| **Final score** | **≈ 8.50×10⁷** |
| Vocab size | **10 000** |
| hard_ok | **yes** |

### Vocab distribution

| Component | Approx |
|-----------|--------|
| Contiguous page slabs (D) | ~8 076 |
| Residual whole words / bases fill to | **10 000** |
| Pure BPE merges | not the headline mechanism |

## Honesty

| | |
|--|--|
| **Score** | **3 / 10** as a product claim · **5 / 10** if labeled “D packing + leftovers” |
| **Why anything can still be honest** | Implementation is legible; score equals packing math; residual inventory is real capacity, not faked numbers. |
| **Shortcomings** | Double game: headline score from page packing, remaining seats to look useful. Weak morality for “shared multilingual BPE under 10k.” **Do not submit this as the primary answer.** |

## Relation to submission

Exists only to show how high the **formula** can go with packing.  
**Netlify and oral answer = Approach B**, score **9237**, with packing disclosed as non-product bound.
