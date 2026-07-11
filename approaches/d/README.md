# Approach D — Equal‑X slice packing (oracle)

**Status:** Experimental score oracle · **not** submitted as product tokenizer  
**Shortcut:** `../approaches/d` → this folder

## Idea (plain English)

**Hard-code multi-word chunks of the training pages** so each language uses almost the **same** pieces-per-word rate \(X\).  
Encode of those pages returns the chunks you stored — no ordinary subword algorithm.

Huge fairness score; weak claim as a general-purpose tokenizer.

## How it works

1. Get \(D_L\) = word counts.  
2. Choose integers \(T_L\) so \(X_L=T_L/D_L\) nearly match, under vocab budget.  
3. Partition each page into \(T_L\) contiguous word spans; each span string → one vocab entry.  
4. Page encode = that list of spans.

## Relevant files

| File | Role |
|------|------|
| **`slice_pack.py`** | Packing + `PackedTokenizer` + CLI |
| **`best_meta.json`** | Saved oracle \(T^*\) snapshot |
| `NOTES.md` | Design + early numbers |

| Shared | Role |
|--------|------|
| `../champion/champion_tokenizer.py` | consumes packing |
| `../datasets/build_full_page/results/D.json` | full_page board |

## Run

```bash
python3 data/grok_2/approaches/d/slice_pack.py
```

## Metrics — full_page board (`D.json`)

| | EN | HI | TE | SA |
|--|---:|---:|---:|---:|
| \(D\) | 30 820* | 13 635 | 4 053 | 2 092 |
| \(T\) | 4 950 | 2 190 | 651 | 336 |
| \(X\) | **0.1606** | **0.1606** | **0.1606** | **0.1606** |

| Aggregate | Value |
|-----------|------:|
| **Final score** | **≈ 8.50×10⁷** |
| gap | ~1.18×10⁻⁵ |
| Vocab size | **≈ 8 076** |
| hard_ok | **yes** |

### Vocab distribution (packing)

| Component | Count |
|-----------|------:|
| Contiguous multi-word slabs EN | ≈ \(T_{\mathrm{EN}}\) entries |
| Contiguous multi-word slabs HI | ≈ \(T_{\mathrm{HI}}\) |
| Contiguous multi-word slabs TE | ≈ \(T_{\mathrm{TE}}\) |
| Contiguous multi-word slabs SA | ≈ \(T_{\mathrm{SA}}\) |
| Sum of slabs | **≈ 8 076** (≤10k) |
| Learned BPE merges | **0** |
| Single-page store generalization | none (encode is the slab list) |

## Honesty

| | |
|--|--|
| **Score (as product tokenizer)** | **3–5 / 10** |
| **Score (as labeled metric oracle)** | **6–7 / 10** if you say so out loud |
| **Why it can be honest** | Numbers are real under the harness; algorithm is simple to re-run; scoring fairness via matched \(X\) is exactly the assignment formula. |
| **Shortcomings** | Not BPE. Not reusable encode for arbitrary new documents. Completely **page-overfit**. Claiming this as “our multilingual tokenizer product” without the warning is **dishonest**. |

## Relation to submission

Written up only for comparison / upper-bound. **Submitted story is B.** Champion = D packing on pages + leftover word fill.
