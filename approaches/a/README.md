# Approach A — Pure BPE (baseline)

**Status:** Experimental baseline · **not** submitted  
**Shortcut:** `../approaches/a` → this folder

## Idea (plain English)

Train classic **byte-pair encoding**: characters → repeated merges of frequent pairs until ~10 000 tokens.  
**No** whole-page word dump. Every page word is segmented only by merges / characters.

## How it works

1. Split text with harness words (Latin / Devanagari / Telugu + digits).  
2. Count pair frequency; merge until vocab full.  
3. Encode with merge application or **DP** min-piece cover.

Variants in `train_bpe.py`: `pure_concat`, `upsample`, `word_bpe`, `word_bpe_upsample`, `word_bpe_closure` (closure hybrid ≠ pure A).

## Relevant files

| File | Role |
|------|------|
| **`train_bpe.py`** | `ClassicBPE` engine + train CLI (**also used by B stage-1**) |
| `NOTES.md` | Variant tables on early corpus |
| `bpe_model_fullpage.json` | ~10k model on full-page data |
| `bpe_model_pure.json` / `bpe_model.json` | Other dumps |

| Shared | Role |
|--------|------|
| `../harness/eval_metrics.py` | \(X\), score |
| `../../datasets/full_page_md/` | pages |
| `../datasets/build_full_page/results/A.json` | board |

## Run

```bash
python3 data/grok_2/approaches/a/train_bpe.py
```

## Metrics — full_page board (`A.json`)

| | EN | HI | TE | SA |
|--|---:|---:|---:|---:|
| \(D\) (words) | 30 820* | 13 635 | 4 053 | 2 092 |
| \(T\) (tokens) | 37 483 | 16 461 | 6 627 | 3 920 |
| \(X=T/D\) | **1.216** | **1.207** | **1.635** | **1.874** |

| Aggregate | Value |
|-----------|------:|
| **Final score** | **≈ 1500.28** |
| gap \(\max X-\min X\) | ≈ 0.6666 |
| Vocab size | **10 000** |
| hard_ok (\(X_{\mathrm{EN}}\le 1.2\), V≤10k) | **no** (\(X_{\mathrm{EN}}>1.2\)) |

\*Board \(D\) may use pre-link regeneration counts; linked corpus is EN 30 789 etc. (same order of magnitude).

### Vocab distribution (pure A)

| Component | Count |
|-----------|------:|
| Character / initial units | hundreds–~2k (script + symbols) |
| Learned BPE merges filling to 10k | rest of budget |
| Dedicated whole-word slots | **0** (pure mode) |
| Total | **10 000** |

## Honesty

| | |
|--|--|
| **Score** | **8 / 10** |
| **Why honest** | Real pair-merge BPE. Metrics are true compression under a shared subword inventory. No multi-word page packing, no pretend that all page words are in the list. Answer to “what does BPE alone do?” |
| **Shortcomings** | Shared merges are EN-heavy → TE/SA stay multi-piece → large gap → low fairness score. On full_page, \(X_{\mathrm{EN}}\) can fail 1.2. Weak competitive for the assignment matric without hybrid tricks. |

## Relation to submission

**B** calls this folder’s `ClassicBPE` for stage-1 only, then spends leftover slots on whole words + deficit fill. Pure A is the baseline **we do not ship** as the final story.
