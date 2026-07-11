# Approach B — Optimized hybrid (**submitted**)

**Status:** ★ PRIMARY submission  
**Shortcut:** `../approaches/b` → this folder

## Idea (plain English)

1. Train **BPE only briefly** (~2 500 slots for letters + subword pieces).  
2. Fill remaining seats with **whole words** from the training pages.  
3. Prefer languages still **above 1.2 pieces/word** until they drop under the bar (or budget ends); then fill leftovers by how many pieces a full-word entry saves.

Common words drop from many pieces to **one**. OWVs still fall back to stage-1 BPE.

## How it works

| Stage | ≈ Slots | Behavior |
|-------|--------:|----------|
| 1 · BPE | 2 500 | `ClassicBPE.train` (optional EN merge weight) |
| 2a · deficit | variable | Add whole words for lang with largest residual above $\tau=1.2$ |
| 2b · leftovers | rest of 7 500 words | Global savings ($freq × pieces\ saved$) |
| Encode | — | In list → 1 token; else DP over BPE pieces |

CLI: `--fill-mode deficit|global`, `--tau 1.2`, `--bpe-size 2500`, `--vocab-cap 10000`.

## Relevant files

| File | Role |
|------|------|
| **`train_optimized_b.py`** | Main trainer |
| **`vocab_optimized_fullpage.json`** | Shipped model (Netlify) |
| `train_fair_bpe.py` | Older fair/minimax experiments |
| `RESILIENCE_X_LE_1_2.md` | Make all langs ≤1.2 on new data |
| `FIX_SA_X_OVER_1_2.md` | Postmortem of SA > 1.2 fix |
| `NOTES.md` | Extra notes |

| Shared | Role |
|--------|------|
| `../a/train_bpe.py` | BPE engine |
| `../harness/eval_metrics.py` | score |
| `../../datasets/full_page_md/` | training pages |
| `../archive/approach_b_vocabs/` | old trial vocabs |
| `../../demo/slides` · `widget` | Netlify packages |
| `../results/PRIMARY.md` | score of record |
| `../datasets/build_full_page/results/B_optimized.json` | board |

## Run

```bash
python3 data/grok_2/approaches/b/train_optimized_b.py \
  --vocab-cap 10000 --bpe-size 2500 \
  --fill-mode deficit --tau 1.2 \
  --out data/grok_2/approaches/b/vocab_optimized_fullpage.json
```

## Metrics — **score of record** (linked `full_page_md`)

| | EN | HI | TE | SA |
|--|---:|---:|---:|---:|
| $D$ (words) | 30 789 | 13 630 | 4 053 | 2 091 |
| $T$ (tokens) | 35 649 | 14 788 | 4 254 | 2 322 |
| $X=T/D$ | **1.158** | **1.085** | **1.050** | **1.110** |

| Aggregate | Value |
|-----------|------:|
| **Final score** | **9237.39** |
| gap | 0.108256 |
| Vocab size | **10 000** |
| hard_ok | **yes** |
| all $X\le 1.2$ | **yes** |
| UNK on train pages | **0** |

### Vocab distribution

| Component | Count |
|-----------|------:|
| Stage-1 chars + merges (budget) | **2 500** |
| Actual merge rules stored | **2 297** |
| Stage-2 whole words | **7 500** |
| Phase A priority seats | EN **2713** · HI **1345** · TE **1278** · SA **867** |
| Phase B after all ≤1.2 | **1 297** |
| **Total** | **10 000** |

### Page types as full words vs BPE fallback

| Lang | Unique on page | In vocab as full word | Not listed (subword) |
|------|---------------:|----------------------:|---------------------:|
| EN | 7 015 | 3 944 | 3 071 |
| HI | 3 589 | 2 790 | 799 |
| TE | 2 155 | 1 982 | 173 |
| SA | 1 275 | 1 110 | 165 |

### Older comparison: fair B without deficit (`B.json`)

| | EN | HI | TE | SA | score |
|--|---:|---:|---:|---:|------:|
| $X$ | 1.336 | 1.000 | 1.000 | 1.000 | ~2975 |
| hard_ok | no | (SA/TE OK but EN over 1.2) | | | |

## Honesty

| | |
|--|--|
| **Score** | **6–7 / 10** |
| **Why somewhat honest** | Still a **tokenizer**: OOV words get BPE pieces; vocab ≤10k; encode is type-level (not multi-word page IDs). Deficit fill is a clear, documentable rule (fix $X>1.2$). Numbers match retrain under the same harness. |
| **Shortcomings** | Stage-2 seats are chosen from **these India pages** — partly closed-set. Transfer to new domains weaker than pure large-BPE. Not “pure BPE.” Fairness coefficients are assignment-aware ($\tau=1.2$). |

## Relation to others

Better score than pure **A** with a real subword floor; fairer hard_ok than naive **C** on full_page; not a packing stunt like **D**/champion.
