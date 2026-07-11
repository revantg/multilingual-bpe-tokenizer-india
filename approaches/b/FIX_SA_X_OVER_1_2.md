# Fix: SA $X > 1.2$ on full_page_md (Approach B)

**Date:** 2026-07-11  
**File changed:** `train_optimized_b.py`  
**Model out:** `vocab_optimized_fullpage.json`

---

## Problem

Global savings fill (stage-2) ranked words by total token savings across languages. EN/HI mass won most whole-word seats. Sanskrit was left with multi-piece OOVs:

| | Before (global fill) | After (deficit fill) |
|--|--:|--:|
| $X_{\mathrm{SA}}$ | **1.226 – 1.227** | **1.110** |
| $X_{\mathrm{EN}}$ | 1.117 | 1.157 |
| hard_ok (X_EN ≤ 1.2) | yes | yes |
| hard_ok_all ($X_L\le 1.2\ \forall L$) | **no** | **yes** |
| score_words | ~9 070 – 9 147 | **~9 298** |

Needed to cut SA: only ≈ **55** tokens ($T_{\mathrm{SA}}$ from ~2567 → ≤ 1.2×2092 ≈ 2510).

---

## What we changed

### 1. New stage-2 mode: `deficit` (default)

Replaced pure  
`score(w) = Σ α_L · f_L(w) · max(0, pieces−1)` fill with **two phases**:

**Phase A — fairness (residual to τ=1.2)**

While any language has residual  
$\max(0,\, T_L - \tau \cdot D_L) > 0$ and seats remain:

1. Pick worst language $L^\star = \arg\max_L$ residual.  
2. Add the unused type that maximizes  
   $f_{L^\star}(w)\cdot\max(0,\,p(w)-1)$  
   (tie-break: global α-weighted savings).  
3. Incrementally subtract saved tokens from running $T_{L}$.

**Phase B — score leftover seats**

Same as legacy: remaining slots by global α-weighted savings, then any leftover types.

### 2. CLI / knobs

| Flag | Default | Meaning |
|------|---------|---------|
| `--fill-mode deficit\|global` | `deficit` | New vs legacy pure savings |
| `--tau` | `1.2` | Phase-A target for every language |

Legacy path preserved: `--fill-mode global`.

### 3. Meta fields in saved model

- `fill_mode`, `tau`  
- `phase_a_added`, `phase_b_added`, `phase_a_by_lang`  
- `hard_ok_all_langs` (every $X_L \le \tau$)

### 4. What we did *not* change

- Stage-1 still ClassicBPE to `bpe_size=2500`, `en_merge_weight=2`.  
- Encode still DP min-cover.  
- Vocab cap still 10 000.  
- Assignment hard gate still X_EN ≤ 1.2 (also satisfied all langs now).

---

## Measured fill on this retrain

```text
fills: word seats after BPE = 7500
phase_a_added = 6176   (until all residual≈0)
phase_a_by_lang:
  EN 2688  HI 1343  TE 1278  SA 867
phase_b_added = 1324   (global savings remnant)
```

Phase A deliberately gave **SA 867 priority seats** (and TE a large round as well) before free-for-all EN mass in phase B. That is the mechanical reason SA dropped under 1.2; EN rose a bit (more competition for savings seats) but stayed under 1.2.

---

## New numbers (authoritative)

```json
{
  "X_words": {
    "EN": 1.157138221933809,
    "HI": 1.0824349101576825,
    "TE": 1.0495928941524797,
    "SA": 1.1104206500956022
  },
  "T": {"EN": 35663, "HI": 14759, "TE": 4254, "SA": 2323},
  "score_words": 9298.404873834106,
  "hard_ok_words": true,
  "hard_ok_all_langs": true,
  "gap_words": 0.10754532778132941,
  "final_vocab": 10000,
  "fill_mode": "deficit"
}
```

Self-score quote: **9298.40**  
calc: `1000 / (1.157138 − 1.049593) ≈ 9298.40`

---

## How to re-run

```bash
cd /home/rev/timepass/transformer-learning/tokenizer
python3 data/grok_2/approach_b/train_optimized_b.py \
  --vocab-cap 10000 --bpe-size 2500 \
  --fill-mode deficit --tau 1.2 \
  --out data/grok_2/approach_b/vocab_optimized_fullpage.json
```

Rollback to old stage-2:

```bash
python3 data/grok_2/approach_b/train_optimized_b.py --fill-mode global ...
```

---

## Design note

This is still **hybrid BPE + page-type inventory**, not pure BPE. The fix only **reallocates** whole-word seats toward languages over τ instead of pure global savings. Documented strategy: `RESILIENCE_X_LE_1_2.md`.
