# Final board — `data/raw` (old) vs `full_md` retrain

**Date:** 2026-07-11  
**Sources:**  
- Old: `grok_2/results/approach_{a..e}.json`, `champion.json`, RED_PILL_REVIEW  
- New: `grok_2/full_rerun/results/{A,B,C,D,E,champion}.json`, `RESULTS_TABLE.md`  
- Corpus STATS: full_md union **7560**; old raw words **EN 12328 / HI 8141 / TE 2532 / SA 1558** (union uniques ~8311 for C-era)

Denoms: **words** $X=T/D_{words}$; **fixed** $X=T/5000$.  
score = $1000/(\max X-\min X)$ (inf if gap≈0).  
hard_ok: $X_{EN}≤1.2$ and vocab≤10k for that denom.

---

## 1. Corpus shift (why numbers move)

| | EN words | HI | TE | SA | Union unique |
|--|--------:|---:|---:|---:|-------------:|
| **old data/raw** | 12328 | 8141 | 2532 | 1558 | ~8311 (C vocab) |
| **full_md** | **10307** | **8111** | **2307** | **1541** | **7560** |
| Δ | −2021 | −30 | −225 | −17 | ~−750 |

- EN thin-out ≈ old paste residue / slightly longer EN body; HI nearly stable.  
- Whole-word ∞ still free both times (union <10k).  
- Packing $r$ search re-fits to new $D$; old $T^*≈(633,418,130,80)$ @ $r≈0.0513$; new $T^*=(1117,879,250,167)$ @ $r≈0.10837$ (score maximizer, denser rate).

---

## 2. Head-to-head (primary reported score_words)

| Approach | old (raw) primary | old score_words | full_md primary | full_md score_words | hard_ok_w old→new |
|----------|-------------------|----------------:|-----------------|--------------------:|-------------------|
| **A** | closure (word dump) | **inf** | **pure BPE only** | **2384** | True→True (but old primary was ∞ junk) |
| **A pure** | buried best_pure | **2247** | primary | **2384** | True→True |
| **B** | full_fair fill | **inf** | whole-word fill | **inf** | True→True |
| **B merges** | (not hard_ok primary) | — | merges_only | 3512 *(X_EN fail)* | —→**False** |
| **C** | words-only | **inf** | words-only | **inf** | True→True |
| **D** | slice oracle | **~1.97e8** | slice oracle | **~1.41e8** | True→True |
| **E** | primary words | **inf** | primary words | **inf** | True→True |
| **E compress** | equalized multiword | **~1.83e5** | multiword ref | **~1.68e4** | True→True |
| **Champion** | D + residual | **~1.97e8** | D + residual | **~1.41e8** | True→True |

Notes:

- full_md **A is castrated of closure** in `run_all.py` — good honesty upgrade vs old shipped A.  
- Packing score fell (1.97e8 → 1.41e8) because $T^*$ search targets max score under new D; different near-equal gap (5.07e-6 → 7.11e-6). Still same species of hack.  
- E compressed much less “fair-equalized” on full_md (gap larger → score 16.8k vs old 183k). Still real non-pack elastic encode.

---

## 3. Fixed-5000 board (hard gate pressure)

### Old raw (representative)

| Approach | $X_{EN}^{fixed}$ | score_fixed | hard_ok_f |
|----------|-------------------:|------------:|:---------:|
| C / B-fill / E-primary ($T=D$) | 12328/5000=**2.4656** | ~464 | **False** |
| A pure (~T_EN from X≈1.17→~14450?) | >1.2 | — | **False** |
| Champion packing T=633 | **0.1266** | **~9042** | **True** |
| D mega T=1 | 0.0002 | **inf** | True |

### full_md

| Approach | $X_{EN}^{fixed}$ | score_fixed | hard_ok_f |
|----------|-------------------:|------------:|:---------:|
| **C / B-fill / E-primary** | **2.0614** | 570.39 | **False** |
| **A pure** | **2.4012** | 518.73 | **False** |
| **D / champion** | **0.2234** | **5263.16** | **True** |
| B merges_only | 3.5818 | 331.06 | False |
| E compressed | 1.7198 | 688.90 | **False** |

**Callout confirmed on full_md:** under fixed-5000, **C fails** $X_{EN}≤1.2$ ($2.0614>1.2$). Same class failure on old raw (2.47). Inventories cannot satisfy “one token per word on a 10k-word page” and “$T_{EN}/5000≤1.2$” simultaneously.

---

## 4. Tight ranking tables — **full_md only** (authoritative for this retrain)

### 4a. score_words (higher better; true maximizers first)

| # | name | vocab | score_words | gap_words | hard_ok_w | T |
|--:|------|------:|------------:|----------:|:---------:|---|
| 1 | **C words** *(honest ∞)* | 7752 | **inf** | 0 | Y | =D (10307/8111/2307/1541) |
| 1 | **B fill** *(∞ softcode)* | 8274 | **inf** | 0 | Y | =D |
| 1 | **E primary** *(∞ softcode)* | 7748 | **inf** | 0 | Y | =D |
| 4 | **D pack** | 2413 | **1.4070e8** | 7.11e-6 | Y | 1117/879/250/167 |
| 4 | **champion** | 10000 | **1.4070e8** | 7.11e-6 | Y | same as D |
| 6 | **E compressed** (ref) | 10000 | **16802.7** | ~0.0595 | Y | 8599/6859/2062/1341 |
| 7 | **B merges_only** | 4200 | 3512.3 | — | **N** | 17909/12460/4189/2806 |
| 8 | **A pure** | 10000 | **2384.5** | 0.419 | Y | 12006/9057/3523/2367 |

*(RESULTS_TABLE folded E compressed into notes; rank list above surfaces it.)*

### 4b. score_fixed

| # | name | vocab | score_fixed | gap_fixed | hard_ok_f | $X_{EN}$ fixed |
|--:|------|------:|------------:|----------:|:---------:|-----------------:|
| 1 | **D pack** | 2413 | **5263.2** | 0.19 | **Y** | 0.2234 |
| 1 | **champion** | 10000 | **5263.2** | 0.19 | **Y** | 0.2234 |
| 3 | E compressed (ref) | 10000 | 688.9 | — | **N** | 1.7198 |
| 4 | C / B fill / E primary | ~7.8k | 570.4 | 1.753 | **N** | **2.0614** |
| 7 | A pure | 10000 | 518.7 | 1.928 | **N** | 2.4012 |
| 8 | B merges_only | 4200 | 331.1 | — | **N** | 3.5818 |

### 4c. Oral honesty ranking (not metric rank)

| # | Approach | Honesty | Submit as |
|--:|----------|--------:|-----------|
| 1 | **C** | 9 | Metric maximizer (words), say so |
| 2 | **A pure** | 8 | “Our BPE” |
| 3 | **E compressed** | 7 | Multiword compression baseline |
| 4 | **B merges_only** | 7 | Fair-BPE attempt (fails EN hard) |
| 5 | **D** labeled oracle | 6 | Packing upper bound |
| 6 | B fill / E primary | 3–4 | Do not brand as BPE |
| 7 | **Champion** as BPE | 3 | Fail oral |

---

## 5. Old raw tight tables (for comparison)

### score_words (old)

| # | name | score_words | hard_ok_w | notes |
|--:|------|------------:|:---------:|-------|
| 1 | C / B fill / E / A-closure | **inf** | Y | $T=D$, X=1 |
| 5 | **D / champion** | **~1.972e8** | Y | T\*=(633,418,130,80) |
| 7 | E compressed | **~1.83e5** | Y | X≈0.50 |
| 8 | A pure | **~2247** | Y | buried under closure in shipped A |
| — | B merges without fill | fails X_EN often | N | as before |

### score_fixed (old packing champion)

| name | score_fixed | hard_ok_f |
|------|------------:|:---------:|
| D mega T=1 | **inf** | Y |
| Champion pack T=633 | **~9042** | Y |
| Word inventory | **~464** | **N** ($X_{EN}=2.47$) |

---

## 6. Wishful ranking vs metric ranking (integrity check)

| Question | Answer on full_md |
|----------|-------------------|
| Does ∞ beat 1.41e8? | **Yes** — RESULTS_TABLE ranks inf first (better than old SWARM that put packing over ∞) |
| Does hard_ok_fixed rescue packing as “best real tokenizer”? | **No** — it only means slabs made $T_{EN}$ small |
| Is conversion cheating unique<10k? | **No** — body-without-refs; 14k generous strip still fails whole-word fit |
| What number is “real BPE”? | **~2384** score_words (A), hard_ok_words only |

---

## 7. Oral-safe final recommendation

**Recommended oral / writeup package**

1. **A pure** as the BPE result: score_words **≈2384**, $X≈(1.16, 1.12, 1.53, 1.54)$, V=10k.  
2. **C** as the words-denom maximizer: **score = ∞**, V≈7752 — and explicitly: **fails fixed hard** ($X_{EN}=2.06$).  
3. **D** as packing **oracle**, score_words **≈1.41×10⁸**, collapses on page mutation.  
4. Do **not** submit champion as “multilingual BPE.”  
5. State denom ambiguity: words → ∞ free if types fit; fixed → inventory far over $X_{EN}=1.2$; packing/mega designed for dramatized finite numbers.

**One-liner:**  
*full_md retrain changes magnitudes, not the species of every approach — pure BPE ~2.4k, inventory ∞, packing ~1e8 page oracle; under fixed-5000, C is hard-fail and packing is still not a tokenizer.*

---

## 8. File map

| Path | Role |
|------|------|
| `/home/rev/timepass/transformer-learning/tokenizer/data/full_md/STATS.json` | Corpus word/unique facts |
| `.../grok_2/full_rerun/results/*.json` | Retrain numbers |
| `.../grok_2/full_rerun/RESULTS_TABLE.md` | Auto side-by-side |
| `.../grok_2/full_rerun/run_all.py` | Encode paths (packing short-circuit; C inventory; A pure) |
| `.../grok_2/full_rerun/reviews/REVIEW_conversion.md` | Conversion greenlight |
| `.../grok_2/full_rerun/reviews/REVIEW_approaches.md` | This retrain’s red pill |
| `.../grok_2/results/*` | Old raw board |
