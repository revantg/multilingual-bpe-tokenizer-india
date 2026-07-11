# Approach E — Hybrid words + multiword phrases + DP

**Status:** Experimental · **not** submitted  
**Shortcut:** `../approaches/e` → this folder

## Idea (plain English)

Build a vocabulary of **characters + whole words + multi-word phrases**, then encode each page with **DP** that picks the fewest pieces (phrase if helpful, else word, else characters).

Can drive $X=1$ (complete inventory) or lower $X$ via phrases. Must respect the 10k cap or hard_ok fails.

## How it works

1. Bases (scripts, digits, specials).  
2. Unique page words.  
3. Optional high-savings multiword n-grams (often worst-$X$ language first).  
4. Filler char n-grams.  
5. Encode: pretok words → DP over word/phrase lattice.

## Relevant files

| File | Role |
|------|------|
| **`hybrid_tokenizer.py`** | Build + DP encode + CLI |
| `vocab.json` | Inventory-style dump (early) |
| `vocab_compressed.json` | Phrase compression build |
| `NOTES.md` | Construction detail |

| Shared | Role |
|--------|------|
| `../datasets/build_full_page/results/E.json` | full_page board |
| `../results/early/approach_e.json` | early board |

## Run

```bash
python3 data/grok_2/approaches/e/hybrid_tokenizer.py
```

## Metrics

### Full_page board (`E.json`) — **invalid under 10k cap as stored**

| | EN | HI | TE | SA |
|--|---:|---:|---:|---:|
| $D$ | 30 820* | 13 635 | 4 053 | 2 092 |
| $T$ | 30 820 | 13 635 | 4 053 | 2 092 |
| $X$ | **1.000** | **1.000** | **1.000** | **1.000** |

| Aggregate | Value |
|-----------|------:|
| **Final score** | **∞** (gap 0) |
| Vocab size | **12 603** |
| hard_ok | **no** (vocab **> 10 000**) |

To use E legitimately on full_page you must **cap** inventory/phrases at 10k (accept $X_{\mathrm{EN}}>1$ or rebuild selection).

### Early corpus (notes) — cap satisfied

| Mode | vocab | $X$ pattern | score |
|------|------:|---------------|------:|
| Full words + bases | ≤10k | all ≈1.0 | **∞** |
| Compressed multiword | 10k | ≈0.50 equalized | ~1.8e5 |

### Vocab distribution (full-inventory style E)

| Component | Role |
|-----------|------|
| Script / char bases | always present |
| Union of page types | main mass (blows past 10k on full_page) |
| Multiword phrases | optional; lowers $T$ unevenly |
| Residual char n-grams | OOV substructure |
| Total must be | **≤ 10 000** for assignment |

## Honesty

| | |
|--|--|
| **Score** | **5–7 / 10** when vocab honestly ≤10k; **2 / 10** if you quote ∞ with a 12k list |
| **Why honest (when capped)** | Hybrid is explicit; DP encode is a real mechanism; ∞ score on small corpora is mathematically correct if inventory fits. |
| **Shortcomings** | Easy to **bust the vocab cap**. Multiword tokens are strongly page-dependent. Overlaps marketing with C/B without keeping BPE subword quality for pure OOV. |

## Relation to submission

Interesting cousin of B (words + structure). We ship **B** because it **respects 10k**, keeps BPE, and passes hard_ok on full_page without packing.
