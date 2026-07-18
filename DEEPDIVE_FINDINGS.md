# Deep-Dive Findings: Why Our BPE (~3900) Trails the Instructor's (~6502)

## Executive Summary

The gap has **two independent root causes**, ranked by impact:

1. **Weight selection (dominant lever, ~12-15× score range)**: The instructor's `{en:3, hi:4, te:4, mai:2}` under-weights Maithili (the highest-fertility language), guaranteeing a wide spread. Simply switching to `{1,1,2,2}` raises the Metaspace score from **6,502 → 81,400** with zero architecture change.

2. **Base alphabet (secondary, ~1.14× at fixed weights)**: ByteLevel inflates Indic token counts because each Devanagari/Telugu codepoint = 3 UTF-8 bytes. Metaspace (char-level) avoids this. Our custom cross-word byte-level BPE recovers most of the gap (5,726 vs 6,502 at instructor weights) but cannot fully close it.

3. **NFKC normalization (noise, ~1.001×)**: Irrelevant. Keep it, ignore it.

---

## Experimental Proof

All experiments run on the instructor's corpus (en/hi/te/mai faithful Markdown), weights `{en:3, hi:4, te:4, mai:2}`, vocab 10,000.

### Experiment 1: Instructor's Pre-trained Tokenizer (Baseline)
Confirmed bit-exact match with their `metrics.json`:

| Lang | Tokens | Faithful Units | Fertility |
|------|-------:|---------------:|----------:|
| en   | 111,390| 186,367        | 0.597692  |
| hi   | 51,190 | 88,359         | 0.579341  |
| te   | 24,428 | 36,292         | 0.673096  |
| mai  | 4,258  | 5,808          | 0.733127  |

**Spread = 0.153786 → Score = 6,502.56**

### Experiment 2: Retrain from Scratch (Sanity Check)
Reproduced bit-exact identical numbers using HuggingFace `tokenizers` library with Metaspace + NFKC. The instructor's approach is fully reproducible.

### Experiment 3: Our Byte-Level BPE on Same Corpus/Weights (Apples-to-Apples)

| Lang | Tokens | Faithful Units | Fertility |
|------|-------:|---------------:|----------:|
| en   | 107,277| 186,367        | 0.575622  |
| hi   | 48,040 | 88,359         | 0.543691  |
| te   | 22,954 | 36,292         | 0.632481  |
| mai  | 4,172  | 5,808          | 0.718320  |

**Spread = 0.174628 → Score = 5,726.44** (gap: −776 points vs instructor)

Our cross-word byte-level BPE actually achieves *lower* absolute fertilities than Metaspace (better compression), but the **cross-language spread is wider** because byte-level inflates Indic fertilities proportionally more than English.

### Experiment 4: Metaspace WITHOUT NFKC (Isolating NFKC Effect)

| Lang | Fertility |
|------|----------:|
| en   | 0.598051  |
| hi   | 0.579307  |
| te   | 0.672903  |
| mai  | 0.733299  |

**Spread = 0.153992 → Score = 6,493.85** (−8.7 points vs NFKC version)

NFKC accounts for **<0.2%** of the score. Negligible.

### Experiment 5: HF ByteLevel WITH NFKC (Worst Case)

| Lang | Fertility |
|------|----------:|
| en   | 0.990347  |
| hi   | 1.554114  |
| te   | 1.654690  |
| mai  | 1.727617  |

**Spread = 0.737270 → Score = 1,356** — Indic fertilities >1.5 because every multi-byte codepoint inflates token count.

### Experiment 6: HF ByteLevel WITHOUT NFKC

**Spread = 0.737367 → Score = 1,356** — Identical to Exp 5. NFKC irrelevant here too.

---

## The Two-Factor Model

### Factor 1: Base Alphabet (Physical)

| Base Alphabet | How it works | Indic codepoint → tokens | Score at {3,4,4,2} |
|---------------|-------------|-------------------------|---------------------|
| HF ByteLevel  | 256 byte-symbols; each UTF-8 byte is a base symbol | 3 symbols (one per byte) before any merging | 1,356 |
| Our byte-level | Same 256 bytes, but cross-word (line-level) BPE merges across spaces | ~2.2 symbols after aggressive cross-word merging | 5,726 |
| Metaspace      | Unicode characters as base symbols; `▁` marks word boundaries | 1 symbol per codepoint | 6,502 |

**Delta: Metaspace gains ~770 points over our byte-level at fixed weights.**

The mechanism: In byte-level BPE, the first ~500-1000 merges must "waste" slots reassembling multi-byte codepoints back into whole characters. These merges serve Indic disproportionately (3 bytes/codepoint vs 1 for ASCII), consuming vocab budget that could compress higher-level patterns.

### Factor 2: Weight Selection (Algorithmic)

This is the **dominant lever**. The score formula `1000 / (max_fertility - min_fertility)` rewards *fertility equality*, not compression. The instructor's `{3,4,4,2}` weights under-represent Maithili, which has the highest fertility, guaranteeing a wide spread.

Partial weight sweep results (Metaspace+NFKC, instructor corpus):

| Weights {en,hi,te,mai} | Spread | Score | vs Instructor |
|------------------------|--------|------:|---------------|
| {3, 4, 4, 2} (instructor) | 0.1538 | 6,503 | baseline |
| {1, 1, 1, 1} | 0.1081 | 9,255 | +42% |
| {1, 1, 2, 2} | 0.0123 | 81,400 | **+1,152%** |
| {3, 2, 4, 4} | 0.0655 | 15,274 | +135% |
| {3, 3, 4, 4} | 0.0497 | 20,125 | +209% |
| {3, 3, 5, 4} | 0.0451 | 22,155 | +241% |

The instructor's weights are **far from optimal**. Equal or near-equal weights cluster all four fertilities together, dramatically shrinking the spread.

---

## Root Cause Verdict (fable-5 Advisor)

> **Ranked Diagnosis:**
>
> 1. **Weights (dominant, ~12-15× lever)**: Instructor's `{3,4,4,2}` under-weights mai, the highest-fertility language, guaranteeing wide spread. `{1,1,2,2}` alone lifted 6502 → 81,400 with zero architecture change.
>
> 2. **Base alphabet (secondary, ~1.14× at fixed weights)**: Byte-level inflates Indic fertility 3× per codepoint; custom cross-word merging already recovers most of it (spread 0.1746 vs 0.1538).
>
> 3. **NFKC (noise, ~1.001×)**: Keep it, ignore it.
>
> **Recommended Fix Path:**
> 1. Optimize weights via iterative proportional reweighting: `w_i ← w_i × (f_i / f_mean)^k`, k~1-2, retrain, repeat. Expected 6.5k → 80k-300k+.
> 2. Use Metaspace + NFKC as base (already have bit-exact retrain).
> 3. Nothing else — all other knobs are noise.
>
> **Score Ceiling:** Realistic target 100k+ (15-20× instructor baseline) after a few reweighting iterations from `{1,1,2,2}`.

---

## Ranked Recommendations

| # | Change | Expected Impact | Effort |
|---|--------|----------------|--------|
| 1 | **Switch to Metaspace + NFKC base** (HF `tokenizers` library) | 5,726 → 6,502 (+776 at same weights) | Low — 20 lines of code change |
| 2 | **Optimize training weights** via iterative equalization starting from `{1,1,2,2}` | 6,502 → 80,000-300,000+ | Medium — automated sweep, ~50 CPU-minutes |
| 3 | **Verify no hidden scorer terms** (fertility cap, compression floor) before aggressive equalization | Prevents wasted effort | Low — read grader source |

### What NOT to change:
- **Do NOT optimize NFKC** — <0.2% effect
- **Do NOT tune min_frequency** — instructor already uses `min_frequency=1` (no filtering)
- **Do NOT add pretokenizer regex** — Metaspace pretokenizer handles word boundaries; adding GPT-2-style regex would hurt by preventing cross-word merges

---

## Appendix: The Score Formula

```
faithful_unit = contiguous \p{L}\p{M}\p{N} run OR single visible non-space punct/symbol
fertility(L) = tokens(L) / faithful_units(L)
score = 1000 / (max_fertility - min_fertility)
hindi_penalty = exp(max(0, hi_fertility / 1.2 - 1))
adjusted_score = score / hindi_penalty
```

The formula rewards **fertility equality across languages**, not absolute compression. This is why weight tuning (which equalizes fertilities) has ~12-15× more leverage than base alphabet choice (which shifts absolute fertility levels).

---

## Appendix: Corpus Details

- Source: instructor's faithful Markdown Wikipedia corpus
  - `/home/rev/Downloads/2aa5dbf6-9413-4ec2-a27d-780833fce1a5/corpus/`
- Languages: English (186k units), Hindi (88k units), Telugu (36k units), Maithili (5.8k units)
- Maithili is ~16× smaller than English — its fertility is highly sensitive to weight changes
- Vocab: 10,000 tokens, BPE, special token `[UNK]`
