# Deep-dive findings: why we score ~3900 and the instructor scores ~6502

**Date:** 2026-07-18 · **Investigator/advisor:** Claude Fable 5 (running as the deep-dive agent; the fable-5 "advisor verdict" in §6 is this model's own strategic analysis, per the brief's note that no external call is needed).
**Scratch artifacts:** `/tmp/claude-1000/-home-rev-timepass-tokenizer-assignment/bb7829c9-d276-417b-8af3-6e0def5ae95e/scratchpad/` (`hf_variant.py`, `ours_on_instructor.py`, `ours_charbase.py`, `e*.json`). No repo or bundle files were modified.

## TL;DR

The gap is **not** our BPE implementation and it is **mostly not** the byte-level base alphabet. It is, in order:

1. **Corpus pollution (dominant): our Markdown corpora keep percent-encoded URLs** (`%E0%A4%AD…`). 61–71% of the "faithful units" in our hi/te/sa files are `%` signs and hex pairs; the instructor's corpus is ~1.5%. This single artifact caps the score at ~2800 on any tokenizer.
2. **Weight choice**: the instructor's {en3,hi4,te4,mai2} is itself far from optimal — equal weights score 9255 on their own pipeline; {2,2,3,2} scores 12868.
3. **Byte-level base**: worth only ~+400–700 points inside our cross-word trainer (272/10000 vocab slots wasted on partial-UTF-8 tokens). The instructor's "ByteLevel wastes tokens on Indic" claim is true only for HF's *word-level* ByteLevel (score 1357); our line-level BPE dodges most of the tax.

**Apples-to-apples, our byte-level trainer on the instructor's exact corpus/langs/weights scores 5726 vs their 6502 — and beats them outright with equal weights (9545) or char base + equal weights (10248).** Our trainer's per-language fertility is *lower than theirs in every language*; our implementation compresses better than their HF Metaspace setup.

## 1. All experiments (10k shared vocab, faithful-unit metric, zero UNK unless noted)

| # | Trainer / base | Corpus | Weights | en | hi | te | 4th | Spread | Score |
|---|---|---|---|---|---|---|---|---|---|
| E1 | HF Metaspace+NFKC (instructor exact) | instr | 3,4,4,2 | .5977 | .5793 | .6731 | mai .7331 | .1538 | **6502.6** ✓ reproduced to 6 decimals |
| E3 | HF **ByteLevel**+NFKC, all else = E1 | instr | 3,4,4,2 | .9910 | 1.5549 | 1.6564 | mai 1.7281 | .7371 | 1356.6 |
| E4a | E1 without NFKC | instr | 3,4,4,2 | — | — | — | — | .1540 | 6493.9 |
| E4b | E1 with min_frequency=2 | instr | 3,4,4,2 | identical to E1 | | | | .1538 | 6502.6 |
| E4c | E1, equal weights | instr | 1,1,1,1 | .5867 | .5977 | .6948 | mai .6624 | .1081 | 9254.9 |
| E4w1 | E1, te-boosted | instr | 1,1,2,1 | .6020 | .6145 | .6025 | mai .6934 | .0913 | 10950.3 |
| E4w2 | E1, tuned | instr | 2,2,3,2 | .5948 | .6054 | .6413 | mai .6725 | .0777 | **12868.3** |
| E4w3 | E1, te 3× | instr | 1,1,3,1 | .6274 | .6419 | .5201 | mai .7359 | .2158 | 4634.6 (overshoot) |
| E4d | E1 but **our sa.txt** replaces mai | instr+sa | 3,4,4,2 | .6005 | .5815 | .6743 | **sa .3138** | .3605 | **2774.1** |
| E2 | **OURS byte-level** (sweep_weighted.Tokenizer) | instr | 3,4,4,2 | .5756 | .5437 | .6325 | mai .7183 | .1746 | **5726.4** |
| E2b | OURS byte-level | instr | 1,1,1,1 | .5580 | .5665 | .6627 | mai .6565 | .1048 | **9545.4** |
| E2c | OURS byte-level | instr | 1,1,2,1 | .5748 | .5877 | .5592 | mai .6810 | .1218 | 8212.2 |
| E3b | OURS **char-base** variant | instr | 3,4,4,2 | .5688 | .5369 | .6345 | mai .6999 | .1630 | 6135.9 |
| E3c | OURS char-base | instr | 1,1,1,1 | .5582 | .5538 | .6514 | mai .6286 | .0976 | **10248.1** |

Notes: E2* round-trip lossless, zero UNK. E3b/E3c have ~1500 UNKs solely because the quick char-base prototype forgot `\n` in its alphabet (newline runs → UNK); this barely affects token counts and is a one-line fix in a real implementation. Reference points: our previous best "instructor corpus + our SA" was adj 2831; on our own corpus adj 3901.

## 2. Root cause #1 (proven): percent-encoded URLs poison the corpora — ours everywhere, and sa.txt in particular

Fraction of faithful units that are URL-encoding artifacts (`%` or a bare hex pair like `E0`, `A4`):

| File | Units | URL-enc fraction |
|---|---|---|
| ours `india_sa.md` / `datasets/instructor/sa.txt` (same content) | 34,751 | **70.9%** |
| ours `india_te.md` | 51,010 | **63.7%** |
| ours `india_hi.md` | 106,766 | **60.7%** |
| ours `india_en.md` | 104,617 | 2.5% |
| instructor `hi.faithful.txt` | 88,359 | 1.5% |
| instructor `te.faithful.txt` | 36,292 | 1.3% |
| instructor `mai.faithful.txt` | 5,808 | 1.7% |

Example from our hi corpus: `(https://hi.wikipedia.org/wiki/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4_%E0%A4%95%E0%A5%80_…)`. Every `%E0` counts as **two** faithful units (`%` + `E0`), and Devanagari/Telugu use only ~30 distinct hex pairs, so BPE learns giant tokens over them → fertility collapses (sa 0.31) → that language becomes the spread's min → score is capped at `1000/(max − 0.31) ≈ 2800` **for any tokenizer**. Proof: the instructor's own exact pipeline scores 2774 (E4d) the moment our sa.txt is included — statistically identical to our 2831.

The instructor's `build_wiki_faithful_markdown.py` pulls Wikipedia REST `page/html`, whose internal links are native-Unicode (`./भारत`), so their corpus never contains these blobs. Our extraction kept fully-qualified percent-encoded hrefs.

This also explains our own-corpus 3901: hi/te are 60%+ polluted there too, dragging all Indic fertilities into artifact-driven territory and making every weight sweep optimize the wrong landscape.

## 3. Root cause #2 (proven): weights — including the instructor's own

Boosting a language's weight lowers its fertility. At the instructor's {3,4,4,2}, mai (weight 2) is stranded at 0.733 while hi (weight 4) is pushed to 0.579 — the weights *widen* the spread. Equal weights already score 9255 on their pipeline (E4c) and 9545 on ours (E2b); a light te boost {2,2,3,2} reaches 12868 (E4w2). {1,1,3,1} overshoots (te becomes the min) — the optimum is shallow and near-equal. **The instructor's 6502 is not a strong target; it's a mediocre point in weight space.**

## 4. Root cause #3 (quantified, minor): byte-level base alphabet

- In **HF's word-level** pipeline, ByteLevel is catastrophic (E3: 1357 vs 6503) — this is the fact the instructor's SOLUTION.md cites.
- In **our cross-word line-level** trainer, the byte tax is small: char base gains +410 (5726→6136 at {3,4,4,2}) and +703 (9545→10248 at equal weights). Per-language fertility deltas are only 0.002–0.028. Direct measurement: just **272 of 10,000** vocab slots in our byte-level tokenizer are partial-UTF-8 (unrecoverable codepoint fragments) — a 2.7% vocab tax, not a 0.15-fertility floor.
- The leading hypothesis in the brief ("byte base inflates Indic fertility floors by 0.15–0.25") is **refuted** for our implementation: our byte-level Indic fertilities (hi .544, te .632) are *below* the instructor's char-based ones (hi .579, te .673).

## 5. Non-factors (measured)

- **NFKC:** Δscore −8.7 without it (E4a). Irrelevant.
- **min_frequency 1→2:** bit-identical results (E4b). Irrelevant.
- **Our BPE implementation quality:** better than HF Metaspace at equal settings — lower fertility in all four languages (E2 vs E1), lossless round-trip, zero UNK.

## 6. Advisor verdict (fable-5)

The team spent its effort optimizing the tokenizer while the metric was being driven by the corpus. Fertility-spread scores are corpus-relative: a single language file whose unit distribution differs structurally (here: 71% URL-encoding artifacts) pins the min-fertility and no trainer change can recover the score. The instructor's stated rationale (Metaspace vs ByteLevel) is real but third-order for your architecture; treating it as the leading hypothesis was a red herring reinforced by the HF word-level framing. Fix the data first, then tune weights — the tokenizer you already have is competitive, and chasing a char-base rewrite before the corpus fix would have produced another ~2800 run and false despair. Ranked fix path below; expected outcomes are measured, not projected.

## 7. Ranked recommendations

| # | Change | Expected score (measured) | Effort |
|---|---|---|---|
| 1 | **Fix the corpus**: URL-decode link targets (`urllib.parse.unquote`) or rebuild via Wikipedia REST `page/html` like the instructor; drop the Sanskrit file or rebuild it the same way. Use en/hi/te/mai to be directly comparable. | unlocks everything below; on clean corpus our existing trainer at {3,4,4,2} = **5726** | ~30 min |
| 2 | **Re-sweep weights near equal** on the clean corpus (candidates around {1,1,1,1}–{2,2,3,2}); boost whichever language floors highest (te/mai), never starve the 4th language. | **9545** (ours, equal) → ~12–13k plausible ({2,2,3,2} on HF = 12868) | reuse `sweep_weighted.py` candidates, ~1 hr of runs |
| 3 | Optional: **char-base variant** of our trainer (base = corpus chars + `▁`-free line-level scheme, byte fallback for unseen chars, include `\n`). | +400–700 on top (10248 measured at equal weights) | ~half day |
| 4 | Skip: NFKC toggling, min_frequency tuning, rewriting to HF Metaspace. | ±10 | — |

Bottom line: with recommendation 1+2 alone, our existing byte-level implementation demonstrably reaches **9.5k+**, comfortably past the instructor's 6502.
