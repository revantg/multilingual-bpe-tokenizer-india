# SCORE-PUSH BRIEF — get the spread lower, fairly

You are **Fable 5**, acting as a creative optimization lead. Goal: **lower the
fertility spread** of our multilingual BPE tokenizer to push the assignment score
as high as honestly possible — **without cheating**. You may **spawn your own
subagents** to parallelize experiments. Be creative, but every gain must survive
the instructor's own re-run.

## The scoring (EXACT — from the instructor's evaluate_tokenizer.py)

```python
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
def faithful_units(text): return len(FAITHFUL_UNIT_RE.findall(text))

for code in ["en","hi","te","mai"]:
    units  = faithful_units(text)
    tokens = len(tokenizer.encode(text).ids)
    ratio  = tokens / units            # this is X_L
spread = max(ratios) - min(ratios)
score  = 1000 / spread
```

- `X_L = tokens(L) / faithful_units(L)` for L in EN, HI, TE, MAI (4 languages).
- **score = 1000 / (max X − min X).** Lower spread ⇒ higher score. That is the
  ONLY objective. (There is a separate reported `english/hindi penalty`, but the
  headline score is `1000/spread`; keep every X ≤ 1.2 anyway — it's a hard gate.)
- Faithful units = contiguous L/M/N runs OR a single visible punctuation char.
  This denominator is FIXED by the instructor; do not redefine it.

## Hard gates (any failure = disqualified, score 0)

1. **vocab == 10000** exactly.
2. **every X_L ≤ 1.2.**
3. **decode(encode(text)) preserves the text** (lossless) on every corpus and on
   the probe `India's population is 1,428,627,663.`
4. **zero UNK** on all corpora.
5. **Reproducible**: the instructor RE-RUNS the tokenizer. No hard-coded numbers,
   no per-corpus special-casing, no test-set peeking that wouldn't generalize.

## Fairness rules (NON-NEGOTIABLE)

- The tokenizer must be a **real, trained BPE** that anyone can retrain and get
  the same vocab/merges. Train ONLY on the India-Wikipedia faithful corpora
  (en/hi/te/mai). No external corpora, no synthetic text engineered to match the
  eval, no vocab entries hand-inserted to fit the test.
- You MAY choose: training granularity (word/sentence/paragraph/line), how the
  languages are **weighted/upsampled** in the training pool, the merge/vocab
  **budget schedule**, and pre-tokenization — as long as it's a principled knob a
  real tokenizer designer would use.
- You may NOT: change the faithful-unit formula, train on the ratios themselves,
  or overfit a knob to these 4 exact files in a way that wouldn't hold if the
  grader used a slightly different India page snapshot. If a technique only works
  because it memorizes THIS text, it's cheating — reject it.

## Current champion (the number to beat)

- Weights EN:2 HI:2 TE:3 MAI:2, **line/paragraph-level** training, vocab 10000.
- X: EN 0.566887, HI 0.577406, TE 0.601152, MAI 0.667355.
- **spread 0.100469 → score 9953.37.** MAI is the ceiling (0.667), EN the floor
  (0.567). The spread is driven mostly by **MAI being high** (its page is small,
  ~85 lines / 5808 units) and EN being low.

## Search already done (don't just repeat it)

- Integer weight sweeps over EN/HI/TE/MAI (1..3 each): best was 2/2/3/2 = 9953.
  1/1/1/1 = 9546, 1/1/2/1 = 8212, etc. See experiments/*sweep*.json.
- Granularity: word-level fails the gate (Indic X≈1.5–1.7); sentence and
  paragraph both pass; paragraph+weights won.
- SA (Sanskrit) was dropped — its page is URL-encoded junk that pins the spread.

## Ideas to explore (get creative, add your own)

- **Finer weighting**: integer weights are coarse. Try **fractional upsampling**
  (e.g. replicate a language's lines a non-integer expected number of times via
  deterministic line sampling) to tune each X more precisely and pull MAI down /
  EN up until they meet. Keep it deterministic + reproducible.
- **Targeted budget**: since MAI is the ceiling, does giving MAI-relevant merges
  more room (or a smarter merge schedule) lower X_MAI without hurting others?
- **Pre-tokenization tweaks** that legitimately help agglutinative Indic scripts
  merge more (e.g. treating whitespace/newline handling consistently), as long as
  it stays lossless and general.
- **Two-stage / balanced merges**: choose merges that reduce the *max* X per step
  (a fairness-aware training objective) rather than pure global frequency — but it
  must still be a deterministic, retrainable procedure.
- Anything else principled. The target: **push spread below 0.100469**, ideally a
  lot, while all four X stay ≤1.2, lossless, 0 UNK, vocab exactly 10000.

## How to work

- Reproduce the baseline first: `cd solution && ../.venv/bin/python train.py`
  writes tokenizer.json + metrics.json and prints spread/score. `train.py
  --weights EN HI TE MAI` tries other integer weights. `tokenizer.py` is the
  trainer; `count_faithful_units` matches the instructor.
- Prototype experiments in **/tmp scratch**; do NOT clobber solution/tokenizer.json
  or metrics.json until you have a verified winner. Score EVERYTHING with the
  instructor's exact formula above (recompute from scratch, don't trust cached).
- **Spawn subagents** to run independent experiment families in parallel. Give
  each a precise, self-contained task and have them report spread/score + the
  four X + gate booleans. Collate.
- Machine limits: watch thermals (`sensors | grep Tctl`, ~95C danger) and memory
  (`free -m`, keep >500M). A 10k retrain is ~2–3 min single-process; don't run
  many in parallel — batch through subagents sensibly.

## Deliverable

Return: the best weighting/method you found, its exact four X + spread + score
under the instructor formula, proof all gates pass, and whether it's a legitimate
improvement over 9953.37. If nothing beats it fairly, say so honestly and explain
why 9953 is near the fair ceiling. Write a report to
`solution/SCORE_PUSH_FINDINGS.md`. Do NOT overwrite the live solution artifacts
unless you have a verified, fair winner — flag it for me to apply instead.
