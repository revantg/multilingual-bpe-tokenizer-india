# SCORE-PUSH FINDINGS — spread 0.100469 → 0.005019, score 9,953 → 199,236

**Date:** 2026-07-18 · **Lead:** Claude Fable 5 (creative optimization lead), with three
subagents (two-stage budget family, idea discovery, adversarial validation).
**Scratch artifacts:** `/tmp/claude-1000/-home-rev-timepass-tokenizer-assignment/ce8b7839-89d4-44f2-9a7f-48f66ba3a5c0/scratchpad/`
(`fractrain.py`, `fractrain_nl.py`, `results/*.json`, `twostage/`, `ideas/`, `validate/`).
No live solution artifacts were modified; the winner is FLAGGED FOR APPLICATION below.

## TL;DR

Two fair, principled changes take the score from **9,953 → 199,236** (20×):

1. **Fractional language upsampling** (deterministic Bresenham line replication) —
   the old integer sweep never pushed MAI above weight 2 even though MAI was the
   spread ceiling. MAI's X responds at roughly −0.04..−0.09 per +1 weight; raising
   it to ~4 and fine-tuning HI/TE aligns all four fertilities near 0.575.
2. **Newline attachment** (train on `line + "\n"`, encode with chunks that keep the
   trailing newline) — previously every `\n` cost exactly one unmergeable token, a
   tax worth 0.0082 (EN) to 0.0146 (MAI) of X, i.e. the tax itself was ~40% of the
   residual spread and fell hardest on the ceiling language.

**Winner:** newline-trailing variant, weights **EN 2 / HI 2.2 / TE 3.7 / MAI 4.05**,
vocab 10000.

| | EN | HI | TE | MAI |
|---|---|---|---|---|
| X (tokens/faithful units) | 0.571995 | 0.573490 | 0.577014 | 0.572658 |

- **spread = 0.005019 → score = 1000/0.005019 = 199,235.89** (instructor formula,
  recomputed from scratch)
- Gates: vocab exactly 10000 ✓ · all X ≤ 1.2 ✓ (max 0.577) · byte-exact
  decode(encode()) on all 4 corpora + probe ✓ · 0 UNK ✓ · deterministic ✓
- Validation: see §5 (independent subagent audit).

## 1. Method (what a grader re-runs)

Training pool: for each language, take the non-empty lines of its corpus file,
**keep a trailing `"\n"` on each line**, and replicate line *i* deterministically
so the language's total replication equals its fractional weight
(Bresenham: line gets ⌊w⌋ copies, +1 whenever the running remainder crosses 1).
Train the existing byte-level greedy BPE (`tokenizer.py`) to vocab 10000.

Encoding: instead of `regex.split(r"(\n+)")` (which strands every newline as its
own token forever), chunk with `[^\n]*\n|[^\n]+\Z` so each chunk carries its
trailing newline and learned `…\n` merges can fire. Chunks partition the text
exactly → byte-lossless, UNK structurally unreachable (256-byte base alphabet
unchanged).

Both knobs are standard tokenizer-design choices: fractional upsampling is
temperature-style sampling for low-resource languages (XLM-R lineage); keeping
newlines mergeable is GPT-2's own behavior. No metric redefinition, no
hand-inserted vocab, no ratio feedback into training.

## 2. Search trajectory (all instructor-formula, all gates checked per run)

| Variant | Weights EN/HI/TE/MAI | X_EN | X_HI | X_TE | X_MAI | Spread | Score |
|---|---|---|---|---|---|---|---|
| plain (old champion) | 2/2/3/2 | 0.566887 | 0.577406 | 0.601152 | 0.667355 | 0.100469 | 9,953 |
| plain | 2/2/3/3 | 0.569908 | 0.575606 | 0.603081 | 0.608127 | 0.038219 | 26,165 |
| plain | 2/2/3/4 | 0.572424 | 0.574950 | 0.603549 | 0.569387 | 0.034162 | 29,272 |
| plain | 2/2/3.5/3.5 | 0.575059 | 0.582838 | 0.579742 | 0.591426 | 0.016367 | 61,100 |
| plain | 2/2.1/3.5/3.8 | 0.575692 | 0.578323 | 0.585363 | 0.589532 | 0.013840 | 72,256 |
| plain | 2/2.1/3.5/4.1 | 0.575649 | 0.579692 | 0.585887 | 0.577135 | 0.010238 | 97,679 |
| plain (best plain) | 2/2.1/3.4/4.0 | 0.576593 | 0.579386 | 0.584289 | 0.583506 | 0.007695 | 129,954 |
| newline | 2/2.1/3.4/4.0 | 0.570739 | 0.573931 | 0.580624 | 0.574725 | 0.009884 | 101,170 |
| newline | 2/2.1/3.7/4.0 | 0.570171 | 0.579296 | 0.571283 | 0.577307 | 0.009125 | 109,587 |
| **newline (WINNER)** | **2/2.2/3.7/4.05** | **0.571995** | **0.573490** | **0.577014** | **0.572658** | **0.005019** | **199,236** |

(15 configs total; full JSON in scratch `results/`. Sanity: our re-run of 2/2/3/2
reproduced the champion's numbers to the last digit before any change was made.)

Below ~0.008 the individual X respond to 0.1-weight moves with ±0.003
merge-discreteness noise, so the search was STOPPED at 0.005 rather than chasing
order statistics — continuing would be fitting noise unique to these exact files.

## 3. Families explored and closed

- **Two-stage merge budgets** (subagent): stage-1 normal pool, last K merges on a
  MAI-heavy pool. Best K=150, MAI-only stage 2 → spread 0.0318 (31,454). Beats the
  old champion but loses to fractional weights, and pushing K up mints tokens that
  are ENTIRE Maithili lines (2 at K=150 → 5 at K=1500) — a memorization smell that
  scales with the knob. Closed: dominated and fairness-fragile at high K.
- **Fairness-aware merge selection** (pick merges minimizing current max X):
  REJECTED without running — that is literally training on the ratios, banned by
  the brief.
- **Idea-discovery agent** measured and rejected: digit splitting (+0.097 to X_EN —
  overshoots the other way), punctuation isolation (punct is 52–56% of units;
  isolating it raises TE/MAI most → widens spread), unused-byte-slot reclamation
  (breaks zero-UNK on unseen bytes), DP/shortest-path encoding (helps EN most →
  WIDENS spread by ~0.004), per-language granularity (gray-zone fairness, weak
  mechanism). Dead-token prune+refill (874/9740 merge tokens never used) was left
  on the table: promising but needs a clean "vocab==10000" accounting story.

## 4. Why this is fair (and what we did to keep it honest)

- **Trained BPE only**, only on the 4 assignment corpora; anyone re-running the
  procedure gets bit-identical merges (greedy frequency + deterministic
  tie-breaking + deterministic replication; verified twice).
- **Knobs used are the ones the brief explicitly allows**: language
  weighting/upsampling (made fractional, still deterministic) and
  pre-tokenization/newline handling (lossless, general — it would help ANY
  line-structured corpus, not just these files).
- **No ratio feedback**: weights were chosen by observing evaluation results and
  picking new weights by hand (coordinate descent), exactly like the original
  integer sweep — the TRAINING PROCEDURE itself never sees the metric.
- **Stopping rule against overfitting**: tuning stopped at 0.1/0.05 weight
  granularity when moves became noise-dominated.
- **Robustness audit** (§5): the recipe was retrained on perturbed corpora
  (simulating a different wiki snapshot) to measure how much of the 0.005 is
  file-specific luck vs structure.

## 5. Independent validation (adversarial subagent)

An adversarial validator retrained everything from scratch with its own driver and
its own instructor-metric evaluator (scratch `validate/`):

- **Reproduction:** the winner retrains bit-identically (merges SHA-256
  `83a2c3bd…92bc7` on both runs) and the independent evaluator reproduces all four
  X and spread 0.005019 / score 199,235.89 **to 6 decimals**.
- **Gates:** vocab exactly 10000 (4 specials + 256 bytes + 9740 merges); max X
  0.577 ≤ 1.2; byte-exact roundtrip on all 4 full corpora, the instructor probe,
  and 12 adversarial probes (emoji ZWJ, Devanagari ZWJ, CRLF, tabs, NUL/control
  bytes, empty string, `\n\n\n`, no-trailing-newline, mixed scripts…); **0 UNK
  everywhere**.
- **Memorization audit:** winner has 4 whole-line vocab tokens vs the champion's 5,
  and **zero MAI-specific memorized lines** (the only whole-line matches are
  `| --- | --- |`-style table markup shared by all four files). Long (~150–200
  char) tokens are HI/EN wiki citation boilerplate present in BOTH configs — a
  base-trainer/corpus property, not a winner exploit. The winner does not memorize
  the tiny MAI page more than the champion does.
- **Robustness (simulated different snapshot, same weights, no re-tuning):**

  | Config | P1: drop every 10th line | P2: drop every 7th line |
  |---|---|---|
  | Winner recipe | spread 0.0387 → 25,869 | spread 0.0185 → 54,169 |
  | Old champion 2/2/3/2 | spread 0.1280 → 7,814 | spread 0.1105 → 9,047 |

  The exact 0.005 spread is snapshot-specific (grows 4–8× under perturbation), but
  the recipe beats the champion by 3.3–6× on every perturbation without touching a
  knob — the improvement is structural, not memorized. Note the champion itself
  degrades to 0.11–0.13 on perturbed data, so the honest comparison is
  0.019–0.039 vs 0.111–0.128.

## 6. To apply the winner (NOT yet applied)

1. `tokenizer.py` encode(): replace the `regex.split(r"(\n+)", text)` loop with
   `for m in regex.compile(r"[^\n]*\n|[^\n]+\Z").finditer(text):` over chunks.
2. `train.py`: build the pool from `line + "\n"` for non-empty lines; replace
   integer replication with Bresenham `repeat_counts` (see scratch
   `fractrain_nl.py`, which is the exact reference implementation).
3. Set weights EN 2 / HI 2.2 / TE 3.7 / MAI 4.05; retrain; expect exactly the §TL;DR
   numbers; regenerate `tokenizer.json` + `metrics.json`.

## 7. Honest verdict

The 9,953 champion was NOT near the fair ceiling — it was an artifact of an
integer-only weight grid that never upsampled the smallest corpus past 2×. Two
principled, reproducible knobs (fractional upsampling + mergeable newlines) fairly
reach **spread 0.005019 / score 199,236** on the instructor's exact formula with
every hard gate passing. The number to quote as robust is more conservative:
under snapshot perturbation the same weights give spread ~0.019–0.039 (score
26k–54k) versus the champion's 0.111–0.128 (7.8k–9.0k) on identical
perturbations — a 3.3–6× structural improvement (see §5). The 0.005 figure is
real, deterministic, and reproducible on these exact corpora — it is what the
instructor's re-run will measure — but the margin below ~0.02 is favorable
merge-discreteness alignment specific to these files; we say so plainly rather
than claim 199k generalizes to a different snapshot.
