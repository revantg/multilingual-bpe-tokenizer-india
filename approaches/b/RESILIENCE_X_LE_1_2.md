# Approach B — resilient design for $X_L \le 1.2$ on every language

**Status:** design / implementation guide (not yet fully wired into every trainer).  
**Code today:** `train_optimized_b.py` (global savings fill).  
**Related:** `train_fair_bpe.py` (temp + script budgets), `NOTES.md`.

---

## Table of contents

1. [Goal](#1-goal)
2. [What B does today](#2-what-b-does-today)
3. [Why it is not all-lang resilient](#3-why-it-is-not-all-lang-resilient)
4. [Evidence on full_page_md](#4-evidence-on-full_page_md)
5. [Feasibility check on a new dataset](#5-feasibility-check-on-a-new-dataset)
6. [Fixes (priority order)](#6-fixes-priority-order)
7. [Recommended algorithm](#7-recommended-algorithm)
8. [Knob → effect cheat sheet](#8-knob--effect-cheat-sheet)
9. [Adaptive corpus modes](#9-adaptive-corpus-modes)
10. [Hard limits](#10-hard-limits)
11. [Implementation sketch](#11-implementation-sketch)
12. [Acceptance tests](#12-acceptance-tests)
13. [Oral one-liner](#13-oral-one-liner)

---

## 1. Goal

Under a **shared** vocab cap $V \le 10\,000$:

$$
X_L = \frac{T_L}{D_L} \le \tau
\quad\text{for every language } L \in \{\mathrm{EN},\mathrm{HI},\mathrm{TE},\mathrm{SA}\}
$$

with default target $\tau = 1.2$.

Secondary (after the constraint is feasible):

$$
\text{maximize } \mathrm{score} = \frac{1000}{\max X - \min X}.
$$

**Note:** The original assignment hard gate is only $X_{\mathrm{EN}} \le 1.2$.  
Targeting **all** languages $\le 1.2$ is a **stronger** product goal. Document both if you report numbers.

Denom: harness pretok words (`extract_words` / `WORD_RE`).  
$T_L$ = token count of encode(page$_L$).

---

## 2. What B does today

Optimized B (`train_optimized_b.py`) is a **two-stage hybrid**:

| Stage | Budget | Mechanism |
|-------|--------|-----------|
| **1 – BPE base** | `bpe_size` (default 2500) | `ClassicBPE`: chars → pair merges. Optional `en_merge_weight` boosts EN unit mass. |
| **2 – whole-word fill** | $V - \lvert\mathrm{vocab}\rvert$ after stage 1 | Rank each type $w$ by **global savings** and insert until full. |

Savings rank (default $\alpha_L = 1$):

$$
\mathrm{score}(w)
  = \sum_L \alpha_L \cdot f_L(w) \cdot \max\bigl(0,\, p_{\mathrm{BPE}}(w) - 1\bigr)
$$

where $p_{\mathrm{BPE}}(w)$ = # pieces classic BPE needs for $w$ **before** $w$ is stored whole.

**Encode:** if $w \in \mathrm{vocab}$ → 1 token; else DP / merge cover of characters (stage-1 merges still apply).

Defaults:

```text
vocab_cap       = 10000
bpe_size        = 2500
en_merge_weight = 2.0
alpha           = EN=HI=TE=SA=1.0
```

---

## 3. Why it is not all-lang resilient

1. **Global savings ≠ per-lang fertility prioritizes** EN/HI mass (high $f_L$).
2. **No constraint awareness** — never reads “$X_L - 1.2$” when picking $w$.
3. **Stop rule is “vocab full”**, not “all $X_L \le \tau$”.
4. **Fixed stage-1 size** — does not adapt when union unique / script mass changes.
5. **EN-only merge boost** can starve TE/SA subword formation when types stay OOV.
6. **Closed-page fill** overfits inventory on these four pages; new domains hurt if stage-1 is thin.

So B today is roughly: “pass $X_{\mathrm{EN}} \le 1.2$ + maximize total compression,” **not** “feasibility program of fertility caps.”

---

## 4. Evidence on full_page_md

Corpus: `data/full_page_md/` (union unique ≈ **12 599 > 10 000**).

| Lang | $D$ (words) | optimized B $X$ | Passes $\tau=1.2$? |
|------|--------------:|------------------:|:--------------------:|
| EN | 30 820 | ≈ 1.117 | yes |
| HI | 13 635 | ≈ 1.120 | yes |
| TE | 4 053 | ≈ 1.158 | yes |
| **SA** | 2 092 | **≈ 1.226** | **no** |

Illustrates the failure mode: **global savings** leaves the **smallest language** slightly over $\tau$ after slots run out.

Meta (typical optimized B run): `merges ≈ 2300`, `words_added_stage2 ≈ 7500`, `final_vocab = 10000`.

---

## 5. Feasibility check on a new dataset

Before retuning knobs, compute:

1. `unique[L]`, `union_U`, `D[L]`.
2. After stage-1 only: $T_L^{(0)}$, $X_L^{(0)}$.
3. Tokens that still must be cut for language $L$:

$$
\Delta_L^{\star} = \max\bigl(0,\; T_L^{(0)} - \tau \cdot D_L\bigr).
$$

4. Upper bound if **every** type of $L$ becomes a single token (ideal inventory):  
   can residual OOVs under a remaining slot budget still leave $\Delta_L^{\star}$ > 0?

Rough regimes:

| Union unique vs $V$ | Likely materialization |
|-----------------------|------------------------|
| $U + \mathrm{chars} \le V$ | Inventory can drive all $X \approx 1$ |
| $U \gg V$ (e.g. full_page ~12.6k) | Need floors + minimax + stage-1 search; some configs infeasible |
| Domain shift / held-out text | Whole-page type dump **does not transfer** — stage-1 quality dominates |

If $\Delta_L^{\star}$ cannot be closed given remaining seats **even with perfect word fills for worst langs**, report **INFEASIBLE** (only packing-style oracles then win round).

---

## 6. Fixes (priority order)

Highest ROI first:

### P1 — Minimax / deficit stage-2 fill (core)

Do **not** only maximize $\sum_L f_L \cdot \mathrm{sav}$. Prefer the language farthest above $\tau$:

```text
while |vocab| < V and max_L X_L > τ:
    L* = argmax_L (X_L − τ)              # worst deficit
    pick w not in vocab that maximizes
        f_{L*}(w) * max(0, pieces(w)−1)
        (+ small global savings tie-break)
    add w; update T_{L*}, X_{L*}
```

Cheap batch form (recompute worst lang every $K$ inserts):

$$
\mathrm{score}_{\mathrm{mm}}(w)
  = \lambda_{\mathrm{worst}}\, f_{L^*}(w)\,\mathrm{sav}(w)
  + \lambda_{\mathrm{def}}\, \sum_L \max(0, X_L-\tau)\, f_L(w)\,\mathrm{sav}(w)
  + \lambda_{\mathrm{glob}}\, \sum_L \alpha_L f_L(w)\,\mathrm{sav}(w)
$$

### P2 — Per-language floors before open market

Reserve a fraction of **word seats** for small / failing languages before global market:

```text
order langs by unique ascending (SA, TE, HI, EN) or by residual Δ*
for L in order:
    fill up to floor[L] types by L-local savings (or until X_L ≤ τ)
then global (or minimax) fill with remaining seats
```

**Adaptive floor idea:**

$$
\mathrm{floor}_L
  = \min\Bigl(
      \mathrm{unique}_L,\;
      \Big\lceil
        \tfrac{\mathrm{unique}_L}{\sum \mathrm{unique}}\cdot r\cdot (V-B)
      \Big\rceil
      + m_L
    \Bigr)
$$

- $B$ = stage-1 vocab size  
- $r \in [0.55, 0.75]$ = share of word budget on floors  
- $m_L$ = bonus proportional to post-stage-1 residual $\Delta_L^{\star}$

### P3 — Search `bpe_size` (do not hard-code 2500)

Trade-off:

| Raise $B$ | Better OOV merges | Fewer word seats |
|------------|-------------------|------------------|
| Lower $B$ | More exact words  | Thin OOV path    |

Grid example: `{800, 1500, 2500, 4000, 6000}`.

**Keep only configs with $\max_L X_L \le \tau$**, then maximize fairness score (or minimize $\max X$).

### P4 — Adaptive $\alpha$ (stage-2 reweight loop)

Start $\alpha_L=1$. After one fill:

$$
\alpha_L \leftarrow \alpha_L \cdot \Bigl(1+\eta\cdot \max\bigl(0,\tfrac{X_L-\tau}{\tau}\bigr)\Bigr)
$$

Re-run **stage-2 only** (keep stage-1) until pass or α saturates. Cheap **cross-dataset adapter**.

### P5 — Multi-language merge weights (stage-1)

Replace EN-only `en_merge_weight` with mass- or residual-aware weights:

$$
w_L \propto \mathrm{mass}_L^{-\gamma}
\quad\text{or}\quad
w_L \propto \max(X_L^{(0)}, 1)
$$

So TE/SA pairs form even when they lose the whole-word auction.

### P6 — Stop over-helping langs already $\ll \tau$

If $X_L \le 1.05$ (configurable), **stop preferring fills for that lang**. Free seats for the worst event.

### P7 — Held-out resilience (optional)

- Train fill on ~80% types / occurrences; measure $X$ on full page.  
- Or leave OOV reserve: do not spend 100% of word seats so stage-1 stays meaningful out of sample.

---

## 7. Recommended algorithm

```text
Input: corpora, V=10000, τ=1.2

1. pretok → words_by[L], D[L], unique[L], union_U

2. for B in grid(bpe_size):
     bpe ← stage1(B, merge_weights_fair)
     X ← encode pages (pieces only)

3. Stations remaining = V − |vocab|
     (optional) apply per-lang floors until each floor met or X_L ≤ τ
     while remaining > 0 and max X > τ:
         L* ← argmax (X_L − τ)
         w* ← best local savings type for L* not in vocab
         add w*; remaining −= 1; update X_{L*}
         every K steps: optional full re-encode for accuracy
     leftover seats → mild global savings fill (do not break τ)

4. if any X_L > τ:
     try next B (usually smaller B → more word seats)
     else if grid exhausted → FAIL with residual Δ_L*

5. among feasible B: pick max score = 1000/(maxX−minX)
```

---

## 8. Knob → effect cheat sheet

| Knob | Raise → | All-lang $X\le 1.2$ |
|------|---------|------------------------|
| `bpe_size` | Better OOV, fewer words | **Search**; don’t fix |
| Word seats $V-B$ | More exact types | Usually helps if floors exist |
| Global savings only | EN/HI win slots | **Hurts** rare scripts |
| Worst-lang / deficit fill | Pulls SA/TE under τ | **Main fix** |
| Per-lang floors by unique | Guarantees small-lang seats | Strong across wiki langs |
| Adaptive $\alpha_L$ | Boost failing $L$ | Easy adapter |
| EN-only `en_merge_weight` | EN OOV better | Can starve TE/SA merges |
| Band $X\in[1,\tau]$ lower stop | Avoids over-covering tiny $X$ | Frees seats + helps score |

CLI surface (once implemented) should expose at least:

```bash
python3 train_optimized_b.py \
  --vocab-cap 10000 \
  --bpe-size 2500 \
  --tau 1.2 \
  --fill-mode minimax \          # savings | floors | minimax
  --floor-frac 0.65 \
  --en-merge-weight 2.0 \
  --alpha-en 1 --alpha-hi 1 --alpha-te 1 --alpha-sa 1
```

---

## 9. Adaptive corpus modes

Auto-pick strategy from dataset shape:

```text
if union_U + base_chars ≤ V:
    MODE = INVENTORY          # X≈1 free; BPE optional/minimal
elif sum_L floor_estimate(L) ≤ (V − B_min):
    MODE = MINIMAX_FILL       # standard resilient B
else:
    MODE = HEAVY_BPE + residual
    # maximize OOV merge quality; report which L still fails τ
```

Always log: mode, $B$, floors used, $\Delta_L^{\star}$ remaining, final $X$, score, hard_ok flags.

---

## 10. Hard limits

1. **$U \gg V$** and large EN mass → all $X_L\le 1.2$ may be **mathematically impossible** with type-level tokens only. Packing (Approach D) is a different species.
2. **Morphologically long types** (TE/SA) with thin stage-1 → need better merges / script-local BPE, not only more word seats.
3. **Domain shift:** dumping training-page inventory does **not** keep $X\le 1.2$ on unrelated corpus; stage-1 + domain-matched pretok matter.
4. Strict all-lang $\tau$ can **shrink** fairness score vs unconstrained global savings.

---

## 11. Implementation sketch

Current path in `train_optimized_b.py`:

```python
bpe = stage1_bpe(...)
scores = word_savings_scores(...)   # global only
stage2_word_fill(bpe, scores, vocab_cap)
```

Target path:

```python
bpe = stage1_bpe(...)               # fair merge weights
stage2_minimax_fill(
    bpe, words_by,
    vocab_cap=V,
    tau=1.2,
    floor_frac=0.65,
    recompute_every=50,
)
# outer:
# for bpe_size in GRID: train; keep feasible; pick best score
```

Suggested new helpers (same file or `fill_strategies.py`):

| Function | Role |
|----------|------|
| `residual_tokens_needed(T, D, tau)` | $\Delta_L^{\star}$ |
| `stage2_floors_fill(...)` | P2 |
| `stage2_minimax_fill(...)` | P1 |
| `search_bpe_size(...)` | P3 |
| `adapt_alphas(X, alpha, tau, eta)` | P4 |

Meta dict should report: `fill_mode`, `floors`, `X_after_stage1`, `X_final`, `feasible`, `bpe_size_chosen`.

---

## 12. Acceptance tests

Run on at least three regimes and assert:

| Dataset | Expectation |
|---------|-------------|
| **wiki_api** (union ≈ 7.7k < 10k) | All $X=1$ (or ≤1.05); score ∞ or large; mode INVENTORY feasible |
| **full_page_md** (union ≈ 12.6k) | Prefer feasible $\max X \le 1.2$; at least reduce SA below 1.2 vs global savings baseline |
| **Synthetic stress** (inflate EN uniques) | Either still pass τ or return `feasible=false` with per-lang $\Delta^\star$ — no silent hard_ok only on EN |

Metrics to log every run:

- $X_L$, $T_L$, $D_L$
- `hard_ok_en` = $X_{\mathrm{EN}}\le 1.2$ and $V\le 10000$
- `hard_ok_all` = $\max_L X_L \le 1.2$ and $V\le 10000$  ← **new gate for this design**
- `score_words`
- stage-1 vs stage-2 slot split

---

## 13. Oral one-liner

> Optimized B today is hybrid BPE + **global savings** word fill — good enough for English hard_ok, weak for smallest languages.  
> To make it **dataset-resilient under $X_L\le 1.2$ for all langs**, treat stage-2 as a **deficit-to-τ solver** (worst lang first + floors), **search merge vs word budget**, and **fail explicitly** when the union cannot fit under the cap.

---

## Quick reference (short points)

- Problem: fill optimizes total savings, not $X_L\le 1.2$.
- Symptom: EN under 1.2, SA slightly over on full_page.
- Fix #1: worst-lang-first / deficit fill.
- Fix #2: per-lang word floors.
- Fix #3: grid-search `bpe_size`; keep only all-$X\le\tau$.
- Fix #4: adaptive $\alpha_L$ after fails.
- Fix #5: multi-lang merge weights (not EN-only).
- Mode-switch: inventory when union fits; minimax when tight; heavy BPE when infeasible for floors.
- Limit: if unique ≫ 10k for one massive language, type-level B may lose — packing is a different method.
- Gate to report: **`hard_ok_all`** vs assignment’s EN-only hard_ok.

---

## Changelog pointer

| Date | Note |
|------|------|
| 2026-07 | Doc written from full_page board: B opt $X\approx(1.12, 1.12, 1.16, 1.23)$, score ≈ 9.1k, hard_ok EN yes / all no. Implementation of minimax fill TBD. |
