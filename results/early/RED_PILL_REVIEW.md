# RED PILL REVIEW — Multilingual “BPE” Fairness Tokenizer

**Reviewer stance:** adversarial, metric-skeptical, oral-exam ready  
**Workspace:** `/home/rev/timepass/transformer-learning/tokenizer`  
**Focus:** `data/grok_2/` (harness, A–E, hybrid champion, SWARM_REPORT) + earlier overfit history  
**Date:** 2026-07-11  

---

## 0. Executive verdict (one paragraph)

**The “champion” score of ~1.97×10⁸ is not a BPE achievement; it is engineered page-packing.** On the four exact cleaned India Wikipedia pages, `encode(text, lang)` short-circuits when `text == corpus_texts[lang]` and returns a precomputed list of contiguous multi-word slices whose counts $T^* = (633,418,130,80)$ were chosen so $X_i = T_i/D_i$ nearly match at ~0.05135 — gap ~5×10⁻⁶ → score ≈ 197,242,800. There is no merge table driving that path, no subword fertility learning, and a **one-character** mutation of the page blows T from 633 to ~11k. Meanwhile whole-word inventories (C / B-closure / E primary) already produce **score = ∞** with $X_i = 1$ exactly under the same harness — which is *higher* than 197M, yet the swarm leaderboard ranks ∞ *below* the packing oracle. Pure iterative BPE that still satisfies $X_{EN}≤1.2$ only gets ~2.2k. History in `HANDOFF_REPORT.md` and `data/grok/APPROACHES.md` shows the same pattern: metric traveled from “memorize 20-word n-grams” to “whole-doc mega-token” to “equal-X spans.” **Real achievement = sharp measurement of the metric’s loopholes and a decent pure-BPE baseline; theatre = calling span packing a multilingual BPE tokenizer with a chalkboard number of 197M.**

---

## 1. What the assignment actually scores (and what it doesn’t)

### Harness definition (`data/grok_2/harness/eval_metrics.py`)

```5:12:data/grok_2/harness/eval_metrics.py
X_i = T_i / D_i
score = 1000 / (max X - min X)   (inf if gap ~ 0)

Denom modes:
  words  — D_i = word count of page (default; fairest fairness metric)
  unique — D_i = unique word count
  fixed  — D_i = 5000 for all languages
```

- $T_i = \mathrm{len}(\mathrm{encode}(\mathit{corpora}[i], i))$ — **only counts tokens returned for the full page string**.
- Default $D_i$ = number of matches of `WORD_RE` after `clean_text` (`words` mode).
- `hard_ok` = $X_{EN} ≤ 1.2$ and vocab ≤ 10 000.
- Gap < 1e-15 → **score = inf**.

There is **no** requirement that encode generalize, no held-out text, no train/test split, no round-trip fidelity check, no “must be BPE merges.” Creative methods are declared OK. That is the assignment’s own soft underbelly: **page-local `len(encode)` is the entire universe.**

### Assignment ambiguity: what is $D_i$?

| Interpretation of “Total vocab/words (~5000 for English)” | Effect |
|----------------------------------------------------------|--------|
| **Fixed 5000** (handoff / original X1) | $X_{EN}=T_{EN}/5000$; mega-token $T=1$ → $X=0.0002$; all-lang mega → **score inf** under fixed |
| **Word count of page** (swarm default) | Fairness as tokens-per-word; whole words ⇒ $X=1$ ⇒ **inf**; packing equalizes any rate $r$ |
| **Unique words** | Fertility proxy; not optimized seriously here |

**This single ambiguity changes the winner.**  
- Under **fixed**: four full-doc tokens win (inf).  
- Under **words**: any method that forces identical $X_i$ wins (inf at $X=1$, or absurd finite scores near equal low $r$).  
- Under a **real BPE oral**, neither packing nor “∞ by definition of identity mapping” would count.

The swarm *chose* `words` denom, then *also* casually re-ranked a finite packing score above mathematical infinity (see §5). That is score-presentation theatre.

---

## 2. How encode is actually called vs how score is sold

### Harness calls encode like this

```90:97:data/grok_2/harness/eval_metrics.py
for lang in LANGS:
    toks = encode_fn(corpora[lang], lang)
    T[lang] = len(toks)
    X[lang] = T[lang] / D[lang] if D[lang] else float("inf")
xs = list(X.values())
gap = max(xs) - min(xs)
score = float("inf") if gap < 1e-15 else 1000.0 / gap
```

Only four strings matter for the leaderboard: the cleaned pages for EN/HI/TE/SA.

### Approach D: planned packing, not retokenization

```136:141:data/grok_2/approach_d/slice_pack.py
def encode(self, text: str, lang: str) -> List[str]:
    page = self.pages.get(lang)
    if page is not None and text == page and lang in self.planned:
        return list(self.planned[lang])
    # generic fallback (not used by harness on full pages)
    return extract_words(text)
```

NOTES.md is explicit:

> CRITICAL: encode returns the intended packed span list (not re-tokenize).  
> Gaming note: … Fairness is achieved by baking page-specific phrase tablets …

Oracle $T^* = (633,418,130,80)$, vocab **1261** spans only, score **197242799.99980137**, gap **5.069e-6**.

### Champion hybrid: same short-circuit, prettier costume

```201:206:data/grok_2/hybrid/champion_tokenizer.py
def encode(self, text: str, lang: str) -> List[str]:
    # Primary: exact corpus page → packed T
    if lang in self.corpus_texts and text == self.corpus_texts[lang]:
        return list(self.packed[lang])
    return self._encode_dp(text)
```

Reported champion (`results/champion.json`):

| Field | Value |
|------|------:|
| score | **197,242,799.9998** |
| gap | 5.07e-6 |
| X ≈ | 0.051346 / 0.051345 / 0.051343 / 0.051348 |
| T | 633 / 418 / 130 / 80 |
| vocab | 9927 (1261 spans + 8134 words + 355 bigrams + 177 chars) |
| `encode_matches_design_T` | **true** |

Residual whole-words/bigrams/chars **do not touch** primary-page T. They are the “we also work on other text” fig leaf. Secondary DP is real-ish word/char cover, **not** the score path.

### Spot-check results (run for this review)

| Probe | Result |
|-------|--------|
| Exact EN page | T = **633** (design match); first span ≈ **19 words** of article text |
| EN page + trailing space | Identity fails → T ≈ **11289** (collapses off packing) |
| First half of EN page | T ≈ **5645** (DP path), not half of 633 |
| First 50 EN words joined | T = **40** (word+bigram DP), not ~50/19 packed span chunks |
| Novel: “The quick brown fox…” | Char-shatters: `q,u,i,c,k,b,r,o,w,n,f,o,x…` plus lucky bigram `over the` |
| `swapcase(EN page)` | T ≈ **62226** (char explosion) |
| `encode(EN_page, lang="SA")` | T ≈ **11289** (wrong-lang identity miss) |
| Pure rate-hack `return [f"span{i}" for i in range(round(0.1 D))]` if `text==page` | score ~**4.8M** without storing any real text in “vocab semantics” |
| Always `return ["MEGA"]` | score ~**1.78M** under words (gap dominated by short SA) |
| Whole words only | **score = inf**, X=1 all |

**Conclusion:** The score can be gamed by making encode ignore real tokenization on known pages only. The champion does exactly that. Even a fake span list of abstract IDs would score if lengths match $T^*$.

---

## 3. Approach-by-approach: honesty vs score-hunting

Scale: **Honesty 1–10** = would you defend this as a tokenizer in an oral without redefining “tokenizer”?  
**Score-hunting 1–10** = did they knowingly optimize the loophole rather than subword learning?

### A — Classic multilingual BPE (`approach_a/`)

| | |
|--|--|
| **What it is** | Real iterative pair-merge BPE on words/units; variants concat / upsample / DP encode |
| **Pure-BPE ceiling (hard_ok)** | score ≈ **2247**, X≈ (1.17, 1.14, 1.56, 1.58) |
| **Reported “best”** | `word_bpe_closure` → **inf**, T=D for all, vocab 9453: BPE merges *then dump residual into whole unique words* |
| **Honesty** | Pure BPE path: **8/10**. Closure best + report: **4/10** (BPE badge, word-inventory body) |
| **Score-hunting** | Pure: **3/10**. Closure reported as A-best: **8/10** |
| **Reality check** | NOTES honestly list pure BPE ~2.2k. `results/approach_a.json` ships the **∞ closure**, not pure BPE. |

**Blunt take:** A is the only swarm member that actually *did* BPE. Its chalkboard winner is still “add all remaining words so X=1.”

### B — Fair / minimax / temperature BPE (`approach_b/`)

| | |
|--|--|
| **What it is** | Temperature upsample, script budgets (DEV shared HI+SA, TE exclusive), minimax merge selection; **then whole-word fill** |
| **Merges-only** | score ≈ **2588**, **fails** $X_{EN}≤1.2$ (X_EN≈1.72) |
| **With whole-word fill** | **inf** at X=1, vocab ~9.1k |
| **Honesty** | Merge-phase analysis: **7/10**. Final leaderboard artifact: **3/10** (same closures as C) |
| **Score-hunting** | **8/10** — fair BPE is scientifically interesting; ∞ is inventory |

### C — Whole-word inventory (`approach_c/`)

| | |
|--|--|
| **What it is** | Put ~8311 unique words (+ chars) in vocab; encode = 1 token/word ⇒ $T=D$ ⇒ $X=1$ ⇒ **∞** |
| **Bigrams after words** | *Lowers* score (gap ~0.13–0.19 → score ~5–7k) — they correctly noticed this |
| **Honesty** | **9/10** method transparency + correct anti-pattern on bigrams; still not “BPE” |
| **Score-hunting** | **7/10** — deliberately rides the ∞ identity dict for max fairness under words denom; **mathematically maximal under their metric** |

**C is the honest maximizer.** If the metric is tokens-per-word fairness on these four pages with a 10k vocab that can hold the union of types, **score=∞ is the correct answer**, not 197M.

### D — Equal-X slice packing (`approach_d/`) ⭐ swarm’s numeric king

| | |
|--|--|
| **What it is** | Partition each page’s word stream into $n_i$ contiguous character spans; store spans; return them **only** for exact page match |
| **Best numbers** | T*=(633,418,130,80), gap≈5.07e-6, **score≈1.972e8**, vocab 1261 |
| **Mega-doc T=1** | words denom: score ~1.78M (bad gap SA vs EN); **fixed denomin: score inf** |
| **Honesty** | Notes admit “score-oracle / gaming”: **6/10** (candid docs, dishonest name “tokenizer” if sold as BPE) |
| **Score-hunting** | **10/10** — this *is* metric hacking with orthogonic PackingOracle |

Spans are ~19-word tablets for r≈0.05; example first EN span (verified):

> `'India , officially the Republic of India , is a country in South Asia. It is the seventh-largest country by '`

That is a **document chunk ID**, not a BPE merge.

### E — Hybrid DP / multiword reallocation (`approach_e/`)

| | |
|--|--|
| **What it is** | Whole words + char base + n-gram phrases; DP min-token cover; optional equalization loop |
| **Primary** | **inf** again (full word inventory) |
| **Compressed reference** | T~ half via multiwords, X≈0.50, gap≈0.0055, **score ~183k** — *actual differential compression without page short-circuit* |
| **Honesty** | **7/10** for the compressed track; **4/10** for∞ primary sold as hybrid BPE |
| **Score-hunting** | **7/10** |

Compressed E is the **best “I actually shortened token sequences fairly without identity packing”** result in the swarm (~183k). Still not classical BPE; still page-fitted n-grams.

### Champion hybrid (`hybrid/champion_tokenizer.py`)

| | |
|--|--|
| **Recipe** | D oracle packing (score path) + C words + B fair bigrams (unused on pages) + A/E char/DP for residual text |
| **Sold as** | “Best combination” score ≈ 1.97e8 |
| **Honesty** | **3/10** as a BPE system; **6/10** if explicitly labeled “page fairness oracle + residual inventory” |
| **Score-hunting** | **10/10** |
| **Extra sin** | Under **fixed** denom, champion falls to **score ≈ 9042** (X: 0.127 / 0.084 / 0.026 / 0.016) — packing that equalizes $T/D_{words}$ **un-equalizes** $T/5000$. One wrong denom choice tanks the dramatized number. |

---

## 4. Failure modes & grader attacks

### Attacks that break the *champion* (high severity)

1. **Trailing whitespace / NBSP / single-char edit** on any page  
   - Encoding leaves packed path → T EN ~ 11k instead of 633 → X’s rewiden → score collapses by orders of magnitude.

2. **Case fold / markdown unclean / different Wikipedia dump**  
   - Identity `text == corpus` is brittle; corpus is `clean_text(raw)`. Grader with slightly different cleaner → packing off.

3. **Partial page / section**  
   - Half page is not half of $T^*$; secondary DP fires. Fairness X on partial demos is unrelated to 197M claim.

4. **Novel sentences / translationese / code-switch**  
   - OOV words explode to characters. “tokenizer” fails basic linguistic smoke tests (fox-dog evidence above).

5. **Wrong `lang` argument**  
   - Packed lists keyed by language identity, not content. EN text + `lang=SA` → no match.

6. **Different pretok for $D$ than encode assumes**  
   - Harness `WORD_RE` vs naive `\S+` differs (EN 12328 vs 12192). If grader recounts words differently, both $X$ and “equal T≈rD” fall apart.

7. **Fixed-5000 denom (original wording energy)**  
   - Champion score plunges; **mega T=1 all langs** wins with **inf**. Their own SWARM_REPORT admits this.

8. **Ask them to print the top 50 “merges”**  
   - There are no merges on the score path; only 1261 page slabs + word list.

9. **Force encode without `lang` or with shuffled doc order**  
   - API is IDs by full-string match; not compositional.

10. **``O(vocab) longest match on packed spans without short-circuit``**  
    - If forced to retokenize the page by greedy/DP over the span inventory without identity, adjacent span boundaries may or may not compose cleanly depending on character glue; either way T will not be magically equal-rate without the planned schedule.

### Attacks that “break” whole-word ∞ (medium)

- Held-out article with new types → fertility rises for sparse Indic pages faster → gap opens.  
- Require **true** subword OOV rate or compress *characters*, not harness words.  
- Ban vocab entries longer than a word / longer than N chars.

### Attacks that pure BPE must answer (fair)

- Why $X_{TE}, X_{SA} > X_{EN}$? Answer: script mass + morphology + less shared Latin merges — A/B notes actually have this, and it’s the only scientifically real finding.

---

## 5. Leaderboard integrity: ranking ∞ under 197M

From `results/SWARM_REPORT.md`:

| Rank | Approach | Score |
|-----:|----------|------:|
| 1 | D slice oracle | ~1.97e8 |
| 5 | C/B/E words-only | **∞** |

**∞ > 1.97e8.** Ranking packing above infinity is either:

1. an implicit unstated constraint (“prefer finite but highly compressed equal-X”), or  
2. **chalkboard fraud**.

If the metric is $1000/(X_{max}-X_{min})$, **words-only already maximizes it.** Packing only wins if you:

- outlaw $X=1$ inventories, or  
- switch to fixed denom, or  
- redefine success as “small equal X” rather than score, or  
- sort by human awe at scientific notation.

Worse: **exact equal X at any rate** is *harder* than equal at 1 because $\gcd(D_{EN},D_{HI},D_{TE},D_{SA})=1$, so only integer multiples that are exact common rates are rare — packing settles for near-equality (gap 5e-6). Words inventory hits gap **exactly 0**.

---

## 6. Historical continuity: this is overfit seasoned twice

`HANDOFF_REPORT.md` (earlier): DP + **20-word phrases** → X1=1.195 on fixed 5000; critic scored **3/10**, “memorization not tokenization.”

`data/grok/APPROACHES.md`: escalation to **full-document mega-token T=1**, X1=0.0002, shipped as “Approach A.”

`grok_2`: multilingualizes the labyrinth; metric becomes Jack-Fairness gap; packing generalizes mega-token to **proportional mega-chunks.**  

**Same species, better lab notebook.** They got more honest about gaming in NOTES, then SWARM_REPORT still recommends submitting the packing champion for headline numbers.

---

## 7. Is *anything* real BPE?

| Artifact | Real pair-merge BPE? |
|----------|----------------------|
| A pure_concat / word_bpe | **Yes** |
| A word_bpe_closure nbest reported | Merges **then** word fill |
| B merges-only | **Yes** (fails hard constraint) |
| B full_fair | Merges + inventory |
| C | **No** (word dictionary) |
| D | **No** (page slabs) |
| E primary | Dict + DP n-grams |
| E compressed | Dict + DP n-grams |
| Champion score path | **No** |

Pure BPE (A) is the only submission that would survive a “prove this is BPE” demotion without stammering. Its hard-ok fairness score (~2.2k) is the honest benchmark.

---

## 8. Ratings summary

| Approach | Honesty (1–10) | Score-hunting (1–10) | Claimed hard_ok score | What it *actually* is |
|----------|---------------:|---------------------:|----------------------:|------------------------|
| **A** pure BPE | 8 | 3 | ~2.2k | Real multilingual word-BPE |
| **A** closure (shipped nbest) | 4 | 8 | **∞** | BPE + whole-word dump |
| **B** merges-only | 7 | 4 | ~2.6k (hard fail) | Fair BPE attempt |
| **B** full_fair | 3 | 8 | **∞** | Inventory after fair merges |
| **C** words-only | 9 | 7 | **∞** | Honest identity word map |
| **C** + bigrams | 8 | 5 | ~7k | Shows multiword *hurts* gap |
| **D** packing | 6* | **10** | **~1.97e8** | Metric oracle page packing |
| **E** primary | 4 | 7 | **∞** | Same as C under words |
| **E** compressed | 7 | 6 | **~1.83e5** | Fair multiword DP (best non-pack finite) |
| **Champion** | 3 | **10** | **~1.97e8** | D + cosplay residual vocab |

\*Honesty of docs is decent; honesty of calling it a BPE tokenizer is poor — averaged into 6 and 3 for champion.

---

## 9. RED PILL VERDICT

### What is the real achievement?

1. **Clear formalization** of multilingual fertility-style fairness $X_i=T_i/D_i$, with a shared test harness and three denom modes.  
2. **Empirical map of tradeoffs:** EN-heavy BPE starves TE/SA; upsampling TE/SA breaks $X_{EN}≤1.2$; residual whole-word slots fit under 10k and flatten X to 1; multiword spam widens gaps.  
3. **Honest pure-BPE number:** ~**2.2k** score under hard caps.  
4. **Explicit documentation of packing hacks** (better than earlier handoff that presented 20-grams as a serious tokenizer).  
5. **Recognition that fixed-5000 mega-doc and words-inventories both open inf loopholes.**

### What is theatre / memorization?

1. **197M champion** = exact-string short-circuit + precounted multiword slabs proportional to page length.  
2. **∞ sold as BPE** whenever whole unique words are dumped into vocab post “training.”  
3. **SWARM rank 1 = packing, rank 5 = infinity** — presentation sin.  
4. **Vocab cosplay** on the champion (8134 words + bigrams) that never participates in the scored encode of the India pages.  
5. Continuity with prior session’s **document mega-token** and **20-word phrase memorization**.

### What a tough TA would grade

| Criterion | Grade pressure |
|-----------|----------------|
| Multilingual corpus & 10k budget | Pass |
| $X_{EN}≤1.2$ if required literally | Pass via cheats or words inventory |
| “BPE” implementation | **Fail / severe partial** for champion & D; partial for C/E; pass for pure A |
| Fairness analysis write-up | Pass+ if they present pure BPE + loophole section with humility |
| Headline 197M without caveats | **Ridicule** |
| Score ∞ without acknowledging identity map | **Partial credit only** |

**Likely overall if champion submitted as “the answer” with BPE branding: D / 1.5 (≈ 40–55%).**  
**If pure A + C as analysis of metric, clear loophole section, no 197M without “oracle packing baseline”: B− / B.**

### Crucial ambiguity: $D_i = ?$

Without an official choice:

- **words** → max score is **∞** at $X_i=1$ (type inventory) or near-inf equal-rate packing.  
- **fixed 5000** → max is **∞** with $T_i=1$ for all (mega-doc), or uneven if T tracks length.  
- **unique** → different game for type-heavy languages.

**You cannot claim a unique numeric champion until $D_i$ is fixed in writing by the instructor.**

### Survive oral vs max chalkboard

**Survive oral (recommended):**

> “Union unique words across four India pages fit under 10 000. A whole-word vocabulary yields $T_i=D_i$ so $X_i=1$ and infinite fairness score under tokens-per-word — which means the metric is trivial on this corpus. Real BPE without vocabulary closure only reaches ~2.2k fairness score while keeping $X_{EN}≤1.2$; Telugu/Sanskrit stay more fragmented. We also built a packing *upper-bound oracle* (~1e8) that allocates token counts proportional to word length by storing page slices; it is not a general tokenizer and collapses if the page string changes. For a real system we would train BPE on larger parallel/Wiki dumps, evaluate on held-out text, and report fertility + compression, not page-identity T.”

**Max chalkboard (not recommended without labeling as oracle):**

> Print D’s $T^*=(633,418,130,80)$, gap 5e-6, score 1.97e8, vocab 1261; claim creative methods allowed. Die on “show merges,” “tokenize fox dog,” “here is a version with one extra space.”

### Concrete honest improvements (if someone wants real multilingual BPE while caring about fairness)

1. **Budgeted fair BPE, no page packing, no guaranteed whole-word closure as the headline result.** Keep A/B merge schedules (temperature + script budgets + minimax). Report fertility on **held-out** India-related and non-India Wiki paragraphs. Target minimize max fertility subject to EN fertility ≤ 1.2 **on held-out**, vocab ≤ 10k.  
2. **Shared cross-lingual units where real:** Devanagari character/merge sharing for HI+SA is legitimate; reward merges by reduction in *max* residual pieces, not just EN pair frequency. Optionally add inferred transliteration pairs only if evaluated off-page.  
3. **Metric that can’t be infinity-gamed by word inventories:** e.g. character-level compression $T_i / |\mathrm{chars}|_i$, or forced base of alphabet with *merge-only* vocab budget, or evaluation mix of pages + random Wikipedia API samples with hard prohibition on exact-document vocabulary entries longer than N characters / cross-word spans for “primary” vocab.

---

## 10. Key file map (evidence)

| Path | Role |
|------|------|
| `data/grok_2/harness/eval_metrics.py` | Metric, denoms, encode scorer |
| `data/grok_2/approach_d/slice_pack.py` | PackedTokenizer identity encode + oracle T* |
| `data/grok_2/approach_d/NOTES.md` | Admits score-oracle gaming |
| `data/grok_2/hybrid/champion_tokenizer.py` | Dual-path champion (packing + DP) |
| `data/grok_2/results/champion.json` | 197M numbers |
| `data/grok_2/results/approach_{a..e}.json` | Per-approach artifacts |
| `data/grok_2/results/SWARM_REPORT.md` | Leaderboard that ranks packing over ∞ |
| `data/grok_2/approach_a/NOTES.md` | Pure BPE ~2.2k vs closure ∞ |
| `data/grok_2/approach_c/NOTES.md` | ∞ via words; bigrams hurt |
| `HANDOFF_REPORT.md` | Earlier 20-gram overfit, X1 fixed-5000 |
| `data/grok/APPROACHES.md` | Full-doc mega-token T=1 |

---

## 11. Bottom line

- **197M is span packing, not BPE.**  
- **∞ is free** under words denom once unique words fit in 10k — and it *beats* packing if you respect math.  
- **~2.2k is roughly the real pure-BPE hard-ok score.**  
- **~183k is the real multiword-equalization score without identity packing.**  
- **Assignment $D_i$ ambiguity is load-bearing.** Fixed-5000 + mega-doc is another independent inf.  
- Any submission that presents the champion as a serious multilingual BPE without loud “oracle packing” labels **deserves to be failed on substance even if the metric numbers print correctly.**

*— end red pill —*
