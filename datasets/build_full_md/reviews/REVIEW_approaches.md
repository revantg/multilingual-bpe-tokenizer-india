# RED PILL review — full_md approach retrain (A–E + champion)

**Date:** 2026-07-11  
**Corpus:** `data/full_md/india_{en,hi,te,sa}.md` (union unique **7560**, words **EN 10307 / HI 8111 / TE 2307 / SA 1541**)  
**Artifacts:** `full_rerun/results/{A,B,C,D,E,champion}.json`, `RESULTS_TABLE.md`, `run_all.py`  
**Prior:** conversion review = **YES retrain**; body-without-refs is fair, not a silent gut to keep unique <10k.

**Reviewer stance:** adversarial, metric-skeptical, oral-exam ready.

---

## 0. Executive verdict (one paragraph)

The full_md retrain **replicates the same loophole map**, with thinner EN than `data/raw` (10307 vs 12328) and the same structural win conditions. Under **words** denom, whole-word inventory (C, B-with-fill, E primary) still hits **exact $X_i=1 \Rightarrow$ score = ∞** because union unique 7560 fits under 10k. Under the same denom, D + champion still **short-circuit on exact page string** and return planned equal-rate spans $T^*=(1117,879,250,167)$ → score **~1.41×10⁸** (not BPE). Under **fixed-5000**, C/B-fill/E-primary all **fail hard** with $X_{EN}=10307/5000=2.0614 > 1.2$; only packing (D/champion) posts `hard_ok_fixed=True` among high scorers — and that is still the packing oracle, not a tokenizer. **Honest pure word-interior BPE (A)** is the only real subword system on the board: score_words **~2384**, hard_ok_words **True**, hard_ok_fixed **False**. Table ranking this time places ∞ above packing (correct math vs old SWARM sin), but B’s *primary* is still whole-word fill marketed as “fair BPE.” **Oral-safe answer remains: metric is trivial once types fit; pure BPE ~2.4k; packing is a page-identity score oracle.**

---

## 1. Corpus / STATS (verified)

From `full_md/STATS.json` + harness re-count (`clean_text` + `WORD_RE`):

| Lang | total_words $D_i$ | unique | chars_cleaned |
|------|-------------------:|-------:|--------------:|
| EN | **10307** | 3122 | 64856 |
| HI | **8111** | 2276 | 43527 |
| TE | **2307** | 1453 | 18301 |
| SA | **1541** | 904 | 12495 |
| **UNION unique** | | **7560** | |

- `union_gt_10000 = false` → whole-word inventory still has ~2.4k headroom under V≤10k.  
- Conversion review already closed: this is **article body**, not ref warehouse; not identity-corrupted solely to keep unique <10k.

Hard gates (as implemented in `run_all.py` / harness):

- `hard_ok_words`: $X_{EN} ≤ 1.2$ under words denom **and** vocab ≤ 10000  
- `hard_ok_fixed`: $X_{EN} ≤ 1.2$ under fixed $D=5000$ **and** vocab ≤ 10000  
- score = `1000 / (maxX − minX)` ; gap < 1e-15 → **inf**

---

## 2. T confirmation (reload / encode path checks)

| Approach | Claimed T | Verification | Honest read of encode |
|----------|-----------|--------------|------------------------|
| **A pure (dp)** | EN 12006 / HI 9057 / TE 3523 / SA 2367 | Arithmetic reload of X/score matches JSON exactly | Word-interior BPE; merge or DP over merges; **no** whole-word dump in this retrain’s primary |
| **B primary (whole-word fill)** | **= D_words** | Spot: identity map | Merges+temperature *then* fill residual unique words → same T as C |
| **B merges_only** | EN 17909 / … | X_EN≈1.74 fails hard | Real(ish) fair BPE, incomplete close |
| **C words** | **= D_words** | Reloaded encode: T≡word count; 100% unique/occ coverage; mode=`all_unique_words_fit` | True whole-word; char fallback only for OOV |
| **D pack** | **1117 / 879 / 250 / 167** = `T_design` | `encode_matches_planned=true`; rebuild `build_slice` → same lens; score reload **140699698.22** | Identity short-circuit `if text == corpora[lang]` |
| **E primary** | **= D_words** | Same as C | All unique words + chars |
| **E compressed** | EN 8599 / HI 6859 / TE 2062 / SA 1341 | X reload matches; multiword phrases=2252 | Realer DP multiword cover (no page short-circuit in notes path) |
| **Champion** | **same as D** | `encode_matches=true`; vocab breakdown spans 2413 + words 7407 + chars 180 = 10000 | **D packing for score path**; residual words are unused on full-page encode |

Spot-checks run for this review:

| Probe | Result |
|-------|--------|
| C: `encode(page)` length | $T_i = D_i$ all langs |
| C: novel “quick brown fox…” | Many chars OOV (`q,u,i,c,k…`); not a fluent subword model |
| D: exact page | $T = T^*$ |
| D: page + trailing space | Identity breaks → falls back to `extract_words` → $T=D_i$ |
| D: first EN span | ~9-word contiguous tablet (avg $10307/1117 ≈ 9.23$) — **document chunk**, not merge |
| A X recalc from T | score_words **2384.4505** matches JSON |

### Cal lout (task §4): under fixed-5000, does C fail $X_{EN}≤1.2$?

**YES.**

$$
X_{EN}^{fixed} = T_{EN}/5000 = 10307/5000 = \mathbf{2.0614} > 1.2
$$

`hard_ok_fixed = false` for C, and for **any** method that returns one token per word on these pages (B-fill, E-primary share the same T).  
Only packing (and champion’s packed page path) gets $X_{EN}^{fixed}=0.2234≤1.2$ among the high-score board entries — by returning far fewer than $D_{EN}$ tokens via multi-word slabs.

---

## 3. Softcoding / gaming still present?

| Pattern | Present on full_md retrain? | Where |
|---------|----------------------------|--------|
| **Page-identity short-circuit packing** | **YES** | `run_all.py` D + champion: `if text == _c[lang]: return planned[lang]` |
| **Whole-word ∞ under words denom** | **YES** | C primary; B `with_whole_word_fill` (**primary**); E primary |
| **Rank packing above ∞** | **No** (improved vs old SWARM) | `RESULTS_TABLE` ranks B/C/E inf at ranks 1–3, D/champ ~1.41e8 after |
| **Primary = gamed variant** | **YES for B/E** | B notes include merges-only (finite) but primary = whole-word fill; E primary = inventory, compressed relegated to notes |
| **Residual vocab cosplay** | **YES for champion** | 7407 leftover words + chars never touch `T` on scored pages |
| **Denom shopping** | **Visible** | D maximizes score under **words** equal-X; then *also* looks strong under fixed only because $T≪D$; inventory wins words-∞ and dies on fixed hard |

**Softcoding verdict:** D/champion = engineered page packing. B/E primaries soft-code the infinite word-identity answer while retaining “BPE/hybrid” labels. A is clean of both cheats in this retrain. C is unapologetic inventory (honest about method, metric-gaming by definition of the score).

---

## 4. Approach ratings (honesty + score)

**Honesty 1–10** = defend as a *tokenizer* in oral without redefining the word.  
**Score-hunting 1–10** = optimize harness loophole over general segmentation.

| Approach | Honesty | Score-hunt | score_words | score_fixed | hard_ok_w | hard_ok_f | What it is on full_md |
|----------|--------:|-----------:|------------:|------------:|:---------:|:---------:|------------------------|
| **A pure** (dp primary) | **8** | **3** | **2384.45** | 518.73 | **True** | False | Real multilingual word-interior BPE, V=10k, no closure |
| **B merges_only** | **7** | **4** | 3512.34 | 331.06 | **False** (X_EN≈1.74) | False | Fair/temp+minimax BPE incomplete |
| **B primary (fill)** | **3** | **8** | **inf** | 570.39 | True | False | Inventory after “fair” merges |
| **C words** | **9** | **7** | **inf** | 570.39 | True | **False** | Honest maximizer under words; fails fixed hard |
| **D packing** | **5–6** | **10** | **~1.407e8** | **5263.16** | True | **True** | Equal-X span oracle + exact-page encode |
| **E primary** | **4** | **7** | **inf** | 570.39 | True | False | Same as C under words |
| **E compressed** | **7** | **6** | **16802.74** | 688.90 | True | False | Multiword DP (best non-pack finite compression here) |
| **Champion** | **3** | **10** | **~1.407e8** | **5263.16** | True | True | D score path + residual word fill cosplay |

\*D honesty ≈5 if sold as “tokenizer”, ≈6–7 if labeled “page fairness oracle” (code/notes do admit packing).

### Per-approach notes (tight)

**A — pure BPE**  
- Merges=9801, encode **dp** slightly beats merge (2384 vs 2319).  
- Gap_words ≈ 0.42; TE/SA fertility still worst ($X≈1.53$).  
- Fixed: $X_{EN}=2.40$ — hard fail (subword still returns >6000 EN tokens).  
- **This retrain’s A is cleaner than old shipped A** (old HQ was closure ∞; pure was buried at ~2247 on raw).

**B — fair BPE**  
- Merges-only: scientifically interesting, **fails** $X_{EN}≤1.2$.  
- With fill: vocab 8274, $T=D$, score inf — **functionally C**.  
- Selecting fill as primary is the softcoding problem.

**C — whole words**  
- Union 7560 fits; vocab 7752; coverage 100%.  
- Under words: **mathematically maximal (∞)**.  
- **Under fixed-5000: hard fail** (callout confirmed).  
- Not BPE; still the correct solution if metric is word-fertility fairness on this closed corpus.

**D — packing**  
- $T^*$ search intentionally maximizes words-score under near-equal $X≈0.10837$.  
- Vocab_spans only **2413** < 10k.  
- Collapse under any page mutation (trailing space verified).  
- Fixed score 5263 looks “best numeric” only because other methods either fail hard or keep large T.

**E — hybrid**  
- Primary = C.  
- Compressed ref score_words ~16.8k (old raw compressed ~183k was tighter equalization at $X≈0.5$); still the best **non-identity, non-pack** finite compress attempt recomputed here.

**Champion**  
- Same T/score as D. Extra words do not change page T. Same oral death as D.

---

## 5. Conversion / unique-under-10k integrity

Already reviewed in `REVIEW_conversion.md`:

- Union 7560 is from **article body** strip (refs/nav/tables out), not a stitched fake that deletes unique words to game fit.  
- Generous main-content strip still shows ~14k union (does **not** fit).  
- Retrain does **not** depend on corrupt identity!conversion — but results **do** enjoy the fact that body union <10k, so ∞ is still free.

Do **not** claim “full Downloads unique inventory covered under 10k.” Claim “cleaned India article-body prose, union 7560.”

---

## 6. Rankings on full_md (verified from JSON)

### By score_words (mathematical metric)

| Rank | Name | vocab | score_words | hard_ok_w | X_words sketch |
|-----:|------|------:|------------:|:---------:|----------------|
| 1–3 (tie ∞) | **B fill / C / E primary** | ~7.7–8.3k | **inf** | True | all $X=1$ |
| 4–5 (tie) | **D / champion** | 2413 / 10k | **1.407e8** | True | $X≈0.10837$ |
| 6 | **A pure** | 10k | **2384** | True | 1.16 / 1.12 / 1.53 / 1.54 |

*If ranking forbids whole-word dumps & packing:* **A wins among survivors.**  
*If true multiword compression allowed but no page ID:* **E compressed ~16.8k** sits between A and packing.

### By score_fixed

| Rank | Name | score_fixed | hard_ok_f | $X_{EN}^{fixed}$ |
|-----:|------|------------:|:---------:|-------------------:|
| 1–2 | **D / champion** | **5263.16** | **True** | 0.2234 |
| 3–5 | B fill / C / E primary | 570.39 | **False** | **2.0614** |
| 6 | A pure | 518.73 | False | 2.4012 |

**Takeaway:** fixed denom + hard $X_{EN}≤1.2$ **outlaws honest inventories** on a page longer than 6000 tokens-worth of words ($D_{EN}=10307$). That pushes the board toward packing/mega-spans if you chase the numeric hard gate under fixed — which is a **metric design bug**, not a tokenization breakthrough.

---

## 7. Oral-safe recommendation

**Say this (survive TA):**

> On cleaned full article bodies, unique words across EN/HI/TE/SA total 7560 < 10k, so a whole-word vocabulary gives $T_i=D_i$, $X_i=1$, and **infinite** fairness score under tokens-per-word — the metric is trivial on this closed set. Under fixed $D_i=5000$, that same inventory has $X_{EN}≈2.06$ and **fails** the stated hard cap $X_{EN}≤1.2$. Pure word-interior BPE (no whole-word closure) reaches **~2.4k** fairness under words with $X_{EN}≈1.16≤1.2$; TE/SA remain more fragmented. Equal-X page packing scores ~**1.4×10⁸** by returning planned multi-word slabs only when `text` equals the training page; one trailing space collapses it. We treat packing as an oracle upper bound, not a BPE submission.

**Do not lead with** champion 1.41e8 or “B fair infinite” without the loophole labels.  
**Chalkboard max only if** explicitly labeled packing oracle / words-identity maximizer.

### What to ship if forced to pick one artifacts

| Goal | Ship |
|------|------|
| Honest “we built BPE” | **A pure** (score ~2384) |
| Honest “metric maximizer under words” | **C** (∞) + fixed-hard fail called out |
| Metric loophole upper bound | **D** as oracle, not toolkit cut |
| Survives fixed hard without packing? | **Nothing real on this board** (needs fewer EN tokens than 6000 ⇒ multiword/mega; or redefine hard gate) |

---

## 8. Bottom line

1. **T check:** C/B-fill/E-primary $T=D$; D/champ $T=T^*$ planned; A subword $T>D$. All consistent with claimed mechanics.  
2. **Gaming still present:** packing short-circuit + whole-word ∞ primary on B/E.  
3. **Honesty stars:** C ~9, A ~8, E-compressed ~7, D ~5–6, B-fill/E-primary/champ ~3.  
4. **Fixed-5000 C fail:** **confirmed** $X_{EN}≈2.0614 > 1.2$.  
5. **Conversion identity:** not fraudulently thinned for 10k; body-without-refs still enables free ∞.  
6. **Oral:** metric trivial + pure BPE ~2.4k + packing = page oracle.

*— end red pill (full_md approaches) —*
