# RED-PILL VALIDATION: prove/refute that the `2/2/3/2` tokenizer passes the assignment

You are a STRICT, adversarial validator. Your job is NOT to confirm the happy
path — it is to try to BREAK this tokenizer against the real grading gate, and
report honestly whether it would score well or get another zero. Assume nothing;
verify everything with code you run yourself. Prototype in /tmp scratch only.
Do NOT modify repo files except your final report.

## Why this matters
The previous submission scored **0** — not on the metric, but on the FAITHFUL
ROUNDTRIP GATE, using a held-out sample the student never tested. We must not
repeat that. Be paranoid about roundtrip and UNK on UNSEEN text.

## The authoritative grading contract (pulled live from Axiom, verbatim)
- **assignment score** = `1000 / (X4 - X1)` where X1..X4 are the four per-language
  fertilities sorted; X4 = max, X1 = min.
- **fertility(L)** = `total_tokens(L) / faithful_units(L)`.
- **faithful_unit** (unitPolicy, exact): "one contiguous Unicode letter/mark/number
  run OR one visible non-space punctuation/symbol character."
  Regex: `regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")`.
- Each of X1..X4 must be **<= 1.2**.
- **faithfulnessRule (THE GATE that caused the zero):** "decode(encode(text)) must
  preserve visible non-whitespace characters." Whitespace may differ; visible
  non-whitespace chars must be IDENTICAL.
- **no UNK tokens** may appear when encoding the corpus (and ideally any text).
- **English penalty:** production score = `raw_score / exp(max(0, en_ratio/1.2 - 1))`.
  (If en_ratio <= 1.2, penalty factor = 1.0.)
- vocab size must be exactly **10000**.
- The grader RE-RUNS the tokenizer themselves and confirms the numbers. Self-reported
  numbers are not trusted.
- The zero's exact failing sample: `"India's population is 1,428,627,663."` decoded to
  `"India ' s population is 1 <UNK> 4 28 <UNK> 6 27 <UNK> 6 63 <UNK>"` on the OLD tokenizer.
  Your first job is to confirm the NEW one does not do this (early check: it passes).

## What is being validated
The candidate is our byte-level BPE tokenizer trained with weights
**EN:2, HI:2, TE:3, MAI:2** on the instructor's CLEAN corpus (en/hi/te/mai).
Self-reported: X_EN 0.567, X_HI 0.577, X_TE 0.601, X_MAI 0.667, spread 0.1005,
adj score 9953, all <1.2, roundtrip True, 0 UNK, vocab 10000.

- Trainer + tokenizer class: `/home/rev/timepass/tokenizer-assignment/sweep_weighted.py`
  (class `Tokenizer`, byte-level base via `bytes_to_unicode`/`token_to_byte_symbols`,
  cross-word line-level BPE). Loader mode `CORPUS=instructor_mai`.
- Clean corpus: `/home/rev/timepass/tokenizer-assignment/datasets/instructor/{en,hi,te,mai}.txt`
  (verified <=1.2% percent-encoded hex; these are the instructor's own faithful files).
- A pre-trained artifact of the exact `2/2/3/2` tokenizer is at `/tmp/tok_2232.json`
  (keys: `vocab` {token->id}, `merges`). You may retrain from scratch to be sure —
  retraining takes ~2-3 min; do it to confirm reproducibility.
- Instructor reference bundle: `/home/rev/Downloads/2aa5dbf6-9413-4ec2-a27d-780833fce1a5/`.

## Adversarial tests you MUST run (break it if you can)
1. **Reproduce** the tokenizer from scratch (CORPUS=instructor_mai, weights 2/2/3/2)
   and confirm vocab=10000, the four fertilities, spread, and the score to 3-4 sig figs.
   Recompute fertility with YOUR OWN independent implementation of the faithful-unit
   regex — do not trust the script's `count_total_vocab`. Report any discrepancy.
2. **Faithful roundtrip on the full corpus**: for each language, assert decoded visible
   non-whitespace chars == original visible non-whitespace chars. Report exact
   mismatch count and first offending sample if any.
3. **Faithful roundtrip on HELD-OUT / adversarial text** the tokenizer never trained on.
   Build a battery of at least ~40 samples covering:
   - The exact grader sample `"India's population is 1,428,627,663."` and variants with
     other large comma-grouped numbers, decimals, percentages, currency (₹, $).
   - Mixed-script lines (en+hi+te+mai in one line), emoji, rare CJK, ZWJ/ZWNJ,
     combining marks, Devanagari conjuncts, newlines/tabs, URLs, Markdown tables.
   - Pure punctuation/symbol runs, control-ish characters, non-BMP codepoints.
   For each: assert visible-char roundtrip AND count UNK. ANY UNK or visible-char change
   is a FAIL — report it loudly.
4. **UNK audit**: confirm `<unk>` id never appears when encoding the corpus. Then try to
   PRODUCE an UNK with adversarial bytes (raw invalid UTF-8, lone surrogates via bytes,
   every byte 0-255). A byte-level tokenizer should never emit UNK; prove it or find the hole.
5. **English penalty**: confirm en_ratio <= 1.2 so penalty factor = 1.0; compute the
   production (penalty-adjusted) score exactly.
6. **Vocab/merges integrity**: vocab size exactly 10000; merges apply deterministically;
   encode is stable (same input -> same ids twice).
7. **Sanity vs instructor**: note our fourth language is now Maithili (matches instructor),
   NOT Sanskrit (the zero recorded Sanskrit). Confirm this is the intended fix.

## Deliverable
Write `/home/rev/timepass/tokenizer-assignment/REDPILL_FINDINGS.md` with:
- PASS/FAIL verdict on EACH of the grading-contract gates (roundtrip, UNK, <=1.2,
  vocab=10000, score formula).
- The independently-recomputed fertilities and score (your numbers, not the script's).
- Every roundtrip/UNK failure found (or an explicit "none found after N adversarial samples").
- A blunt final judgment: would this get a zero again? If any risk remains, name it exactly
  and give the concrete fix.
Return a concise summary too.

## Hard operational limits
- **NEVER more than 4 worker processes at once; prefer 2-3.** The machine has hung before.
- Monitor thermals (`sensors | grep Tctl`, throttle ~95C) and memory (`free -m`, keep
  >500M free) around any training/parallel work; kill and back off if breached.
- One 10k training run ~2-3 min and ~2.5G RSS; do not run many in parallel.
