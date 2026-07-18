# FINAL RED-PILL VALIDATION — submitted solution vs the assignment

You are a STRICT, adversarial validator. Do NOT rubber-stamp. Your job is to try
to BREAK this submission against the assignment below, and report honestly what
passes and what fails. Prototype/verify in /tmp scratch only.

## The assignment (verbatim)

> Pick India's Wikipedia page in English, Hindi, Telugu, and one more language of
> your choice. Design a BPE tokenizer such that:
> - 10000 tokens (vocab) overall for all languages,
> - (Total English tokens)/(Total English faithful units) must be 1.2 or less = X1
> - Similarly X2, X3, X4 for Hindi, Telugu, another language
> - Sort X1..X4; score = 1000 / (X_max - X_min).
>
> Submitting:
> 1. A Widget that shows these ratios, token statistics and calculations and your
>    self score
> 2. Allows me to see and download your tokenizer (list of all tokens)
> 3. URL for this widget (Netlify or anywhere)
> 4. "We'll run your tokenizer ourselves to confirm your scores" — numbers must
>    be reproducible.

## What was submitted (verify each independently)

- Live widget: https://india-bpe-tokenizer.surge.sh/  (also widget.html, slides.html, notebook.html)
- Downloadable tokenizer: vocab.json (10000 tokens + 9740 merges + scores) and
  tokens.txt (all 10000 tokens, one per line) — both linked with download buttons.
- Hosted marimo notebook (static HTML with baked outputs + code): notebook.html
- Repo: /home/rev/timepass/tokenizer-assignment/solution

## Claimed numbers (4 languages: EN/HI/TE/MAI)

| Lang | X (fertility) | tokens | units |
|---|---|---|---|
| EN | 0.5669 | 105649 | 186367 |
| HI | 0.5774 | 51019 | 88359 |
| TE | 0.6012 | 21817 | 36292 |
| MAI | 0.6674 | 3876 | 5808 |

spread = 0.1005, **score = 9953**, vocab = 10000, merges = 9740, 0 UNK, lossless.

## What to check (be adversarial)

1. **Download the LIVE vocab.json + tokens.txt from surge.** Rebuild encode/decode
   from scratch (do NOT import the repo's code). Re-encode each corpus file
   (solution/corpus/{en,hi,te,mai}.txt) and independently recompute X per language,
   the spread, and 1000/spread. Do they match the claims exactly?
2. **Vocab == 10000?** tokens.txt has exactly 10000 lines? merges == 9740?
3. **Every X ≤ 1.2?** Is the score really 1000/(max-min)?
4. **Zero UNK / lossless?** decode(encode(text)) preserves text on every corpus
   AND on the probe "India's population is 1,428,627,663."?
5. **Requirement 1** — does the widget actually SHOW the 4 ratios, token stats,
   the calculation, and the self-score (not just assert them)?
6. **Requirement 2** — can you SEE and DOWNLOAD the full token list (all 10000)?
7. **Requirement 3** — is the URL public over HTTPS?
8. **Requirement 4** — are the widget's displayed numbers exactly what an
   independent re-run produces from the downloaded artifact? Any fabrication or
   rounding that misleads?
9. **Notebook** — does notebook.html render (marimo static export) and show the
   same 9953 and the hand-written BPETokenizer code?
10. **Dead links** — crawl the site; any href/src that 404s?
11. **Faithful-unit definition** — is "faithful units" a defensible denominator
    (not gamed to inflate the score)? Check how count_faithful_units works.

## Report

Write findings to /home/rev/timepass/tokenizer-assignment/solution/FINAL_VALIDATION_FINDINGS.md
and return a compact JSON verdict: per-requirement PASS/FAIL, recomputed vs
claimed table, gate results, dead-link list, and residual risks.

## Machine notes

Read-only host is fine. Watch thermals (sensors | grep Tctl, ~95C danger) and
memory (free -m, keep >500M). Applying the downloaded merges needs NO retrain —
just apply merges in rank order to score. One 10k retrain is ~2-3 min if you
choose to retrain. Don't run many processes in parallel.
