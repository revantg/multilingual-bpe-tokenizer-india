# RED-PILL VALIDATION: the deployed widget vs the assignment's submission rules

You are a STRICT, adversarial validator. Do NOT rubber-stamp. Try to find every
way this submission fails the assignment's stated requirements, and report
honestly. Prototype/verify in /tmp scratch only; do not modify repo files except
your final report.

## Live URL to validate
https://india-bpe-tokenizer.surge.sh/  (widget = index.html; story = slides.html)

Local source: /home/rev/timepass/tokenizer-assignment/solution/frontend/
Downloadable artifacts served: vocab.json, tokens.txt, tokenizer.json.

## The assignment's submission requirements (validate EACH, verbatim)
1. "A Widget that shows these **ratios, token statistics and calculations and
   your self score**." → The widget must clearly display all four fertility
   ratios, per-language token statistics, the score calculation, and the final
   self-score.
2. "Allows me to **see and download your tokenizer (list of all tokens)**." →
   The widget must let a grader view AND download the full token list. Confirm
   the download actually delivers all 10,000 tokens (not truncated), by fetching
   the live file.
3. "URL for this widget on Netlify (or anywhere else you want to host)." →
   A working public URL. Confirm it loads over HTTPS and serves the widget.
4. "**We'll run your tokenizer ourselves to confirm your scores**, so be careful
   about your numbers." → THE critical gate: every number the widget claims must
   be independently reproducible. Re-derive them yourself.

## What you MUST independently verify (do not trust the widget's own text)
- Fetch the live vocab.json and tokens.txt from the surge URL. Confirm:
  - vocab.json contains exactly 10000 tokens and 9740 merges.
  - tokens.txt lists all tokens (count == 10000), one per line, id<TAB>token.
- Reconstruct the tokenizer from the downloaded vocab.json+merges IN A FRESH
  SCRIPT (do not import the repo's tokenizer.py — build encode/decode yourself
  from the byte-level scheme, or reuse only the minimal decode) and RE-RUN it on
  the corpus at /home/rev/timepass/tokenizer-assignment/solution/corpus/{en,hi,te,mai}.txt
  to recompute: tokens per language, faithful units per language (regex
  [\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]), the four fertilities, spread, and
  score = 1000/(X4-X1). They MUST match the widget's claims:
  EN 0.5669, HI 0.5774, TE 0.6012, MAI 0.6674, spread 0.1005, score 9953.
- Confirm the gates the widget asserts: vocab == 10000, every X <= 1.2, 0 UNK on
  the corpus, and decode(encode(text)) is lossless (test on the corpus AND on the
  held-out digit sample "India's population is 1,428,627,663.").
- Cross-check that the downloaded tokenizer is the SAME one the scores describe
  (i.e., the vocab.json a grader downloads actually reproduces 9953, not a stale
  or different artifact).
- Check the numbers shown on slides.html too (approach tables: word 1374 fail,
  sentence 2648, paragraph 2605; instructor 6503) against VERIFIED_NUMBERS.md.

## Adversarial checks
- Does any claimed number appear NOWHERE in a reproducible source? Flag it.
- Is the "download tokenizer" truly the full list, or a sample/truncation?
- If the widget computes anything client-side (e.g., a live encode demo), does it
  agree with the Python reference on a few probe strings? If it fakes/precomputes,
  is it labeled honestly?
- Is the self-score consistent with the ratios shown (1000/(0.6674-0.5669)=9953)?

## Deliverable
Write /home/rev/timepass/tokenizer-assignment/solution/frontend/WIDGET_VALIDATION_FINDINGS.md
with a PASS/FAIL per requirement (1-4) and per gate, your independently
recomputed numbers vs the widget's claims (table), any discrepancy found (or an
explicit "none"), and a blunt final judgment: is this submission acceptable, and
would the grader's own re-run confirm the claimed 9953? Name any residual risk.
Return a concise summary.

## Limits
Max 3 worker processes; watch thermals (sensors|grep Tctl ~95C) and memory
(free -m, >500M). One 10k retrain ~2-3 min if you retrain; you can also just
apply the downloaded merges (no retrain needed) to score.
