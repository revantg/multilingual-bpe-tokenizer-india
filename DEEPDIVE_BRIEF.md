# Deep-dive brief: why our BPE score (~3900) trails the instructor's (~6500)

## Goal
Find, with evidence, EXACTLY why our tokenizer cannot reach the instructor's
fertility-spread score on the same corpus, and return concrete, ranked,
implementable recommendations. You MUST consult the `cliproxy/claude-fable-5`
advisor model (via `agent(..., agent="task", model="cliproxy/claude-fable-5")`
or `completion(..., model=...)`) for the strategic analysis and include its
verdict. You are an investigator — produce a written report, do not modify
notebook/production files (scratch under /tmp only).

## The metric (frozen)
```
faithful_unit = one contiguous \p{L}\p{M}\p{N} run  OR  one visible non-space punct/symbol char
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
fertility(L) = tokens(L) / faithful_units(L)
score = 1000 / (max_fertility - min_fertility)
hindi_penalty = exp(max(0, hi_fertility/1.2 - 1));  adj = score / penalty
```
Constraints: shared vocab = 10000, decode(encode(text)) lossless on visible
chars (zero UNK on the corpus), all fertilities <= 1.2.

## Instructor result (verified, from their metrics.json)
- Base: **HuggingFace `tokenizers` BPE with Metaspace** pretokenizer/decoder
  (`▁` space marker), Normalizer **NFKC**, `min_frequency=1`, vocab 10000.
- Languages: en, hi, te, **mai (Maithili)**. Weights {en:3, hi:4, te:4, mai:2}.
- ratios: en 0.5977, hi 0.5793, te 0.6731, mai 0.7331
- spread 0.15379 -> score 6502.6, hindi penalty 1.0
- Explicit quote from SOLUTION.md: "Metaspace is used instead of ByteLevel
  because ByteLevel spends too many tokens on UTF-8 bytes for Indic scripts."
- Full bundle: /home/rev/Downloads/2aa5dbf6-9413-4ec2-a27d-780833fce1a5/
  (SOLUTION.md, train_tokenizer.py, evaluate_tokenizer.py, tokenizer.json,
  metrics.json, corpus/*.faithful.txt)

## Our result (best so far)
- Base: **byte-level (GPT-2 style)** — `bytes_to_unicode()` -> 256 byte-symbols;
  every token symbolized via `token_to_byte_symbols` (UTF-8 bytes). Cross-word
  (line-level) BPE, greedy optimized trainer. See:
  - /home/rev/timepass/tokenizer-assignment/sweep_weighted.py (standalone impl)
  - /home/rev/timepass/tokenizer-assignment/poc_gpt2.py (notebook, class BPETokenizer / PhraseBPETokenizer)
- On instructor corpus (en/hi/te instructor + our SA): best {en:5,hi:3,te:3,sa:1}
  -> X en 0.53, hi 0.61, te 0.72, sa 0.37; spread 0.35; adj 2831.
- On our own corpus: best {en:5,hi:2,te:2,sa:1} -> spread 0.256, adj 3901.
- Our Indic fertilities floor much higher than his (te 0.72 vs 0.673, hi 0.61
  vs 0.579). TE is our spread-driving outlier.

## Leading hypothesis (verify or refute with experiments)
ByteLevel base inflates Indic token counts because each Devanagari/Telugu
codepoint = 3 UTF-8 bytes -> merges wasted reassembling codepoints -> Indic
fertility floors ~0.15-0.25 higher than a char/Metaspace base. Switching our
base alphabet from bytes to a Metaspace/char scheme (unicode chars + `▁`,
byte fallback only for unseen bytes to keep zero-UNK) should close most of the
gap.

## What to actually do
1. Reproduce the instructor tokenizer with `tokenizers` lib on their corpus;
   confirm ~6502. (Deps: `tokenizers regex` in the repo .venv — check first:
   /home/rev/timepass/tokenizer-assignment/.venv/bin/python -c "import tokenizers".)
2. Run OUR byte-level impl on the SAME 4 languages/weights the instructor used
   (en/hi/te/mai, weights {3,4,4,2}) so it is an apples-to-apples comparison
   (removes the SA-substitution confound). Report our spread/score there.
3. Isolate the base-alphabet effect: build a Metaspace/char variant (either via
   `tokenizers` or a minimal char-level version of our trainer) and measure the
   Indic fertility drop vs our byte-level, all else equal.
4. Quantify secondary factors: NFKC normalization, min_frequency, weight choice,
   Maithili-vs-Sanskrit as the 4th language.
5. Consult **fable-5** with the full numeric picture; get its ranked diagnosis
   and recommended fix path. Include its response.

## Hard operational limits
- **NEVER run more than 4 worker processes at any time.** Prefer 3. The machine
  has hung before at high load; watch it.
- Monitor thermals (`sensors | grep Tctl`, throttle ~95C) and memory
  (`free -m`, keep >500M available) if you launch any parallel sweep; kill and
  back off if either is breached.
- Training one 10k tokenizer on the pooled corpus takes ~30-120s and ~2.5G RSS
  per process; size parallelism accordingly (3x2.5G ~ 7.5G).

## Deliverable
A tight report: root-cause (proven, not asserted), the apples-to-apples number,
the measured base-alphabet delta, fable-5's verdict, and a ranked list of
changes with expected score impact and effort. Write it to
/home/rev/timepass/tokenizer-assignment/DEEPDIVE_FINDINGS.md.
