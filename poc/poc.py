import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    # ── Imports ──────────────────────────────────────────────────────────────
    # Building a from-scratch BPE tokenizer for India Wikipedia (EN/HI/TE/SA).
    # heapq / defaultdict added for the optimized (word-dedup + heap) trainer.
    from dataclasses import dataclass
    from collections import Counter
    from pathlib import Path
    import heapq
    from collections import defaultdict

    from pprint import pprint

    return Counter, Path, dataclass, heapq


@app.cell
def _(Path):
    # ── Load raw corpora ─────────────────────────────────────────────────────
    # Four India-Wikipedia pages, one per language, used as the training corpus.
    en_path = Path("datasets/full_page_md/india_en.md")
    hi_path = Path("datasets/full_page_md/india_hi.md")
    te_path = Path("datasets/full_page_md/india_te.md")
    sa_path = Path("datasets/full_page_md/india_sa.md")

    def read_file(path: Path):
        with open(path) as f:
            txt = f.read()
            return txt

    en_txt = read_file(en_path)
    hi_txt = read_file(hi_path)
    te_txt = read_file(te_path)
    sa_txt = read_file(sa_path)
    return en_txt, hi_txt, sa_txt, te_txt


@app.cell
def _(en_txt):
    # eyeball the raw English text
    print(en_txt)
    return


@app.cell
def _(en_txt, hi_txt, sa_txt, te_txt):
    # ── Naive adjacent-pair construction (flat character stream) ────────────
    # creating all adjacent symbols list
    # list1 -> some_string
    # list2 -> ome_string
    # zip(list1, list2)
    # NOTE: this scans the ENTIRE raw text char-by-char, no word splitting/
    # dedup yet -- kept here as the first exploratory pass per language.

    en_pairs = list(zip(list(en_txt[:-1]), list(en_txt[1:])))
    en_pairs = ["".join(p) for p in en_pairs]

    hi_pairs = list(zip(list(hi_txt[:-1]), list(hi_txt[1:])))
    hi_pairs = ["".join(p) for p in hi_pairs]

    te_pairs = list(zip(list(te_txt[:-1]), list(te_txt[1:])))
    te_pairs = ["".join(p) for p in te_pairs]

    sa_pairs = list(zip(list(sa_txt[:-1]), list(sa_txt[1:])))
    sa_pairs = ["".join(p) for p in sa_pairs]
    return (hi_pairs,)


@app.cell
def _(Counter, hi_pairs, hi_txt):
    # just trying to see where do the top most pairs occur in the dataset
    for ch, cnt in Counter(hi_pairs).most_common(10):
        print("-" * 100)
        print("-" * 100)
        print(ch)
        prev = 0
        for i_ind, i in enumerate(range(min(cnt, 10))):
            ind = hi_txt.find(ch, prev)
            prev = ind + 1
            print(" " * 5, i_ind, ":", hi_txt[ind: ind+200])

        print("-" * 100)
        print()
    return


@app.cell
def _(hi_txt):
    # ── Base vocab / symbol scratch ──────────────────────────────────────────
    # initial vocab = every distinct character seen in hi_txt (starting alphabet
    # before any merges). `symbols` kept as a separate copy for reference.
    vocab: set[str] = set(list(hi_txt))
    symbols: set[str] = vocab.copy()

    merges = []
    vocab
    return


@app.cell
def _(dataclass):
    # ── Artifact: result of one merge operation ──────────────────────────────
    # Bundles the updated sequence together with the symbol that was actually
    # created, so callers never need to recompute sym1+sym2 independently
    # (keeps merge() as the single source of truth for the new token string).
    @dataclass(frozen=True)
    class MergeResult:
        updated_seq: list[str]
        created_symbol: str

    return (MergeResult,)


@app.cell
def _(MergeResult):
    # ── merge_naive: reference/correctness implementation ────────────────────
    # Operates on ONE FLAT sequence (whole text as a single list of symbols).
    # Compares whole adjacent LIST ELEMENTS (sym1, sym2) directly -- not string
    # character-indexing -- so it stays correct even once elements are already
    # multi-character merged tokens (e.g. matching ('lo','w'), not decomposing
    # "low" into characters).
    def merge_naive(seq: list[str], sym1: str, sym2) -> MergeResult:
        """
        merge adjacent element in seq equal to symbol to one symbol
        for eg. for seq = ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r']
                 symbol = "lo"
                new_seq = ['lo', 'w', ' ', 'lo', 'w', 'e', 'r']
        """
        symbol_len = len(sym1) + len(sym2)
        merge_list: list[int] = []

        for i in range(0, len(seq) - 1):
            match = True
            if seq[i] == sym1 and seq[i + 1] == sym2:
                merge_list.append(i)

        if len(merge_list) == 0:
            return MergeResult(updated_seq=seq, created_symbol="")

        # print(f"{merge_list=}")

        new_seq: list[str] = []
        prev = 0
        for ind in merge_list:
            new_seq += seq[prev: ind]
            new_seq.append(sym1 + sym2)
            prev = ind + 2

        new_seq += seq[prev:]

        return MergeResult(updated_seq=new_seq, created_symbol=sym1+sym2)

    # sanity check: "l"+"o" -> "lo", then "lo"+"w" -> "low", chained correctly
    sequence = list("low lower lowest flow")
    iter1 = merge_naive(sequence, "l", "o")
    print(f"after iter1={iter1}")
    assert iter1 == MergeResult(
        updated_seq=['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ' ', 'lo', 'w', 'e', 's', 't', ' ', 'f', 'lo', 'w'],
        created_symbol='lo'
    )
    iter2 = merge_naive(iter1.updated_seq, "lo", "w")
    print(f"after iter2={iter2}")
    assert iter2 == MergeResult(
        updated_seq=['low', ' ', 'low', 'e', 'r', ' ', 'low', 'e', 's', 't', ' ', 'f', 'low'],
        created_symbol='low'
    )
    return (merge_naive,)


@app.cell
def _(dataclass):
    # ── Trained tokenizer artifact ───────────────────────────────────────────
    # vocab  : token string -> integer id (used for encode's final id lookup)
    # merges : ORDERED list of (sym1, sym2) rules -- order matters, encode()
    #          must replay these in the exact order they were learned.
    @dataclass
    class BytePairEncoder:
        vocab: dict[str, int]
        merges: list[tuple[str, str]]

    return (BytePairEncoder,)


@app.cell
def _(Counter, MergeResult, en_txt, heapq, max_heap, train_bpe, word):
    # ── OPTIMIZED path (word-dedup + inverted index + max-heap) — WORK IN PROGRESS ──
    # Goal: avoid rescanning the full corpus every merge iteration.
    #   1. word_freq        : dedupe corpus into {unique word: frequency}
    #   2. pair_counts       : pair frequencies weighted by word_freq, computed
    #                          ONCE per unique word (not once per occurrence)
    #   3. pair_to_word_map  : inverted index, pair -> {words containing it},
    #                          so a merge only needs to touch AFFECTED words
    #   4. max-heap          : O(log n) "find current best pair" instead of a
    #                          fresh O(n) Counter.most_common() scan every step
    #
    # STATUS: not yet working -- known open bugs (see chat):
    #   - merge_optimized(): inner loop still uses `word` instead of `word_seq`,
    #     and does not skip the second element of a matched pair (no pointer
    #     advance-by-2), so a match can double-count the consumed element.
    #   - the main training while-loop body is not implemented yet (`pass`).
    #   - trailing call still says `train_bpe(en_txt, 1)` -- should be
    #     `train_bpe_optimized(...)` once the loop body exists.
    def merge_optimized(seq_words: list[list[str]], sym1: str, sym2: str):
        """
        merge adjacent element in seq equal to symbol to one symbol
        for eg. for seq = ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r']
                 symbol = "lo"
                new_seq = ['lo', 'w', ' ', 'lo', 'w', 'e', 'r']
        """
        symbol_len = len(sym1) + len(sym2)
        updated_seq: list[list[str]] = []

        new_symbol = sym1 + sym2

        for word_seq in seq_words:
            # word_seq is a list of tokens
            new_seq = []
            for ind in range(0, len(word) - 1):
                if word_seq[ind] == sym1 and word_seq[ind + 1] == sym2:
                    new_seq.append(new_symbol)
                else:
                    new_seq.append(word_seq[ind])
                    if ind == len(word) - 2:
                        new_seq.append(word_seq[ind + 1])
            updated_seq.append(new_seq)
            
        return MergeResult(updated_seq=updated_seq, created_symbol=new_symbol)

    def train_bpe_optimized(txt: str, vocab_size: int):# -> BytePairEncoder:
        merges: tuple[str, str] = []

        vocab: dict[str, int] = {ch: ind for ind, ch in enumerate(set(txt))}
        latest_vocab_id = len(vocab) - 1

        sequence = list(txt)
        new_seq: list[str] = []

        # dedupe: word -> frequency count (avoids re-deriving the same word's
        # internal pairs once per occurrence -- e.g. "India" appearing ~500x
        # in en_txt only needs its pairs computed once, then weighted by freq)
        word_freq: Counter[str] = Counter()
        for word in txt.split():
            word_freq[word] += 1
        words_set: set[str] = word_freq.keys()

        # pair_counts: frequency-weighted adjacent-pair tally, built ONCE from
        # unique words only.
        # pair_to_word_map: inverted index so a future merge only needs to
        # revisit words that actually contain the winning pair.
        pair_counts: Counter[str] = Counter()
        pair_to_word_map: dict[tuple[str, str], set[str]] = {}
        for word in words_set:
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += word_freq[word]
                pair_to_word_map[(word[i], word[i + 1])].add(word)

        pair_counts = [(cnt, pair) for pair, cnt in pair_counts.items()]
        heapq.heapify_max(pair_counts)
    
        print(f"{max_heap=}")

        while len(vocab) < vocab_size:
            pass
        

        return None

    inp = "low lower lowest flow"
    train_bpe(en_txt, 1)
    return


@app.cell
def _(BytePairEncoder, Counter, en_txt, merge_naive):
    # ── NAIVE trainer: correct, but full O(n) rescan every merge iteration ──
    # Trains on a single flat character sequence (no word-dedup, no heap).
    # Confirmed working end-to-end: records merges in order AND reads
    # created_symbol from merge_naive()'s own return value (no independent
    # recompute of the new token -- vocab and sequence stay consistent).
    # This is the reference/baseline; train_bpe_optimized above is the
    # in-progress faster replacement for real-corpus scale.
    def train_bpe_naive(txt: str, vocab_size: int) -> BytePairEncoder:
        symbols: set[str] = set(list(txt))
        merges: tuple[str, str] = []

        vocab: dict[str, int] = {ch: ind for ind, ch in enumerate(symbols)}
        latest_vocab_id = len(vocab) - 1

        sequence = list(txt)
        new_seq: list[str] = []

        while len(vocab) < vocab_size:
            pairs = list(zip(list(sequence[:-1]), list(sequence[1:])))
            (sym1, sym2), _cnt = Counter(pairs).most_common(1)[0]
            print(f"{(sym1, sym2)=} {_cnt=}")

            # print(f"before merge: {sequence=}")
            merge_res = merge_naive(sequence, sym1, sym2)
            new_seq, created_symbol = (merge_res.updated_seq, merge_res.created_symbol)
            # print(f"after merge: {new_seq=}")

            vocab[created_symbol] = latest_vocab_id + 1
            latest_vocab_id += 1
            merges.append((sym1, sym2))

            sequence = new_seq

        return BytePairEncoder(
            vocab=vocab, 
            merges=merges
        )

    bpe = train_bpe_naive(en_txt, 1000)
    return


@app.function
# ── encode: NOT YET IMPLEMENTED ──────────────────────────────────────────
# Should: take raw text + a trained BytePairEncoder (vocab, merges), replay
# every rule in `merges` IN ORDER (reusing merge_naive/merge_optimized per
# rule) to turn the text into token strings, then look each token up in
# `vocab` to produce the final list of integer ids. This is the missing
# piece that lets the tokenizer run on text it was NOT trained on.
def encode(txt: str, vocab: dict[str, int], merges: list[tuple[str, str]]) -> list[int]:
    pass


if __name__ == "__main__":
    app.run()
