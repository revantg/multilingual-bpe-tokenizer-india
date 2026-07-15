import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    # ── Imports ──────────────────────────────────────────────────────────────
    # Building a from-scratch BPE tokenizer for India Wikipedia (EN/HI/TE/SA).
    # heapq / defaultdict added for the optimized (word-dedup + heap) trainer.
    from dataclasses import dataclass
    from collections import Counter, defaultdict
    from pathlib import Path
    from heapq import heappop_max, heapify_max, heappush_max
    from collections import defaultdict

    from pprint import pprint

    return (
        Counter,
        Path,
        dataclass,
        defaultdict,
        heapify_max,
        heappop_max,
        heappush_max,
    )


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
            print(" " * 5, i_ind, ":", hi_txt[ind : ind + 200])

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
            new_seq += seq[prev:ind]
            new_seq.append(sym1 + sym2)
            prev = ind + 2

        new_seq += seq[prev:]

        return MergeResult(updated_seq=new_seq, created_symbol=sym1 + sym2)

    # sanity check: "l"+"o" -> "lo", then "lo"+"w" -> "low", chained correctly
    sequence = list("low lower lowest flow")
    iter1 = merge_naive(sequence, "l", "o")
    print(f"after iter1={iter1}")
    assert iter1 == MergeResult(
        updated_seq=[
            "lo",
            "w",
            " ",
            "lo",
            "w",
            "e",
            "r",
            " ",
            "lo",
            "w",
            "e",
            "s",
            "t",
            " ",
            "f",
            "lo",
            "w",
        ],
        created_symbol="lo",
    )
    iter2 = merge_naive(iter1.updated_seq, "lo", "w")
    print(f"after iter2={iter2}")
    assert iter2 == MergeResult(
        updated_seq=[
            "low",
            " ",
            "low",
            "e",
            "r",
            " ",
            "low",
            "e",
            "s",
            "t",
            " ",
            "f",
            "low",
        ],
        created_symbol="low",
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
def _(MergeResult):
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
            ind = 0
            while ind + 1 < len(word_seq):
                if word_seq[ind] == sym1 and word_seq[ind + 1] == sym2:
                    new_seq.append(new_symbol)
                    ind += 1
                else:
                    new_seq.append(word_seq[ind])
                ind += 1

            if ind == len(word_seq) - 1:
                new_seq.append(word_seq[ind])

            updated_seq.append(new_seq)

        return MergeResult(updated_seq=updated_seq, created_symbol=new_symbol)

    return (merge_optimized,)


@app.cell
def _(
    Counter,
    dataclass,
    defaultdict,
    en_txt,
    heapify_max,
    heappop_max,
    heappush_max,
    merge_optimized,
):
    @dataclass
    class GeneratePairsResult:
        pair_counts: Counter[tuple[str, str]]
        rev_pair_index: defaultdict[tuple[int, int], list[list[str]]]

    def gen_pairs_reverse_index(
        words: list[list[str]], words_freq: Counter[str]
    ) -> dict[str, set[list[str]]]:
        """
        generate pairs for a list of words (already split)
        example input:
        [
            ["lo", "w"],
            ["lo", "w", "e", "r"],
            ["lo", "w", "e", "s", "t"]
        ]

        output ->
        {
            ("lo", "w") -> [
                ["lo", "w"],
                ["lo", "w", "e", "r"],
                ["lo", "w", "e", "s", "t"]
            ],
            ("w", "e") -> [
                ["lo", "w", "e", "r"],
                ["lo", "w", "e", "s", "t"]
            ]
        }
        """
        pair_counts: Counter[tuple[str, str]] = Counter()
        rev_pair_index: defaultdict[tuple[int, int], list[str]] = defaultdict(
            set
        )

        for word in words:
            for ind in range(len(word) - 1):
                pair_counts[(word[ind], word[ind + 1])] += words_freq.get(
                    "".join(word)
                )
                rev_pair_index[(word[ind], word[ind + 1])].add("".join(word))

        return GeneratePairsResult(pair_counts, rev_pair_index)

    def train_bpe_optimized(txt: str, vocab_size: int):  # -> BytePairEncoder:
        merges: tuple[str, str] = []

        vocab: dict[str, int] = {ch: ind for ind, ch in enumerate(set(txt))}
        latest_vocab_id = len(vocab) - 1

        sequence = list(txt)
        new_seq: list[str] = []

        word_freq: Counter[str] = Counter()
        for word in txt.split():
            word_freq[word] += 1
        words_set: set[str] = word_freq.keys()

        pair_counts: Counter[str] = Counter()
        word_to_pair_map: dict[str, list[str]] = {
            word: list(word) for word in words_set
        }

        pairs_reverse_idx_res = gen_pairs_reverse_index(
            [list(w) for w in words_set], word_freq
        )
        pair_counts: Counter[tuple[str, str]] = (
            pairs_reverse_idx_res.pair_counts
        )
        pair_to_word_map: defaultdict[tuple[str, str], list[list[str]]] = (
            pairs_reverse_idx_res.rev_pair_index
        )

        pair_cnt_max_heap: list[tuple[int, tuple[str, str]]] = [
            (cnt, pair) for pair, cnt in pair_counts.items()
        ]
        heapify_max(pair_cnt_max_heap)

        merges: list[tuple[str, str]] = []

        while len(vocab) < vocab_size:
            while len(pair_cnt_max_heap):
                cnt, top_pair = heappop_max(pair_cnt_max_heap)
                if pair_counts[top_pair] == cnt:
                    break

            updated_pairs = set()
            for word in pair_to_word_map[top_pair]:
                word_pairs = word_to_pair_map[word]
                for ind in range(len(word_pairs) - 1):
                    updated_pairs.add((word_pairs[ind], word_pairs[ind + 1]))

            merges.append(top_pair)

            # words containing the top pair
            top_words_tokens = [
                word_to_pair_map[full_word]
                for full_word in pair_to_word_map[top_pair]
            ]

            # merge the pair inside all the selected words
            result = merge_optimized(
                top_words_tokens, sym1=top_pair[0], sym2=top_pair[1]
            )

            # update the word to their latest pair mapping
            for word in result.updated_seq:
                word_to_pair_map["".join(word)] = word

            # recompute the counts and inverse index mapping of all the changed words

            # remove the pair counts of the pre-merge words
            for word in top_words_tokens:
                for pair in zip(word[:-1], word[1:]):
                    pair_counts[pair] -= word_freq["".join(word)]
                    pair_to_word_map[pair].remove("".join(word))

            # generate pairs after the merged sequence for each word and update pair_counts
            for word_tokens in result.updated_seq:
                word_tokens_pairs = zip(word_tokens[:-1], word_tokens[1:])

                for word_token_pair in word_tokens_pairs:
                    pair_counts[word_token_pair] += word_freq[
                        "".join(word_tokens)
                    ]
                    pair_to_word_map[word_token_pair].add("".join(word_tokens))
                    updated_pairs.add(word_token_pair)

            # generate_inv_pairs_result = gen_pairs_reverse_index(
            #     result.updated_seq, words_freq
            # )

            # for (
            #     pair,
            #     words,
            # ) in generate_inv_pairs_result.rev_pair_index.items():
            #     for word in words:
            #         pair_to_word_map[pair].add(word_tokens)

            vocab[result.created_symbol] = len(vocab) + 1

            # update the heap
            for pair in updated_pairs:
                heappush_max(pair_cnt_max_heap, (pair_counts[pair], pair))

        return vocab, merges

    inp = "low lower lowest flow"
    train_bpe_optimized(en_txt, 1000)
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
            new_seq, created_symbol = (
                merge_res.updated_seq,
                merge_res.created_symbol,
            )
            # print(f"after merge: {new_seq=}")

            vocab[created_symbol] = latest_vocab_id + 1
            latest_vocab_id += 1
            merges.append((sym1, sym2))

            sequence = new_seq

        return BytePairEncoder(vocab=vocab, merges=merges)

    bpe = train_bpe_naive(en_txt, 1000)
    return


@app.function
# ── encode: NOT YET IMPLEMENTED ──────────────────────────────────────────
# Should: take raw text + a trained BytePairEncoder (vocab, merges), replay
# every rule in `merges` IN ORDER (reusing merge_naive/merge_optimized per
# rule) to turn the text into token strings, then look each token up in
# `vocab` to produce the final list of integer ids. This is the missing
# piece that lets the tokenizer run on text it was NOT trained on.
def encode(
    txt: str, vocab: dict[str, int], merges: list[tuple[str, str]]
) -> list[int]:
    pass


if __name__ == "__main__":
    app.run()
