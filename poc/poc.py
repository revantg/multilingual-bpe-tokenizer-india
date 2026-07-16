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
    def merge_optimized(
        word_seq: list[str], sym1: str, sym2: str
    ) -> MergeResult:
        """Merge every non-overlapping occurrence of ``(sym1, sym2)``."""
        new_symbol = sym1 + sym2
        updated_seq: list[str] = []

        index = 0
        while index < len(word_seq):
            if (
                index + 1 < len(word_seq)
                and word_seq[index] == sym1
                and word_seq[index + 1] == sym2
            ):
                updated_seq.append(new_symbol)
                index += 2
            else:
                updated_seq.append(word_seq[index])
                index += 1

        return MergeResult(updated_seq=updated_seq, created_symbol=new_symbol)

    return (merge_optimized,)


@app.cell
def _(Counter, defaultdict):
    from typing import TypeAlias

    class TokenizerStatsV2:
        TokenPair: TypeAlias = tuple[str, str]

        def __init__(
            self,
            pair_to_token_map: dict[TokenPair, set[str]],
            token_to_symbols_map: dict[str, list[str]],
            token_freq: Counter[str],
            pair_freq: Counter[TokenPair],
        ) -> None:
            self.pair_to_token_map = pair_to_token_map
            self.token_to_symbols_map = token_to_symbols_map
            self.token_freq = token_freq
            self.pair_freq = pair_freq

        @staticmethod
        def build_stats(raw_tokens: list[str]) -> "TokenizerStatsV2":
            token_freq: Counter[str] = Counter(raw_tokens)
            pair_to_token_map: dict[TokenizerStatsV2.TokenPair, set[str]] = (
                defaultdict(set)
            )
            token_to_symbols_map: dict[str, list[str]] = {}
            pair_freq: Counter[TokenizerStatsV2.TokenPair] = Counter()

            for token, frequency in token_freq.items():
                symbols: list[str] = list(token)
                token_to_symbols_map[token] = symbols

                for pair in zip(symbols, symbols[1:]):
                    pair_freq[pair] += frequency
                    pair_to_token_map[pair].add(token)

            return TokenizerStatsV2(
                pair_to_token_map=dict(pair_to_token_map),
                token_to_symbols_map=token_to_symbols_map,
                token_freq=token_freq,
                pair_freq=pair_freq,
            )

    return TokenizerStatsV2, TypeAlias


@app.cell
def _(
    TokenizerStatsV2,
    TypeAlias,
    en_txt,
    heapify_max,
    heappop_max,
    heappush_max,
    merge_optimized,
):
    class BPETokenizer:
        TokenPair: TypeAlias = tuple[str, str]

        def __init__(
            self,
            corpus: list[str],
            vocab: dict[str, int] | None = None,
            merges: list[TokenPair] | None = None,
        ) -> None:
            self.vocab = {} if vocab is None else vocab
            self.merges = [] if merges is None else merges
            self.corpus = corpus

        def train_bpe_optimized(self, vocab_size: int) -> None:
            stats = TokenizerStatsV2.build_stats(raw_tokens=self.corpus)
            pair_to_token_map = stats.pair_to_token_map
            token_to_symbols_map = stats.token_to_symbols_map
            token_freq = stats.token_freq
            pair_freq = stats.pair_freq

            pair_freq_max_heap = [
                (count, pair) for pair, count in pair_freq.items()
            ]
            heapify_max(pair_freq_max_heap)

            while len(self.vocab) < vocab_size and pair_freq_max_heap:
                top_pair: BPETokenizer.TokenPair | None = None
                while pair_freq_max_heap:
                    count, candidate_pair = heappop_max(pair_freq_max_heap)
                    if count > 0 and pair_freq[candidate_pair] == count:
                        top_pair = candidate_pair
                        break

                if top_pair is None:
                    break

                merged_token = "".join(top_pair)
                if merged_token not in self.vocab:
                    self.vocab[merged_token] = len(self.vocab)
                self.merges.append(top_pair)

                tokens = pair_to_token_map.get(top_pair, set()).copy()
                updated_pairs: set[BPETokenizer.TokenPair] = set()

                for token in tokens:
                    symbols = token_to_symbols_map[token]
                    for pair in zip(symbols, symbols[1:]):
                        pair_freq[pair] -= token_freq[token]
                        pair_to_token_map[pair].discard(token)
                        updated_pairs.add(pair)

                    updated_seq = merge_optimized(
                        symbols, *top_pair
                    ).updated_seq
                    token_to_symbols_map[token] = updated_seq

                    for pair in zip(updated_seq, updated_seq[1:]):
                        pair_freq[pair] += token_freq[token]
                        pair_to_token_map.setdefault(pair, set()).add(token)
                        updated_pairs.add(pair)

                for pair in updated_pairs:
                    if pair_freq[pair] > 0:
                        heappush_max(
                            pair_freq_max_heap, (pair_freq[pair], pair)
                        )

            print("=" * 20, "FINAL", "=" * 20)
            print(f"vocabulary size: {len(self.vocab)}")
            print(f"merges learned: {len(self.merges)}")

    inp = "low lower lowest flow"
    vocab = {symbol: index for index, symbol in enumerate(set(en_txt))}
    print(f"initial len vocab=", len(vocab))
    tokenizer = BPETokenizer(corpus=en_txt.split(), vocab=vocab)
    tokenizer.train_bpe_optimized(2000)
    print(tokenizer.merges)
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
