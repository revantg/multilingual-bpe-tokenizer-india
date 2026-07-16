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
    return (en_txt,)


@app.cell
def _(en_txt):
    # eyeball the raw English text
    print(en_txt)
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
    print(tokenizer.vocab)
    return


if __name__ == "__main__":
    app.run()
