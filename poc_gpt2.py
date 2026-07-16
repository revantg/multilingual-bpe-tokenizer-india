import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import regex
    from dataclasses import dataclass
    from collections import Counter, defaultdict
    from pathlib import Path
    from heapq import heappop_max, heapify_max, heappush_max
    from typing import Callable, TypeAlias


@app.cell
def _():
    # ── Load raw corpora ─────────────────────────────────────────────────────
    # Four India-Wikipedia pages, one per language, used as the training corpus.
    en_path = Path("datasets/full_page_md/india_en.md")
    hi_path = Path("datasets/full_page_md/india_hi.md")
    te_path = Path("datasets/full_page_md/india_te.md")
    sa_path = Path("datasets/full_page_md/india_sa.md")

    def read_file(path: Path):
        with open(path) as f:
            return f.read()

    en_txt = read_file(en_path)
    hi_txt = read_file(hi_path)
    te_txt = read_file(te_path)
    sa_txt = read_file(sa_path)
    return en_txt, hi_txt, sa_txt, te_txt


@app.cell
def _():
    # ── Byte-level base alphabet + regex pre-tokenizer ───────────────────────
    # Fixes issue #1 (no byte-level fallback) and #4 (no pre-tokenizer regex).
    # Every one of the 256 possible byte values gets a unique, printable
    # unicode stand-in symbol (GPT-2 scheme) so *any* UTF-8 text -- any script,
    # emoji, or byte sequence never seen in training -- can always be encoded.
    def bytes_to_unicode() -> dict[int, str]:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    BYTE_ENCODER: dict[int, str] = bytes_to_unicode()
    BYTE_DECODER: dict[str, int] = {
        symbol: b for b, symbol in BYTE_ENCODER.items()
    }

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # GPT-2 style pre-tokenizer: keeps a leading space attached to the following
    # word/number/punctuation run (fixes #3, whitespace is part of a token
    # instead of being discarded by str.split()) and never lets a BPE merge
    # cross a word/punctuation boundary (fixes #4). \p{L}/\p{N} are unicode
    # letter/number classes, so this works identically for Latin, Devanagari,
    # and Telugu script without special-casing any language.
    PRETOKENIZE_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def pretokenize(text: str) -> list[str]:
        return PRETOKENIZE_PATTERN.findall(text)

    def token_to_byte_symbols(token: str) -> list[str]:
        return [
            BYTE_ENCODER[byte_value] for byte_value in token.encode("utf-8")
        ]

    return (
        BYTE_DECODER,
        BYTE_ENCODER,
        SPECIAL_TOKENS,
        pretokenize,
        token_to_byte_symbols,
    )


@app.class_definition
# ── Artifact: result of one merge operation ──────────────────────────────
# Bundles the updated sequence together with the symbol that was actually
# created, so callers never need to recompute sym1+sym2 independently
# (keeps merge() as the single source of truth for the new token string).
@dataclass(frozen=True)
class MergeResult:
    updated_seq: list[str]
    created_symbol: str


@app.function
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


@app.class_definition
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
    def build_stats(
        raw_tokens: list[str], symbolize: Callable[[str], list[str]]
    ) -> "TokenizerStatsV2":
        token_freq: Counter[str] = Counter(raw_tokens)
        pair_to_token_map: dict[TokenizerStatsV2.TokenPair, set[str]] = (
            defaultdict(set)
        )
        token_to_symbols_map: dict[str, list[str]] = {}
        pair_freq: Counter[TokenizerStatsV2.TokenPair] = Counter()

        for token, frequency in token_freq.items():
            symbols: list[str] = symbolize(token)
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


@app.cell
def _(
    BYTE_DECODER: dict[str, int],
    BYTE_ENCODER: dict[int, str],
    SPECIAL_TOKENS,
    pretokenize,
    token_to_byte_symbols,
):
    class BPETokenizer:
        TokenPair: TypeAlias = tuple[str, str]

        BOS = "<bos>"
        EOS = "<eos>"
        PAD = "<pad>"
        UNK = "<unk>"

        def __init__(
            self,
            corpus: list[str],
            vocab: dict[str, int] | None = None,
            merges: list[TokenPair] | None = None,
        ) -> None:
            if vocab is None:
                # Base alphabet = special tokens + all 256 byte symbols, always.
                # Fixes #6 (no special tokens) and, combined with byte symbols,
                # #1 (encode() can never hit an out-of-vocab character).
                vocab = {
                    token: index for index, token in enumerate(SPECIAL_TOKENS)
                }
                for byte_value in range(256):
                    vocab.setdefault(BYTE_ENCODER[byte_value], len(vocab))
            self.vocab = vocab
            self.merges = [] if merges is None else merges
            self.corpus = corpus
            self.merge_rank: dict[BPETokenizer.TokenPair, int] = {
                pair: rank for rank, pair in enumerate(self.merges)
            }

        def train_bpe_optimized(self, vocab_size: int) -> None:
            # corpus is now pre-tokenized (regex) byte-symbol strings pooled
            # across ALL languages together, so merges/vocab are shared and
            # priced consistently across scripts (fixes #5).
            stats = TokenizerStatsV2.build_stats(
                raw_tokens=self.corpus, symbolize=token_to_byte_symbols
            )
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
                self.merge_rank[top_pair] = len(self.merges)
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

        def _apply_merges(self, symbols: list[str]) -> list[str]:
            symbols = list(symbols)
            while len(symbols) > 1:
                ranked = [
                    (self.merge_rank[pair], pair)
                    for pair in zip(symbols, symbols[1:])
                    if pair in self.merge_rank
                ]
                if not ranked:
                    break
                _, best_pair = min(ranked, key=lambda item: item[0])
                symbols = merge_optimized(symbols, *best_pair).updated_seq
            return symbols

        def encode(
            self, text: str, add_special_tokens: bool = False
        ) -> list[int]:
            # Fixes #2 (training-only, no encode/decode path).
            ids: list[int] = []
            if add_special_tokens:
                ids.append(self.vocab[self.BOS])

            for pretoken in pretokenize(text):
                symbols = self._apply_merges(token_to_byte_symbols(pretoken))
                ids.extend(self.vocab[symbol] for symbol in symbols)

            if add_special_tokens:
                ids.append(self.vocab[self.EOS])
            return ids

        def decode(self, ids: list[int]) -> str:
            id_to_token = {index: token for token, index in self.vocab.items()}
            special_ids = {
                self.vocab[token]
                for token in SPECIAL_TOKENS
                if token in self.vocab
            }
            pieces = [
                id_to_token[token_id]
                for token_id in ids
                if token_id in id_to_token and token_id not in special_ids
            ]
            raw_bytes = bytes(
                BYTE_DECODER[symbol] for piece in pieces for symbol in piece
            )
            return raw_bytes.decode("utf-8", errors="replace")

    return (BPETokenizer,)


@app.cell
def _(BPETokenizer, en_txt, hi_txt, pretokenize, sa_txt, te_txt):
    # ── Train on the pooled multilingual corpus, then round-trip demo ───────
    raw_pretokens: list[str] = []
    for lang_text in (en_txt, hi_txt, te_txt, sa_txt):
        raw_pretokens.extend(pretokenize(lang_text))

    tokenizer = BPETokenizer(corpus=raw_pretokens)
    tokenizer.train_bpe_optimized(2000)

    print("=" * 20, "FINAL", "=" * 20)
    print(f"vocabulary size: {len(tokenizer.vocab)}")
    print(f"merges learned: {len(tokenizer.merges)}")

    for lang_name, lang_sample in (
        ("EN", en_txt[:300]),
        ("HI", hi_txt[:300]),
        ("TE", te_txt[:300]),
        ("SA", sa_txt[:300]),
    ):
        ids = tokenizer.encode(lang_sample)
        roundtrip_ok = tokenizer.decode(ids) == lang_sample
        print(f"{lang_name}: tokens={len(ids)} roundtrip_ok={roundtrip_ok}")

    # Byte-level fallback: emoji + accented + Devanagari text never in training.
    fallback_demo = "🎉 BPE across scripts: ¡Hola! नमस्ते"
    fallback_ids = tokenizer.encode(fallback_demo)
    print(
        "fallback roundtrip_ok:",
        tokenizer.decode(fallback_ids) == fallback_demo,
    )
    return


if __name__ == "__main__":
    app.run()
