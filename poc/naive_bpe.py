import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import Counter
    from dataclasses import dataclass
    from pathlib import Path

    corpus_path = Path("datasets/full_page_md/india_en.md")
    corpus_text = corpus_path.read_text()

    @dataclass(frozen=True)
    class MergeResult:
        updated_seq: list[str]
        created_symbol: str

    @dataclass(frozen=True)
    class BytePairEncoder:
        vocab: dict[str, int]
        merges: list[tuple[str, str]]

    def merge_naive(
        sequence: list[str], left: str, right: str
    ) -> MergeResult:
        """Merge every non-overlapping adjacent ``(left, right)`` pair."""
        merged_symbol = left + right
        updated_sequence: list[str] = []
        index = 0

        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == left
                and sequence[index + 1] == right
            ):
                updated_sequence.append(merged_symbol)
                index += 2
            else:
                updated_sequence.append(sequence[index])
                index += 1

        return MergeResult(
            updated_seq=updated_sequence,
            created_symbol=merged_symbol,
        )

    def train_bpe_naive(text: str, vocab_size: int) -> BytePairEncoder:
        """Reference BPE trainer that rescans the entire stream per merge."""
        vocab = {
            symbol: token_id
            for token_id, symbol in enumerate(sorted(set(text)))
        }
        merges: list[tuple[str, str]] = []
        sequence = list(text)

        while len(vocab) < vocab_size and len(sequence) >= 2:
            pair_counts = Counter(zip(sequence, sequence[1:]))
            if not pair_counts:
                break

            (left, right), _ = pair_counts.most_common(1)[0]
            result = merge_naive(sequence, left, right)

            vocab[result.created_symbol] = len(vocab)
            merges.append((left, right))
            sequence = result.updated_seq

        return BytePairEncoder(vocab=vocab, merges=merges)

    # Keep the default demo small: naive BPE rescans all input for each merge.
    demo_text = corpus_text[:20_000]
    naive_bpe = train_bpe_naive(demo_text, vocab_size=300)
    print(f"vocabulary size: {len(naive_bpe.vocab)}")
    print(f"merges learned: {len(naive_bpe.merges)}")
    return BytePairEncoder, MergeResult, merge_naive, naive_bpe, train_bpe_naive


if __name__ == "__main__":
    app.run()
