import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import dataclass
    from collections import Counter
    from pathlib import Path

    from pprint import pprint

    return Counter, Path, dataclass, pprint


@app.cell
def _(Path):
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
    print(en_txt)
    return


@app.cell
def _(en_txt, hi_txt, sa_txt, te_txt):
    # creating all adjacent symbols list
    # list1 -> some_string
    # list2 -> ome_string
    # zip(list1, list2)

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
    vocab: set[str] = set(list(hi_txt))
    symbols: set[str] = vocab.copy()

    merges = []
    vocab
    return


@app.cell
def _(dataclass):
    @dataclass(frozen=True)
    class MergeResult:
        updated_seq: list[str]
        created_symbol: str
    
    def merge(seq: list[str], sym1: str, sym2) -> MergeResult:
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
    
    sequence = list("low lower lowest flow")
    iter1 = merge(sequence, "l", "o")
    print(f"after iter1={iter1}")
    assert iter1 == MergeResult(
        updated_seq=['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ' ', 'lo', 'w', 'e', 's', 't', ' ', 'f', 'lo', 'w'],
        created_symbol='lo'
    )
    iter2 = merge(iter1.updated_seq, "lo", "w")
    print(f"after iter2={iter2}")
    assert iter2 == MergeResult(
        updated_seq=['low', ' ', 'low', 'e', 'r', ' ', 'low', 'e', 's', 't', ' ', 'f', 'low'],
        created_symbol='low'
    )
    return (merge,)


@app.cell
def _(Counter, dataclass, merge, pprint):
    @dataclass
    class BytePairEncoder:
        vocab: dict[str, int]
        merges: list[tuple[str, str]]
    
    def train_bpe(txt: str, vocab_size: int) -> BytePairEncoder:
        symbols: set[str] = set(list(txt))
        merges: tuple[str, str] = []
    
        vocab: dict[str, int] = {ch: ind for ind, ch in enumerate(symbols)}
        latest_vocab_id = len(vocab) - 1

        sequence = list(txt)
        new_seq: list[str] = []

        print(f"initial vocab size = {len(vocab)}")
        while len(vocab) < vocab_size:
            pairs = list(zip(list(sequence[:-1]), list(sequence[1:])))
            (sym1, sym2), _cnt = Counter(pairs).most_common(1)[0]
            print(f"{(sym1, sym2)=} {_cnt=}")

            # print(f"before merge: {sequence=}")
            merge_res = merge(sequence, sym1, sym2)
            new_seq, created_symbol = (merge_res.updated_seq, merge_res.created_symbol)
            # print(f"after merge: {new_seq=}")
        
            vocab[created_symbol] = latest_vocab_id + 1
            latest_vocab_id += 1
            merges.append((sym1, sym2))

            sequence = new_seq
            # break
            print()

        return BytePairEncoder(
            vocab=vocab, 
            merges=merges
        )

    tokenizer_input = "low lower lowest flow"
    pprint(train_bpe(tokenizer_input, 15))
    return


@app.function
# incorrect merge algo
# needlessly complex
def merge_wrong(seq: list[str], symbol: str) -> list[str]: 
    """
    merge adjacent element in seq equal to symbol to one symbol
    for eg. for seq = ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r']
             symbol = "lo"
            new_seq = ['lo', 'w', ' ', 'lo', 'w', 'e', 'r']
    """
    symbol_len = len(symbol)
    merge_list: list[int] = []
    
    for i in range(0, len(seq) - symbol_len + 1):
        match = True
        for j in range(0, symbol_len):
            
            sym_ind = j
            seq_ind = i + j
            
            if seq[seq_ind] != symbol[sym_ind]:
                match = False
                break

        if match:
            merge_list.append(i)
            i += symbol_len

    if len(merge_list) == 0:
        return seq

    print(f"{merge_list=}")
    prev = 0
    new_seq: list[str] = []
    for ind in merge_list:
        if prev == 0:
            new_seq.append(symbol)
            prev = ind + symbol_len
            continue
        new_seq += seq[prev: ind]
        new_seq.append(symbol)
        prev = ind + symbol_len

    new_seq += seq[prev: ]

    return new_seq


@app.cell
def _():
    #incorrect merge algo
    # made it complex needlessly
    return


if __name__ == "__main__":
    app.run()
