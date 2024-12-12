import re
import os
import torch
import numpy as np
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from collections import Counter, OrderedDict

AMINO_MAP = {
    "<pad>": 24,
    "*": 23,
    "A": 0,
    "C": 4,
    "B": 20,
    "E": 6,
    "D": 3,
    "G": 7,
    "F": 13,
    "I": 9,
    "H": 8,
    "K": 11,
    "M": 12,
    "L": 10,
    "N": 2,
    "Q": 5,
    "P": 14,
    "S": 15,
    "R": 1,
    "T": 16,
    "W": 17,
    "V": 19,
    "Y": 18,
    "X": 22,
    "Z": 21,
}

AMINO_MAP_REV = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "B",
    "Z",
    "X",
    "*",
    "@",
]

AMINO_MAP_REV_ = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "N",
    "Q",
    "*",
    "*",
    "@",
]


def tokenizer(sequence):
    sequence = re.sub(r"\s+", "", str(sequence))
    sequence = re.sub(r"[^ARNDCQEGHILKMFPSTWYVBZX]", "*", sequence)
    return list(sequence)


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(
        self,
        X_pep,
        X_tcr,
        y=None,
        maxlen_pep=None,
        maxlen_tcr=None,
        amino_map=None,
        padding="mid",
    ):
        self.X_pep = X_pep
        self.X_tcr = X_tcr
        self.y = y
        self.amino_map = amino_map
        self.padding = padding

        self.maxlen_pep = maxlen_pep or max(len(x) for x in X_pep)
        self.maxlen_tcr = maxlen_tcr or max(len(x) for x in X_tcr)

    def pad_sequence(self, sequence, max_length):
        if len(sequence) >= max_length:
            return sequence[:max_length]
        pad_length = max_length - len(sequence)
        if self.padding == "front":
            return [self.amino_map["<pad>"]] * pad_length + sequence
        elif self.padding == "end":
            return sequence + [self.amino_map["<pad>"]] * pad_length
        elif self.padding == "mid":
            mid = len(sequence) // 2
            left_pad = pad_length // 2
            right_pad = pad_length - left_pad
            return (
                [self.amino_map["<pad>"]] * left_pad
                + sequence[:mid]
                + [self.amino_map["<pad>"]] * right_pad
                + sequence[mid:]
            )
        else:
            raise ValueError('Padding type must be "front", "end", or "mid".')

    def __len__(self):
        return len(self.X_pep)

    def __getitem__(self, idx):
        pep = [
            self.amino_map.get(x, self.amino_map["*"]) for x in self.X_pep[idx]
        ]
        tcr = [
            self.amino_map.get(x, self.amino_map["*"]) for x in self.X_tcr[idx]
        ]

        pep = self.pad_sequence(pep, self.maxlen_pep)
        tcr = self.pad_sequence(tcr, self.maxlen_tcr)

        if self.y is not None:
            label = self.y[idx]

            # Ensure label is numeric
            if isinstance(label, np.str_):
                label = float(label)  # Convert string to float

            return (
                torch.tensor(pep),
                torch.tensor(tcr),
                torch.tensor(label, dtype=torch.float32),
            )

        return torch.tensor(pep), torch.tensor(tcr)


# Vocabulary builder
def build_vocab_from_mapping(amino_map):
    counter = Counter(amino_map.keys())
    return vocab(counter, specials=["<pad>", "<unk>"])


# DataLoader definition
def define_dataloader(
    X_pep,
    X_tcr,
    y=None,
    maxlen_pep=None,
    maxlen_tcr=None,
    padding="mid",
    batch_size=50,
    device="cuda",
):
    amino_map = AMINO_MAP  # Use your predefined mapping
    dataset = CustomDataset(
        X_pep, X_tcr, y, maxlen_pep, maxlen_tcr, amino_map, padding
    )

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Metadata for downstream use
    loader_metadata = {
        "pep_amino_idx": AMINO_MAP_REV,
        "tcr_amino_idx": AMINO_MAP_REV,
        "tensor_type": (
            torch.cuda.LongTensor if device == "cuda" else torch.LongTensor
        ),
        "pep_length": maxlen_pep or max(len(x) for x in X_pep),
        "tcr_length": maxlen_tcr or max(len(x) for x in X_tcr),
        "loader": data_loader,
    }

    return loader_metadata


# Example: Load embedding matrix
def load_embedding(filename):
    if filename is None or filename.lower() == "none":
        filename = "data/blosum/BLOSUM45"

    with open(filename, "r") as embedding_file:
        lines = embedding_file.readlines()[7:]

    embedding = [[float(x) for x in l.strip().split()[1:]] for l in lines]
    embedding.append(
        [0.0] * len(embedding[0])
    )  # Add zero embedding for padding
    return embedding


# Example: Splitting data
def load_data_split(x_pep, x_tcr, args):
    split_type = args.split_type
    idx_test_remove = idx_test = idx_train = None

    if split_type == "random":
        n_total = len(x_pep)
    elif split_type == "epitope":
        unique_peptides = np.unique(x_pep)
        n_total = len(unique_peptides)
    elif split_type == "tcr":
        unique_tcrs = np.unique(x_tcr)
        n_total = len(unique_tcrs)

    indexfile = re.sub(".csv", f"_{split_type}_data_shuffle.txt", args.infile)
    if os.path.exists(indexfile):
        idx_shuffled = np.loadtxt(indexfile, dtype=np.int32)
    else:
        idx_shuffled = np.arange(n_total)
        np.random.shuffle(idx_shuffled)
        np.savetxt(indexfile, idx_shuffled, fmt="%d")

    n_test = int(round(n_total / args.n_fold))
    test_fold_start_index = args.idx_test_fold * n_test
    test_fold_end_index = (args.idx_test_fold + 1) * n_test

    if split_type == "random":
        if args.idx_val_fold < 0:
            idx_test = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_train = list(set(idx_shuffled) - set(idx_test))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove = idx_shuffled[
                test_fold_start_index:test_fold_end_index
            ]
            idx_test = idx_shuffled[
                validation_fold_start_index:validation_fold_end_index
            ]
            idx_train = list(
                set(idx_shuffled) - set(idx_test) - set(idx_test_remove)
            )
    elif split_type == "epitope":
        unique_peptides = np.unique(x_pep)
        idx_test_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
        test_peptides = unique_peptides[idx_test_pep]
        idx_test = [i for i, pep in enumerate(x_pep) if pep in test_peptides]
        idx_train = list(set(range(len(x_pep))) - set(idx_test))

    return idx_train, idx_test, idx_test_remove
