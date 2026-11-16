import torch
from torch.utils.data import Dataset
from datasets import load_dataset


MINIMUM_ACCURACY = 0.01


def loader(transform_fn, task):
    dataset_name = "Salesforce/wikitext"
    config = "wikitext-2-raw-v1"
    seq_length = 128

    raw = load_dataset(dataset_name, config, split="train")
    text_all = "\n".join(raw["text"]).lower()
    ds = TextDatasetPreparation(text_all, seq_length)

    return (ds.vocab_size,), MINIMUM_ACCURACY, ds, ds


class TextDatasetPreparation(Dataset):
    """
    Sliding window of length `seq_length` with offset of 1 character
        X – indices of characters of form [S]
        y – indices of "next" character of form [S]
    """

    def __init__(self, txt: str, seq_length: int = 100):
        self.seq_length = seq_length
        self.chars = sorted(set(txt))
        self.vocab_size = len(self.chars)

        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.text_idx = [self.char2idx[c] for c in txt]

        self.n_samples = len(self.text_idx) - seq_length - 1
        if self.n_samples <= 0:
            raise ValueError("Text is too short for the given seq_length")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        if not 0 <= idx < self.n_samples:
            raise IndexError

        start = idx
        end = start + self.seq_length

        x_idx = self.text_idx[start:end]            # [S]
        y_idx = self.text_idx[start + 1:end + 1]    # [S]

        x = torch.tensor(x_idx, dtype=torch.long)
        y = torch.tensor(y_idx, dtype=torch.long)
        return x, y
