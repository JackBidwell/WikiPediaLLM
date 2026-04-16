import torch
from torch.utils.data import Dataset


class WikipediaDataset(Dataset):
    def __init__(self, tokenizer, block_size=256):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.data = tokenizer.encode(tokenizer.text)
        self.data = torch.tensor(self.data, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y