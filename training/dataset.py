import torch
from torch.utils.data import DataLoader, Dataset


class NextTokenDataset(Dataset):
    """Create fixed-length input/target windows from one token stream."""

    def __init__(self, token_ids, block_size):
        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

        if self.token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional token stream.")
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")
        if len(self.token_ids) <= block_size:
            raise ValueError("token stream must contain more tokens than block_size.")

    def __len__(self):
        # Each sample needs block_size input tokens and one following target.
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index):
        x = self.token_ids[index : index + self.block_size]
        y = self.token_ids[index + 1 : index + self.block_size + 1]
        return x, y


def split_token_stream(token_ids, train_fraction=0.9):
    """Split a stream once, preserving text order and preventing leakage."""
    tokens = torch.as_tensor(token_ids, dtype=torch.long)
    split_index = int(len(tokens) * train_fraction)

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if split_index == 0 or split_index == len(tokens):
        raise ValueError("split must leave tokens for both training and validation.")

    return tokens[:split_index], tokens[split_index:]


def create_dataloaders(train_tokens, validation_tokens, block_size, batch_size):
    """Build CPU-friendly loaders for independent training and validation sets."""
    train_dataset = NextTokenDataset(train_tokens, block_size)
    validation_dataset = NextTokenDataset(validation_tokens, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader
