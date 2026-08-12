from pathlib import Path

import torch

from model.minigpt import MiniGPT
from tokenizer.tokenizer import SimpleTokenizer


def load_minigpt(checkpoint_path, device=None):
    """Load a trained MiniGPT and the exact character vocabulary it used."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["configuration"]
    saved_characters = checkpoint["tokenizer_characters"]

    # Training used SimpleTokenizer's sorted character vocabulary. Rebuilding
    # from exactly those saved characters recreates the same token-ID mapping.
    tokenizer = SimpleTokenizer("".join(saved_characters))
    if tokenizer.characters != saved_characters:
        raise ValueError("Checkpoint tokenizer vocabulary could not be reconstructed.")

    model = MiniGPT(
        vocab_size=len(tokenizer.characters),
        embedding_dim=configuration["embedding_dim"],
        num_heads=configuration["num_heads"],
        hidden_dim=configuration["hidden_dim"],
        num_layers=configuration["num_layers"],
        max_sequence_length=configuration["block_size"],
        dropout=configuration["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer
