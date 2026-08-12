from pathlib import Path

import torch

from generation.generate import generate
from model.minigpt import MiniGPT
from tokenizer.tokenizer import SimpleTokenizer


project_root = Path(__file__).resolve().parents[1]
checkpoint = torch.load(
    project_root / "checkpoints" / "minigpt.pt", map_location="cpu", weights_only=False
)
configuration = checkpoint["configuration"]

# Recreate the same sorted character vocabulary saved during training.
tokenizer = SimpleTokenizer("".join(checkpoint["tokenizer_characters"]))
model = MiniGPT(
    vocab_size=len(tokenizer.characters),
    embedding_dim=configuration["embedding_dim"],
    num_heads=configuration["num_heads"],
    hidden_dim=configuration["hidden_dim"],
    num_layers=configuration["num_layers"],
    max_sequence_length=configuration["block_size"],
    dropout=configuration["dropout"],
)
model.load_state_dict(checkpoint["model_state_dict"])

prompt = "artificial intelligence"
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
output_ids = generate(model, input_ids, max_new_tokens=80, temperature=0.8, top_k=8)

print("Prompt:")
print(prompt)
print("\nGenerated text:")
print(tokenizer.decode(output_ids[0].tolist()))
print("\nNew tokens generated:")
print(output_ids.size(1) - input_ids.size(1))
