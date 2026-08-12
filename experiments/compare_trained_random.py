from pathlib import Path

import torch

from evaluation.evaluate import load_trained_model
from generation.generate import generate
from model.minigpt import MiniGPT


project_root = Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model, tokenizer, configuration, _ = load_trained_model(
    project_root / "checkpoints" / "minigpt.pt", device
)
random_model = MiniGPT(
    vocab_size=len(tokenizer.characters),
    embedding_dim=configuration["embedding_dim"],
    num_heads=configuration["num_heads"],
    hidden_dim=configuration["hidden_dim"],
    num_layers=configuration["num_layers"],
    max_sequence_length=configuration["block_size"],
    dropout=configuration["dropout"],
).to(device)
prompt = "artificial intelligence"
input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

for label, model in [("Random model", random_model), ("Trained model", trained_model)]:
    torch.manual_seed(0)
    output = generate(model, input_ids, max_new_tokens=60, temperature=0.8, top_k=8)
    print(f"\n{label}:\n{tokenizer.decode(output[0].tolist())}")
