from pathlib import Path

import torch

from evaluation.evaluate import load_trained_model
from generation.generate import generate


project_root = Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, _, _ = load_trained_model(project_root / "checkpoints" / "minigpt.pt", device)
prompt = "artificial intelligence"
input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
settings = [
    ("Greedy", {"do_sample": False}),
    ("Temperature 0.5", {"temperature": 0.5}),
    ("Temperature 1.0", {"temperature": 1.0}),
    ("Temperature 1.5", {"temperature": 1.5}),
    ("Top-k sampling (k=8)", {"temperature": 1.0, "top_k": 8}),
]

for seed, (label, options) in enumerate(settings):
    torch.manual_seed(seed)
    output = generate(model, input_ids, max_new_tokens=60, **options)
    print(f"\n{label}:\n{tokenizer.decode(output[0].tolist())}")
