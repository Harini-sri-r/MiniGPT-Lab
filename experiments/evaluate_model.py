from pathlib import Path

import torch

from evaluation.evaluate import evaluate_loss, load_trained_model, model_report, perplexity
from generation.generate import generate
from training.dataset import NextTokenDataset, split_token_stream


project_root = Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, configuration, _ = load_trained_model(
    project_root / "checkpoints" / "minigpt.pt", device
)
text = (project_root / "data" / "training.txt").read_text(encoding="utf-8")
_, validation_tokens = split_token_stream(tokenizer.encode(text))
validation_dataset = NextTokenDataset(validation_tokens, configuration["block_size"])
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=configuration["batch_size"], shuffle=False
)
loss = evaluate_loss(model, validation_loader, device)
prompt = "artificial intelligence"
input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
generated = generate(model, input_ids, max_new_tokens=60, temperature=0.8, top_k=8)
report = model_report(model, device)

print("MINIGPT EVALUATION\n------------------")
print("Parameters:", report["parameters"])
print("Vocabulary:", report["vocabulary_size"])
print("Validation Loss:", f"{loss:.4f}")
print("Perplexity:", f"{perplexity(loss):.4f}")
print("\nSample Prompt:")
print(prompt)
print("\nGenerated:")
print(tokenizer.decode(generated[0].tolist()))
