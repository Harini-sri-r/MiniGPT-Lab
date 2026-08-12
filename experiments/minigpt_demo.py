import torch

from model.minigpt import MiniGPT
from tokenizer.tokenizer import SimpleTokenizer


text = "hello world"
tokenizer = SimpleTokenizer(text)
input_text = "hello worl"
token_ids = tokenizer.encode(input_text)
input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

model = MiniGPT(
    vocab_size=len(tokenizer.characters),
    embedding_dim=64,
    num_heads=4,
    hidden_dim=256,
    num_layers=4,
    max_sequence_length=128,
    dropout=0.1,
)
logits = model(input_ids)

last_logits = logits[:, -1, :]
probabilities = torch.softmax(last_logits, dim=-1)
predicted_token_id = torch.argmax(probabilities, dim=-1).item()
predicted_token = tokenizer.decode([predicted_token_id])
trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Input text:")
print(input_text)
print("\nToken IDs:")
print(token_ids)
print("\nInput shape:")
print(input_ids.shape)
print("\nLogits shape:")
print(logits.shape)
print("\nVocabulary size:")
print(model.vocab_size)
print("\nEmbedding dimension:")
print(model.embedding_dim)
print("\nNumber of Transformer layers:")
print(model.num_layers)
print("\nNumber of attention heads:")
print(model.num_heads)
print("\nTotal trainable parameters:")
print(trainable_parameters)
print("\nPredicted next token:")
print(predicted_token)
print("\nNote: this prediction is meaningless until the model is trained.")
