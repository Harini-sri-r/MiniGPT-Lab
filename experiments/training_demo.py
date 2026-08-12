from pathlib import Path

from model.minigpt import MiniGPT
from tokenizer.tokenizer import SimpleTokenizer
from training.dataset import create_dataloaders, split_token_stream
from training.train import get_device, save_checkpoint, train_model


configuration = {
    "batch_size": 16,
    "block_size": 32,
    "learning_rate": 3e-4,
    "num_epochs": 5,
    "embedding_dim": 32,
    "num_heads": 4,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.1,
}

project_root = Path(__file__).resolve().parents[1]
text = (project_root / "data" / "training.txt").read_text(encoding="utf-8")
tokenizer = SimpleTokenizer(text)
token_ids = tokenizer.encode(text)
train_tokens, validation_tokens = split_token_stream(token_ids)
train_loader, validation_loader = create_dataloaders(
    train_tokens,
    validation_tokens,
    block_size=configuration["block_size"],
    batch_size=configuration["batch_size"],
)
device = get_device()

model = MiniGPT(
    vocab_size=len(tokenizer.characters),
    embedding_dim=configuration["embedding_dim"],
    num_heads=configuration["num_heads"],
    hidden_dim=configuration["hidden_dim"],
    num_layers=configuration["num_layers"],
    max_sequence_length=configuration["block_size"],
    dropout=configuration["dropout"],
)

print("Device:", device)
print("Vocabulary size:", len(tokenizer.characters))
print("Vocabulary:", tokenizer.characters)
print("Total characters:", len(text))
print("Total tokens:", len(token_ids))
print("Training tokens:", len(train_tokens))
print("Validation tokens:", len(validation_tokens))

history, optimizer = train_model(
    model,
    train_loader,
    validation_loader,
    num_epochs=configuration["num_epochs"],
    learning_rate=configuration["learning_rate"],
    device=device,
)
save_checkpoint(
    project_root / "checkpoints" / "minigpt.pt",
    model,
    optimizer,
    configuration,
    tokenizer,
    history,
)

print("Initial training loss:", history[0]["train_loss"])
print("Final training loss:", history[-1]["train_loss"])
print("Initial validation loss:", history[0]["validation_loss"])
print("Final validation loss:", history[-1]["validation_loss"])
