from pathlib import Path
import os
import tempfile

# Use a file-only backend so this experiment also works on machines without a
# graphical Tk installation. Keep Matplotlib's cache outside the repository.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "minigpt-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from evaluation.evaluate import history_records


project_root = Path(__file__).resolve().parents[1]
checkpoint = torch.load(
    project_root / "checkpoints" / "minigpt.pt", map_location="cpu", weights_only=False
)
history = history_records(checkpoint["training_history"])
epochs = [record["epoch"] for record in history]
train_loss = [record["train_loss"] for record in history]
validation_loss = [record["validation_loss"] for record in history]
results_directory = project_root / "experiments" / "results"
results_directory.mkdir(exist_ok=True)

plt.plot(epochs, train_loss, marker="o", label="Training loss")
plt.plot(epochs, validation_loss, marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("MiniGPT training history")
plt.legend()
plt.tight_layout()
plt.savefig(results_directory / "training_loss.png")
print("Saved experiments/results/training_loss.png")
