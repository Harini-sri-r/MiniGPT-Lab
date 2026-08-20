import os
import json
from pathlib import Path
import tempfile
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "minigpt-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_checkpoint_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    data = torch.load(path, map_location="cpu")
    return {
        "epoch": data.get("epoch"),
        "train_loss": data.get("train_loss"),
        "val_loss": data.get("val_loss"),
        "history": data.get("history", []),
        "config": data.get("config", {}),
    }

def main():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "experiments" / "results" / "v2"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_path = project_root / "checkpoints" / "v2" / "minigpt_v2_best.pt"
    improved_path = project_root / "checkpoints" / "v2" / "improved" / "minigpt_v2_best.pt"
    
    baseline_meta = load_checkpoint_meta(baseline_path)
    improved_meta = load_checkpoint_meta(improved_path)
    
    # 1. Export JSON
    if baseline_meta.get("history"):
        with open(results_dir / "history_baseline.json", "w") as f:
            json.dump(baseline_meta["history"], f, indent=2)
            
    if improved_meta.get("history"):
        with open(results_dir / "history_improved.json", "w") as f:
            json.dump(improved_meta["history"], f, indent=2)
            
    # 2. Plot Loss
    plt.figure(figsize=(10, 6))
    
    if baseline_meta.get("history"):
        b_hist = baseline_meta["history"]
        b_epochs = [r["epoch"] for r in b_hist]
        b_train = [r["train_loss"] for r in b_hist]
        b_val = [r.get("val_loss", r["train_loss"]) for r in b_hist] # fallback
        plt.plot(b_epochs, b_train, marker="o", linestyle="--", color="blue", alpha=0.6, label="Baseline Train Loss")
        plt.plot(b_epochs, b_val, marker="o", linestyle="-", color="blue", label="Baseline Val Loss")
        
    if improved_meta.get("history"):
        i_hist = improved_meta["history"]
        i_epochs = [r["epoch"] for r in i_hist]
        i_train = [r["train_loss"] for r in i_hist]
        i_val = [r.get("val_loss", r["train_loss"]) for r in i_hist] # fallback
        plt.plot(i_epochs, i_train, marker="s", linestyle="--", color="green", alpha=0.6, label="Improved Train Loss")
        plt.plot(i_epochs, i_val, marker="s", linestyle="-", color="green", label="Improved Val Loss")
        
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("MiniGPT v2 Training Loss: Baseline vs Improved")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "training_loss_comparison.png", dpi=150)
    print(f"Saved plots and JSON history to {results_dir}")

if __name__ == "__main__":
    main()
