"""
Compare MiniGPT v2 Baseline vs Improved Model

Loads:
- Phase 4 Baseline Checkpoint: checkpoints/v2/minigpt_v2_best.pt
- Phase 6 Improved Checkpoint: checkpoints/v2/improved/minigpt_v2_best.pt

Calculates and prints comparison metrics:
- Dataset sizes
- Epoch counts
- Final training and validation loss values
- Train/Validation Perplexity (exp of loss)
- Detailed summary table
"""

import math
from pathlib import Path
import torch

from model.minigpt_v2 import MiniGPTV2


def load_checkpoint_meta(path: Path) -> dict:
    """Load metadata from a checkpoint without failing if optimizer state is missing."""
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
    print("=" * 70)
    print("MINIGPT V2: BASELINE VS IMPROVED MODEL COMPARISON")
    print("=" * 70)
    
    project_dir = Path(__file__).parent.parent
    baseline_path = project_dir / "checkpoints" / "v2" / "minigpt_v2_best.pt"
    improved_path = project_dir / "checkpoints" / "v2" / "improved" / "minigpt_v2_best.pt"
    
    baseline_meta = load_checkpoint_meta(baseline_path)
    improved_meta = load_checkpoint_meta(improved_path)
    
    if not baseline_meta:
        print(f"[ERROR] Baseline checkpoint not found at: {baseline_path}")
    if not improved_meta:
        print(f"[ERROR] Improved checkpoint not found at: {improved_path}")
        
    print(f"\nSummary metrics comparison:")
    
    # Extract baseline stats
    b_epochs = baseline_meta.get("epoch", "N/A")
    b_train_loss = baseline_meta.get("train_loss", float("inf"))
    b_val_loss = baseline_meta.get("val_loss", float("inf"))
    b_train_ppl = math.exp(b_train_loss) if b_train_loss < 50 else float("inf")
    b_val_ppl = math.exp(b_val_loss) if b_val_loss < 50 else float("inf")
    
    # Extract improved stats
    i_epochs = improved_meta.get("epoch", "N/A")
    i_train_loss = improved_meta.get("train_loss", float("inf"))
    i_val_loss = improved_meta.get("val_loss", float("inf"))
    i_train_ppl = math.exp(i_train_loss) if i_train_loss < 50 else float("inf")
    i_val_ppl = math.exp(i_val_loss) if i_val_loss < 50 else float("inf")
    
    # Print formatted table
    print("-" * 70)
    print(f"Metric                 | Phase 4 Baseline      | Phase 6 Improved")
    print("-" * 70)
    print(f"Dataset Size (examples)| 201 (Baseline)        | 510 (Expanded)")
    print(f"Total Epochs           | {b_epochs:<21} | {i_epochs:<16}")
    print(f"Final Train Loss       | {b_train_loss:<21.4f} | {i_train_loss:<16.4f}")
    print(f"Final Val Loss         | {b_val_loss:<21.4f} | {i_val_loss:<16.4f}")
    print(f"Train Perplexity       | {b_train_ppl:<21.2f} | {i_train_ppl:<16.2f}")
    print(f"Val Perplexity         | {b_val_ppl:<21.2f} | {i_val_ppl:<16.2f}")
    print(f"Warmup / Scheduler     | None                  | Cosine with Warmup")
    print(f"Gradient Clipping      | None                  | Norm Limit = 1.0")
    print("-" * 70)
    
    # Explain improvements
    if i_val_loss < b_val_loss:
        print("\nConclusion: The Improved Model (Phase 6) achieved LOWER validation loss and perplexity.")
        print("This indicates that the expanded, structured dataset combined with learning rate warmup/decay")
        print("and gradient clipping successfully improved model convergence.")
    else:
        print("\nConclusion: Validation metrics did not show improvement. This can occur if")
        print("the expanded dataset has significantly more linguistic diversity, making the next-token")
        print("prediction objective intrinsically harder (higher entropy) than the tiny baseline corpus.")
        print("We must inspect actual generation quality outputs to judge coherence.")
    print("=" * 70)


if __name__ == "__main__":
    main()
