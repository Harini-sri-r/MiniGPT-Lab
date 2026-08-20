"""
MiniGPT v2 Improved Training Pipeline

Implements advanced training techniques for Phase 6, including:
- AdamW Optimizer.
- Learning rate linear warmup and cosine decay scheduler.
- Configurable gradient clipping.
- Metric tracking (training/validation loss and perplexity).
- Isolated checkpoint saving under `checkpoints/v2/improved/`.
- Fully reproducible random seeds.
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from training.train_v2 import get_device, calculate_loss, validate_v2, save_checkpoint


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_warmup_cosine_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of steps for linear warmup
        total_steps: Total number of training steps
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Keep progress bounded between 0.0 and 1.0
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_v2_improved_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
    warmup_epochs: int = 2,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints/v2/improved",
    pad_token_id: int = 0,
    tokenizer_info: Optional[dict] = None,
    max_grad_norm: float = 1.0,
    seed: int = 42,
) -> Tuple[List[dict], str]:
    """
    Train MiniGPT v2 with warmup, cosine decay, and gradient clipping.
    
    Args:
        model: MiniGPTV2 model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Number of epochs
        learning_rate: Base target learning rate
        warmup_epochs: Number of epochs for linear warmup
        device: PyTorch device
        checkpoint_dir: Saving directory for checkpoints
        pad_token_id: PAD token ID to ignore in loss
        tokenizer_info: Tokenizer metadata dict
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (training_history, best_checkpoint_path)
    """
    set_seed(seed)
    
    if device is None:
        device = get_device()
        
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Calculate step counts
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    
    best_val_loss = float("inf")
    best_checkpoint_path = str(Path(checkpoint_dir) / "minigpt_v2_best.pt")
    history = []
    
    config = model.get_config() if hasattr(model, "get_config") else {}
    
    print(f"Total training steps: {total_steps} | Warmup steps: {warmup_steps}")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        # Log learning rate at the start of the epoch
        current_lr = optimizer.param_groups[0]["lr"]
        
        for input_ids, targets in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            _, loss = calculate_loss(model, input_ids, targets, criterion)
            loss.backward()
            
            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
        epoch_train_loss = total_train_loss / max(steps_per_epoch, 1)
        epoch_val_loss = validate_v2(model, val_loader, criterion, device)
        
        # Validation Perplexity = exp(val_loss)
        val_perplexity = math.exp(epoch_val_loss) if epoch_val_loss < 50 else float("inf")
        
        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_perplexity": val_perplexity,
            "learning_rate": current_lr,
        }
        history.append(epoch_record)
        
        print(
            f"Epoch {epoch:2d}/{num_epochs:2d} | "
            f"Train loss: {epoch_train_loss:.4f} | "
            f"Val loss: {epoch_val_loss:.4f} | "
            f"Val PPL: {val_perplexity:.2f} | "
            f"LR: {current_lr:.6f}"
        )
        
        # Save best checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            
            # Prepare state dict checkpointers
            path = Path(best_checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "tokenizer_info": tokenizer_info or {},
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "val_perplexity": val_perplexity,
                "history": history,
            }
            torch.save(checkpoint_data, path)
            
    return history, best_checkpoint_path
