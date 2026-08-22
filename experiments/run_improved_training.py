"""
MiniGPT v2 Phase 6 Training Pipeline Script

Executes:
1. Overfitting sanity check: trains a tiny model on 3 Q&A examples for 30 steps
   to verify optimizer, scheduler, and loss backward updates work correctly.
2. Improved training run: trains MiniGPT v2 on the expanded 510-example dataset
   (qa_training_v2.txt) for 20 epochs with linear warmup, cosine decay, and gradient clipping.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

from tokenizer.bpe_tokenizer import BPETokenizer
from model.minigpt_v2 import MiniGPTV2
from training.qa_dataset_v2 import (
    load_qa_examples,
    split_qa_examples,
    QADatasetV2,
    create_v2_dataloaders,
)
from training.train_v2 import get_device, calculate_loss
from training.train_v2_improved import train_v2_improved_model


def run_overfit_test(tokenizer: BPETokenizer, device: torch.device):
    """Run a tiny overfit test on 3 examples for 30 steps."""
    print("=" * 70)
    print("STEP 1: PHASE 6 OVERFITTING SANITY CHECK")
    print("=" * 70)
    
    data_path = Path(__file__).parent.parent / "data" / "qa_training_v2.txt"
    all_examples = load_qa_examples(str(data_path))
    tiny_examples = all_examples[:3]
    
    dataset = QADatasetV2(tiny_examples, tokenizer, max_sequence_length=128)
    loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)
    
    model = MiniGPTV2(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=64,
        num_heads=4,
        hidden_dim=256,
        num_layers=2,
        max_sequence_length=128,
        dropout=0.0,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    input_ids, targets = next(iter(loader))
    input_ids, targets = input_ids.to(device), targets.to(device)
    
    # Measure initial loss
    with torch.no_grad():
        _, initial_loss = calculate_loss(model, input_ids, targets, criterion)
    initial_loss_val = initial_loss.item()
    print(f"Initial loss on 3 examples: {initial_loss_val:.4f}")
    
    # Train for 30 steps
    for _ in range(30):
        optimizer.zero_grad()
        _, loss = calculate_loss(model, input_ids, targets, criterion)
        loss.backward()
        optimizer.step()
        
    final_loss_val = loss.item()
    print(f"Final loss after 30 steps:   {final_loss_val:.4f}")
    print(f"Loss reduction:              {initial_loss_val - final_loss_val:.4f}")
    assert final_loss_val < initial_loss_val
    print("[OK] Overfitting sanity check PASSED.")


def run_improved_training():
    """Run full 20-epoch training on the expanded dataset."""
    print("\n" + "=" * 70)
    print("STEP 2: FULL IMPROVED TRAINING (20 EPOCHS)")
    print("=" * 70)
    
    device = get_device()
    
    # Load BPE Tokenizer
    tokenizer_dir = Path(__file__).parent.parent / "tokenizer"
    vocab_path = tokenizer_dir / "bpe_vocab.json"
    merges_path = tokenizer_dir / "bpe_merges.json"
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(vocab_path), str(merges_path))
    vocab_size = tokenizer.vocab_size()
    
    # Load Q&A dataset
    data_path = Path(__file__).parent.parent / "data" / "qa_training_v2.txt"
    all_examples = load_qa_examples(str(data_path))
    
    # Split examples 80% / 10% / 10% deterministically
    train_ex, val_ex, test_ex = split_qa_examples(
        all_examples, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    
    # Create MiniGPT v2 model (keep same sizes to verify training/data improvement)
    config = {
        "vocab_size": vocab_size,
        "embedding_dim": 128,
        "num_heads": 4,
        "hidden_dim": 512,
        "num_layers": 4,
        "max_sequence_length": 256,
        "dropout": 0.1,
    }
    
    model = MiniGPTV2(**config)
    
    print("\nTRAINING DETAILS:")
    print(f"  Device:               {device}")
    print(f"  Vocabulary size:      {vocab_size}")
    print(f"  Training examples:    {len(train_ex)}")
    print(f"  Validation examples:  {len(val_ex)}")
    print(f"  Test examples:        {len(test_ex)}")
    print(f"  Total parameters:     {model.get_parameter_count():,}")
    
    # DataLoaders
    train_loader, val_loader, test_loader = create_v2_dataloaders(
        train_ex, val_ex, test_ex, tokenizer, max_sequence_length=256, batch_size=16
    )
    
    # Run improved training
    history, best_path = train_v2_improved_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        learning_rate=3e-4,
        warmup_epochs=2,
        device=device,
        checkpoint_dir="checkpoints/v2/improved",
        pad_token_id=0,
        tokenizer_info={"vocab_size": vocab_size},
        max_grad_norm=1.0,
        seed=42,
    )
    
    print("\nIMPROVED TRAINING SUMMARY:")
    print(f"  Best checkpoint path:   {best_path}")
    print(f"  Initial training loss:  {history[0]['train_loss']:.4f}")
    print(f"  Final training loss:    {history[-1]['train_loss']:.4f}")
    print(f"  Initial validation loss: {history[0]['val_loss']:.4f}")
    print(f"  Final validation loss:   {history[-1]['val_loss']:.4f}")
    print(f"  Final Val Perplexity:   {history[-1]['val_perplexity']:.2f}")


def main():
    tokenizer_dir = Path(__file__).parent.parent / "tokenizer"
    vocab_path = tokenizer_dir / "bpe_vocab.json"
    merges_path = tokenizer_dir / "bpe_merges.json"
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(vocab_path), str(merges_path))
    
    device = get_device()
    
    # 1. Run overfitting check
    run_overfit_test(tokenizer, device)
    
    # 2. Run full improved training
    run_improved_training()


if __name__ == "__main__":
    main()
