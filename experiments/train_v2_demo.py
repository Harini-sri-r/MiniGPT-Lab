"""
MiniGPT v2 Phase 4 Training Demo

Demonstrates:
1. Overfitting sanity check on a tiny subset (3 Q&A examples) to verify
   that tokenizer -> dataset -> model -> loss -> backprop -> optimizer is connected correctly.
2. Complete 5-epoch training session on the full Q&A dataset with an 80/10/10 split.
3. Checkpoint saving and history tracking.

Note:
This initial model is trained for 5 epochs to verify pipeline operation.
It is not expected to produce fluent English responses; full multi-epoch training
and evaluation occur in later phases.
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
from training.train_v2 import (
    get_device,
    calculate_loss,
    train_v2_model,
    save_checkpoint,
    load_checkpoint,
)


def run_overfit_sanity_check(tokenizer: BPETokenizer, device: torch.device):
    """
    Perform a tiny overfit sanity check on 3 Q&A examples for 30 steps.
    
    Verifies that backprop and optimization reduce training loss significantly.
    """
    print("\n" + "=" * 70)
    print("STEP 1: OVERFITTING SANITY CHECK (3 EXAMPLES)")
    print("=" * 70)
    
    data_path = Path(__file__).parent.parent / "data" / "qa_training.txt"
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
    
    # Train for 30 iterations
    for step in range(1, 31):
        optimizer.zero_grad()
        _, loss = calculate_loss(model, input_ids, targets, criterion)
        loss.backward()
        optimizer.step()
        
    final_loss_val = loss.item()
    print(f"Final loss after 30 steps:   {final_loss_val:.4f}")
    
    loss_reduction = initial_loss_val - final_loss_val
    print(f"Loss reduction:              {loss_reduction:.4f}")
    
    assert final_loss_val < initial_loss_val, (
        f"Overfit test failed: initial loss ({initial_loss_val:.4f}) <= final loss ({final_loss_val:.4f})"
    )
    print("[OK] Tiny overfitting sanity check PASSED successfully.")


def run_full_training_demo():
    """Run full 5-epoch training session on full Q&A dataset."""
    print("\n" + "=" * 70)
    print("STEP 2: MINIGPT V2 Q&A TRAINING PIPELINE (5 EPOCHS)")
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
    data_path = Path(__file__).parent.parent / "data" / "qa_training.txt"
    all_examples = load_qa_examples(str(data_path))
    
    # Split examples 80% / 10% / 10% deterministically
    train_ex, val_ex, test_ex = split_qa_examples(
        all_examples, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    
    # Create MiniGPT v2 model
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
    param_count = model.get_parameter_count()
    
    print("\nTRAINING METADATA:")
    print(f"  Device:               {device}")
    print(f"  Training examples:    {len(train_ex)}")
    print(f"  Validation examples:  {len(val_ex)}")
    print(f"  Test examples:        {len(test_ex)}")
    print(f"  Vocabulary size:      {vocab_size}")
    print(f"  Parameter count:      {param_count:,}")
    
    # DataLoaders
    train_loader, val_loader, test_loader = create_v2_dataloaders(
        train_ex, val_ex, test_ex, tokenizer, max_sequence_length=256, batch_size=8
    )
    
    print("\nSTARTING 5-EPOCH TRAINING RUN:")
    history, checkpoint_path = train_v2_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=1e-3,
        device=device,
        checkpoint_dir="checkpoints/v2",
        pad_token_id=0,
        tokenizer_info={"vocab_size": vocab_size},
    )
    
    initial_train_loss = history[0]["train_loss"]
    final_train_loss = history[-1]["train_loss"]
    initial_val_loss = history[0]["val_loss"]
    final_val_loss = history[-1]["val_loss"]
    best_val_loss = min(record["val_loss"] for record in history)
    
    print("\nTRAINING SUMMARY:")
    print(f"  Initial training loss:    {initial_train_loss:.4f}")
    print(f"  Final training loss:      {final_train_loss:.4f}")
    print(f"  Initial validation loss:  {initial_val_loss:.4f}")
    print(f"  Final validation loss:    {final_val_loss:.4f}")
    print(f"  Best validation loss:     {best_val_loss:.4f}")
    print(f"  Checkpoint saved path:    {checkpoint_path}")
    
    # Verify checkpoint loading
    print("\nVERIFYING CHECKPOINT RESTORATION:")
    eval_model = MiniGPTV2(**config)
    loaded_ckpt = load_checkpoint(checkpoint_path, eval_model, device=device)
    print(f"  [OK] Restored checkpoint from epoch {loaded_ckpt['epoch']} with val loss {loaded_ckpt['val_loss']:.4f}")
    
    print("\n" + "=" * 70)
    print("STATUS: MiniGPT v2 Phase 4 Q&A Training Pipeline verified.")
    print("=" * 70)


def main():
    """Run training demo."""
    # 1. BPE Tokenizer
    tokenizer_dir = Path(__file__).parent.parent / "tokenizer"
    vocab_path = tokenizer_dir / "bpe_vocab.json"
    merges_path = tokenizer_dir / "bpe_merges.json"
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(vocab_path), str(merges_path))
    
    device = get_device()
    
    # Run Overfit Sanity Check first
    run_overfit_sanity_check(tokenizer, device)
    
    # Run Full Training Demo
    run_full_training_demo()


if __name__ == "__main__":
    main()
