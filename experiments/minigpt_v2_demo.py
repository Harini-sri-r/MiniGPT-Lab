"""
MiniGPT v2 Demo — BPE-Based Model Verification

Demonstrates:
1. Loading the trained BPE tokenizer artifacts.
2. Initializing the MiniGPT v2 Transformer model.
3. Encoding example input text ("What is machine learning?") with BPE.
4. Converting token IDs to a PyTorch tensor with a batch dimension.
5. Running a forward pass through MiniGPT v2 to generate logits.
6. Displaying model parameter breakdown and shape verifications.

IMPORTANT NOTE:
The MiniGPT v2 model is randomly initialized at this stage.
It cannot answer questions yet; learning happens in V2 Phase 4.
"""

from pathlib import Path
import torch

from model.minigpt_v2 import MiniGPTV2
from tokenizer.bpe_tokenizer import BPETokenizer


def main():
    """Run the MiniGPT v2 demo."""
    
    # 1. Load BPE Tokenizer Artifacts
    print("=" * 70)
    print("MINIGPT V2 PHASE 3: BPE-BASED MODEL DEMO")
    print("=" * 70)
    
    tokenizer_dir = Path(__file__).parent.parent / "tokenizer"
    vocab_path = tokenizer_dir / "bpe_vocab.json"
    merges_path = tokenizer_dir / "bpe_merges.json"
    
    print("\n1. LOADING BPE TOKENIZER ARTIFACTS")
    print(f"   Vocab path:  {vocab_path}")
    print(f"   Merges path: {merges_path}")
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(vocab_path), str(merges_path))
    vocab_size = tokenizer.vocab_size()
    print(f"   BPE Vocabulary Size: {vocab_size}")

    # 2. Create MiniGPT v2 Model Configuration
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
    model.eval()
    
    total_params = model.get_parameter_count()

    # 3. Model Parameter Report
    print("\n2. MODEL PARAMETER REPORT")
    print(f"   Vocabulary size:            {config['vocab_size']}")
    print(f"   Embedding dimension:        {config['embedding_dim']}")
    print(f"   Number of heads:            {config['num_heads']}")
    print(f"   Hidden dimension:           {config['hidden_dim']}")
    print(f"   Number of layers:           {config['num_layers']}")
    print(f"   Maximum sequence length:    {config['max_sequence_length']}")
    print(f"   Dropout:                    {config['dropout']}")
    print(f"   Total trainable parameters: {total_params:,}")

    # 4. Tokenization & Forward Pass Demo
    text = "What is machine learning?"
    print("\n3. EXAMPLE FORWARD PASS")
    
    token_ids = tokenizer.encode(text)
    bpe_tokens = [tokenizer.id_to_token.get(tid, "<UNK>") for tid in token_ids]
    
    print(f"   Original text:   {text}")
    print(f"   BPE tokens:      {bpe_tokens}")
    print(f"   Token IDs:       {token_ids}")
    
    # Convert token IDs to PyTorch tensor with batch dimension (batch_size=1, sequence_length=6)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    print(f"   Input shape:     {tuple(input_ids.shape)}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"   Logits shape:    {tuple(logits.shape)}")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Parameter count: {total_params:,}")

    # Verification assertions
    batch_size, seq_len, vocab_dim = logits.shape
    assert batch_size == 1, f"Expected batch size 1, got {batch_size}"
    assert seq_len == len(token_ids), f"Expected sequence length {len(token_ids)}, got {seq_len}"
    assert vocab_dim == vocab_size, f"Expected vocab dimension {vocab_size}, got {vocab_dim}"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values"
    
    print("\n[OK] Single-sample forward pass completed successfully.")

    # 5. Batch Processing Verification
    additional_texts = [
        "What is machine learning?",
        "What is a Transformer?",
        "How does artificial intelligence work?",
    ]
    batch_encoded = [tokenizer.encode(t) for t in additional_texts]
    max_len = max(len(ids) for ids in batch_encoded)
    
    # Pad sequences to max length in batch
    padded_batch = [ids + [0] * (max_len - len(ids)) for ids in batch_encoded]
    batch_tensor = torch.tensor(padded_batch, dtype=torch.long)
    
    with torch.no_grad():
        batch_logits = model(batch_tensor)
        
    print("\n4. BATCH PROCESSING VERIFICATION")
    print(f"   Batch input shape:  {tuple(batch_tensor.shape)}")
    print(f"   Batch logits shape: {tuple(batch_logits.shape)}")
    assert batch_logits.shape == (len(additional_texts), max_len, vocab_size)
    print("[OK] Batch processing shape verification passed.")

    print("\n" + "=" * 70)
    print("STATUS: MiniGPT v2 Phase 3 model & tokenizer integration verified.")
    print("Note: Weights are randomly initialized. Training will occur in Phase 4.")
    print("=" * 70)


if __name__ == "__main__":
    main()
