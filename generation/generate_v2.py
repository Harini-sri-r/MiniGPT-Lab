"""
MiniGPT v2 Text Generation Module

Implements autoregressive text generation using the trained BPE-based MiniGPT v2 model.
Supports greedy decoding, temperature scaling, top-k sampling, and top-p (nucleus) sampling.

Key Features:
- Temperature scaling: controls random sampling diversity (lower = more deterministic).
- Greedy decoding: selects the token with the highest logit.
- Top-k sampling: restricts sampling to the top k highest probability tokens.
- Top-p sampling: restricts sampling to the minimal set of tokens whose cumulative
  probability exceeds threshold p.
- Stop conditions: stops when <EOS> (ID 3) is generated, max_new_tokens is reached,
  or maximum context length is reached.
"""

from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.minigpt_v2 import MiniGPTV2
from tokenizer.bpe_tokenizer import BPETokenizer


def load_minigpt_v2(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[MiniGPTV2, BPETokenizer]:
    """
    Load a trained MiniGPT v2 model and its BPE tokenizer.
    
    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: PyTorch device
        
    Returns:
        Tuple of (MiniGPTV2 model, BPETokenizer instance)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    # Reconstruct the model from configuration
    model = MiniGPTV2(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        max_sequence_length=config["max_sequence_length"],
        dropout=config.get("dropout", 0.1),
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load BPE tokenizer from the standard vocabulary directory
    tokenizer_dir = Path(__file__).parent.parent / "tokenizer"
    vocab_path = tokenizer_dir / "bpe_vocab.json"
    merges_path = tokenizer_dir / "bpe_merges.json"
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(vocab_path), str(merges_path))
    
    if tokenizer.vocab_size() != config["vocab_size"]:
        raise ValueError(
            f"Tokenizer vocabulary size ({tokenizer.vocab_size()}) "
            f"mismatches checkpoint vocabulary size ({config['vocab_size']})."
        )
        
    return model, tokenizer


def generate_v2(
    model: nn.Module,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
) -> str:
    """
    Generate text autoregressively using MiniGPT v2 given a text prompt.
    
    Args:
        model: Trained MiniGPTV2 model instance
        tokenizer: BPETokenizer instance
        prompt: Question text (formatted as Question: ...\nAnswer:)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Softmax temperature scale (> 0)
        top_k: Keep only top k tokens for sampling (>= 1)
        top_p: Cumulative probability threshold for nucleus sampling (0 < top_p <= 1)
        do_sample: If True, use sampling; if False, use greedy decoding
        
    Returns:
        Full decoded generated string (prompt + generated answer)
    """
    # 1. Parameter Validation
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero.")
    if do_sample:
        if temperature is not None and temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 when provided.")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be between 0 and 1 (exclusive of 0, inclusive of 1).")
        
    # Get model device
    device = next(model.parameters()).device
    
    # Save training state and set to eval
    was_training = model.training
    model.eval()
    
    # 2. Tokenize Input Prompt
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) == 0:
        raise ValueError("Prompt must contain at least one token.")
        
    generated = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_token_id = tokenizer.SPECIAL_TOKENS.get("<EOS>", 3)
    
    # 3. Autoregressive Loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window if sequence length exceeds maximum configuration
            if generated.size(1) >= model.max_sequence_length:
                context = generated[:, -model.max_sequence_length:]
            else:
                context = generated
                
            # Forward pass: extract logits of the final sequence position
            # Logits shape: (batch_size, context_len, vocab_size) -> final: (vocab_size,)
            logits = model(context)[:, -1, :]
            
            # Greedy decoding
            if not do_sample or temperature is None:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Apply temperature scaling
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    val, _ = torch.topk(logits, k, dim=-1)
                    min_val = val[:, -1:]
                    logits = logits.masked_fill(logits < min_val, float("-inf"))
                    
                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Identify elements to remove: cumulative prob exceeds top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift mask right to ensure at least 1 token is preserved
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    # Map masks back to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))
                    
                # Softmax and sampling
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                
            # Append generated token to input IDs
            generated = torch.cat((generated, next_token), dim=-1)
            
            # Check stop conditions
            if next_token.item() == eos_token_id:
                break
            if generated.size(1) >= model.max_sequence_length:
                break
                
    # Restore model training mode
    if was_training:
        model.train()
        
    # Decode full sequence
    decoded_text = tokenizer.decode(generated[0].tolist())
    return decoded_text
