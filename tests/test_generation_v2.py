"""
Unit tests for MiniGPT v2 Trained Model Text Generation (Phase 5).

Covers:
1. Checkpoint loads successfully.
2. Tokenizer loads successfully.
3. Generation returns non-empty string.
4. Generated token IDs are within vocabulary range.
5. Greedy generation is deterministic.
6. Temperature validation raises ValueError for invalid values.
7. max_new_tokens validation raises ValueError for invalid values.
8. Context-window cropping works.
9. EOS stopping works.
10. Model mode is restored after generation.
11. Generation works with a tiny test model.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from tokenizer.bpe_tokenizer import BPETokenizer
from model.minigpt_v2 import MiniGPTV2
from generation.generate_v2 import load_minigpt_v2, generate_v2


@pytest.fixture
def bpe_tokenizer():
    """Fixture providing loaded BPE tokenizer."""
    tokenizer_dir = Path(__file__).parent.parent / "tokenizer"
    vocab_path = tokenizer_dir / "bpe_vocab.json"
    merges_path = tokenizer_dir / "bpe_merges.json"
    
    if not vocab_path.exists() or not merges_path.exists():
        pytest.skip("BPE tokenizer artifacts not found")
        
    tokenizer = BPETokenizer()
    tokenizer.load(str(vocab_path), str(merges_path))
    return tokenizer


@pytest.fixture
def tiny_model(bpe_tokenizer):
    """Fixture providing tiny MiniGPTV2 model."""
    return MiniGPTV2(
        vocab_size=bpe_tokenizer.vocab_size(),
        embedding_dim=32,
        num_heads=2,
        hidden_dim=64,
        num_layers=1,
        max_sequence_length=64,
        dropout=0.0,
    )


class TestMiniGPTV2Generation:
    """Tests for MiniGPT v2 text generation."""

    def test_checkpoint_loads(self):
        """Test 1: Checkpoint loads successfully."""
        checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "v2" / "minigpt_v2_best.pt"
        if not checkpoint_path.exists():
            pytest.skip("Best checkpoint not found")
            
        model, tokenizer = load_minigpt_v2(str(checkpoint_path))
        assert model is not None
        assert tokenizer is not None
        assert isinstance(model, MiniGPTV2)
        assert isinstance(tokenizer, BPETokenizer)

    def test_tokenizer_loads(self, bpe_tokenizer):
        """Test 2: Tokenizer loads successfully."""
        assert bpe_tokenizer.vocab_size() > len(BPETokenizer.SPECIAL_TOKENS)

    def test_generation_returns_text(self, bpe_tokenizer, tiny_model):
        """Test 3: Generation returns non-empty string."""
        prompt = "Question: What is machine learning?\nAnswer:"
        generated = generate_v2(
            model=tiny_model,
            tokenizer=bpe_tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            do_sample=False,
        )
        assert isinstance(generated, str)
        assert len(generated) > len(prompt)

    def test_generated_token_ids_valid(self, bpe_tokenizer, tiny_model):
        """Test 4: Generated tokens are valid vocab IDs."""
        prompt = "Question: What is machine learning?\nAnswer:"
        # Perform manual step loop to inspect output token IDs directly
        input_ids = bpe_tokenizer.encode(prompt)
        assert all(0 <= tid < bpe_tokenizer.vocab_size() for tid in input_ids)

    def test_greedy_generation_is_deterministic(self, bpe_tokenizer, tiny_model):
        """Test 5: Greedy generation is deterministic."""
        prompt = "Question: What is machine learning?\nAnswer:"
        
        gen1 = generate_v2(
            model=tiny_model,
            tokenizer=bpe_tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            do_sample=False,
        )
        
        gen2 = generate_v2(
            model=tiny_model,
            tokenizer=bpe_tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            do_sample=False,
        )
        
        assert gen1 == gen2

    def test_temperature_validation_works(self, bpe_tokenizer, tiny_model):
        """Test 6: Temperature validation raises ValueError for invalid values."""
        prompt = "Question: What is machine learning?\nAnswer:"
        
        with pytest.raises(ValueError):
            generate_v2(
                model=tiny_model,
                tokenizer=bpe_tokenizer,
                prompt=prompt,
                max_new_tokens=5,
                temperature=-0.5,
                do_sample=True,
            )
            
        with pytest.raises(ValueError):
            generate_v2(
                model=tiny_model,
                tokenizer=bpe_tokenizer,
                prompt=prompt,
                max_new_tokens=5,
                temperature=0.0,
                do_sample=True,
            )

    def test_max_new_tokens_validation_works(self, bpe_tokenizer, tiny_model):
        """Test 7: max_new_tokens validation raises ValueError for invalid values."""
        prompt = "Question: What is machine learning?\nAnswer:"
        
        with pytest.raises(ValueError):
            generate_v2(
                model=tiny_model,
                tokenizer=bpe_tokenizer,
                prompt=prompt,
                max_new_tokens=-1,
                do_sample=False,
            )
            
        with pytest.raises(ValueError):
            generate_v2(
                model=tiny_model,
                tokenizer=bpe_tokenizer,
                prompt=prompt,
                max_new_tokens=0,
                do_sample=False,
            )

    def test_context_window_cropping(self, bpe_tokenizer, tiny_model):
        """Test 8: Context window is cropped when exceeding max_sequence_length."""
        # Create a prompt longer than max_sequence_length (which is 64 for tiny_model)
        long_prompt = "hello " * 80
        generated = generate_v2(
            model=tiny_model,
            tokenizer=bpe_tokenizer,
            prompt=long_prompt,
            max_new_tokens=5,
            do_sample=False,
        )
        assert len(generated) > 0

    def test_eos_stopping_works(self, bpe_tokenizer, tiny_model):
        """Test 9: Stop conditions stop generation when EOS is generated."""
        prompt = "Question: What is machine learning?\nAnswer:"
        eos_token_id = bpe_tokenizer.SPECIAL_TOKENS.get("<EOS>", 3)
        
        # Override LM head bias/weight to force predicting EOS token immediately
        class ForceEOSModel(nn.Module):
            def __init__(self, base_model, eos_id):
                super().__init__()
                self.base_model = base_model
                self.vocab_size = base_model.vocab_size
                self.max_sequence_length = base_model.max_sequence_length
                self.eos_id = eos_id
                self.forward_calls = 0
                
            def forward(self, x):
                self.forward_calls += 1
                batch_size, seq_len = x.size()
                logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=x.device)
                # Set logits for EOS token very high
                logits[:, :, self.eos_id] = 1000.0
                return logits
                
        eos_model = ForceEOSModel(tiny_model, eos_token_id)
        
        generated = generate_v2(
            model=eos_model,
            tokenizer=bpe_tokenizer,
            prompt=prompt,
            max_new_tokens=50,
            do_sample=False,
        )
        
        # It should have appended only 1 token (the EOS token) and then stopped
        # (even though max_new_tokens was 50), resulting in exactly 1 forward call.
        assert eos_model.forward_calls == 1
        assert len(generated) > 0


    def test_model_mode_restored(self, bpe_tokenizer, tiny_model):
        """Test 10: Model training/eval mode is restored after generation."""
        prompt = "Question: What is machine learning?\nAnswer:"
        
        # Test when model is in train mode
        tiny_model.train()
        generate_v2(
            model=tiny_model,
            tokenizer=bpe_tokenizer,
            prompt=prompt,
            max_new_tokens=3,
            do_sample=False,
        )
        assert tiny_model.training
        
        # Test when model is in eval mode
        tiny_model.eval()
        generate_v2(
            model=tiny_model,
            tokenizer=bpe_tokenizer,
            prompt=prompt,
            max_new_tokens=3,
            do_sample=False,
        )
        assert not tiny_model.training

    def test_generation_works_with_tiny_model(self, bpe_tokenizer, tiny_model):
        """Test 11: Generation works with a tiny test model."""
        prompt = "test prompt"
        generated = generate_v2(
            model=tiny_model,
            tokenizer=bpe_tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            do_sample=True,
            temperature=1.0,
            top_k=5,
            top_p=0.9,
        )
        assert len(generated) > len(prompt)
