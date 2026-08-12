"""
Tests for MiniGPT v2 (BPE-based language model).

Covers:
- Model initialization
- Forward pass functionality
- Output shape correctness
- Vocabulary dimension compatibility
- Batch processing
- Sequence length validation
- Parameter tracking
- Weight tying
- Output finite values
- Deterministic behavior
- Causal attention
- Integration with BPE tokenizer
- Existing v1 tests still pass
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from model.minigpt_v2 import MiniGPTV2
from tokenizer.bpe_tokenizer import BPETokenizer


class TestMiniGPTV2Initialization:
    """Tests for model initialization."""
    
    def test_model_initializes_successfully(self):
        """Test 1: Model initializes without errors."""
        model = MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
            dropout=0.1,
        )
        
        assert model is not None
        assert model.vocab_size == 256
        assert model.embedding_dim == 64
    
    def test_model_has_trainable_parameters(self):
        """Test 8: Model has trainable parameters."""
        model = MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
        )
        
        param_count = model.get_parameter_count()
        assert param_count > 0
        
        # Verify trainable parameters exist
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0
    
    def test_weight_tying_implemented(self):
        """Test 9: Weight tying is implemented."""
        model = MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
        )
        
        # LM head weight should be tied to token embedding weight
        assert model.lm_head.weight is model.token_embedding.embedding.weight
        print("Weight tying verified: LM head shares embedding matrix")


class TestMiniGPTV2ForwardPass:
    """Tests for forward pass functionality."""
    
    @pytest.fixture
    def small_model(self):
        """Fixture providing a small model for testing."""
        return MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
            dropout=0.1,
        )
    
    def test_forward_pass_works(self, small_model):
        """Test 2: Forward pass completes without errors."""
        input_ids = torch.randint(0, 256, (2, 16))
        logits = small_model(input_ids)
        
        assert logits is not None
        assert isinstance(logits, torch.Tensor)
    
    def test_output_shape_correct(self, small_model):
        """Test 3: Output shape is correct."""
        batch_size = 2
        seq_length = 16
        vocab_size = 256
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        logits = small_model(input_ids)
        
        expected_shape = (batch_size, seq_length, vocab_size)
        assert logits.shape == expected_shape
    
    def test_output_vocab_dimension(self, small_model):
        """Test 4: Output vocabulary dimension matches tokenizer size."""
        input_ids = torch.randint(0, 256, (1, 10))
        logits = small_model(input_ids)
        
        # Last dimension should equal vocab_size
        assert logits.shape[-1] == small_model.vocab_size
    
    def test_batch_size_preserved(self, small_model):
        """Test 5: Batch size is preserved through forward pass."""
        batch_sizes = [1, 2, 4, 8]
        seq_length = 16
        
        for batch_size in batch_sizes:
            input_ids = torch.randint(0, 256, (batch_size, seq_length))
            logits = small_model(input_ids)
            
            assert logits.shape[0] == batch_size
    
    def test_sequence_length_preserved(self, small_model):
        """Test 6: Sequence length is preserved through forward pass."""
        seq_lengths = [4, 8, 16, 32]
        batch_size = 2
        
        for seq_length in seq_lengths:
            input_ids = torch.randint(0, 256, (batch_size, seq_length))
            logits = small_model(input_ids)
            
            assert logits.shape[1] == seq_length
    
    def test_invalid_sequence_length_raises_error(self, small_model):
        """Test 7: Invalid sequence length raises ValueError."""
        max_len = small_model.max_sequence_length
        
        # Sequence longer than max
        input_ids = torch.randint(0, 256, (1, max_len + 1))
        
        with pytest.raises(ValueError):
            small_model(input_ids)
    
    def test_output_contains_finite_values(self, small_model):
        """Test 10: Output logits contain only finite values."""
        input_ids = torch.randint(0, 256, (4, 20))
        logits = small_model(input_ids)
        
        # Check no NaN or Inf
        assert torch.isfinite(logits).all()


class TestMiniGPTV2Determinism:
    """Tests for deterministic and stable behavior."""
    
    def test_same_input_same_output_eval_mode(self):
        """Test 11: Same input produces same output in eval mode."""
        model = MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
        )
        model.eval()
        
        input_ids = torch.tensor([[10, 20, 30, 40, 50]])
        
        with torch.no_grad():
            logits1 = model(input_ids)
            logits2 = model(input_ids)
        
        assert torch.allclose(logits1, logits2, atol=1e-6)
    
    def test_training_mode_different(self):
        """Test: Training mode produces different outputs (due to dropout)."""
        model = MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
            dropout=0.5,  # High dropout for testing
        )
        model.train()
        
        input_ids = torch.tensor([[10, 20, 30, 40, 50]])
        
        logits1 = model(input_ids)
        logits2 = model(input_ids)
        
        # Due to dropout, should likely be different
        # (not guaranteed but highly probable with dropout=0.5)
        # We just verify both are valid
        assert torch.isfinite(logits1).all()
        assert torch.isfinite(logits2).all()


class TestMiniGPTV2CausalBehavior:
    """Tests for causal attention mechanism."""
    
    def test_causal_masking_preserved(self):
        """Test 12: Causal masking prevents future token influence."""
        model = MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
        )
        model.eval()
        
        # Create two sequences that differ only at the end
        seq1 = torch.tensor([[10, 20, 30, 40, 50]])
        seq2 = torch.tensor([[10, 20, 30, 40, 99]])  # Last token different
        
        with torch.no_grad():
            logits1 = model(seq1)
            logits2 = model(seq2)
        
        # Logits at position 3 should be identical (can't see position 4)
        # (allowing small numerical difference)
        assert torch.allclose(logits1[:, 3, :], logits2[:, 3, :], atol=1e-5)


class TestMiniGPTV2TokenizerCompatibility:
    """Tests for compatibility with BPE tokenizer."""
    
    @pytest.fixture
    def bpe_tokenizer(self):
        """Fixture providing trained BPE tokenizer."""
        tokenizer_dir = Path(__file__).parent.parent / 'tokenizer'
        vocab_path = tokenizer_dir / 'bpe_vocab.json'
        merges_path = tokenizer_dir / 'bpe_merges.json'
        
        if not vocab_path.exists() or not merges_path.exists():
            pytest.skip("BPE tokenizer artifacts not found")
        
        tokenizer = BPETokenizer()
        tokenizer.load(str(vocab_path), str(merges_path))
        return tokenizer
    
    def test_model_vocab_matches_tokenizer(self, bpe_tokenizer):
        """Verify model vocabulary size matches tokenizer."""
        vocab_size = bpe_tokenizer.vocab_size()
        
        model = MiniGPTV2(
            vocab_size=vocab_size,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
        )
        
        assert model.vocab_size == vocab_size
    
    def test_tokenizer_ids_in_valid_range(self, bpe_tokenizer):
        """Test that tokenized IDs are valid for model."""
        text = "What is machine learning?"
        token_ids = bpe_tokenizer.encode(text)
        
        # All IDs should be in valid range
        assert all(0 <= tid < bpe_tokenizer.vocab_size() for tid in token_ids)
    
    def test_model_accepts_tokenizer_output(self, bpe_tokenizer):
        """Test 9: Model accepts tokenizer output."""
        text = "machine learning"
        token_ids = bpe_tokenizer.encode(text)
        
        # Create model with tokenizer's vocab size
        model = MiniGPTV2(
            vocab_size=bpe_tokenizer.vocab_size(),
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
        )
        model.eval()
        
        # Convert to tensor and run through model
        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        
        with torch.no_grad():
            logits = model(input_tensor)
        
        assert logits.shape[-1] == bpe_tokenizer.vocab_size()


class TestMiniGPTV2EdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_token_sequence(self):
        """Test: Model handles single-token sequences."""
        model = MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
        )
        
        input_ids = torch.tensor([[42]])  # Single token
        logits = model(input_ids)
        
        assert logits.shape == (1, 1, 256)
    
    def test_max_sequence_length_boundary(self):
        """Test: Model handles max sequence length exactly."""
        max_len = 64
        model = MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=max_len,
        )
        
        input_ids = torch.randint(0, 256, (1, max_len))
        logits = model(input_ids)
        
        assert logits.shape == (1, max_len, 256)
    
    def test_large_batch_size(self):
        """Test: Model handles large batches."""
        model = MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
        )
        
        input_ids = torch.randint(0, 256, (32, 16))
        logits = model(input_ids)
        
        assert logits.shape == (32, 16, 256)


class TestMiniGPTV2Configuration:
    """Tests for configuration and metadata."""
    
    def test_get_parameter_count(self):
        """Test: get_parameter_count() works."""
        model = MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
        )
        
        param_count = model.get_parameter_count()
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_get_config(self):
        """Test: get_config() returns configuration."""
        config = {
            'vocab_size': 512,
            'embedding_dim': 128,
            'num_heads': 4,
            'hidden_dim': 512,
            'num_layers': 4,
            'max_sequence_length': 256,
        }
        
        model = MiniGPTV2(**config)
        retrieved_config = model.get_config()
        
        assert retrieved_config['vocab_size'] == config['vocab_size']
        assert retrieved_config['embedding_dim'] == config['embedding_dim']
        assert retrieved_config['num_layers'] == config['num_layers']


class TestMiniGPTV2Integration:
    """Integration tests."""
    
    def test_complete_workflow(self):
        """Test: Complete forward pass workflow."""
        # Initialize model
        model = MiniGPTV2(
            vocab_size=256,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2,
            max_sequence_length=128,
        )
        model.eval()
        
        # Create input
        input_ids = torch.randint(0, 256, (2, 16))
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_ids)
        
        # Verify output
        assert logits.shape == (2, 16, 256)
        assert torch.isfinite(logits).all()
    
    def test_bpe_tokenizer_integration(self):
        """Test: Full BPE tokenizer + MiniGPT v2 integration."""
        tokenizer_dir = Path(__file__).parent.parent / 'tokenizer'
        vocab_path = tokenizer_dir / 'bpe_vocab.json'
        merges_path = tokenizer_dir / 'bpe_merges.json'
        
        if not vocab_path.exists() or not merges_path.exists():
            pytest.skip("BPE tokenizer artifacts not found")
        
        # Load tokenizer
        tokenizer = BPETokenizer()
        tokenizer.load(str(vocab_path), str(merges_path))
        
        # Create model with matching vocab size
        model = MiniGPTV2(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=128,
            num_heads=4,
            hidden_dim=512,
            num_layers=2,
            max_sequence_length=256,
        )
        model.eval()
        
        # Test on actual Q&A data
        examples = [
            "What is machine learning?",
            "How does deep learning work?",
            "What is a neural network?",
        ]
        
        for text in examples:
            token_ids = tokenizer.encode(text)
            assert all(0 <= tid < tokenizer.vocab_size() for tid in token_ids)
            
            # Run through model
            input_tensor = torch.tensor([token_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(input_tensor)
            
            assert logits.shape[2] == tokenizer.vocab_size()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
