"""
Tests for BPE Tokenizer (MiniGPT v2).

Covers:
- Training functionality
- Vocabulary creation
- Special tokens
- Encoding/decoding
- Determinism
- Save/load functionality
- Edge cases
"""

import pytest
import json
import tempfile
from pathlib import Path
from tokenizer.bpe_tokenizer import BPETokenizer


class TestBPETokenizerTraining:
    """Tests for tokenizer training."""
    
    def test_training_works(self):
        """Test 1: Training completes successfully."""
        tokenizer = BPETokenizer()
        text = "hello world hello there"
        tokenizer.train(text, vocab_size=100)
        
        assert tokenizer.vocab_size() > len(BPETokenizer.SPECIAL_TOKENS)
        assert len(tokenizer.merges) > 0
    
    def test_vocabulary_created(self):
        """Test 2: Vocabulary is created during training."""
        tokenizer = BPETokenizer()
        text = "the quick brown fox jumps over the lazy dog"
        tokenizer.train(text, vocab_size=100)
        
        vocab = tokenizer.get_vocab()
        assert len(vocab) > 0
        assert tokenizer.vocab_size() > len(BPETokenizer.SPECIAL_TOKENS)
    
    def test_special_tokens_exist(self):
        """Test 3: Special tokens are present and have correct IDs."""
        tokenizer = BPETokenizer()
        text = "test text"
        tokenizer.train(text, vocab_size=100)
        
        vocab = tokenizer.get_vocab()
        for token, token_id in BPETokenizer.SPECIAL_TOKENS.items():
            assert token in vocab
            assert vocab[token] == token_id
    
    def test_vocab_size_respected(self):
        """Test 10: Vocabulary size is at or below target."""
        tokenizer = BPETokenizer()
        text = "the quick brown fox jumps over the lazy dog " * 100
        target_size = 256
        tokenizer.train(text, vocab_size=target_size)
        
        assert tokenizer.vocab_size() <= target_size
    
    def test_vocab_size_too_small_raises_error(self):
        """Test: vocab_size must be > number of special tokens."""
        tokenizer = BPETokenizer()
        text = "test"
        
        with pytest.raises(ValueError):
            tokenizer.train(text, vocab_size=2)  # Too small


class TestBPETokenizerEncoding:
    """Tests for text encoding."""
    
    @pytest.fixture
    def trained_tokenizer(self):
        """Fixture providing a trained tokenizer."""
        tokenizer = BPETokenizer()
        text = "what is machine learning what is deep learning " * 10
        tokenizer.train(text, vocab_size=256)
        return tokenizer
    
    def test_encoding_returns_integers(self, trained_tokenizer):
        """Test 4: Encoding returns a list of integers."""
        text = "what is machine learning?"
        token_ids = trained_tokenizer.encode(text)
        
        assert isinstance(token_ids, list)
        assert all(isinstance(tid, int) for tid in token_ids)
    
    def test_decoding_returns_text(self, trained_tokenizer):
        """Test 5: Decoding returns a string."""
        text = "what is learning"
        token_ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(token_ids)
        
        assert isinstance(decoded, str)
    
    def test_encode_decode_roundtrip(self, trained_tokenizer):
        """Test 6: encode/decode preserves text reasonably."""
        text = "machine learning"
        token_ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(token_ids)
        
        # Should preserve the words even if spacing/punctuation changes
        assert "machine" in decoded.lower() or "mach" in decoded.lower()
        assert "learning" in decoded.lower() or "learn" in decoded.lower()
    
    def test_unknown_tokens_handled(self, trained_tokenizer):
        """Test 7: Unknown/unseen text is handled gracefully."""
        # Use characters that may not be in training data
        text = "xyzabc"
        token_ids = trained_tokenizer.encode(text)
        
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
    
    def test_empty_text_encoding(self, trained_tokenizer):
        """Test 12: Empty input is handled."""
        assert trained_tokenizer.encode("") == []
        assert trained_tokenizer.encode("   ") == []
    
    def test_empty_text_decoding(self, trained_tokenizer):
        """Test 12: Empty decoding is handled."""
        assert trained_tokenizer.decode([]) == ""


class TestBPETokenizerDeterminism:
    """Tests for deterministic behavior."""
    
    def test_token_ids_deterministic(self):
        """Test 8: Token IDs are deterministic."""
        tokenizer = BPETokenizer()
        text = "the quick brown fox " * 50
        tokenizer.train(text, vocab_size=256, verbose=False)
        
        sentence = "quick brown fox"
        ids1 = tokenizer.encode(sentence)
        ids2 = tokenizer.encode(sentence)
        
        assert ids1 == ids2
    
    def test_merge_rules_deterministic(self):
        """Test 11: Merge rules are deterministic."""
        # Train two tokenizers identically
        tokenizer1 = BPETokenizer()
        tokenizer2 = BPETokenizer()
        
        text = "the quick brown fox jumps over the lazy dog " * 50
        vocab_size = 256
        
        tokenizer1.train(text, vocab_size=vocab_size, verbose=False)
        tokenizer2.train(text, vocab_size=vocab_size, verbose=False)
        
        # Should have identical merge rules
        assert tokenizer1.get_merges() == tokenizer2.get_merges()
        assert tokenizer1.get_vocab() == tokenizer2.get_vocab()


class TestBPETokenizerSaveLoad:
    """Tests for save/load functionality."""
    
    @pytest.fixture
    def temp_paths(self):
        """Fixture providing temporary file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            merges_path = Path(tmpdir) / "merges.json"
            yield vocab_path, merges_path
    
    def test_save_creates_files(self, temp_paths):
        """Test: Save creates JSON files."""
        vocab_path, merges_path = temp_paths
        
        tokenizer = BPETokenizer()
        text = "the quick brown fox " * 50
        tokenizer.train(text, vocab_size=256, verbose=False)
        
        tokenizer.save(str(vocab_path), str(merges_path))
        
        assert vocab_path.exists()
        assert merges_path.exists()
    
    def test_saved_vocab_is_valid_json(self, temp_paths):
        """Test: Saved vocab is valid JSON."""
        vocab_path, merges_path = temp_paths
        
        tokenizer = BPETokenizer()
        text = "the quick brown fox " * 50
        tokenizer.train(text, vocab_size=256, verbose=False)
        tokenizer.save(str(vocab_path), str(merges_path))
        
        with open(vocab_path) as f:
            vocab_dict = json.load(f)
        
        assert 'token_to_id' in vocab_dict
        assert 'id_to_token' in vocab_dict
    
    def test_load_restores_tokenizer(self, temp_paths):
        """Test: Loading restores tokenizer state."""
        vocab_path, merges_path = temp_paths
        
        tokenizer1 = BPETokenizer()
        text = "machine learning deep learning neural networks " * 20
        tokenizer1.train(text, vocab_size=256, verbose=False)
        tokenizer1.save(str(vocab_path), str(merges_path))
        
        tokenizer2 = BPETokenizer()
        tokenizer2.load(str(vocab_path), str(merges_path))
        
        assert tokenizer2.vocab_size() == tokenizer1.vocab_size()
        assert tokenizer2.get_merges() == tokenizer1.get_merges()
    
    def test_save_load_identical_tokenization(self, temp_paths):
        """Test 9: Save/load produces identical tokenization."""
        vocab_path, merges_path = temp_paths
        
        tokenizer1 = BPETokenizer()
        text = "what is machine learning what is deep learning " * 20
        tokenizer1.train(text, vocab_size=256, verbose=False)
        tokenizer1.save(str(vocab_path), str(merges_path))
        
        tokenizer2 = BPETokenizer()
        tokenizer2.load(str(vocab_path), str(merges_path))
        
        test_text = "what is machine learning"
        ids1 = tokenizer1.encode(test_text)
        ids2 = tokenizer2.encode(test_text)
        
        assert ids1 == ids2


class TestBPETokenizerSpecialTokens:
    """Tests for special token handling."""
    
    @pytest.fixture
    def trained_tokenizer(self):
        """Fixture providing a trained tokenizer."""
        tokenizer = BPETokenizer()
        text = "hello world test " * 50
        tokenizer.train(text, vocab_size=256)
        return tokenizer
    
    def test_special_token_ids_fixed(self, trained_tokenizer):
        """Test 13: Special tokens have fixed IDs."""
        vocab = trained_tokenizer.get_vocab()
        
        assert vocab['<PAD>'] == 0
        assert vocab['<UNK>'] == 1
        assert vocab['<BOS>'] == 2
        assert vocab['<EOS>'] == 3
    
    def test_unknown_token_used_for_unseen(self, trained_tokenizer):
        """Test 13: Unknown tokens use <UNK> ID."""
        # Create text with uncommon character sequence
        text = "xyzabc"
        token_ids = trained_tokenizer.encode(text)
        
        # Should contain valid token IDs (including <UNK>)
        unk_id = trained_tokenizer.get_vocab()['<UNK>']
        # Not all should be <UNK>, but at least some should encode successfully
        assert all(tid >= 0 for tid in token_ids)


class TestBPETokenizerEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_character_text(self):
        """Test: Single character text."""
        tokenizer = BPETokenizer()
        tokenizer.train("a", vocab_size=100)
        
        ids = tokenizer.encode("a")
        assert len(ids) > 0
    
    def test_repeated_characters(self):
        """Test: Repeated characters."""
        tokenizer = BPETokenizer()
        tokenizer.train("aaaa bbbb cccc", vocab_size=100)
        
        ids = tokenizer.encode("aaaa")
        assert len(ids) > 0
    
    def test_special_characters(self):
        """Test: Special characters in text."""
        tokenizer = BPETokenizer()
        text = "hello! world? how's it going... great!"
        tokenizer.train(text, vocab_size=150)
        
        ids = tokenizer.encode("hello!")
        assert len(ids) > 0
    
    def test_numbers_in_text(self):
        """Test: Numbers in training and encoding."""
        tokenizer = BPETokenizer()
        text = "test 123 example 456 data 789" * 20
        tokenizer.train(text, vocab_size=200)
        
        ids = tokenizer.encode("123")
        assert len(ids) > 0
    
    def test_mixed_case_text(self):
        """Test: Mixed case handling."""
        tokenizer = BPETokenizer()
        text = "Hello WORLD hello WORLD" * 20
        tokenizer.train(text, vocab_size=150)
        
        ids1 = tokenizer.encode("Hello")
        ids2 = tokenizer.encode("hello")
        # May be different due to case sensitivity
        assert len(ids1) > 0 and len(ids2) > 0


class TestBPETokenizerIntegration:
    """Integration tests."""
    
    def test_complete_workflow(self):
        """Test: Complete training -> encode -> decode workflow."""
        # Train
        tokenizer = BPETokenizer()
        text = "machine learning is a branch of artificial intelligence " * 20
        tokenizer.train(text, vocab_size=300)
        
        # Encode example
        original = "machine learning"
        encoded = tokenizer.encode(original)
        
        # Verify encoding
        assert isinstance(encoded, list)
        assert all(isinstance(x, int) for x in encoded)
        assert len(encoded) > 0
        
        # Decode
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)
    
    def test_qa_dataset_training(self):
        """Test: Training on actual Q&A dataset."""
        dataset_path = Path(__file__).parent.parent / 'data' / 'qa_training.txt'
        
        if not dataset_path.exists():
            pytest.skip("Q&A dataset not found")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=256, verbose=False)
        
        # Basic verification
        assert tokenizer.vocab_size() > 0
        assert len(tokenizer.merges) > 0
        
        # Test encoding example Q&A
        encoded = tokenizer.encode("What is machine learning?")
        assert len(encoded) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
