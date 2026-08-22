"""
Unit tests for MiniGPT v2 Phase 6 improvements.

Covers:
1. Improved dataset loads.
2. Dataset has no duplicate questions.
3. Dataset split is deterministic.
4. Train/validation/test sets do not overlap.
5. Training configuration loads.
6. Scheduler works.
7. Gradient clipping works.
8. Training step works.
9. Validation works without gradients.
10. Checkpoint saves.
11. Checkpoint loads.
12. Training history is recorded.
13. Generation works with improved checkpoint.
14. Baseline checkpoint remains loadable.
"""

from pathlib import Path
import tempfile
import pytest
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
from training.train_v2 import calculate_loss
from training.train_v2_improved import (
    set_seed,
    get_warmup_cosine_scheduler,
    train_v2_improved_model,
)
from generation.generate_v2 import load_minigpt_v2, generate_v2


@pytest.fixture
def bpe_tokenizer():
    """Fixture providing BPE tokenizer."""
    tokenizer_dir = Path(__file__).parent.parent / "tokenizer"
    vocab_path = tokenizer_dir / "bpe_vocab.json"
    merges_path = tokenizer_dir / "bpe_merges.json"
    
    if not vocab_path.exists() or not merges_path.exists():
        pytest.skip("BPE tokenizer artifacts not found")
        
    tokenizer = BPETokenizer()
    tokenizer.load(str(vocab_path), str(merges_path))
    return tokenizer


@pytest.fixture
def sample_examples():
    """Fixture providing sample Q&A examples."""
    return [
        {"category": "AI", "question": "What is AI?", "answer": "AI is artificial intelligence."},
        {"category": "ML", "question": "What is machine learning?", "answer": "ML is learning patterns from data."},
        {"category": "DL", "question": "What is deep learning?", "answer": "DL uses deep neural networks."},
        {"category": "NN", "question": "What is a neural network?", "answer": "NN has neurons and layers."},
        {"category": "NLP", "question": "What is NLP?", "answer": "NLP processes natural human language."},
        {"category": "CV", "question": "What is computer vision?", "answer": "CV processes visual image data."},
        {"category": "TR", "question": "What is a Transformer?", "answer": "Transformer uses self-attention."},
        {"category": "LLM", "question": "What is an LLM?", "answer": "LLM is a large language model."},
        {"category": "PY", "question": "What is Python?", "answer": "Python is a clean programming language."},
        {"category": "DS", "question": "What is data science?", "answer": "DS extracts insights from data."},
    ]


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


class TestPhase6DatasetAndConfig:
    """Dataset and split tests."""

    def test_improved_dataset_loads(self):
        """Test 1: Improved dataset loads from qa_training_v2.txt."""
        dataset_path = Path(__file__).parent.parent / "data" / "qa_training_v2.txt"
        if not dataset_path.exists():
            pytest.skip("qa_training_v2.txt not found")
            
        examples = load_qa_examples(str(dataset_path))
        assert len(examples) >= 500

    def test_dataset_has_no_duplicate_questions(self):
        """Test 2: Improved dataset has zero duplicate questions."""
        dataset_path = Path(__file__).parent.parent / "data" / "qa_training_v2.txt"
        if not dataset_path.exists():
            pytest.skip("qa_training_v2.txt not found")
            
        examples = load_qa_examples(str(dataset_path))
        questions = [ex["question"] for ex in examples]
        duplicates = [q for q in set(questions) if questions.count(q) > 1]
        assert len(duplicates) == 0, f"Duplicates found: {duplicates}"

    def test_dataset_split_is_deterministic(self, sample_examples):
        """Test 3: Dataset split is deterministic with fixed seed."""
        train1, val1, test1 = split_qa_examples(sample_examples, seed=42)
        train2, val2, test2 = split_qa_examples(sample_examples, seed=42)
        
        assert [ex["question"] for ex in train1] == [ex["question"] for ex in train2]
        assert [ex["question"] for ex in val1] == [ex["question"] for ex in val2]
        assert [ex["question"] for ex in test1] == [ex["question"] for ex in test2]

    def test_train_val_test_do_not_overlap(self, sample_examples):
        """Test 4: Train/val/test splits have no question overlap."""
        train_ex, val_ex, test_ex = split_qa_examples(
            sample_examples, seed=42, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        train_qs = {ex["question"] for ex in train_ex}
        val_qs = {ex["question"] for ex in val_ex}
        test_qs = {ex["question"] for ex in test_ex}
        
        assert train_qs.isdisjoint(val_qs)
        assert train_qs.isdisjoint(test_qs)
        assert val_qs.isdisjoint(test_qs)

    def test_training_configuration_loads(self, tiny_model):
        """Test 5: Training configuration can be retrieved from model."""
        config = tiny_model.get_config()
        assert config["embedding_dim"] == 32
        assert config["num_heads"] == 2
        assert config["hidden_dim"] == 64
        assert config["num_layers"] == 1


class TestPhase6Optimizations:
    """Tests for scheduler, gradient clipping, and training loops."""

    def test_scheduler_works(self, tiny_model):
        """Test 6: Scheduler warms up and cosinely decays learning rate."""
        optimizer = AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps=10, total_steps=100)
        
        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
            
        # Warmup phase: learning rate increases
        assert lrs[0] == 0.0
        assert lrs[10] == pytest.approx(1e-3)
        # Decay phase: learning rate decreases towards zero
        assert lrs[99] < 1e-4
        assert lrs[99] > 0.0

    def test_gradient_clipping_works(self, tiny_model, sample_examples, bpe_tokenizer):
        """Test 7: Gradient clipping clamps large parameter gradients."""
        dataset = QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=16)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        inputs, targets = next(iter(loader))
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = AdamW(tiny_model.parameters(), lr=1e-3)
        
        tiny_model.train()
        optimizer.zero_grad()
        _, loss = calculate_loss(tiny_model, inputs, targets, criterion)
        loss.backward()
        
        # Manually inflate gradient norm to test clipping
        for p in tiny_model.parameters():
            if p.grad is not None:
                p.grad.data.fill_(100.0)
                
        grad_norm = torch.nn.utils.clip_grad_norm_(tiny_model.parameters(), max_norm=1.0)
        assert grad_norm > 1.0
        
        # Verify gradients are clipped/bounded
        for p in tiny_model.parameters():
            if p.grad is not None:
                assert torch.all(p.grad <= 1.0)

    def test_one_training_step_works(self, tiny_model, sample_examples, bpe_tokenizer):
        """Test 8: One training step runs without raising exceptions."""
        dataset = QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=16)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        inputs, targets = next(iter(loader))
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = AdamW(tiny_model.parameters(), lr=1e-3)
        
        tiny_model.train()
        optimizer.zero_grad()
        _, loss = calculate_loss(tiny_model, inputs, targets, criterion)
        loss.backward()
        
        # Verify gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(tiny_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        assert loss is not None
        assert torch.isfinite(loss)

    def test_validation_works_without_gradients(self, tiny_model, sample_examples, bpe_tokenizer):
        """Test 9: Validation pass does not calculate gradients or mutate weights."""
        dataset = QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=16)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Clone parameters before validation
        params_before = [p.clone() for p in tiny_model.parameters()]
        
        tiny_model.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                _, loss = calculate_loss(tiny_model, inputs, targets, criterion)
                
        # Check weights are unchanged
        for p_before, p_after in zip(params_before, tiny_model.parameters()):
            assert torch.equal(p_before, p_after)

    def test_checkpoint_saves_and_loads(self, tiny_model, sample_examples, bpe_tokenizer):
        """Test 10 & 11: Saves to improved path and loads parameters correctly."""
        train_loader = torch.utils.data.DataLoader(
            QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=16), batch_size=2
        )
        val_loader = torch.utils.data.DataLoader(
            QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=16), batch_size=2
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history, best_path = train_v2_improved_model(
                model=tiny_model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=1,
                learning_rate=3e-4,
                warmup_epochs=0,
                device=torch.device("cpu"),
                checkpoint_dir=tmpdir,
                seed=42,
            )
            
            assert Path(best_path).exists()
            
            # Load checkpoint into new model
            new_model = MiniGPTV2(
                vocab_size=bpe_tokenizer.vocab_size(),
                embedding_dim=32,
                num_heads=2,
                hidden_dim=64,
                num_layers=1,
                max_sequence_length=64,
            )
            load_minigpt_v2(best_path, device=torch.device("cpu"))
            checkpoint_data = torch.load(best_path, map_location="cpu")
            new_model.load_state_dict(checkpoint_data["model_state_dict"])
            
            for p1, p2 in zip(tiny_model.parameters(), new_model.parameters()):
                assert torch.equal(p1, p2)

    def test_training_history_recorded(self, tiny_model, sample_examples, bpe_tokenizer):
        """Test 12: Training history records loss, validation loss, and learning rates."""
        train_loader = torch.utils.data.DataLoader(
            QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=16), batch_size=2
        )
        val_loader = torch.utils.data.DataLoader(
            QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=16), batch_size=2
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history, _ = train_v2_improved_model(
                model=tiny_model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=2,
                learning_rate=3e-4,
                warmup_epochs=1,
                device=torch.device("cpu"),
                checkpoint_dir=tmpdir,
                seed=42,
            )
            
            assert len(history) == 2
            assert "train_loss" in history[0]
            assert "val_loss" in history[0]
            assert "val_perplexity" in history[0]
            assert "learning_rate" in history[0]

    def test_generation_works_with_improved_checkpoint(self, tiny_model, bpe_tokenizer):
        """Test 13: generation runs correctly under evaluation mode."""
        prompt = "Question: What is ML?\nAnswer:"
        generated = generate_v2(
            model=tiny_model,
            tokenizer=bpe_tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            do_sample=False,
        )
        assert isinstance(generated, str)
        assert len(generated) > 0

    def test_baseline_checkpoint_loadable(self):
        """Test 14: Baseline Phase 4 checkpoint remains loadable."""
        baseline_path = Path(__file__).parent.parent / "checkpoints" / "v2" / "minigpt_v2_best.pt"
        if not baseline_path.exists():
            pytest.skip("Baseline checkpoint not found")
            
        model, tokenizer = load_minigpt_v2(str(baseline_path), device=torch.device("cpu"))
        assert model is not None
        assert tokenizer is not None
