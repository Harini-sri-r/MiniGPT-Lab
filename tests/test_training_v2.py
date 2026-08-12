"""
Unit tests for MiniGPT v2 Q&A Training Pipeline (Phase 4).

Covers:
1. Dataset loads from text / Q&A examples.
2. Dataset split is deterministic.
3. Train/validation/test sets are non-empty.
4. Input and target shapes match.
5. DataLoader works.
6. Padding is handled correctly.
7. PAD tokens are ignored by loss (ignore_index=0).
8. MiniGPT v2 forward pass works with dataset output.
9. One training step works.
10. Loss is finite.
11. Model parameters change after optimizer.step().
12. Validation does not update parameters.
13. Checkpoint can be saved.
14. Checkpoint can be loaded.
15. Training history is recorded.
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
    format_qa_example,
    split_qa_examples,
    QADatasetV2,
    create_v2_dataloaders,
)
from training.train_v2 import (
    calculate_loss,
    validate_v2,
    train_v2_model,
    save_checkpoint,
    load_checkpoint,
)


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
def sample_examples():
    """Fixture providing sample Q&A examples."""
    return [
        {
            "category": "AI",
            "question": "What is AI?",
            "answer": "AI is artificial intelligence.",
        },
        {
            "category": "ML",
            "question": "What is machine learning?",
            "answer": "ML is learning patterns from data.",
        },
        {
            "category": "DL",
            "question": "What is deep learning?",
            "answer": "DL uses neural networks.",
        },
        {
            "category": "NLP",
            "question": "What is NLP?",
            "answer": "NLP processes natural language.",
        },
        {
            "category": "CV",
            "question": "What is computer vision?",
            "answer": "CV processes visual data.",
        },
    ]


@pytest.fixture
def tiny_model(bpe_tokenizer):
    """Fixture providing tiny MiniGPTV2 model for fast testing."""
    return MiniGPTV2(
        vocab_size=bpe_tokenizer.vocab_size(),
        embedding_dim=32,
        num_heads=2,
        hidden_dim=64,
        num_layers=1,
        max_sequence_length=64,
        dropout=0.0,
    )


class TestQADatasetV2:
    """Tests for dataset processing, formatting, and splitting."""

    def test_dataset_loads_from_file(self):
        """Test 1: Q&A examples load from text file."""
        data_path = Path(__file__).parent.parent / "data" / "qa_training.txt"
        if not data_path.exists():
            pytest.skip("qa_training.txt not found")
            
        examples = load_qa_examples(str(data_path))
        assert len(examples) > 0
        assert "question" in examples[0]
        assert "answer" in examples[0]

    def test_dataset_split_is_deterministic(self, sample_examples):
        """Test 2: Dataset split is deterministic with fixed seed."""
        train1, val1, test1 = split_qa_examples(sample_examples, seed=42)
        train2, val2, test2 = split_qa_examples(sample_examples, seed=42)
        
        assert [ex["question"] for ex in train1] == [ex["question"] for ex in train2]
        assert [ex["question"] for ex in val1] == [ex["question"] for ex in val2]
        assert [ex["question"] for ex in test1] == [ex["question"] for ex in test2]

    def test_splits_are_non_empty(self, sample_examples):
        """Test 3: Train/val/test splits are non-empty."""
        train_ex, val_ex, test_ex = split_qa_examples(
            sample_examples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        assert len(train_ex) > 0
        assert len(val_ex) > 0
        assert len(test_ex) > 0

    def test_input_and_target_shapes_match(self, sample_examples, bpe_tokenizer):
        """Test 4: Input IDs and Target IDs have identical shapes."""
        dataset = QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=32)
        assert len(dataset) > 0
        
        inputs, targets = dataset[0]
        assert inputs.shape == targets.shape
        assert inputs.shape[0] == 32

    def test_dataloader_works(self, sample_examples, bpe_tokenizer):
        """Test 5: DataLoader iterates and yields correct batch dimensions."""
        dataset = QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=32)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        
        batch_inputs, batch_targets = next(iter(loader))
        assert batch_inputs.shape == (2, 32)
        assert batch_targets.shape == (2, 32)

    def test_padding_handled_correctly(self, sample_examples, bpe_tokenizer):
        """Test 6: Short sequences are padded with PAD token ID (0)."""
        dataset = QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=64)
        inputs, targets = dataset[0]
        
        # Check that trailing positions are padded with 0
        assert inputs[-1].item() == 0
        assert targets[-1].item() == 0


class TestV2TrainingPipeline:
    """Tests for MiniGPT v2 training loop, loss calculation, and checkpointing."""

    def test_pad_tokens_ignored_by_loss(self, tiny_model):
        """Test 7: PAD tokens (ID 0) are ignored by CrossEntropyLoss."""
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        logits = torch.randn(1, 4, tiny_model.vocab_size)
        # Target with 2 valid tokens and 2 PAD tokens (0)
        targets_with_pad = torch.tensor([[10, 20, 0, 0]])
        targets_without_pad = torch.tensor([[10, 20]])
        
        flat_logits = logits.view(-1, tiny_model.vocab_size)
        
        loss_with_pad = criterion(flat_logits, targets_with_pad.view(-1))
        
        # Logits corresponding to non-pad targets
        valid_flat_logits = logits[:, :2, :].reshape(-1, tiny_model.vocab_size)
        loss_without_pad = criterion(valid_flat_logits, targets_without_pad.view(-1))
        
        assert torch.allclose(loss_with_pad, loss_without_pad, atol=1e-5)

    def test_minigpt_v2_forward_pass(self, sample_examples, bpe_tokenizer, tiny_model):
        """Test 8: MiniGPT v2 forward pass works with dataset batch."""
        dataset = QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=32)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        inputs, targets = next(iter(loader))
        tiny_model.eval()
        
        with torch.no_grad():
            logits = tiny_model(inputs)
            
        assert logits.shape == (2, 32, bpe_tokenizer.vocab_size())

    def test_one_training_step_works(self, sample_examples, bpe_tokenizer, tiny_model):
        """Test 9: One training step completes without error."""
        dataset = QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=32)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        inputs, targets = next(iter(loader))
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = AdamW(tiny_model.parameters(), lr=1e-3)
        
        tiny_model.train()
        optimizer.zero_grad()
        logits, loss = calculate_loss(tiny_model, inputs, targets, criterion)
        loss.backward()
        optimizer.step()
        
        assert loss is not None

    def test_loss_is_finite(self, sample_examples, bpe_tokenizer, tiny_model):
        """Test 10: Loss is finite (not NaN or Inf)."""
        dataset = QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=32)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        inputs, targets = next(iter(loader))
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        _, loss = calculate_loss(tiny_model, inputs, targets, criterion)
        
        assert torch.isfinite(loss).all()

    def test_parameters_change_after_optimizer_step(self, sample_examples, bpe_tokenizer, tiny_model):
        """Test 11: Model parameters change after optimizer step."""
        dataset = QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=32)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        inputs, targets = next(iter(loader))
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = AdamW(tiny_model.parameters(), lr=1e-2)
        
        # Clone initial weights
        init_param = next(tiny_model.parameters()).clone()
        
        tiny_model.train()
        optimizer.zero_grad()
        _, loss = calculate_loss(tiny_model, inputs, targets, criterion)
        loss.backward()
        optimizer.step()
        
        updated_param = next(tiny_model.parameters())
        assert not torch.equal(init_param, updated_param)

    def test_validation_does_not_update_parameters(self, sample_examples, bpe_tokenizer, tiny_model):
        """Test 12: Validation pass does not update model parameters."""
        dataset = QADatasetV2(sample_examples, bpe_tokenizer, max_sequence_length=32)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        device = torch.device("cpu")
        
        # Clone parameters before validation
        param_before = [p.clone() for p in tiny_model.parameters()]
        
        validate_v2(tiny_model, val_loader, criterion, device)
        
        # Check parameters after validation
        for p_before, p_after in zip(param_before, tiny_model.parameters()):
            assert torch.equal(p_before, p_after)

    def test_checkpoint_can_be_saved(self, tiny_model):
        """Test 13: Checkpoint can be saved to file."""
        optimizer = AdamW(tiny_model.parameters(), lr=1e-3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            save_checkpoint(
                str(ckpt_path),
                model=tiny_model,
                optimizer=optimizer,
                config=tiny_model.get_config(),
                tokenizer_info={"vocab_size": tiny_model.vocab_size},
                epoch=1,
                train_loss=2.5,
                val_loss=2.6,
                history=[{"epoch": 1, "train_loss": 2.5, "val_loss": 2.6}],
            )
            assert ckpt_path.exists()

    def test_checkpoint_can_be_loaded(self, bpe_tokenizer, tiny_model):
        """Test 14: Checkpoint can be loaded and restores model weights."""
        optimizer = AdamW(tiny_model.parameters(), lr=1e-3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            save_checkpoint(
                str(ckpt_path),
                model=tiny_model,
                optimizer=optimizer,
                config=tiny_model.get_config(),
                tokenizer_info={"vocab_size": tiny_model.vocab_size},
                epoch=2,
                train_loss=1.8,
                val_loss=1.9,
                history=[{"epoch": 1, "train_loss": 2.5, "val_loss": 2.6}, {"epoch": 2, "train_loss": 1.8, "val_loss": 1.9}],
            )
            
            # Create a new model instance and load checkpoint
            new_model = MiniGPTV2(
                vocab_size=bpe_tokenizer.vocab_size(),
                embedding_dim=32,
                num_heads=2,
                hidden_dim=64,
                num_layers=1,
                max_sequence_length=64,
            )
            loaded_data = load_checkpoint(str(ckpt_path), new_model)
            
            assert loaded_data["epoch"] == 2
            assert loaded_data["train_loss"] == 1.8
            assert loaded_data["val_loss"] == 1.9
            
            # Check model weights match original tiny_model
            for p1, p2 in zip(tiny_model.parameters(), new_model.parameters()):
                assert torch.equal(p1, p2)

    def test_training_history_is_recorded(self, sample_examples, bpe_tokenizer, tiny_model):
        """Test 15: Multi-epoch training returns history for each epoch."""
        train_ex, val_ex, test_ex = split_qa_examples(sample_examples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        train_loader, val_loader, _ = create_v2_dataloaders(
            train_ex, val_ex, test_ex, bpe_tokenizer, max_sequence_length=32, batch_size=2
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history, _ = train_v2_model(
                model=tiny_model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=2,
                learning_rate=1e-3,
                device=torch.device("cpu"),
                checkpoint_dir=tmpdir,
            )
            
            assert len(history) == 2
            assert history[0]["epoch"] == 1
            assert history[1]["epoch"] == 2
            assert "train_loss" in history[0]
            assert "val_loss" in history[0]
