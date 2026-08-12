"""
MiniGPT v2 Q&A Dataset Pipeline

Handles parsing, tokenizing, and batching Q&A dataset examples for MiniGPT v2 training.

Features:
- Parses `data/qa_training.txt` into structured Q&A examples.
- Formats Q&A examples consistently: "Question: ...\nAnswer: ...".
- Appends `<EOS>` token at the end of each example.
- Splits dataset deterministically (80% train, 10% validation, 10% test) by Q&A examples.
- Generates shifted input/target sequences for next-token prediction.
- Handles padding with `<PAD>` token ID (0) and truncation to `max_sequence_length`.
- Creates PyTorch `DataLoader` instances for training, validation, and testing.
"""

import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer.bpe_tokenizer import BPETokenizer


def load_qa_examples(file_path: str) -> List[Dict[str, str]]:
    """
    Load and parse Q&A examples from a text file.
    
    Expected file format:
    Category: ...
    Question: ...
    Answer: ...
    (separated by blank lines)

    Args:
        file_path: Path to qa_training.txt
        
    Returns:
        List of dictionaries containing 'category', 'question', and 'answer'
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Q&A data file not found at: {file_path}")
    
    text = path.read_text(encoding="utf-8")
    blocks = text.strip().split("\n\n")
    
    examples = []
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        category = ""
        question = ""
        answer = ""
        
        for line in lines:
            if line.startswith("Category:"):
                category = line[len("Category:"):].strip()
            elif line.startswith("Question:"):
                question = line[len("Question:"):].strip()
            elif line.startswith("Answer:"):
                answer = line[len("Answer:"):].strip()
                
        if question and answer:
            examples.append({
                "category": category,
                "question": question,
                "answer": answer,
            })
            
    return examples


def format_qa_example(example: Dict[str, str]) -> str:
    """
    Format a Q&A dictionary into a standardized text string for tokenization.
    
    Format:
    Question: <question_text>
    Answer: <answer_text>
    """
    return f"Question: {example['question']}\nAnswer: {example['answer']}"


def split_qa_examples(
    examples: List[Dict[str, str]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Deterministically split Q&A examples into train, validation, and test sets.
    
    Args:
        examples: List of Q&A dictionaries
        seed: Random seed for deterministic shuffling
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for testing (default 0.1)
        
    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")
        
    # Copy and shuffle deterministically
    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_examples = shuffled[:train_end]
    val_examples = shuffled[train_end:val_end]
    test_examples = shuffled[val_end:]
    
    return train_examples, val_examples, test_examples


class QADatasetV2(Dataset):
    """
    PyTorch Dataset for MiniGPT v2 next-token prediction.
    
    Takes Q&A text examples, encodes them using the BPE tokenizer,
    appends an <EOS> token (ID 3), and creates shifted input/target sequences:
    
    Tokens:  [T_0, T_1, T_2, ..., T_k, <EOS>]
    Input:   [T_0, T_1, T_2, ..., T_k]         (padded/truncated to max_sequence_length)
    Target:  [T_1, T_2, ..., T_k, <EOS>]        (padded/truncated to max_sequence_length)
    
    Padding uses <PAD> token ID (0).
    """

    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer: BPETokenizer,
        max_sequence_length: int = 256,
    ):
        """
        Initialize dataset.
        
        Args:
            examples: List of Q&A dictionaries
            tokenizer: BPETokenizer instance
            max_sequence_length: Maximum sequence length for input/target tensors
        """
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = tokenizer.SPECIAL_TOKENS.get("<PAD>", 0)
        self.eos_token_id = tokenizer.SPECIAL_TOKENS.get("<EOS>", 3)
        
        inputs_list = []
        targets_list = []
        
        for ex in examples:
            formatted_text = format_qa_example(ex)
            token_ids = tokenizer.encode(formatted_text)
            
            # Append EOS token
            token_ids.append(self.eos_token_id)
            
            if len(token_ids) < 2:
                # Sequence too short for next-token prediction
                continue
                
            # Shift by 1: input = tokens[:-1], target = tokens[1:]
            raw_input = token_ids[:-1]
            raw_target = token_ids[1:]
            
            # Truncate or Pad to max_sequence_length
            if len(raw_input) > max_sequence_length:
                seq_input = raw_input[:max_sequence_length]
                seq_target = raw_target[:max_sequence_length]
            else:
                pad_len = max_sequence_length - len(raw_input)
                seq_input = raw_input + [self.pad_token_id] * pad_len
                seq_target = raw_target + [self.pad_token_id] * pad_len
                
            inputs_list.append(seq_input)
            targets_list.append(seq_target)
            
        if not inputs_list:
            self.input_ids = torch.empty((0, max_sequence_length), dtype=torch.long)
            self.target_ids = torch.empty((0, max_sequence_length), dtype=torch.long)
        else:
            self.input_ids = torch.tensor(inputs_list, dtype=torch.long)
            self.target_ids = torch.tensor(targets_list, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def create_v2_dataloaders(
    train_examples: List[Dict[str, str]],
    val_examples: List[Dict[str, str]],
    test_examples: List[Dict[str, str]],
    tokenizer: BPETokenizer,
    max_sequence_length: int = 256,
    batch_size: int = 8,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        train_examples: Training Q&A examples
        val_examples: Validation Q&A examples
        test_examples: Test Q&A examples
        tokenizer: BPETokenizer instance
        max_sequence_length: Maximum sequence length
        batch_size: Batch size for DataLoaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = QADatasetV2(train_examples, tokenizer, max_sequence_length)
    val_dataset = QADatasetV2(val_examples, tokenizer, max_sequence_length)
    test_dataset = QADatasetV2(test_examples, tokenizer, max_sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
