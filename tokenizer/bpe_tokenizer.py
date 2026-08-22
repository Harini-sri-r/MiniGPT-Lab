"""
Byte Pair Encoding (BPE) Tokenizer for MiniGPT v2.

A small educational BPE tokenizer that learns subword units from training text.
Implements the core BPE algorithm: start with characters, iteratively merge the
most frequent adjacent pairs until reaching target vocabulary size.

Features:
- Training vocabulary from text
- Deterministic merging algorithm
- Special tokens: <PAD>, <UNK>, <BOS>, <EOS>
- Encode/decode functionality
- Save/load vocabulary and merge rules
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Set


class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer.
    
    Learns frequent subword units from training text through iterative merging
    of the most common adjacent symbol pairs.
    """
    
    # Special token IDs (fixed, do not change after training)
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3,
    }
    
    def __init__(self):
        """Initialize tokenizer with special tokens."""
        self.token_to_id = self.SPECIAL_TOKENS.copy()
        self.id_to_token = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        
        # Track the next available ID for new tokens
        self.next_id = len(self.SPECIAL_TOKENS)
        
        # Merge rules: list of (token1, token2) tuples in order applied
        self.merges = []
        
        # Vocabulary: set of all known tokens
        self.vocab = set(self.SPECIAL_TOKENS.keys())
    
    def train(self, text: str, vocab_size: int = 256, verbose: bool = False):
        """
        Train the BPE tokenizer on text.
        
        Algorithm:
        1. Initialize with character-level tokens
        2. Count adjacent pair frequencies
        3. Find most frequent pair
        4. Merge the pair across all word splits
        5. Repeat until vocab_size is reached
        
        Args:
            text: Training text corpus
            vocab_size: Target vocabulary size (including special tokens)
            verbose: Print progress during training
        """
        if vocab_size <= len(self.SPECIAL_TOKENS):
            raise ValueError(
                f"vocab_size ({vocab_size}) must be > number of special tokens "
                f"({len(self.SPECIAL_TOKENS)})"
            )
        
        # Step 1: Initialize with character-level tokens
        # Add all unique characters to vocabulary
        unique_chars = set(text)
        for char in sorted(unique_chars):
            if char not in self.token_to_id:
                self.token_to_id[char] = self.next_id
                self.id_to_token[self.next_id] = char
                self.vocab.add(char)
                self.next_id += 1
        
        if verbose:
            print(f"Starting vocabulary size: {len(self.token_to_id)}")
        
        # Step 2: Split text into words (by whitespace)
        words = text.split()
        
        # Convert words to character sequences
        word_splits = []
        for word in words:
            # Represent word as tuple of characters, mark end with </w>
            split = tuple(char for char in word) + ('</w>',)
            word_splits.append(split)
        
        # Ensure end-of-word token is in vocabulary
        if '</w>' not in self.token_to_id:
            self.token_to_id['</w>'] = self.next_id
            self.id_to_token[self.next_id] = '</w>'
            self.vocab.add('</w>')
            self.next_id += 1
        
        # Step 3: Iteratively merge most frequent pairs
        iteration = 0
        while len(self.token_to_id) < vocab_size:
            iteration += 1
            
            # Count adjacent pairs
            pair_counts = self._count_pairs(word_splits)
            
            if not pair_counts:
                # No more pairs to merge (shouldn't happen with real text)
                if verbose:
                    print(f"No more pairs at iteration {iteration}")
                break
            
            # Find most frequent pair
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            frequency = pair_counts[most_frequent_pair]
            
            if verbose and iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: merging {most_frequent_pair} "
                    f"(freq={frequency}), vocab_size={len(self.token_to_id)}"
                )
            
            # Create new token for this pair
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            self.token_to_id[new_token] = self.next_id
            self.id_to_token[self.next_id] = new_token
            self.vocab.add(new_token)
            self.next_id += 1
            
            # Store merge rule
            self.merges.append(most_frequent_pair)
            
            # Merge the pair in all word splits
            word_splits = self._merge_pair(word_splits, most_frequent_pair)
        
        if verbose:
            print(f"Training complete. Final vocab size: {len(self.token_to_id)}")
    
    def _count_pairs(self, word_splits: List[Tuple]) -> Counter:
        """
        Count all adjacent pairs in word splits.
        
        Args:
            word_splits: List of word tuples
            
        Returns:
            Counter of adjacent pairs and their frequencies
        """
        pairs = Counter()
        for word in word_splits:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += 1
        return pairs
    
    def _merge_pair(self, word_splits: List[Tuple], pair: Tuple[str, str]) -> List[Tuple]:
        """
        Merge a specific pair throughout all word splits.
        
        Args:
            word_splits: List of word tuples
            pair: (token1, token2) pair to merge
            
        Returns:
            Updated list of word tuples with pair merged
        """
        merged = []
        for word in word_splits:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    # Merge this pair
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            merged.append(tuple(new_word))
        return merged
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Algorithm:
        1. Split text into words
        2. Convert to character sequences
        3. Apply merge rules in order
        4. Convert tokens to IDs
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        # Handle empty text
        if not text or not text.strip():
            return []
        
        # Split into words
        words = text.split()
        
        token_ids = []
        for word in words:
            # Start with character-level split
            split = list(word) + ['</w>']
            
            # Apply merge rules in order
            for token1, token2 in self.merges:
                i = 0
                while i < len(split) - 1:
                    if split[i] == token1 and split[i + 1] == token2:
                        # Merge
                        split[i:i + 2] = [token1 + token2]
                    else:
                        i += 1
            
            # Convert tokens to IDs
            for token in split:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    # Unknown token
                    token_ids.append(self.token_to_id['<UNK>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back into text.
        
        Limitations:
        - Reconstructs words but may not preserve exact original spacing/punctuation
        - Whitespace is reconstructed between tokens marked with </w>
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if not token_ids:
            return ""
        
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append('<UNK>')
        
        # Join tokens, handling </w> as word boundary
        text = ""
        for token in tokens:
            # Skip special control tokens
            if token.startswith('<') and token.endswith('>'):
                continue
            # Replace </w> with space
            elif token.endswith('</w>'):
                text += token[:-4]  # Remove </w>
                text += " "
            else:
                text += token
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def save(self, vocab_path: str, merges_path: str) -> None:
        """
        Save tokenizer vocabulary and merge rules.
        
        Args:
            vocab_path: Path to save vocabulary JSON
            merges_path: Path to save merge rules JSON
        """
        # Save vocabulary (token -> ID mapping)
        vocab_dict = {
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
        }
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
        
        # Save merge rules as list of [token1, token2] pairs
        merges_list = [list(pair) for pair in self.merges]
        with open(merges_path, 'w', encoding='utf-8') as f:
            json.dump(merges_list, f, indent=2, ensure_ascii=False)
    
    def load(self, vocab_path: str, merges_path: str) -> None:
        """
        Load tokenizer vocabulary and merge rules.
        
        Args:
            vocab_path: Path to vocabulary JSON
            merges_path: Path to merge rules JSON
        """
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        
        self.token_to_id = vocab_dict['token_to_id']
        # Convert string IDs back to integers
        self.id_to_token = {int(k): v for k, v in vocab_dict['id_to_token'].items()}
        self.vocab = set(self.token_to_id.keys())
        self.next_id = max(self.id_to_token.keys()) + 1
        
        # Load merge rules
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_list = json.load(f)
        
        self.merges = [tuple(pair) for pair in merges_list]
    
    def vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.token_to_id)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get token to ID mapping."""
        return self.token_to_id.copy()
    
    def get_merges(self) -> List[Tuple[str, str]]:
        """Get merge rules."""
        return self.merges.copy()
