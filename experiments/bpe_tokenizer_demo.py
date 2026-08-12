"""
BPE Tokenizer Demo for MiniGPT v2.

Demonstrates:
1. Loading the Q&A dataset
2. Training the BPE tokenizer
3. Encoding example sentences
4. Decoding token IDs back to text
5. Analyzing learned tokens
"""

from pathlib import Path
from tokenizer.bpe_tokenizer import BPETokenizer


def main():
    """Run the BPE tokenizer demo."""
    
    # Load dataset
    dataset_path = Path(__file__).parent.parent / 'data' / 'qa_training.txt'
    print(f"Loading dataset from {dataset_path}...\n")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset size: {len(text):,} characters\n")
    
    # Train tokenizer
    print("=" * 70)
    print("TRAINING BPE TOKENIZER")
    print("=" * 70)
    
    tokenizer = BPETokenizer()
    vocab_size = 512  # Reasonable size for educational purposes
    tokenizer.train(text, vocab_size=vocab_size, verbose=True)
    
    print(f"\nFinal vocabulary size: {tokenizer.vocab_size()}")
    print(f"Number of merge rules: {len(tokenizer.merges)}\n")
    
    # Show sample learned tokens
    print("=" * 70)
    print("LEARNED TOKENS (sample)")
    print("=" * 70)
    
    learned_tokens = [
        token for token in tokenizer.get_vocab().keys()
        if token not in BPETokenizer.SPECIAL_TOKENS and len(token) > 1
    ]
    learned_tokens.sort(key=len, reverse=True)
    
    print("\nLongest learned tokens:")
    for token in learned_tokens[:20]:
        print(f"  {token!r}")
    
    print("\nSpecial tokens:")
    for token, token_id in BPETokenizer.SPECIAL_TOKENS.items():
        print(f"  {token} (ID: {token_id})")
    
    # Test encoding/decoding on example sentences
    print("\n" + "=" * 70)
    print("TOKENIZATION EXAMPLES")
    print("=" * 70)
    
    test_sentences = [
        "What is machine learning?",
        "What is a Transformer?",
        "How does artificial intelligence work?",
    ]
    
    for sentence in test_sentences:
        print(f"\n{'-' * 70}")
        print(f"Original:  {sentence}")
        
        # Encode
        token_ids = tokenizer.encode(sentence)
        print(f"Token IDs: {token_ids}")
        
        # Show tokens
        tokens = [tokenizer.id_to_token.get(tid, '<UNK>') for tid in token_ids]
        print(f"Tokens:    {tokens}")
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        print(f"Decoded:   {decoded}")
        
        # Show compression ratio
        num_tokens = len(token_ids)
        num_chars = len(sentence)
        print(f"Compression: {num_chars} chars -> {num_tokens} tokens "
              f"({num_tokens/num_chars*100:.1f}% of original)")
    
    # Save tokenizer
    print("\n" + "=" * 70)
    print("SAVING TOKENIZER")
    print("=" * 70)
    
    tokenizer_dir = Path(__file__).parent.parent / 'tokenizer'
    vocab_path = tokenizer_dir / 'bpe_vocab.json'
    merges_path = tokenizer_dir / 'bpe_merges.json'
    
    print(f"\nSaving vocabulary to: {vocab_path}")
    print(f"Saving merges to: {merges_path}")
    tokenizer.save(str(vocab_path), str(merges_path))
    
    print(f"[OK] Tokenizer saved successfully")
    
    # Test loading
    print("\n" + "=" * 70)
    print("TESTING LOAD AND ENCODE")
    print("=" * 70)
    
    tokenizer2 = BPETokenizer()
    tokenizer2.load(str(vocab_path), str(merges_path))
    
    test_text = "What is machine learning?"
    ids1 = tokenizer.encode(test_text)
    ids2 = tokenizer2.encode(test_text)
    
    print(f"\nOriginal tokenizer: {ids1}")
    print(f"Loaded tokenizer:   {ids2}")
    if ids1 == ids2:
        print(f"Match: True [OK]")
    else:
        print(f"Match: False [FAIL]")


if __name__ == '__main__':
    main()
