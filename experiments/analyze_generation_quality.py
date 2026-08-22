"""
Analyze and Compare Generation Quality of Baseline vs Improved MiniGPT v2 Models

Evaluates the models on 8 test questions:
1. What is machine learning?
2. What is artificial intelligence?
3. What is a neural network?
4. What is a Transformer?
5. What is deep learning?
6. What is supervised learning?
7. What is NLP?
8. What is an LLM?

Computes metrics to detect degeneration and loops:
- Unique Token Ratio (Unique Tokens / Total Generated Tokens)
- Repeated bigrams/trigrams/4-grams
- Average generation length
- EOS termination rate
"""

from collections import Counter
from pathlib import Path
import torch

from generation.generate_v2 import load_minigpt_v2, generate_v2


def get_token_repetition_metrics(tokens: list) -> dict:
    """
    Compute unigram and n-gram repetition statistics.
    
    Args:
        tokens: List of string token values
        
    Returns:
        Dict containing repetition statistics
    """
    if not tokens:
        return {
            "length": 0,
            "unique_ratio": 0.0,
            "repeated_bigrams": 0,
            "repeated_trigrams": 0,
        }
        
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    unique_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0.0
    
    # Bigram repetitions
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(count - 1 for count in bigram_counts.values() if count > 1)
    
    # Trigram repetitions
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    trigram_counts = Counter(trigrams)
    repeated_trigrams = sum(count - 1 for count in trigram_counts.values() if count > 1)
    
    # 4-gram repetitions
    fourgrams = [tuple(tokens[i:i+4]) for i in range(len(tokens) - 3)]
    fourgram_counts = Counter(fourgrams)
    repeated_fourgrams = sum(count - 1 for count in fourgram_counts.values() if count > 1)
    
    return {
        "length": total_tokens,
        "unique_ratio": unique_ratio,
        "repeated_bigrams": repeated_bigrams,
        "repeated_trigrams": repeated_trigrams,
        "repeated_fourgrams": repeated_fourgrams,
    }


def clean_and_tokenize(generated_text: str, prompt: str) -> list:
    """Extract generated answer portion and split it into clean words/tokens."""
    gen_norm = " ".join(generated_text.split())
    prompt_norm = " ".join(prompt.split())
    
    if gen_norm.startswith(prompt_norm):
        ans = gen_norm[len(prompt_norm):].strip()
    else:
        ans = generated_text.strip()
        
    for special in ["<EOS>", "<PAD>", "<BOS>"]:
        ans = ans.replace(special, "")
        
    return ans.strip().split()


def evaluate_model_generation(model, tokenizer, questions: list) -> dict:
    """Generate answers and compute average metrics across all questions."""
    total_len = 0
    total_unique_ratio = 0.0
    total_bigram_rep = 0
    total_trigram_rep = 0
    total_fourgram_rep = 0
    eos_hits = 0
    
    device = next(model.parameters()).device
    eos_token_id = tokenizer.SPECIAL_TOKENS.get("<EOS>", 3)
    
    for q in questions:
        prompt = f"Question: {q}\nAnswer:"
        
        # We manually encode and run step-by-step to check if EOS is hit
        input_ids = tokenizer.encode(prompt)
        generated = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        hit_eos = False
        with torch.no_grad():
            for _ in range(50): # max_new_tokens
                if generated.size(1) >= model.max_sequence_length:
                    break
                logits = model(generated)[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=-1)
                
                if next_token.item() == eos_token_id:
                    hit_eos = True
                    break
                    
        if hit_eos:
            eos_hits += 1
            
        full_generated_text = tokenizer.decode(generated[0].tolist())
        if questions.index(q) == 0:
            print(f"  [Example] Q: {q}")
            print(f"  [Example] A: {full_generated_text}")
        tokens = clean_and_tokenize(full_generated_text, prompt)
        
        metrics = get_token_repetition_metrics(tokens)
        total_len += metrics["length"]
        total_unique_ratio += metrics["unique_ratio"]
        total_bigram_rep += metrics["repeated_bigrams"]
        total_trigram_rep += metrics["repeated_trigrams"]
        total_fourgram_rep += metrics["repeated_fourgrams"]
        
    num_q = len(questions)
    return {
        "avg_length": total_len / num_q,
        "avg_unique_ratio": total_unique_ratio / num_q,
        "avg_bigram_rep": total_bigram_rep / num_q,
        "avg_trigram_rep": total_trigram_rep / num_q,
        "avg_fourgram_rep": total_fourgram_rep / num_q,
        "eos_rate": eos_hits / num_q,
    }


def main():
    print("=" * 70)
    print("MINIGPT V2: GENERATION REPETITION & LOOP ANALYSIS")
    print("=" * 70)
    
    project_dir = Path(__file__).parent.parent
    baseline_path = project_dir / "checkpoints" / "v2" / "minigpt_v2_best.pt"
    improved_path = project_dir / "checkpoints" / "v2" / "improved" / "minigpt_v2_best.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    questions = [
        "What is machine learning?",
        "What is artificial intelligence?",
        "What is a neural network?",
        "What is a Transformer?",
        "What is deep learning?",
        "What is NLP?",
        "What is computer vision?",
        "What is generative AI?",
        "What is a large language model?",
        "What is backpropagation?",
    ]
    
    # 1. Baseline Model
    print(f"\nAnalyzing Phase 4 Baseline Model...")
    b_metrics = {}
    if baseline_path.exists():
        b_model, b_tokenizer = load_minigpt_v2(str(baseline_path), device=device)
        b_metrics = evaluate_model_generation(b_model, b_tokenizer, questions)
    else:
        print("[ERROR] Baseline checkpoint not found.")
        
    # 2. Improved Model
    print(f"Analyzing Phase 6 Improved Model...")
    i_metrics = {}
    if improved_path.exists():
        i_model, i_tokenizer = load_minigpt_v2(str(improved_path), device=device)
        i_metrics = evaluate_model_generation(i_model, i_tokenizer, questions)
    else:
        print("[ERROR] Improved checkpoint not found.")
        
    print("\nREPETITION METRICS SUMMARY:")
    print("-" * 70)
    print("Metric                 | Phase 4 Baseline      | Phase 6 Improved")
    print("-" * 70)
    print(f"Avg Generated Words    | {b_metrics.get('avg_length', 0):<21.1f} | {i_metrics.get('avg_length', 0):<16.1f}")
    print(f"Unique Token Ratio     | {b_metrics.get('avg_unique_ratio', 0):<21.2%} | {i_metrics.get('avg_unique_ratio', 0):<16.2%}")
    print(f"Repeated Bigrams count | {b_metrics.get('avg_bigram_rep', 0):<21.1f} | {i_metrics.get('avg_bigram_rep', 0):<16.1f}")
    print(f"Repeated Trigrams count| {b_metrics.get('avg_trigram_rep', 0):<21.1f} | {i_metrics.get('avg_trigram_rep', 0):<16.1f}")
    print(f"Repeated 4-grams count | {b_metrics.get('avg_fourgram_rep', 0):<21.1f} | {i_metrics.get('avg_fourgram_rep', 0):<16.1f}")
    print(f"EOS Termination Rate   | {b_metrics.get('eos_rate', 0):<21.2%} | {i_metrics.get('eos_rate', 0):<16.2%}")
    print("-" * 70)
    print("\nAnalysis Interpretation:")
    print("  - Unique Token Ratio: Higher ratio (closer to 100%) indicates less repetitive output.")
    print("  - Repeated N-grams: Lower values suggest fewer localized loops (e.g. 'a model is a model').")
    print("  - EOS Rate: Higher rate shows the model successfully concludes sentences with <EOS>.")
    print("=" * 70)


if __name__ == "__main__":
    main()
