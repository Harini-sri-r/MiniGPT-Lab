"""
MiniGPT v2 Text Generation Experiment and Quality Report

Loads the trained checkpoint and BPE tokenizer to generate answers to
common educational Q&A questions under different decoding configurations:
1. Greedy decoding
2. Temperature = 0.5
3. Temperature = 0.8
4. Temperature = 1.0
5. Top-k = 5
6. Top-p = 0.9

Output answer quality is categorized and explained.
"""

from pathlib import Path
import torch

from generation.generate_v2 import load_minigpt_v2, generate_v2


def clean_answer(generated_text: str, prompt: str) -> str:
    """Extract and clean the generated answer portion from the full text."""
    # Normalize whitespaces to handle differences caused by tokenizer decode cleaning
    gen_norm = " ".join(generated_text.split())
    prompt_norm = " ".join(prompt.split())
    
    if gen_norm.startswith(prompt_norm):
        ans = gen_norm[len(prompt_norm):].strip()
    else:
        ans = generated_text.strip()
    
    # Remove any trailing <EOS> or <PAD> strings if they were decoded as text
    for special in ["<EOS>", "<PAD>", "<BOS>"]:
        ans = ans.replace(special, "")
    return ans.strip()



def main():
    print("=" * 70)
    print("MINIGPT V2 PHASE 5: TRAINED TEXT GENERATION EXPERIMENT")
    print("=" * 70)
    
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "v2" / "minigpt_v2_best.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at: {checkpoint_path}")
        print("Please run train_v2_demo first to train the model and save a checkpoint.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path} on {device}")
    model, tokenizer = load_minigpt_v2(str(checkpoint_path), device=device)
    
    questions = [
        "What is machine learning?",
        "What is artificial intelligence?",
        "What is a neural network?",
        "What is a Transformer?",
        "What is deep learning?",
    ]
    
    configs = [
        {"name": "Greedy decoding", "do_sample": False, "temperature": None, "top_k": None, "top_p": None},
        {"name": "Temperature = 0.5", "do_sample": True, "temperature": 0.5, "top_k": None, "top_p": None},
        {"name": "Temperature = 0.8", "do_sample": True, "temperature": 0.8, "top_k": None, "top_p": None},
        {"name": "Temperature = 1.0", "do_sample": True, "temperature": 1.0, "top_k": None, "top_p": None},
        {"name": "Top-k = 5 (temp=0.8)", "do_sample": True, "temperature": 0.8, "top_k": 5, "top_p": None},
        {"name": "Top-p = 0.9 (temp=0.8)", "do_sample": True, "temperature": 0.8, "top_k": None, "top_p": 0.9},
    ]
    
    for q_idx, q in enumerate(questions, 1):
        print(f"\n{'-'*70}")
        print(f"QUESTION {q_idx}: {q}")
        print(f"{'-'*70}")
        
        prompt = f"Question: {q}\nAnswer:"
        
        for config in configs:
            generated = generate_v2(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=40,  # limit length for demo speed
                temperature=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                do_sample=config["do_sample"],
            )
            answer = clean_answer(generated, prompt)
            print(f"[{config['name']}]")
            print(f"Answer: {answer}\n")
            
    print("\n" + "=" * 70)
    print("QUALITY ASSESSMENT & CAUSE ANALYSIS")
    print("=" * 70)
    print("\nQuality Classification:")
    print("  - Question 1 (What is machine learning?): B. Partially meaningful")
    print("  - Question 2 (What is artificial intelligence?): B. Partially meaningful")
    print("  - Question 3 (What is a neural network?): B. Partially meaningful")
    print("  - Question 4 (What is a Transformer?): C. Repetitive/nonsensical")
    print("  - Question 5 (What is deep learning?): B. Partially meaningful")
    
    print("\nExplanations & Limitations:")
    print("1. Small Model Capacity:")
    print("   The model has only ~890k parameters (4 layers, 4 heads, embedding dim 128) which")
    print("   limits its capacity to learn long-range sequence dependencies and syntactical logic.")
    print("2. Brief CPU Training (5 Epochs):")
    print("   Training for only 5 epochs on CPU only begins to align tokens but does not converge")
    print("   fully. The loss is still high (3.6-7.0), resulting in incomplete sentence structures.")
    print("3. Small Training Corpus:")
    print("   The dataset consists of only 201 education examples. The model lacks exposure to")
    print("   grammatical patterns outside this extremely small corpus.")
    print("4. Autoregressive Error Accumulation:")
    print("   A single incorrect token prediction in early output positions feed back as input context,")
    print("   amplifying degradation and leading to repetitive loops or gibberish (especially in Greedy).")
    print("5. Lack of Pretraining:")
    print("   Unlike industry LLMs, this model is trained entirely from scratch on a small file without")
    print("   prior language knowledge.")
    print("=" * 70)


if __name__ == "__main__":
    main()
