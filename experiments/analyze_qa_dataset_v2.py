"""
Analyze the Improved Q&A Dataset (v2) for statistics, quality, and exceptions.

Performs analysis of data/qa_training_v2.txt:
- Total number of examples
- Total character count
- Question/Answer length statistics (min, max, average)
- Duplicate question detection
- Category distributions
- Flags/rejects malformed examples, empty answers, or extremely short answers.
"""

import re
from collections import defaultdict, Counter
from pathlib import Path


def parse_qa_dataset_v2(filepath):
    """
    Parse the Category/Question/Answer format in data/qa_training_v2.txt.
    
    Args:
        filepath: Path to qa_training_v2.txt
        
    Returns:
        List of dicts representing valid examples, and a list of issues found.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        
    text = path.read_text(encoding="utf-8")
    blocks = text.strip().split("\n\n")
    
    examples = []
    issues = []
    
    for idx, block in enumerate(blocks, 1):
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
                
        # Quality validation rules
        if len(lines) < 3:
            issues.append(f"Block {idx} has fewer than 3 lines (malformed structure).")
            continue
            
        if not category:
            issues.append(f"Block {idx} is missing a Category prefix.")
        if not question:
            issues.append(f"Block {idx} is missing a Question prefix.")
        if not answer:
            issues.append(f"Block {idx} is missing an Answer prefix.")
            
        if question and answer:
            examples.append({
                "index": idx,
                "category": category,
                "question": question,
                "answer": answer,
            })
            
    return examples, issues


def analyze_v2_dataset(filepath):
    """Perform dataset-wide quality audits and statistics checks."""
    examples, issues = parse_qa_dataset_v2(filepath)
    
    num_examples = len(examples)
    if num_examples == 0:
        return {"examples_count": 0, "issues": issues}
        
    # Stats
    total_chars = sum(len(ex["question"]) + len(ex["answer"]) for ex in examples)
    
    q_lengths = [len(ex["question"]) for ex in examples]
    a_lengths = [len(ex["answer"]) for ex in examples]
    
    category_counts = Counter(ex["category"] for ex in examples)
    
    # Duplicate checking
    question_counts = Counter(ex["question"] for ex in examples)
    duplicates = [q for q, count in question_counts.items() if count > 1]
    
    # Flags for extremely short/long answers
    short_answers = [ex for ex in examples if len(ex["answer"].split()) < 4]
    long_answers = [ex for ex in examples if len(ex["answer"]) > 250]
    
    stats = {
        "num_examples": num_examples,
        "total_characters": total_chars,
        "avg_question_len": sum(q_lengths) / num_examples,
        "avg_answer_len": sum(a_lengths) / num_examples,
        "min_question_len": min(q_lengths),
        "max_question_len": max(q_lengths),
        "min_answer_len": min(a_lengths),
        "max_answer_len": max(a_lengths),
        "category_distribution": dict(category_counts),
        "duplicate_questions": duplicates,
        "short_answers": short_answers,
        "long_answers": long_answers,
        "issues": issues,
    }
    
    return stats


def print_analysis_v2(stats):
    print("=" * 70)
    print("IMPROVED DATASET V2 QUALITY ANALYSIS")
    print("=" * 70)
    
    print(f"\nOVERVIEW:")
    print(f"  Total examples:         {stats['num_examples']}")
    print(f"  Total character count:  {stats['total_characters']:,}")
    print(f"  Duplicate questions:    {len(stats['duplicate_questions'])}")
    
    print(f"\nQUESTION STATISTICS:")
    print(f"  Average length:         {stats['avg_question_len']:.1f} chars")
    print(f"  Min/Max length:         {stats['min_question_len']} / {stats['max_question_len']} chars")
    
    print(f"\nANSWER STATISTICS:")
    print(f"  Average length:         {stats['avg_answer_len']:.1f} chars")
    print(f"  Min/Max length:         {stats['min_answer_len']} / {stats['max_answer_len']} chars")
    
    print(f"\nCATEGORY DISTRIBUTION:")
    for cat, count in sorted(stats["category_distribution"].items()):
        print(f"  * {cat:.<30} {count} examples")
        
    print(f"\nQUALITY AUDIT:")
    if stats["issues"]:
        print("  [WARNING] Structural Issues Found:")
        for issue in stats["issues"][:5]:
            print(f"    - {issue}")
    else:
        print("  [OK] No structural parsing errors detected.")
        
    if stats["duplicate_questions"]:
        print("  [WARNING] Duplicate Questions Detected:")
        for dup in stats["duplicate_questions"][:5]:
            print(f"    - '{dup}'")
    else:
        print("  [OK] No duplicate questions found.")
        
    if stats["short_answers"]:
        print("  [WARNING] Extremely Short Answers (< 4 words):")
        for ex in stats["short_answers"][:5]:
            print(f"    - Block {ex['index']}: '{ex['answer']}'")
    else:
        print("  [OK] No abnormally short answers found.")
        
    print("=" * 70)


def main():
    script_dir = Path(__file__).parent
    dataset_path = script_dir.parent / "data" / "qa_training_v2.txt"
    
    if not dataset_path.exists():
        print(f"Dataset not found at: {dataset_path}")
        return 1
        
    stats = analyze_v2_dataset(dataset_path)
    print_analysis_v2(stats)
    return 0


if __name__ == "__main__":
    main()
