"""
Analyze the Q&A dataset for statistics and quality metrics.

This script provides comprehensive analysis of data/qa_training.txt including:
- Total number of examples
- Character count
- Question length statistics (min, max, average)
- Answer length statistics (min, max, average)
- Category breakdown (if present)
"""

import re
from collections import defaultdict
from pathlib import Path


def parse_qa_dataset(filepath):
    """
    Parse the Q&A dataset file and extract questions and answers.
    
    Args:
        filepath: Path to the qa_training.txt file
        
    Returns:
        Tuple of (questions, answers, categories)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    questions = []
    answers = []
    categories = []
    
    # Find all Question-Answer pairs
    # Pattern: "Question: <text>\nAnswer: <text>"
    qa_pattern = r'Question:\s*(.+?)\nAnswer:\s*(.+?)(?=\n(?:Category:|Question:|$))'
    
    matches = re.finditer(qa_pattern, content, re.DOTALL)
    for match in matches:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        questions.append(question)
        answers.append(answer)
    
    # Find all categories
    category_pattern = r'Category:\s*(.+?)(?=\n)'
    category_matches = re.finditer(category_pattern, content)
    for match in category_matches:
        category = match.group(1).strip()
        categories.append(category)
    
    return questions, answers, categories


def calculate_stats(texts):
    """
    Calculate length statistics for a list of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with min, max, average length stats
    """
    if not texts:
        return {
            'min_length': 0,
            'max_length': 0,
            'avg_length': 0.0,
        }
    
    lengths = [len(text) for text in texts]
    return {
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_length': sum(lengths) / len(lengths),
    }


def analyze_dataset(filepath):
    """
    Analyze the Q&A dataset and return comprehensive statistics.
    
    Args:
        filepath: Path to the qa_training.txt file
        
    Returns:
        Dictionary with analysis results
    """
    # Parse the dataset
    questions, answers, categories = parse_qa_dataset(filepath)
    
    # Calculate basic stats
    num_examples = len(questions)
    
    # Count total characters
    total_chars = sum(len(q) for q in questions) + sum(len(a) for a in answers)
    
    # Calculate length statistics
    question_stats = calculate_stats(questions)
    answer_stats = calculate_stats(answers)
    
    # Category breakdown
    category_counts = defaultdict(int)
    for category in categories:
        category_counts[category] += 1
    
    # Check for unique questions
    unique_questions = len(set(questions))
    duplicate_questions = num_examples - unique_questions
    
    return {
        'num_examples': num_examples,
        'total_characters': total_chars,
        'question_stats': question_stats,
        'answer_stats': answer_stats,
        'category_counts': dict(category_counts),
        'unique_questions': unique_questions,
        'duplicate_questions': duplicate_questions,
    }


def print_analysis(analysis):
    """
    Pretty print the analysis results.
    
    Args:
        analysis: Dictionary with analysis results from analyze_dataset()
    """
    print("=" * 70)
    print("Q&A DATASET ANALYSIS")
    print("=" * 70)
    
    print(f"\nDATASET OVERVIEW:")
    print(f"  Total Q&A examples:     {analysis['num_examples']}")
    print(f"  Total characters:       {analysis['total_characters']:,}")
    print(f"  Unique questions:       {analysis['unique_questions']}")
    print(f"  Duplicate questions:    {analysis['duplicate_questions']}")
    
    print(f"\nQUESTION LENGTH STATISTICS:")
    print(f"  Minimum length:         {analysis['question_stats']['min_length']} characters")
    print(f"  Maximum length:         {analysis['question_stats']['max_length']} characters")
    print(f"  Average length:         {analysis['question_stats']['avg_length']:.1f} characters")
    
    print(f"\nANSWER LENGTH STATISTICS:")
    print(f"  Minimum length:         {analysis['answer_stats']['min_length']} characters")
    print(f"  Maximum length:         {analysis['answer_stats']['max_length']} characters")
    print(f"  Average length:         {analysis['answer_stats']['avg_length']:.1f} characters")
    
    if analysis['category_counts']:
        print(f"\nCATEGORY BREAKDOWN:")
        sorted_categories = sorted(
            analysis['category_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for category, count in sorted_categories:
            print(f"  {category:.<40} {count:>3} examples")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point for dataset analysis."""
    # Determine dataset path relative to this script
    script_dir = Path(__file__).parent
    dataset_path = script_dir.parent / 'data' / 'qa_training.txt'
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return 1
    
    # Analyze the dataset
    try:
        analysis = analyze_dataset(dataset_path)
        print_analysis(analysis)
        return 0
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
