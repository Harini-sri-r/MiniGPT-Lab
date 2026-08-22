"""
Tests for the Q&A dataset.

This module tests:
1. Dataset file existence and readability
2. Minimum number of examples (200)
3. Presence of Questions and Answers
4. Non-empty questions and answers
5. Question uniqueness
6. Answer meaningfulness
7. Parse robustness
"""

import re
from pathlib import Path
import pytest


DATASET_PATH = Path(__file__).parent.parent / 'data' / 'qa_training.txt'


@pytest.fixture
def dataset_content():
    """Fixture to load dataset content once per test session."""
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def parsed_dataset(dataset_content):
    """Fixture to parse questions and answers from dataset."""
    questions = []
    answers = []
    
    qa_pattern = r'Question:\s*(.+?)\nAnswer:\s*(.+?)(?=\n(?:Category:|Question:|$))'
    matches = re.finditer(qa_pattern, dataset_content, re.DOTALL)
    
    for match in matches:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        questions.append(question)
        answers.append(answer)
    
    return {'questions': questions, 'answers': answers}


class TestDatasetFile:
    """Tests for dataset file existence and readability."""
    
    def test_dataset_file_exists(self):
        """Test 1: Dataset file exists."""
        assert DATASET_PATH.exists(), f"Dataset file not found at {DATASET_PATH}"
    
    def test_dataset_file_is_readable(self):
        """Test: Dataset file can be read."""
        try:
            with open(DATASET_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
            assert len(content) > 0, "Dataset file is empty"
        except Exception as e:
            pytest.fail(f"Failed to read dataset file: {e}")
    
    def test_dataset_is_not_empty(self, dataset_content):
        """Test 2: Dataset is not empty."""
        assert len(dataset_content) > 0, "Dataset content is empty"


class TestDatasetStructure:
    """Tests for dataset structure and format."""
    
    def test_minimum_number_of_examples(self, parsed_dataset):
        """Test 3: Dataset contains at least 200 examples."""
        num_examples = len(parsed_dataset['questions'])
        assert num_examples >= 200, (
            f"Dataset has {num_examples} examples, but minimum is 200"
        )
    
    def test_equal_questions_and_answers(self, parsed_dataset):
        """Test: Number of questions equals number of answers."""
        num_questions = len(parsed_dataset['questions'])
        num_answers = len(parsed_dataset['answers'])
        assert num_questions == num_answers, (
            f"Mismatch: {num_questions} questions but {num_answers} answers"
        )
    
    def test_every_example_has_question(self, dataset_content):
        """Test 4: Every example contains a Question."""
        question_count = len(re.findall(r'Question:', dataset_content))
        answer_count = len(re.findall(r'Answer:', dataset_content))
        assert question_count == answer_count, (
            f"Found {question_count} questions but {answer_count} answers"
        )
    
    def test_every_example_has_answer(self, dataset_content):
        """Test 5: Every example contains an Answer."""
        # Same as above - if all questions have matching answers
        question_count = len(re.findall(r'Question:', dataset_content))
        answer_count = len(re.findall(r'Answer:', dataset_content))
        assert question_count == answer_count, (
            f"Found {question_count} questions but {answer_count} answers"
        )


class TestQuestionQuality:
    """Tests for question quality and validity."""
    
    def test_questions_not_empty(self, parsed_dataset):
        """Test 6: Questions are not empty."""
        for i, question in enumerate(parsed_dataset['questions'], 1):
            assert len(question) > 0, f"Question {i} is empty"
    
    def test_questions_have_minimum_length(self, parsed_dataset):
        """Test: Questions have reasonable minimum length."""
        min_length = 5
        for i, question in enumerate(parsed_dataset['questions'], 1):
            assert len(question) >= min_length, (
                f"Question {i} is too short ({len(question)} chars): {question}"
            )
    
    def test_questions_are_unique(self, parsed_dataset):
        """Test 8: Questions are unique (no duplicates)."""
        questions = parsed_dataset['questions']
        unique_questions = set(questions)
        num_duplicates = len(questions) - len(unique_questions)
        assert num_duplicates == 0, (
            f"Found {num_duplicates} duplicate questions"
        )
    
    def test_questions_formatted_correctly(self, parsed_dataset):
        """Test: Questions are properly formatted."""
        for i, question in enumerate(parsed_dataset['questions'], 1):
            # Questions should start with capital letter
            assert question[0].isupper(), (
                f"Question {i} does not start with capital letter: {question}"
            )
            # Questions should end with question mark
            assert question.endswith('?'), (
                f"Question {i} does not end with '?': {question}"
            )


class TestAnswerQuality:
    """Tests for answer quality and validity."""
    
    def test_answers_not_empty(self, parsed_dataset):
        """Test 7: Answers are not empty."""
        for i, answer in enumerate(parsed_dataset['answers'], 1):
            assert len(answer) > 0, f"Answer {i} is empty"
    
    def test_answers_have_minimum_length(self, parsed_dataset):
        """Test: Answers have reasonable minimum length."""
        min_length = 10
        for i, answer in enumerate(parsed_dataset['answers'], 1):
            assert len(answer) >= min_length, (
                f"Answer {i} is too short ({len(answer)} chars): {answer}"
            )
    
    def test_answers_contain_meaningful_text(self, parsed_dataset):
        """Test 9: Answers contain meaningful text."""
        for i, answer in enumerate(parsed_dataset['answers'], 1):
            # Answer should have at least one word
            words = answer.split()
            assert len(words) >= 3, (
                f"Answer {i} has too few words ({len(words)}): {answer}"
            )
            # Answer should not be all numbers
            non_number_chars = sum(1 for c in answer if not c.isdigit() and not c.isspace())
            assert non_number_chars > 0, (
                f"Answer {i} contains only numbers/spaces: {answer}"
            )
    
    def test_answers_start_with_capital(self, parsed_dataset):
        """Test: Answers start with capital letter."""
        for i, answer in enumerate(parsed_dataset['answers'], 1):
            assert answer[0].isupper(), (
                f"Answer {i} does not start with capital: {answer}"
            )
    
    def test_answers_end_with_period(self, parsed_dataset):
        """Test: Answers end with punctuation."""
        for i, answer in enumerate(parsed_dataset['answers'], 1):
            assert answer[-1] in '.!?', (
                f"Answer {i} does not end with punctuation: {answer}"
            )


class TestDatasetParsing:
    """Tests for dataset parsing and robustness."""
    
    def test_dataset_can_be_parsed_without_errors(self, dataset_content):
        """Test 10: Dataset can be parsed without errors."""
        try:
            qa_pattern = r'Question:\s*(.+?)\nAnswer:\s*(.+?)(?=\n(?:Category:|Question:|$))'
            matches = re.finditer(qa_pattern, dataset_content, re.DOTALL)
            parsed_count = sum(1 for _ in matches)
            assert parsed_count > 0, "No Q&A pairs could be parsed"
        except Exception as e:
            pytest.fail(f"Dataset parsing failed: {e}")
    
    def test_all_lines_have_valid_format(self, dataset_content):
        """Test: Dataset lines follow expected format."""
        lines = dataset_content.strip().split('\n')
        valid_line_types = {'Category:', 'Question:', 'Answer:', ''}
        
        for i, line in enumerate(lines, 1):
            # Skip empty lines
            if not line.strip():
                continue
            
            # Check if line starts with expected prefix
            found_valid = False
            for prefix in valid_line_types:
                if line.startswith(prefix):
                    found_valid = True
                    break
            
            assert found_valid, (
                f"Line {i} has invalid format: {line[:50]}"
            )
    
    def test_dataset_consistency(self, parsed_dataset):
        """Test: Dataset maintains consistency across questions and answers."""
        questions = parsed_dataset['questions']
        answers = parsed_dataset['answers']
        
        # Basic consistency checks
        assert len(questions) > 0, "No questions found"
        assert len(answers) > 0, "No answers found"
        assert len(questions) == len(answers), "Questions/answers count mismatch"


class TestDatasetCoverage:
    """Tests for dataset topic coverage and diversity."""
    
    def test_multiple_categories_present(self, dataset_content):
        """Test: Dataset covers multiple categories."""
        categories = set(re.findall(r'Category:\s*(.+)', dataset_content))
        assert len(categories) >= 5, (
            f"Only {len(categories)} categories found, expected at least 5"
        )
    
    def test_question_variety(self, parsed_dataset):
        """Test: Questions are varied, not just one template."""
        questions = parsed_dataset['questions']
        
        # Count question lengths
        lengths = [len(q) for q in questions]
        length_variance = max(lengths) - min(lengths)
        
        assert length_variance > 30, (
            "Questions show too little length variation"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
