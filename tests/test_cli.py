from pathlib import Path

import pytest

from app.minigpt_cli import generate_text, parse_generation_settings
from generation.load_model import load_minigpt


CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "minigpt.pt"


def test_model_loading_reconstructs_evaluation_ready_model_and_tokenizer():
    model, tokenizer = load_minigpt(CHECKPOINT_PATH)

    assert not model.training
    assert len(tokenizer.characters) == model.vocab_size


def test_generation_returns_a_string():
    model, tokenizer = load_minigpt(CHECKPOINT_PATH)

    output = generate_text(model, tokenizer, "artificial intelligence", max_new_tokens=1)

    assert isinstance(output, str)
    assert output.startswith("artificial intelligence")


@pytest.mark.parametrize(
    "settings", [("0", "0.8", ""), ("10", "0", ""), ("10", "0.8", "0")]
)
def test_invalid_parameters_are_handled(settings):
    with pytest.raises(ValueError):
        parse_generation_settings(*settings)
