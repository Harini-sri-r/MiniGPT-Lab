"""Interactive command-line interface for the trained educational MiniGPT."""

from pathlib import Path

import torch

from generation.generate import generate
from generation.load_model import load_minigpt


def parse_generation_settings(max_tokens_text, temperature_text, top_k_text):
    """Parse optional CLI settings and raise friendly errors for bad input."""
    try:
        max_new_tokens = int(max_tokens_text or 100)
        temperature = float(temperature_text or 0.8)
        top_k = int(top_k_text) if top_k_text else None
    except ValueError as error:
        raise ValueError("Max tokens and top-k must be integers; temperature must be a number.") from error

    if max_new_tokens < 1:
        raise ValueError("Max new tokens must be at least 1.")
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if top_k is not None and top_k < 1:
        raise ValueError("Top-k must be at least 1 when provided.")
    return max_new_tokens, temperature, top_k


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=None):
    """Encode a prompt, generate tokens, and decode the complete result."""
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    try:
        token_ids = tokenizer.encode(prompt)
    except KeyError as error:
        raise ValueError(
            f"Prompt contains a character outside this model's vocabulary: {error.args[0]!r}."
        ) from error

    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
    )
    return tokenizer.decode(output_ids[0].tolist())


def run_cli(model, tokenizer, input_function=input, output_function=print):
    """Run the prompt loop; injectable I/O keeps its noninteractive parts testable."""
    output_function("MiniGPT Text Generator\n======================")
    output_function("Type 'exit' or 'quit' to leave. Press Enter for defaults.")

    while True:
        prompt = input_function("MiniGPT> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            output_function("Goodbye.")
            return
        if not prompt:
            output_function("Please enter a prompt.")
            continue

        try:
            max_tokens = input_function("Max new tokens [100]: ").strip()
            temperature = input_function("Temperature [0.8]: ").strip()
            top_k = input_function("Top-k [none]: ").strip()
            max_new_tokens, temperature, top_k = parse_generation_settings(
                max_tokens, temperature, top_k
            )
            text = generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k)
            output_function(f"Generated:\n{text}")
        except ValueError as error:
            output_function(f"Invalid input: {error}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = project_root / "checkpoints" / "minigpt.pt"
    model, tokenizer = load_minigpt(checkpoint_path)
    run_cli(model, tokenizer)


if __name__ == "__main__":
    main()
