import torch
import torch.nn.functional as F


def generate(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    do_sample=True,
):
    """Autoregressively append tokens predicted by a decoder-only model.

    The returned tensor includes the original prompt followed by newly
    generated token IDs.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch_size, sequence_length).")
    if input_ids.size(1) == 0:
        raise ValueError("input_ids must contain at least one prompt token.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative.")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 when provided.")

    was_training = model.training
    model.eval()
    generated_ids = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # The model only has positional embeddings for this many tokens.
            context = generated_ids[:, -model.max_sequence_length :]
            logits = model(context)[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            probabilities = F.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(probabilities, dim=-1, keepdim=True)

            generated_ids = torch.cat((generated_ids, next_token), dim=1)

    if was_training:
        model.train()
    return generated_ids
