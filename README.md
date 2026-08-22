# MiniGPT-Lab

An **educational GPT-style language model built from scratch** with Python and PyTorch. MiniGPT implements the core decoder-only Transformer mechanics manually: character tokenization, embeddings, causal multi-head attention, feed-forward layers, normalization, residual paths, training, evaluation, and autoregressive generation.

It does not use pretrained models, external LLM APIs, or `nn.MultiheadAttention`.

## What it does

MiniGPT learns next-character prediction from a small original educational corpus, then appends predicted characters one at a time to a prompt. It is deliberately small so that every major LLM component is inspectable.

## Architecture

```text
Text
  ↓
Character tokenizer
  ↓
Token embedding + positional embedding
  ↓
Transformer block × N
  ├─ LayerNorm → multi-head causal attention → dropout → residual add
  └─ LayerNorm → feed-forward network → dropout → residual add
  ↓
Final LayerNorm
  ↓
Weight-tied LM head
  ↓
Vocabulary logits
  ↓
Autoregressive generation
```

The model predicts vocabulary logits at every sequence position. Causal masking prevents a token from attending to future tokens.

## Project structure

```text
MiniGPT/
├─ app/minigpt_cli.py          # Interactive MiniGPT CLI
├─ checkpoints/                # Local trained checkpoints (ignored by Git)
├─ data/training.txt           # Small educational corpus
├─ evaluation/evaluate.py      # Loss, perplexity, and reports
├─ experiments/                # Demos, comparisons, and plotting
├─ generation/
│  ├─ generate.py              # Autoregressive decoding
│  └─ load_model.py            # Checkpoint and tokenizer loader
├─ model/                      # Embeddings, attention, blocks, and MiniGPT
├─ tests/                      # MiniGPT unit tests
├─ tokenizer/tokenizer.py      # Character-level tokenizer
├─ training/
│  ├─ dataset.py               # Shifted next-token windows
│  └─ train.py                 # AdamW training loop
├─ requirements.txt
└─ README.md
```

## Technology

- Python 3.11+
- PyTorch
- Matplotlib for loss plots
- Pytest for tests

## Trained experiment

The included local checkpoint was trained with a compact learning configuration:

```text
Vocabulary size: 36 characters
Embedding dimension: 32
Attention heads: 4
Hidden dimension: 128
Transformer layers: 2
Context length: 32
Trainable parameters: 27,648
Dataset: 2,081 characters
```

Training is next-token prediction:

```text
training.txt → tokenizer → token IDs → shifted input/target pairs
→ MiniGPT → logits → CrossEntropyLoss → backpropagation → AdamW → updated weights
```

For an input such as `"The "`, the target is `"he c"`: each character position learns to predict the immediately following character. `CrossEntropyLoss` measures the mismatch between vocabulary logits and target tokens. Backpropagation computes gradients, and AdamW updates the learnable weights.

## Results

Verified training and evaluation results from the short educational run:

```text
Tests:                    61 passed (before the final CLI tests)
Initial training loss:    12.2258
Final training loss:       3.0707
Initial validation loss:   5.2288
Final validation loss:     2.8085
Validation perplexity:    16.5849
```

Lower loss and perplexity generally mean less next-token uncertainty, though neither metric alone proves good generated text.

Example output is intentionally rough because this is a tiny model trained on a tiny corpus:

```text
Prompt: artificial intelligence
Generated: artificial intelligencetaato e cea ceresee ...
```

## Install and run

Create and activate a virtual environment, then install the MiniGPT dependencies:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Train the small model (creates an ignored local checkpoint):

```bash
python -m experiments.training_demo
```

Run the interactive generator:

```bash
python -m app.minigpt_cli
```

Run evaluation and generation experiments:

```bash
python -m experiments.evaluate_model
python -m experiments.generation_comparison
python -m experiments.plot_training
```

Run the MiniGPT test suite:

```bash
python -m pytest tests/test_multi_head_attention.py tests/test_feed_forward.py tests/test_layer_norm.py tests/test_residual.py tests/test_transformer_block.py tests/test_transformer.py tests/test_minigpt.py tests/test_dataset.py tests/test_training.py tests/test_generation.py tests/test_evaluation.py tests/test_cli.py
```

## Limitations

MiniGPT is a learning project, not a modern production LLM. It has a very small dataset and model, a character-level tokenizer, CPU-oriented short training, limited context, and poor language quality. It has no instruction tuning, RLHF, large-scale pretraining, factual retrieval, or external knowledge source.

## Phase 6: Training Improvements

In Phase 6, we improved the model's generation quality without changing its unde![Architecture Diagram](docs/assets/architecture.png) The original model generated poor, repetitive answers because its dataset was extremely small (201 examples) and the learning strategy was naive. 

To fix this, we implemented several improvements:
- **Expanded Dataset**: We increased the dataset to 510 high-quality, structured Q&A examples. More data gives the model a wider linguistic distribution to learn from, reducing overfitting and repetitive loops.
- **AdamW & Cosine Learning Rate Decay**: We adopted the AdamW optimizer with a cosine learning rate decay scheduler. This allows the model to take large steps early on to escape bad local minima, then fine-tune its weights as the learning rate smoothly decays towards zero.
- **Linear Warmup**: We added a warmup phase that linearly increases the learning rate at the start of training. This prevents early massive gradients from destabilizing the randomly initialized weights.
- **Gradient Clipping**: We clamp gradients to a maximum norm to prevent exploding gradients.
- **Evaluation Isolation**: We keep the baseline checkpoint to explicitly compare metrics and generation outputs. We measure objective improvements via validation perplexity and generation metrics like unique token ratio and repeated n-grams.

Despite these improvements, there are intrinsic limitations: training a 1M parameter model from scratch on CPU with limited context will not rival modern LLMs.

## Future improvements

Reasonable next steps are a larger corpus, a subword tokenizer, a larger model, longer GPU training, learning-rate scheduling, checkpoint resume, stronger evaluation, a longer context window, instruction tuning, and additional generation strategies.

## MiniGPT Commerce Agent (Phase 7)

Phase 7 adds an isolated, deterministic product-comparison foundation. It does not alter MiniGPT v2's BPE tokenizer, Transformer, training pipeline, or checkpoints.

```text
User Query
    ↓
Requirement Parser
    ↓
Product Search
    ↓
Product Normalization
    ↓
Cross-Platform Comparison
    ↓
Recommendation Engine
    ↓
Recommendation
```

**CURRENT PHASE USES MOCK DATA.** The catalog has 102 synthetic listings across Amazon, Flipkart, and Meesho labels. Amazon, Flipkart, and Meesho are **not** queried live; prices, ratings, URLs, and availability are fabricated solely for safe, repeatable testing.

The foundation supports category, maximum-price, minimum-rating, brand, and platform filtering; deterministic name grouping; and a configurable recommendation formula:

```text
score = 0.35 × price_score + 0.30 × rating_score
      + 0.15 × review_score + 0.20 × requirement_match
```

The rule-based requirement parser can identify common categories, INR budgets, simple product features, and use cases such as programming or gaming. Run the demonstration with:

```bash
python -m experiments.commerce_demo
```

The eventual architecture can support real product sources only through compliant APIs or otherwise permitted data sources.

## Commerce Assistant (Phase 8)

Phase 8 evolves the Phase 7 comparison foundation into an offline, explainable shopping-assistant architecture. It remains completely separate from MiniGPT v1/v2 training, BPE, Transformer code, and checkpoints.

```text
Shopping query → Rule-based query understanding → Marketplace provider adapters
              → Normalized listings → Likely-product matching → Transparent scoring
              → Best match, alternatives, comparison table, and explanation
```

The provider interface (`MarketplaceProvider`) supplies `search_products`, `get_product`, and `normalize_product`. The Amazon, Flipkart, and Meesho adapters all use the same reusable mock-provider implementation and the existing synthetic catalog—there is no web scraping, browser automation, or live marketplace request.

The richer parser extracts category, INR budget (including `70k`), brand, minimum rating, use case, and simple requested features. Likely equivalence is a deterministic confidence heuristic based on normalized brand/title/category and feature similarity; it is not claimed to be perfect matching.

Phase 8 recommendation scoring is configurable and reports every component:

```text
30% price + 25% rating + 15% reviews + 20% feature match + 10% requirement match
```

Use the interactive terminal assistant:

```bash
python -m app.commerce_cli
```

For the ten-query offline evaluation run:

```bash
python -m experiments.commerce_phase8_evaluation
```

**Current:** offline synthetic product data only. Prices and availability are not live, verified marketplace information.

**Future:** a legitimate marketplace API or permitted data provider can implement the same provider interface. Live prices must not be claimed unless such a source is actually connected.

## Commerce Decision Agent (Phase 9)

Phase 9 adds a modular, explicit multi-step decision agent over the same offline catalog:

```text
Query → plan → validate → provider search → hard-constraint filter
      → equivalent-product matching → comparison → transparent scoring
      → recommendation, explanation, and alternatives
```

```bash
python -m experiments.commerce_phase9_evaluation
```

**Current:** offline synthetic marketplace data. **Not implemented:** live APIs, scraping, browser automation, purchasing, payments, account access, or external LLM APIs.

## Explainable Commerce Decision Assistant (Phase 10)

Phase 10 polishes the offline agent into an evaluation-driven decision assistant without changing its mock-data boundary. Rich comparisons expose each listing’s product, brand, category, platform, price, rating, review count, relevant structured features, requirement match, price/rating difference, and deterministic value score.

Every recommendation now exposes price/rating/review/feature/requirement score components plus the configured weights. Its factual explanation covers hard constraints, use case, price, rating, reviews, features, a trade-off, and a score-based reason alternatives rank lower. The structured decision trace reports concise actions and factual metadata only; it does not expose hidden reasoning.

The terminal agent supports `Why this one?`, `Show alternatives`, `Only HP`, `Make the budget 70000`, `What if I need 16GB RAM?`, and `Which is cheapest?`. No-result responses retain hard constraints and provide only safe relaxation suggestions.

The provider contract now includes offline `search`, `get_product`, `availability`, and `price` operations, so compliant future sources can adopt the same interface. Run the 20-scenario Phase 10 evaluation with:

```bash
python -m experiments.commerce_phase10_evaluation
```

**Current Amazon/Flipkart/Meesho data is synthetic/offline mock data. No live marketplace prices or availability are retrieved.** Future providers must use legitimate permitted APIs/data sources; scraping, browser automation, purchasing, payments, and external LLM APIs are not implemented.
