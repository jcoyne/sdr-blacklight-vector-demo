#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.1.0",
#   "transformers>=4.52.0",
#   "tokenizers>=0.21.0",
#   "huggingface-hub>=0.27.0",
#   "accelerate>=0.34.0",
#   "numpy>=1.24.0",
# ]
# ///
"""
Local embedding helper for Qwen/Qwen3-Embedding-0.6B.

Reads a JSON array of strings from stdin, writes a JSON array of float
arrays (one per input) to stdout, and exits.

Usage (called by Ruby via Open3):
    echo '["hello world", "foo bar"]' | uv run script/embed.py
"""

import json
import sys


def last_token_pool(last_hidden_states, attention_mask):
    """Pool the last non-padding token from each sequence."""
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "Qwen/Qwen3-Embedding-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()

    return tokenizer, model


def encode(texts, tokenizer, model, max_length=8192):
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = last_token_pool(
            outputs.last_hidden_state, inputs["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.float().tolist()


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print("[]")
        return

    try:
        texts = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON input: {exc}"}), file=sys.stderr)
        sys.exit(1)

    if not isinstance(texts, list):
        print(
            json.dumps({"error": "Input must be a JSON array of strings"}),
            file=sys.stderr,
        )
        sys.exit(1)

    if len(texts) == 0:
        print("[]")
        return

    tokenizer, model = load_model()
    embeddings = encode(texts, tokenizer, model)
    print(json.dumps(embeddings))


if __name__ == "__main__":
    main()
