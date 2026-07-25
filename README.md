# 🤖 Transformer from Scratch

A clean, well-documented implementation of the **Transformer architecture** from the seminal paper [*"Attention Is All You Need"* (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), built entirely in **PyTorch** from the ground up.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Components](#-components)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dependencies](#-dependencies)
- [Reference](#-reference)

---

## 🌟 Overview

This project implements the full Transformer model — including the Encoder, Decoder, Multi-Head Attention, Positional Encoding, Feed-Forward Network, Layer Normalization, and Residual Connections — from scratch using PyTorch primitives.

The goal is educational: every module is thoroughly commented to explain *why* each design choice exists, not just *what* the code does.

---

## 🏗️ Architecture

The Transformer follows an Encoder-Decoder structure:

```
Input Tokens
     │
     ▼
Input Embeddings  ──── scaled by √d_model
     │
     ▼
Positional Encoding  ── sinusoidal (fixed, non-learnable)
     │
     ▼
┌─────────────────────────────┐
│         Encoder (×N)        │
│  ┌───────────────────────┐  │
│  │  Multi-Head Self-Attn │  │
│  │  + Residual + LayerNorm│  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Feed-Forward Network │  │
│  │  + Residual + LayerNorm│  │
│  └───────────────────────┘  │
└─────────────────────────────┘
     │  (encoder output)
     ▼
┌─────────────────────────────┐
│         Decoder (×N)        │
│  ┌───────────────────────┐  │
│  │  Masked Self-Attention│  │
│  │  + Residual + LayerNorm│  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Cross-Attention      │  │  ◄── attends to encoder output
│  │  + Residual + LayerNorm│  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Feed-Forward Network │  │
│  │  + Residual + LayerNorm│  │
│  └───────────────────────┘  │
└─────────────────────────────┘
     │
     ▼
Projection Layer (Linear + log_softmax)
     │
     ▼
Output Probabilities
```

**Default Hyperparameters** (matching the original paper):

| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `N` (layers) | 6 |
| `h` (attention heads) | 8 |
| `d_ff` (FFN inner dim) | 2048 |
| `dropout` | 0.1 |

---

## 📁 Project Structure

```
transformer/
├── README.md
├── main.py                        # Entry point
├── model.py                       # Transformer, Encoder, Decoder, ProjectionLayer, build_transformer()
├── encoder_component/
│   ├── input_embedding.py         # InputEmbeddings — token ID → dense vector
│   ├── positional_encoding.py     # PositionalEncoding — sinusoidal position signals
│   ├── layer_normalization.py     # LayerNormalization — with learnable γ and β
│   ├── feed_forward_network.py    # FeedForwardBlock — two linear layers with ReLU
│   ├── Multihead_attention.py     # MultiHeadAttentionBlock — scaled dot-product attention
│   ├── residual_connection.py     # ResidualConnection — skip connection + layer norm
│   └── encoder_block.py          # EncoderBlock — self-attention + FFN + residuals
├── decoder_component/
│   └── decoder_block.py          # DecoderBlock — masked self-attn + cross-attn + FFN
├── requirements.txt
├── pyproject.toml
├── .python-version
└── .gitignore
```

---

## 🧩 Components

### `InputEmbeddings` — `encoder_component/input_embedding.py`

Converts integer token IDs into dense embedding vectors using a learnable lookup table (`nn.Embedding`). Following the paper, embeddings are scaled by **√d_model** so their magnitude is comparable to the positional encodings before addition.

```
(batch, seq_len)  →  (batch, seq_len, d_model)
```

---

### `PositionalEncoding` — `encoder_component/positional_encoding.py`

Injects position information into token embeddings using fixed sinusoidal signals:

```
PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
```

- The encoding matrix is pre-computed once in `__init__` and stored as a **non-trainable buffer** (saved in `state_dict`, moves with the model to GPU, but not updated by the optimizer).
- Dropout is applied after adding positional encoding to token embeddings.

---

### `LayerNormalization` — `encoder_component/layer_normalization.py`

Normalizes across the embedding dimension for each token independently. Includes two learnable scalar parameters:
- **α (alpha)** — scale (multiplied), initialized to 1
- **β (bias)** — shift (added), initialized to 0

A small epsilon (`1e-6`) is added for numerical stability.

---

### `FeedForwardBlock` — `encoder_component/feed_forward_network.py`

A position-wise two-layer fully connected network applied identically to each token:

```
FFN(x) = Linear2( Dropout( ReLU( Linear1(x) ) ) )

(batch, seq_len, d_model) → (batch, seq_len, d_ff) → (batch, seq_len, d_model)
```

---

### `MultiHeadAttentionBlock` — `encoder_component/Multihead_attention.py`

Implements scaled dot-product attention across `h` heads in parallel:

1. Projects Q, K, V with separate linear layers (`W_q`, `W_k`, `W_v`).
2. Splits into `h` heads, each of dimension `d_k = d_model / h`.
3. Computes attention scores: `softmax(QK^T / √d_k) · V`.
4. Applies an optional mask (set to `-1e9` before softmax to effectively zero out masked positions).
5. Concatenates heads and projects with `W_o`.

The last computed attention scores are stored in `self.attention_score` for inspection.

---

### `ResidualConnection` — `encoder_component/residual_connection.py`

Implements the **Add & Norm** pattern used throughout the Transformer:

```
output = x + Dropout( sublayer( LayerNorm(x) ) )
```

> **Note:** This implementation uses **Pre-LN** (normalization before the sublayer), which differs slightly from the original paper's **Post-LN**, but is widely used in practice for improved training stability.

---

### `EncoderBlock` — `encoder_component/encoder_block.py`

A single Transformer encoder layer combining:
1. Self-attention with residual connection
2. Feed-forward network with residual connection

Takes an optional `src_mask` to ignore padding tokens.

---

### `DecoderBlock` — `decoder_component/decoder_block.py`

A single Transformer decoder layer with three sub-layers:
1. **Masked self-attention** (with `tgt_mask` to prevent attending to future tokens)
2. **Cross-attention** over encoder output (with `src_mask` to ignore source padding)
3. **Feed-forward network**

Each sub-layer is wrapped in a residual connection.

---

### `Transformer` & `build_transformer()` — `model.py`

The top-level model that wires all components together. The `build_transformer()` factory function:
- Creates embedding layers for source and target vocabularies.
- Stacks `N` encoder and decoder blocks.
- Adds a final projection layer (linear → log_softmax) to produce output token log-probabilities.
- Initializes all parameters with **Xavier uniform initialization** for stable training.

---

## ⚙️ Installation

### Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### With `uv` (recommended)

```bash
uv sync
```

### With `pip`

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python main.py
```

To build and use the transformer in your own script:

```python
from model import build_transformer

model = build_transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    src_seq_len=512,
    tgt_seq_len=512,
    d_model=512,
    N=6,
    h=8,
    droupout=0.1,
    d_ff=2048
)

# Encode source sequence
encoder_output = model.encode(src_tokens, src_mask)

# Decode target sequence
decoder_output = model.decode(encoder_output, src_mask, tgt_tokens, tgt_mask)

# Project to vocabulary log-probabilities
log_probs = model.project(decoder_output)
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Core deep learning framework (PyTorch) |
| `numpy` | Numerical utilities |
| `transformers` | Tokenizers and pretrained models (HuggingFace) |
| `nltk` | Natural language processing utilities |
| `ipykernel` | Jupyter notebook / interactive kernel support |

---

## 📄 Reference

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
> **Attention Is All You Need.**
> *Advances in Neural Information Processing Systems*, 30.
> https://arxiv.org/abs/1706.03762
