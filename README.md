![Image alt](https://github.com/mhassan-278/GPT-from-scratch/blob/main/decoder_only_transformer.png?raw=true)

# 🧠 GPT from Scratch (Decoder-Only Transformer)

## 🚀 Overview
This project implements a **GPT-style decoder-only Transformer** from scratch using PyTorch, focusing on understanding the core mechanics behind modern large language models.

Instead of using high-level libraries, this implementation builds every component manually, including attention, masking, and autoregressive generation.

---

## 🎯 Objectives
- Understand the internal working of GPT architectures
- Implement Transformer components from first principles
- Build a minimal yet functional language model
- Gain intuition for training dynamics and sequence modeling

---

## 🏗️ Architecture

This project follows the **decoder-only Transformer architecture**

### Key Components:
- Token Embeddings
- Positional Embeddings
- Multi-Head Self-Attention
- Causal Masking (prevents future token leakage)
- Feedforward Neural Network (MLP)
- Residual Connections & Layer Normalization

---

## ⚙️ How It Works

1. Input text is converted into token indices
2. Tokens are embedded into dense vectors
3. Causal self-attention learns relationships between tokens
4. Model predicts the next token in an autoregressive manner
5. Text is generated iteratively

---
