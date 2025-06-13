# Shakespeare.GPT

Shakespeare.GPT is a tiny language model trained on the complete works of William Shakespeare. Inspired by Andrej Karpathy's "GPT from scratch" series, this project builds a character-level generative transformer using PyTorch, with the goal of generating realistic Shakespearean-style text.

---

## 📌 Features

- Character-level language modeling
- Bigram and Transformer-based architectures
- Custom training loop using PyTorch
- GPU-accelerated training (if available)
- Sample generation with temperature and top-k sampling (optional for future)

---

## 🧠 Model Architecture

We start from a basic Bigram Language Model and gradually evolve to a self-attention based Transformer. The final model resembles the decoder-only part of the Transformer as shown in the original paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762).

Below is the reference architecture used:

![image](https://github.com/user-attachments/assets/197719c2-de8f-41c6-9bce-0efa81f53a52)


### 🔍 How This Was Used

- We implemented the **right half** of the above architecture (the decoder block).
- Key components:
  - **Input Embedding + Positional Encoding**: Converts characters into vectors with positional context.
  - **Masked Multi-Head Attention**: Allows the model to attend to previous characters only.
  - **Add & Norm**: Residual connections with Layer Normalization.
  - **Feed Forward**: A simple two-layer MLP applied to each position.
  - **Linear + Softmax**: Projects output to vocabulary logits and produces probabilities.

This architecture allows the model to learn contextual patterns and generate coherent Shakespeare-style text character by character.

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/uday-gandhi04/Shakespeare.GPT.git
cd Shakespeare.GPT
