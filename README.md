# gpt∀

> A from-scratch GPT implementation in PyTorch — supporting both a custom-trained language model and OpenAI's pretrained GPT-2 weights.

---

## Overview

**gpt∀** is a clean, modular implementation of the GPT architecture built entirely from scratch using PyTorch. It is designed to deepen understanding of how large language models work — from tokenization and attention all the way through training and text generation.

The project supports two modes of operation:

| Mode | Script | Description |
|------|--------|-------------|
| Custom GPT | `custom_gpt.py` | Runs your own trained model from a saved checkpoint |
| Pretrained GPT-2 | `gpt2.py` | Downloads and loads official OpenAI GPT-2 weights |

---

## Features

- **Text Generation** — Autoregressive text generation with temperature scaling and top-k sampling
- **Custom Training Pipeline** — Train a GPT model on any plain-text dataset with configurable hyperparameters
- **Pretrained Weight Loading** — Automatically downloads and loads OpenAI GPT-2 weights (124M, 355M, 774M, 1558M)
- **CLI-Based Prompt Input** — Pass prompts and generation parameters directly via command line using `argparse`
- **Modular Architecture** — Cleanly separated components: attention, feed-forward, normalization, transformer blocks

---

## Project Structure

```
gpt-core/
│
├── GPT/
│   └── gpt_model.py              # Full GPT model definition
│
├── core/
│   ├── attention/
│   │   └── multi_head_attention.py   # Multi-head causal self-attention
│   ├── feedforward/
│   │   └── feed_forward.py           # Position-wise feed-forward network
│   ├── layers/
│   │   └── normalization.py          # Layer normalization
│   ├── activations/
│   │   └── gelu.py                   # GELU activation function
│   └── transformer/
│       └── transformer_block.py      # Transformer block (Attn + FFN + Norm)
│
├── config/
│   └── config.py                 # Model configs (GPT_CONFIG, NEW_CONFIG)
│
├── tokenizer/
│   └── tokenizer_utils.py        # Tokenization via tiktoken (GPT-2 BPE)
│
├── dataset/
│   ├── prepare/
│   │   ├── fetch_data.py         # Fetch raw text data
│   │   └── merge_dataset.py      # Merge and analyze text data
│   ├── dataloaders/
│   │   └── dataloaders.py        # Sliding-window dataset + DataLoader
│   ├── loss/
│   │   └── calculate_loss.py     # Batch and loader loss computation
│   ├── training/
│   │   └── train_gpt.py          # Core training loop
│   ├── train_val_split.py        # Train/validation split
│   └── train.py                  # Training entry point
│
├── utils/
│   ├── device.py                 # CUDA/CPU device selection
│   ├── text_generation.py        # Greedy text generation
│   └── text_generation_top_k.py  # Top-k + temperature generation
│
├── preTrainedWeights/
│   ├── load_weights.py           # Weight mapping from GPT-2 to this model
│   └── gpt_2/
│       └── download_and_load.py  # Download + parse GPT-2 TF checkpoints
│
├── main.py                       # Quick forward-pass test
├── custom_gpt.py                 # Run custom-trained GPT model
├── gpt2.py                       # Run pretrained GPT-2 model
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SankalpNaranje/gpt-core.git
cd gpt-core
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `tensorflow` is required only for loading OpenAI's original GPT-2 checkpoints. PyTorch is used for all model training and inference.

---

## Dataset & Training

### Dataset Preparation

Place your plain-text `.txt` dataset(s) inside the `dataset/prepare/` directory. The `merge_dataset.py` script will read and merge them into a single training corpus.

### Training Workflow

```
Raw Text Files
     │
     ▼
merge_dataset.py  →  Tokenized Text
     │
     ▼
dataloaders.py    →  Sliding-window DataLoader (input/target pairs)
     │
     ▼
train_gpt.py      →  Training loop (loss, backprop, optimizer step)
     │
     ▼
train.py          →  Entry point (saves checkpoint: model_and_optimizer.pth)
```

### Start Training

```bash
python -m dataset.train
```

The trained model is saved as `model_and_optimizer.pth` in the project root.

---

## How to Run

### Run Custom-Trained GPT

Uses your own trained checkpoint (`model_and_optimizer.pth`):

```bash
python custom_gpt.py
```

To also print model statistics (parameter count, size in MB):

```bash
python custom_gpt.py --stats
```

---

### Run Pretrained GPT-2

Uses OpenAI's pretrained GPT-2 (124M) weights. Downloads them automatically on first run and caches as `gpt2_pretrained.pth`:

```bash
python gpt2.py
```

With a custom prompt:

```bash
python gpt2.py --prompt "The future of AI is" --max_tokens 50 --temperature 1.0 --top_k 40
```

---

## CLI Reference (`gpt2.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prompt` | `str` | `"Hi, Greetings!"` | Input prompt for text generation |
| `--max_tokens` | `int` | `25` | Maximum number of tokens to generate |
| `--temperature` | `float` | `1.4` | Sampling temperature (higher = more random) |
| `--top_k` | `int` | `25` | Top-k candidates to sample from |

**Example:**

```bash
python gpt2.py --prompt "Once upon a time" --max_tokens 100 --temperature 0.8 --top_k 50
```

---

## Model Configuration

Defined in `config/config.py`:

```python
GPT_CONFIG = {
    "vocab_size": 50257,     # GPT-2 BPE vocabulary
    "context_length": 256,   # Reduced for faster custom training
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Transformer depth
    "drop_rate": 0.1,        # Dropout regularization
    "qkv_bias": False        # No QKV bias for custom model
}
```

For pretrained GPT-2, `NEW_CONFIG` overrides context length to 1024 and enables QKV bias to match OpenAI's architecture.

---

## Requirements

```
torch
tiktoken
numpy
requests
tensorflow>=2.15.0
tqdm>=4.66
```

---

## License

This project is licensed under the [MIT License](LICENSE).
