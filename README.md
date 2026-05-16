# GSEC - Generative Semantic Guidance and Bi-Layer Ensemble for Image Clustering

An unsupervised image clustering framework that leverages multimodal LLMs to generate adaptive semantic descriptions and weighted semantic embeddings to reduce clustering bias. It combines inner-layer BatchEnsemble cross-modal integration with outer-layer alignment to reduce model variance, achieving more robust and accurate clustering on unlabeled image data.

## Project Structure

```
GSEC/
├── clip_culster.py          # Step 1: Feature clustering + LLM description generation + CLIP text encoding
├── retrieve_text.py         # Step 2: Text feature
├── ensemble_en4_jian.py     # Step 3: BatchEnsemble ensemble clustering training
└── data/
    ├── path/                # Image path files per CLIP backbone
    │   ├── clipvitB32/
    │   ├── clipRN50/
    │   ├── clipRN101/
    │   └── clipRN50x4/
    ├── representations/     # Pre-extracted feature vectors (.npy)
    ├── labels/              # Ground-truth validation labels (.npy)
    └── text/                # CLIP-encoded text embeddings
```

## Pipeline Overview

### Step 1: Clustering and Description Generation (`clip_culster.py`)

1. Load pre-extracted image features
2. Perform K-means clustering on training features
3. Uniformly select 5 representative samples from each cluster
4. Generate structured descriptions for each sample using Ollama (`llama3.2-vision:11b`)
5. Encode descriptions into 512-dim vectors using the CLIP text encoder

Output is saved to `./cluster/` and `./text/`.

### Step 2: Text Feature (`retrieve_text.py`)

1. Load text embeddings and image embeddings
2. Compute image-text similarity (softmax-weighted)
3. Generate per-image text embeddings (weighted average of text embeddings)
4. Concatenate image embeddings with text embeddings to form augmented features

Output is saved to `./data/representations/`.

### Step 3: Ensemble Training (`ensemble_en4_jian.py`)

1. Load augmented training/validation features and labels
2. Build a BatchEnsemble multi-modal classifier (image branch + text branch)
3. Training components:
   - Task encoder (linear layers + softmax pseudo-label generation)
   - Inner classifier (BatchEnsemble + knowledge distillation)
4. Evaluation metrics: Acc, NMI, ARI

## Usage

### Step 1: Clustering and Description Generation

```bash
python clip_culster.py --dataset cifar10 --phis clipvitB32
```

Arguments:
- `--dataset`: Dataset name
- `--phis`: Feature space (CLIP backbone)

Requires a locally running Ollama instance with the `llama3.2-vision:11b` model.

### Step 2: Text Feature

```bash
python retrieve_text.py --dataset cifar10 --phis clipvitB32 --tau 0.005
```

Arguments:
- `--dataset`: Dataset name
- `--phis`: Feature space
- `--tau`: Temperature parameter

### Step 3: Ensemble Training

```bash
python ensemble_en4_jian.py --dataset cifar10 --phis clipvitB32
```

Key training arguments:
- `--dataset`: Dataset name
- `--phis`: Feature space(s)
- `--gamma`: Entropy regularization strength (default: `50.0`)
- `--T`: Total iterations (default: `6000`)
- `--inner_lr`: Inner loop learning rate (default: `0.001`)
- `--outer_lr`: Outer loop learning rate (default: `0.01`)
- `--batch_size`: Batch size (default: `1000`)
- `--M`: Inner loop steps (default: `10`)
- `--k`: BatchEnsemble members (default: `32`)
- `--topk`: Nearest neighbors count (default: `3`)
- `--grad_clip`: Gradient clipping (default: `1.0`)

## Dependencies

- Python 3.8+
- PyTorch
- CLIP (`openai/CLIP`)
- scikit-learn
- NumPy
- Ollama (local deployment with `llama3.2-vision:11b` model)
- tabm
- tqdm

## Data Directory

- `data/path/<backbone>/` — Image file paths per sample (used for MLLM visual description generation)
- `data/representations/<backbone>/` — Pre-extracted feature vectors (`.npy` format)
- `data/labels/` — Ground-truth validation labels
- `data/text/` — CLIP-encoded text embeddings
- `cluster/` — Clustering results (labels, centers, representative samples, descriptions)
