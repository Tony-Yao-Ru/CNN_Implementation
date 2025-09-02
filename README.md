# CNN (NumPy & PyTorch) — CIFAR-10 Cats vs Dogs (Small)

This repo contains two implementations of an N-D Convolutional Neural Network:

- **`ConvolutionalNN_Numpy`** — a from-scratch NumPy CNN with forward/backward conv/pool/FC, ReLU, softmax/BCE, and SGD/Adam optimizers.
- **`ConvolutionalNN_Pytorch`** — a PyTorch CNN using Conv/BN/ReLU blocks with optional MaxPool and a GAP (+ optional MLP) head.

A lightweight runner `cifar10_cnn_runner.py` downloads CIFAR-10, **subsamples only “cat” and “dog”**, and trains/evaluates either backend. Both backends share a similar CLI and plotting.

---

## Features

- **Dimensionality:** 1D/2D/3D conv support in the NumPy model (`conv_dim ∈ {1,2,3}`).
- **Layers/ops (NumPy):** Conv, MaxPool, ReLU, softmax or sigmoid, cross-entropy / BCE-with-logits.
- **Optimizers (NumPy):** SGD and Adam (with weight decay option).
- **PyTorch model:** Conv+BN+ReLU stacks, optional MaxPool frequency (`pool_every`), **GAP head** (shape-agnostic), optional MLP (via `hidden_units`), Dropout.
- **Runner:** Consistent CLI across backends; live loss curves; simple visualization of predictions.

---

## File Layout

├── ConvolutionalNN_Numpy.py # NumPy CNN (forward/backward + trainer)
├── ConvolutionalNN_Pytorch.py # PyTorch CNN (Conv/BN/ReLU + GAP/MLP)
├── cifar10_cnn_runner.py # CLI runner; builds small cat/dog dataset
└── README.md


---

## Requirements

### Core
- Python 3.9+ recommended
- NumPy
- Matplotlib

### PyTorch backend
- PyTorch (with CUDA support optional)
- torchvision

Install (CPU example):

```bash
pip install numpy matplotlib torch torchvision
```

If you only use the NumPy backend, torch/torchvision are not strictly needed, but the runner uses torchvision to fetch CIFAR-10. If you prefer, you can pre-download or replace the data loader.

---

## Quick Start
1. Train PyTorch backend (recommended for speed)

```bash
python cifar10_cnn_runner.py \
  --backend pytorch \
  --epochs 10 \
  --optimizer momentum \
  --lr 1e-3 \
  --beta 0.9 \
  --conv_channels 8 16 \
  --hidden_units 64 \
  --pool_every 1
```

2. Train NumPy backend (educational; slower)

```bash
python cifar10_cnn_runner.py \
  --backend numpy \
  --epochs 5 \
  --optimizer adam \
  --lr 1e-3 \
  --conv_channels 8 16 \
  --hidden_units 64 \
  --pool_every 1
```

The script will:

1. Download CIFAR-10 into ./data.

2. Build a small “cat vs dog” dataset (defaults: 200 train/class, 20 test/class).

3. Train for the requested epochs, showing a live loss plot.

4. Visualize a few predictions at the end.

---

## CLI Arguments (Runner)

```bash
--backend            {numpy,pytorch}  Backend to use. (default: pytorch)

--batch_size         int              Training batch size. (default: 128)
--epochs             int              Number of epochs. (default: 10)

--optimizer          {sgd,momentum,adam}
                                      Optimizer choice.
--lr                 float            Learning rate. (default: 1e-3)
--beta               float            Momentum (for momentum) or Adam β1. (default: 0.9)
--beta2              float            Adam β2. (default: 0.999)
--weight_decay       float            L2 weight decay. (default: 0.0)

--conv_channels      ints...          Conv channels per block. (default: 8 16)
--kernel_size        int              Conv kernel size. (default: 3)
--pool_size          int              MaxPool kernel/stride. (default: 2)
--hidden_units       ints...          FC/MLP hidden sizes in the head. (default: 64)

--use_padding                         Use “same-ish” padding (odd kernels). (default: True)
--no_padding                          Disable padding (overrides --use_padding).
--dropout            float            Dropout in head MLP (PyTorch). (default: 0.2)
--pool_every         int              Insert MaxPool after every k conv blocks. (default: 1)

--num_workers        int              DataLoader workers (PyTorch). (default: 2)
```

> Notes
• The dataset is binary (num_classes=2).
• Runner normalizes CIFAR-10 by default (mean/std used by common recipes).

---

## Model Notes

NumPy model (ConvolutionalNN_Numpy)

- conv_dim: 1/2/3 controls spatial dimensionality of conv/pool.

- Padding/stride: Per-layer conv uses META_conv{i} with stride/padding; runtime arguments conv_stride/conv_padding are broadcast to all convs.

- Pooling: pool_every=k applies MaxPool after each k-th conv block.

- Head: Builds an FC stack (hidden layers via hidden_units) on first forward once the flatten dimension is known.

- Loss:

    num_classes==1: sigmoid + BCE-with-logits.

    num_classes>1: softmax + cross-entropy.

    Optimizers: SGD or Adam; optional weight decay.


PyTorch model (ConvolutionalNN_Pytorch)

- Conv/BN/ReLU stacks; optional MaxPool inserted every pool_every blocks.

- GAP head: AdaptiveAvgPool to 1×…×1; then Flatten → MLP (hidden_units) → Linear to num_classes.

- Set pooling/stride in __init__. (The Train() signature accepts these for API compatibility but ignores them; build-time only.)

- Loss:

    num_classes==1: BCEWithLogitsLoss.

    else CrossEntropyLoss.

---

## Examples
### Deeper conv stack (PyTorch)

```bash
python cifar10_cnn_runner.py \
  --backend pytorch \
  --epochs 12 \
  --optimizer adam \
  --lr 5e-4 \
  --conv_channels 16 32 64 \
  --hidden_units 128 \
  --pool_every 1
```

### NumPy, 3D conv prototype

```bash
from ConvolutionalNN_Numpy import ConvolutionalNN_Numpy
import numpy as np

# Fake (N, C, D, H, W)
x = np.random.randn(4, 1, 8, 16, 16).astype(np.float32)
y = np.random.randint(0, 2, size=(4,))  # binary

model = ConvolutionalNN_Numpy(input_channels=1, conv_channels=[4,8], conv_dim=3,
                              kernel_size=3, pool_size=2, num_classes=1)
loss, probs = model.Backward_Propagation(x, y.reshape(-1,1).astype(np.float32))
model.train_loop(lr=1e-3, optimizer="adam")
```
