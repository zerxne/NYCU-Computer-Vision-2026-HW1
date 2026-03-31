# NYCU Computer Vision 2026 HW1

- **Student ID**: 314554032
- **Name**: 江怜儀

---

## Introduction

Image classification on a 100-class fine-grained dataset using an advanced ResNet backbone. The goal is to exceed the strong baseline (~0.94) while keeping the parameter count under the 100M limit. 

Instead of relying on heavy multi-model ensembles, this solution maximizes the representational capacity of a single ResNet-152 model through architectural enhancements, robust regularization, and a progressive training strategy.

---

## Environment Setup

```bash
pip install -r requirements.txt
```

---

## Usage

1. Stage 1: Train on 224 x 224 images

    ```bash
    python train.py \
    --model resnet152 \
    --img_size 224 \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-4 \
    --use_sam \
    --save_dir checkpoints/r152_224
    ```

2. Stage 2: Fine-tune on 320 x 320 images

    ```bash
    python train.py \
    --model resnet152 \
    --img_size 320 \
    --epochs 20 \
    --batch_size 16 \
    --lr 1e-4 \
    --use_sam \
    --resume checkpoints/r152_224/best_resnet152.pth \
    --save_dir checkpoints/r152_320
    ```

3. Inference

    ```bash
    python inference.py \
    --checkpoint checkpoints/r152_320/best_resnet152.pth \
    --model resnet152 \
    --img_size 320 \
    --tta \
    --output prediction.csv

    zip submission.zip prediction.csv
    ```

---

## Performance Snapshot
![alt text](image.png)