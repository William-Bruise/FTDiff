#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
MODEL_CONFIG=${MODEL_CONFIG:-configs/imagenet_model_config.yaml}
DIFFUSION_CONFIG=${DIFFUSION_CONFIG:-configs/diffusion_config.yaml}
ADAPTER_CKPT=${ADAPTER_CKPT:-./models/hsi_adapter/hsi_adapter_best.pt}
NUM_SAMPLES=${NUM_SAMPLES:-8}
IMAGE_SIZE=${IMAGE_SIZE:-256}
SAVE_DIR=${SAVE_DIR:-./results_hsi_uncond}

python sample_unconditional_hsi.py \
  --model_config "$MODEL_CONFIG" \
  --diffusion_config "$DIFFUSION_CONFIG" \
  --adapter_ckpt "$ADAPTER_CKPT" \
  --gpu "$GPU" \
  --num_samples "$NUM_SAMPLES" \
  --image_size "$IMAGE_SIZE" \
  --save_dir "$SAVE_DIR"
