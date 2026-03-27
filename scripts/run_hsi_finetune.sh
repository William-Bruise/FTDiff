#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
MODEL_CONFIG=${MODEL_CONFIG:-configs/model_config.yaml}
DIFFUSION_CONFIG=${DIFFUSION_CONFIG:-configs/diffusion_config.yaml}
DATASET_NAME=${DATASET_NAME:-cave}
DATA_ROOT=${DATA_ROOT:-./data/hsi/cave}
SAVE_DIR=${SAVE_DIR:-./models/hsi_adapter}
HSI_CHANNELS=${HSI_CHANNELS:-31}

python scripts/download_hsi_dataset.py --dataset "$DATASET_NAME" --output "$DATA_ROOT"

python train_hsi_adapter.py \
  --model_config "$MODEL_CONFIG" \
  --diffusion_config "$DIFFUSION_CONFIG" \
  --data_root "$DATA_ROOT" \
  --save_dir "$SAVE_DIR" \
  --gpu "$GPU" \
  --hsi_channels "$HSI_CHANNELS" \
  --image_size 256 \
  --random_crop_size 256 \
  --repeats_per_scene 8 \
  --adapter_hidden_channels 64 \
  --epochs 20 \
  --batch_size 2 \
  --num_workers 4 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --amp
