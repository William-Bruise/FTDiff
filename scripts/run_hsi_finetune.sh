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
  --image_size 128 \
  --random_crop_size 128 \
  --use_grid_patches \
  --grid_patch_size 128 \
  --rotation_aug \
  --repeats_per_scene 1 \
  --adapter_hidden_channels 256 \
  --adapter_num_blocks 4 \
  --core_peft none \
  --epochs 400 \
  --batch_size 32 \
  --num_workers 4 \
  --lr 2e-4 \
  --weight_decay 5e-5 \
  --grad_clip 1.0 \
  --grad_accum_steps 1 \
  --warmup_ratio 0.05 \
  --min_lr_scale 0.1 \
  --log_file train_log.csv \
  --log_interval 20 \
  --t_max_start_ratio 0.35 \
  --t_max_end_ratio 1.0
