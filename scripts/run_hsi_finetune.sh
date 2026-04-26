#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
MODEL_CONFIG=${MODEL_CONFIG:-configs/imagenet_model_config.yaml}
DIFFUSION_CONFIG=${DIFFUSION_CONFIG:-configs/diffusion_config.yaml}
DATASET_NAME=${DATASET_NAME:-cave}
DATA_ROOT=${DATA_ROOT:-./data/hsi/cave}
SAVE_DIR=${SAVE_DIR:-./models/hsi_adapter}
HSI_CHANNELS=${HSI_CHANNELS:-31}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

python scripts/download_hsi_dataset.py --dataset "$DATASET_NAME" --output "$DATA_ROOT"

python train_hsi_adapter.py \
  --model_config "$MODEL_CONFIG" \
  --diffusion_config "$DIFFUSION_CONFIG" \
  --data_root "$DATA_ROOT" \
  --save_dir "$SAVE_DIR" \
  --gpu "$GPU" \
  --hsi_channels "$HSI_CHANNELS" \
  --image_size 256 \
  --random_crop_size 0 \
  --repeats_per_scene 1 \
  --adapter_hidden_channels 256 \
  --adapter_num_blocks 4 \
  --core_peft none \
  --epochs 400 \
  --batch_size 4 \
  --num_workers 0 \
  --lr 2e-4 \
  --weight_decay 5e-5 \
  --grad_clip 1.0 \
  --grad_accum_steps 8 \
  --warmup_ratio 0.05 \
  --min_lr_scale 0.1 \
  --amp \
  --log_file train_log.csv \
  --log_interval 20 \
  --t_max_start_ratio 1.0 \
  --t_max_end_ratio 1.0
