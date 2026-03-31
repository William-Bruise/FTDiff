#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
MODEL_CONFIG=${MODEL_CONFIG:-configs/model_config.yaml}
DIFFUSION_CONFIG=${DIFFUSION_CONFIG:-configs/diffusion_config.yaml}
ADAPTER_CKPT=${ADAPTER_CKPT:-./models/hsi_adapter/hsi_adapter_best.pt}
HSI_CHANNELS=${HSI_CHANNELS:-31}

TASK_CONFIGS=(
  "configs/hsi/inpainting_config.yaml"
  "configs/hsi/denoise_config.yaml"
  "configs/hsi/super_resolution_config.yaml"
  "configs/hsi/snapshot_csi_config.yaml"
  "configs/hsi/deblur_config.yaml"
)

for TASK_CFG in "${TASK_CONFIGS[@]}"; do
  echo "[Run] ${TASK_CFG}"
  python sample_condition_hsi.py \
    --model_config "$MODEL_CONFIG" \
    --diffusion_config "$DIFFUSION_CONFIG" \
    --task_config "$TASK_CFG" \
    --adapter_ckpt "$ADAPTER_CKPT" \
    --hsi_channels "$HSI_CHANNELS" \
    --adapter_hidden_channels 256 \
    --adapter_num_blocks 4 \
    --core_peft none \
    --gpu "$GPU" \
    --save_dir ./results_hsi

done
