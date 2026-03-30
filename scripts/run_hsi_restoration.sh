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
    --core_peft lora \
    --lora_rank 1 \
    --lora_alpha 1.0 \
    --gpu "$GPU" \
    --save_dir ./results_hsi

done
