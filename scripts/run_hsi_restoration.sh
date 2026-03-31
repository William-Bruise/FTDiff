#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
MODEL_CONFIG=${MODEL_CONFIG:-configs/imagenet_model_config.yaml}
DIFFUSION_CONFIG=${DIFFUSION_CONFIG:-configs/diffusion_config.yaml}
ADAPTER_CKPT=${ADAPTER_CKPT:-./models/hsi_adapter/hsi_adapter_best.pt}
HSI_CHANNELS=${HSI_CHANNELS:-31}
ICVL_RAW_ROOT=${ICVL_RAW_ROOT:-/home/wuweihao/FTDiff/data/hsi/icvl}
DATA_ROOT=${DATA_ROOT:-./data/samples/icvl}
PREP_SIZE=${PREP_SIZE:-256}
MAX_SAMPLES=${MAX_SAMPLES:-0}
ICVL_LOCAL_ZIP=${ICVL_LOCAL_ZIP:-}
FALLBACK_DATASET=${FALLBACK_DATASET:-ehu}

mkdir -p "$DATA_ROOT"
if ! find "$DATA_ROOT" -type f -name "*.npy" | grep -q .; then
  echo "[Prepare] building resized ICVL samples into $DATA_ROOT from $ICVL_RAW_ROOT"
  python scripts/prepare_icvl_samples.py \
    --input_root "$ICVL_RAW_ROOT" \
    --output_root "$DATA_ROOT" \
    --size "$PREP_SIZE" \
    --max_samples "$MAX_SAMPLES"
fi

TASK_CONFIGS=(
  "configs/hsi/inpainting_config.yaml"
  "configs/hsi/denoise_config.yaml"
  "configs/hsi/super_resolution_config.yaml"
  "configs/hsi/snapshot_csi_config.yaml"
  "configs/hsi/deblur_config.yaml"
)

for TASK_CFG in "${TASK_CONFIGS[@]}"; do
  echo "[Run] ${TASK_CFG}"
  EXTRA_ARGS=()
  if [[ -n "${ICVL_LOCAL_ZIP}" ]]; then
    EXTRA_ARGS+=(--icvl_local_zip "${ICVL_LOCAL_ZIP}")
  fi
  python sample_condition_hsi.py \
    --model_config "$MODEL_CONFIG" \
    --diffusion_config "$DIFFUSION_CONFIG" \
    --task_config "$TASK_CFG" \
    --adapter_ckpt "$ADAPTER_CKPT" \
    --hsi_channels "$HSI_CHANNELS" \
    --adapter_hidden_channels 256 \
    --adapter_num_blocks 4 \
    --core_peft none \
    --data_root_override "$DATA_ROOT" \
    --auto_download_icvl \
    --download_fallback_dataset "$FALLBACK_DATASET" \
    --gpu "$GPU" \
    --save_dir ./results_hsi \
    "${EXTRA_ARGS[@]}"

done
