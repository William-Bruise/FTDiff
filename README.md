# Diffusion Posterior Sampling for General Noisy Inverse Problems (ICLR 2023 spotlight)

![result-gif1](./figures/motion_blur.gif)
![result-git2](./figures/super_resolution.gif)
<!-- See more results in the [project-page](https://jeongsol-kim.github.io/dps-project-page) -->

## Abstract
In this work, we extend diffusion solvers to efficiently handle general noisy (non)linear inverse problems via the approximation of the posterior sampling. Interestingly, the resulting posterior sampling scheme is a blended version of the diffusion sampling with the manifold constrained gradient without strict measurement consistency projection step, yielding more desirable generative path in noisy settings compared to the previous studies.

![cover-img](./figures/cover.jpg)


## Prerequisites
- python 3.8

- pytorch 1.11.0

- CUDA 11.3.1

- nvidia-docker (if you use GPU in docker container)

It is okay to use lower version of CUDA with proper pytorch version.

Ex) CUDA 10.2 with pytorch 1.7.0

<br />

## Getting started 

### 1) Clone the repository

```
git clone https://github.com/DPS2022/diffusion-posterior-sampling

cd diffusion-posterior-sampling
```

<br />

### 2) Download pretrained checkpoint
From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" and paste it to ./models/
```
mkdir models
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

:speaker: Checkpoint for imagenet is uploaded.

<br />


### 3) Set environment
### [Option 1] Local environment setting

We use the external codes for motion-blurring and non-linear deblurring.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

Install dependencies

```
conda create -n DPS python=3.8

conda activate DPS

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

<br />

### [Option 2] Build Docker image

Install docker engine, GPU driver and proper cuda before running the following commands.

Dockerfile already contains command to clone external codes. You don't have to clone them again.

--gpus=all is required to use local GPU device (Docker >= 19.03)

```
docker build -t dps-docker:latest .

docker run -it --rm --gpus=all dps-docker
```

<br />

### 4) Inference

```
python3 sample_condition.py \
--model_config=configs/model_config.yaml \
--diffusion_config=configs/diffusion_config.yaml \
--task_config={TASK-CONFIG};
```


:speaker: For imagenet, use configs/imagenet_model_config.yaml

<br />

## Possible task configurations

```
# Linear inverse problems
- configs/super_resolution_config.yaml
- configs/gaussian_deblur_config.yaml
- configs/motion_deblur_config.yaml
- configs/inpainting_config.yaml

# Non-linear inverse problems
- configs/nonlinear_deblur_config.yaml
- configs/phase_retrieval_config.yaml
```

### Structure of task configurations
You need to write your data directory at data.root. Default is ./data/samples which contains three sample images from FFHQ validation set.

```
conditioning:
    method: # check candidates in guided_diffusion/condition_methods.py
    params:
        scale: 0.5

data:
    name: ffhq
    root: ./data/samples/

measurement:
    operator:
        name: # check candidates in guided_diffusion/measurements.py

noise:
    name:   # gaussian or poisson
    sigma:  # if you use name: gaussian, set this.
    (rate:) # if you use name: poisson, set this.
```

## Citation
If you find our work interesting, please consider citing

```
@inproceedings{
chung2023diffusion,
title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
author={Hyungjin Chung and Jeongsol Kim and Michael Thompson Mccann and Marc Louis Klasky and Jong Chul Ye},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=OnD9zGAGT0k}
}
```



## Hyperspectral fine-tuning (parameter-efficient + frozen diffusion core)

This repo now includes an HSI fine-tuning path that keeps the pretrained diffusion U-Net frozen and only trains parameter-efficient modules:
- replaced CNN head at diffusion input: HSI -> core stem channels
- replaced CNN tail at diffusion output: core output channels -> HSI
- frozen middle diffusion layers (optional LoRA if you explicitly enable it)

### 1) Download hyperspectral dataset (default: CAVE)

```bash
## default uses CAVE (official Columbia zip)
python scripts/download_hsi_dataset.py --dataset cave --output ./data/hsi/cave

# optional datasets
python scripts/download_hsi_dataset.py --dataset ehu --output ./data/hsi/ehu

# ICVL via SharePoint URL or custom source_urls
python scripts/download_hsi_dataset.py --dataset icvl --output ./data/hsi/icvl

# if you manually downloaded ICVL zip from SharePoint, use local zip directly
python scripts/download_hsi_dataset.py --dataset icvl --output ./data/hsi/icvl --local_zip /path/to/icvl.zip --only_mat
```

### 2) Fine-tune adapter on HSI data (256x256)

`run_hsi_finetune.sh` will skip dataset download automatically if `.mat/.npy` files already exist in `DATA_ROOT`.

Note: by default `train_hsi_adapter.py` keeps the model config checkpoint behavior; pass `--disable_checkpoint` only if you explicitly want to turn it off.

```bash
bash scripts/run_hsi_finetune.sh
```

Training logs are written to:
- `SAVE_DIR/train_log.csv` (step + epoch metrics, LR, timestep range)

### 3) Run HSI restoration tasks

```bash
bash scripts/run_hsi_restoration.sh
```

By default the restoration runner now targets ICVL (`DATA_ROOT=./data/hsi/icvl`) and will auto-download ICVL `.mat` files if missing.  
After sampling, per-image metrics are saved to `results_hsi/<operator>/metrics.csv` with PSNR and SSIM(global).
If ICVL access fails (e.g., SharePoint 403), the runner falls back to `FALLBACK_DATASET=ehu` by default.

Tasks covered in `scripts/run_hsi_restoration.sh`:
- Inpainting
- Denoising
- Super-resolution
- Snapshot compressive imaging
- Deblurring

Main added scripts:
- `train_hsi_adapter.py`
- `sample_condition_hsi.py`
- `scripts/download_hsi_dataset.py`
- `scripts/run_hsi_finetune.sh`
- `scripts/run_hsi_restoration.sh`


Adapter defaults use replaced CNN input/output layers and freeze the diffusion middle.


Recommended stable HSI fine-tuning defaults (for better convergence):
- `batch_size=32` + `grad_accum_steps=1`
- `epochs=400`, `lr=2e-4`, `weight_decay=5e-5`
- cosine LR schedule with `warmup_ratio=0.05`, `min_lr_scale=0.1`
- timestep curriculum: `t_max_start_ratio=0.35`, `t_max_end_ratio=1.0`, `t_curriculum_power=2.0`
- `use_grid_patches + grid_patch_size=128` with `rotation_aug`


HSI augmentation defaults for stronger fine-tuning:
- 90°/180°/270° rotation augmentation (`--rotation_aug`)
- Grid patch augmentation (`--use_grid_patches --grid_patch_size 128`)
  - for 512x512 scenes this yields 16 non-overlap patches (128x128 each)
- Train directly on 128x128 patches (`--image_size 128`, `--batch_size 32`)


Core PEFT option:
- default is to train replaced CNN head/tail only:
  - `--core_peft none`
- optional LoRA on frozen diffusion core conv layers:
  - `--core_peft lora --lora_rank 1 --lora_alpha 1.0`
- memory-safe default only injects LoRA into 1x1 Conv2d layers:
  - `--lora_conv2d_target 1x1`
- if you want stronger adaptation and have enough GPU memory, you can increase coverage:
  - `--lora_conv2d_target all --lora_enable_conv1d`
