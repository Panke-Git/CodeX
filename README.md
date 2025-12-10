# Underwater Image Enhancement with Conditional DDPM

This repository trains a conditional diffusion model (DDPM) to enhance underwater images. The model learns to denoise target clean images while conditioning on the raw underwater inputs.

## Project layout
- `configs/default.yaml`: Example training configuration.
- `train.py`: Entry point that builds the dataset, model, diffusion process, and training loop.
- `src/data.py`: Dataset and dataloader helpers for paired underwater images.
- `src/diffusion.py`: DDPM implementation with conditional sampling utilities.
- `src/models/unet.py`: U-Net backbone that consumes noisy images, time steps, and conditioning inputs.
- `src/metrics.py`: SSIM and PSNR utilities for evaluation.
- `src/utils.py`: Training helpers (logging, checkpoints, seeds).

## Quick start
1. Update `configs/default.yaml` to match your dataset layout (`/dataset/Train/input`, `/dataset/Train/GT`, `/dataset/Val/input`, `/dataset/Val/GT`). Ensure each split has both `input_dir` and `target_dir` (you may also use `gt_dir` as an alias for `target_dir`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (The project relies only on PyTorch and torchvision; install CUDA-enabled wheels if available.)
3. Launch training:
   ```bash
   python train.py --config configs/default.yaml
   ```

Training will save checkpoints for the best SSIM, best PSNR, and lowest loss. After training completes, the script automatically generates enhanced validation images for each of the three best checkpoints.
