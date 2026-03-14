# LingBot-VA Agent Notes

## Project Overview

LingBot-VA is a robot video-action foundation model built around the `wan_va/` package. The repository contains:

- `wan_va/`: core model, configs, distributed utilities, training, and inference server code.
- `evaluation/robotwin/`: RoboTwin evaluation launch scripts.
- `script/`: launch helpers for inference and training workflows.
- `example/`: sample inputs for demos and evaluation.

## Supported Environment Assumptions

- Python `3.10.x`
- PyTorch `2.9.0`
- CUDA runtime packages matching the `cu126` wheel set
- `diffusers==0.36.0`
- `transformers==4.55.2`

## Attention Backend Behavior

- `attn_mode='torch'` is the safest default for inference in this workspace.
- `attn_mode='flashattn'` now requires a working `flash-attn` install at runtime instead of crashing during module import.
- `attn_mode='flex'` remains training-specific as described in the upstream README.

## Current Workspace Notes

- The local `lingbot-va` conda environment was created for this repository.
- RoboTwin evaluation is wired to `/home/zaijia001/vam/RoboTwin-lingbot` by default and can be overridden with `ROBOTWIN_ROOT`.
- `lerobot==0.3.3` remains installed for post-training support, but its published dependency constraints conflict with the upstream LingBot-VA PyTorch requirement. Follow the repository README and treat it as a `--no-deps` style add-on.
- If `flash-attn` is unavailable or ABI-incompatible, use `attn_mode='torch'`.
