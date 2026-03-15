# LingBot-VA Agent Notes

## Project Overview

LingBot-VA is a robot video-action foundation model built around the `wan_va/` package. The repository contains:

- `wan_va/`: core model, configs, distributed utilities, training, and inference server code.
- `evaluation/robotwin/`: RoboTwin evaluation launch scripts.
- `script/`: launch helpers for inference and training workflows.
- `example/`: sample inputs for demos and evaluation.
- `script/prepare_robotwin_posttrain.py`: local RoboTwin raw-data to LingBot post-training bundle converter.

## Supported Environment Assumptions

- Python `3.10.x`
- PyTorch `2.9.0+cu128`
- CUDA runtime packages matching the `cu128` wheel set
- `diffusers==0.36.0`
- `transformers==4.55.2`

## Attention Backend Behavior

- `attn_mode='torch'` is the safest default for inference in this workspace.
- `attn_mode='flashattn'` now requires a working `flash-attn` install at runtime instead of crashing during module import.
- `attn_mode='flex'` remains training-specific as described in the upstream README.

## Current Workspace Notes

- The local `lingbot-va` conda environment was created for this repository.
- The RoboTwin evaluation config points to `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`.
- The RoboTwin checkpoint currently in use is the post-trained RoboTwin model `robbyant/lingbot-va-posttrain-robotwin`; the pre-RoboTwin base model is `robbyant/lingbot-va-base`.
- RoboTwin evaluation is wired to `/home/zaijia001/vam/RoboTwin-lingbot` by default and can be overridden with `ROBOTWIN_ROOT`.
- `lerobot==0.3.3` remains installed for post-training support, but its published dependency constraints conflict with the upstream LingBot-VA PyTorch requirement. Follow the repository README and treat it as a `--no-deps` style add-on.
- If `flash-attn` is unavailable or ABI-incompatible, use `attn_mode='torch'`.
- The most detailed local description of the current RoboTwin model input/output contract, latent flow, and evaluation conclusions is in `agent-read/lingbot-v0.md`.
- The latest end-to-end RoboTwin eval run with success-tagged videos and latent decoder outputs is documented in `agent-read/eval-test-decoder-v1.md`.
- The current raw-data-to-posttrain workflow and the concrete `place_can_basket` processing run are documented in `agent-read/posttrain-data-v1.md`.
- The post-training converter now assumes recollected `Large_D435` RoboTwin data and validates raw camera frames at `480x640` before conversion.
- Post-training WandB behavior now preserves existing auth, defaults the project name to `lingbot`, and supports custom run names via `WANDB_RUN_NAME`.

## Current Evaluation Conclusions

- `click_bell` minimal eval was validated end-to-end earlier.
- `click_alarmclock` completed `10/10` successfully under the current `lingbot-va + RoboTwin-lingbot` setup.
- A fresh `click_alarmclock` eval rerun on March 15, 2026 also completed `10/10`, now with success-tagged `eval_result` videos, a per-episode latent decode manifest, and decoded latent videos saved alongside the raw eval videos.
- `press_stapler` had reached `9/9` success in `res.json` before the user's interruption/abnormal close.
- Heavier tasks such as `turn_switch` are still runnable in principle, but are much slower on this machine because RoboTwin must fall back from `curobo` to `MPLib`.
