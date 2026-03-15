# Changelog

## 2026-03-14

- Created the project-specific `lingbot-va` conda environment.
- Installed the base LingBot-VA runtime stack around Python `3.10.16` and PyTorch `2.9.0+cu126`.
- Added a runtime-safe fallback in `wan_va/modules/model.py` so missing or broken `flash-attn` no longer crashes module import when `attn_mode='torch'` is used.
- Added a minimal `.gitignore` entry for Python bytecode caches generated during validation.
- Pointed RoboTwin evaluation to the dedicated `/home/zaijia001/vam/RoboTwin-lingbot` worktree by default, with `ROBOTWIN_ROOT` as an override.
- Downloaded the `robbyant/lingbot-va-posttrain-robotwin` checkpoint into `checkpoints/` and wired `va_robotwin_cfg.py` to use it.
- Documented the environment and attention-backend compatibility notes in `agent-read/`.
- Updated the `lingbot-va` runtime to the `cu128` PyTorch wheel set so Blackwell GPUs on this machine can run inference.
- Added `agent-read/lingbot-v0.md` to document the current RoboTwin model input/output contract, latent path, server-side preprocessing, returned action semantics, and the evaluation conclusions observed in this workspace.
- Confirmed a complete `click_alarmclock` RoboTwin eval at `10/10` success; `press_stapler` had reached `9/9` success before the interrupted session ended.
- Added a server-to-client eval manifest flow so each RoboTwin episode records its server latent directory and saves success-tagged `eval_result` videos instead of anonymous `episode*.mp4`.
- Added `evaluation/robotwin/decode_saved_latents.py` to decode saved `latents_*.pt` files into per-episode videos after eval.
- Completed a fresh `click_alarmclock` eval run at `10/10` success with decoded latent videos saved into RoboTwin `eval_result/`.
- Added `agent-read/eval-test-decoder-v1.md` as the concrete report for the March 15, 2026 eval + latent decoder run.
- Expanded `agent-read/eval-test-decoder-v1.md` with a step-by-step description of how eval-time `latents_*.pt` files are produced, mapped into a manifest, decoded by VAE, and exported into final visualization videos.
- Added `script/prepare_robotwin_posttrain.py` to convert RoboTwin raw data into a LingBot-ready post-training bundle with LeRobot videos, `action_config`, latents, and `empty_emb.pt`.
- Added CLI overrides to `wan_va/train.py` so local post-training can pass dataset/model paths without editing config files.
- Added `agent-read/posttrain-data-v1.md` to document the concrete `place_can_basket` raw-data-to-posttrain workflow.
- Tightened `script/prepare_robotwin_posttrain.py` to enforce the recollected `Large_D435` raw camera size (`480x640`) instead of adapting to smaller legacy frames.
- Expanded `agent-read/posttrain-data-v1.md` with the recommended RoboTwin recollection command plus the processing and post-training commands for the `Large_D435` workflow.
- Updated post-training WandB handling so the launcher no longer overwrites `WANDB_*`, defaults `WANDB_PROJECT` to `lingbot`, and supports `WANDB_RUN_NAME` for custom run names.
- Updated `script/run_va_posttrain.sh` to auto-detect the Python interpreter from `PYTHON_BIN`, `CONDA_PREFIX`, `python`, or `python3`, and documented the explicit `PYTHON_BIN` fallback command.
- Reduced local post-training dataset initialization fan-out by bounding LeRobot repo init workers and exposing `--dataset-init-worker` as a training override.
- Cast floating-point training batches to `config.param_dtype` before the transformer forward pass so local post-training no longer fails on `Float` vs `BFloat16` input mismatches.
- Overrode local RobotWin post-training to use `attn_mode='torch'` by default and added `--attn-mode` so training does not hit the `flex_attention` block-mask failure from the base checkpoint config.
- Downloaded the clean base checkpoint `robbyant/lingbot-va-base` into `checkpoints/lingbot-va-base` for local post-training starts.
- Verified local post-training on March 16, 2026: `NGPU=1` still OOMs at optimizer-state initialization, while a 2-GPU smoke run completed `num_steps=1` and saved `checkpoint_step_1`.
