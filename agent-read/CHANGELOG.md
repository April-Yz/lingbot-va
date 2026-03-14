# Changelog

## 2026-03-14

- Created the project-specific `lingbot-va` conda environment.
- Installed the base LingBot-VA runtime stack around Python `3.10.16` and PyTorch `2.9.0+cu126`.
- Added a runtime-safe fallback in `wan_va/modules/model.py` so missing or broken `flash-attn` no longer crashes module import when `attn_mode='torch'` is used.
- Added a minimal `.gitignore` entry for Python bytecode caches generated during validation.
- Pointed RoboTwin evaluation to the dedicated `/home/zaijia001/vam/RoboTwin-lingbot` worktree by default, with `ROBOTWIN_ROOT` as an override.
- Downloaded the `robbyant/lingbot-va-posttrain-robotwin` checkpoint into `checkpoints/` and wired `va_robotwin_cfg.py` to use it.
- Documented the environment and attention-backend compatibility notes in `agent-read/`.
