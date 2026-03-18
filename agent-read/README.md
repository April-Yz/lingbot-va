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
- Common server path map:
  - `servery`: `/home/e230112/vam`
  - `serverd`: `/home/zaijia001/vam`
- The RoboTwin evaluation config should point to the local server's LingBot checkpoint root:
  - `servery`: `/home/e230112/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`
  - `serverd`: `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`
- The RoboTwin checkpoint currently in use is the post-trained RoboTwin model `robbyant/lingbot-va-posttrain-robotwin`; the pre-RoboTwin base model is `robbyant/lingbot-va-base`.
- RoboTwin evaluation should use the matching local RoboTwin repo:
  - `servery`: `/home/e230112/vam/RoboTwin-lingbot`
  - `serverd`: `/home/zaijia001/vam/RoboTwin-lingbot`
  and can still be overridden with `ROBOTWIN_ROOT`.
- `lerobot==0.3.3` remains installed for post-training support, but its published dependency constraints conflict with the upstream LingBot-VA PyTorch requirement. Follow the repository README and treat it as a `--no-deps` style add-on.
- If `flash-attn` is unavailable or ABI-incompatible, use `attn_mode='torch'`.
- Baseline eval and post-train notes now live under `agent-read/baseline/`.
- The most detailed local description of the current RoboTwin model input/output contract, latent flow, and evaluation conclusions is in `agent-read/baseline/lingbot-v0.md`.
- The latest end-to-end RoboTwin eval run with success-tagged videos and latent decoder outputs is documented in `agent-read/baseline/eval-test-decoder-v1.md`.
- The current raw-data-to-posttrain workflow, concrete `place_can_basket` processing run, and direct checkpoint-eval method are documented in `agent-read/baseline/posttrain-data-v1.md`.
- The March 16, 2026 `place_can_basket` post-train eval debugging record is documented in `agent-read/baseline/debug-posttrain-eval-place_can_basket-v1.md`.
- A bilingual quick command lookup now lives in:
  - `agent-read/command-index.md`
  - `agent-read/command-index_ZH.md`
- The command index now also includes:
  - a light-task official-checkpoint sanity-check command (`click_bell`)
  - a note explaining why `place_can_basket` can be much slower than earlier light-task evals
  - explicit parallel-server port override examples using `START_PORT` and `MASTER_PORT`
  - a CUDA device-remapping note explaining why `CUDA_VISIBLE_DEVICES=2` still appears as local `GPU 0` in PyTorch OOM logs
  - a note explaining the non-fatal `OIDN Error: invalid handle` renderer logs plus the recommended `SAPIEN_RT_DENOISER=none` and `LINGBOT_SKIP_RENDER_TEST=1` client launch pattern
- RoboTwin eval now accepts an optional `model_tag` field in the client overrides; when present, the tag is added to the `eval_result/...` directory path, manifest, console summary, and `_result.txt`.
- The baseline docs and command index now explicitly record that multi-episode eval is controlled by the client-side `--test_num`, not by the server command.
- The baseline docs and command index now also include the post-eval latent decoder command for rerunning `decode_saved_latents.py` from a saved `latent_decode_manifest.json`.
- The latent decoder examples now also pin a GPU explicitly with `CUDA_VISIBLE_DEVICES=...` and document that the `clip_output` message is a benign warning rather than a fatal error.
- Project-local `AGENTS.md` files now record the standing rules for debug logs, command-index synchronization, and scope protection for `lingbot-va` and `RoboTwin-lingbot`.
- Added a dedicated baseline data-processing audit at `agent-read/baseline/v1.5_dataprocess.md` tracing RoboTwin raw `endpose` format through LingBot preprocessing, training data loading, and eval reconstruction.
- The current audit conclusion is that translation ordering looks consistent, but quaternion handling is very likely inconsistent: RoboTwin stores `wxyz`, while LingBot preprocessing/eval helpers currently treat those values as `xyzw`.
- The RoboTwin eval client now exposes a temporary `quat_order_mode` debug switch so the same checkpoint can be evaluated with legacy quaternion handling or a `robowin_wxyz` compatibility path.
- A dedicated quaternion-order smoke-test debug record now exists at `agent-read/baseline/debug-quat-order-place_can_basket-v1.md` and the Chinese mirror `agent-read/baseline/debug-quat-order-place_can_basket-v1_ZH.md`.
- Agent-oriented environment bootstrap steps are documented in `agent-read/env-setup-agent.md`.
- Chinese baseline mirrors are now available at:
  - `agent-read/baseline/lingbot-v0_ZH.md`
  - `agent-read/baseline/eval-test-decoder-v1_ZH.md`
  - `agent-read/baseline/posttrain-data-v1_ZH.md`
  - `agent-read/baseline/debug-posttrain-eval-place_can_basket-v1_ZH.md`
- The post-training converter now assumes recollected `Large_D435` RoboTwin data and validates raw camera frames at `480x640` before conversion.
- Post-training WandB behavior now preserves existing auth, defaults the project name to `lingbot`, defaults the team/entity to `haoyuan-lingbot`, and supports custom run names via `WANDB_RUN_NAME`.
- The post-training launcher now auto-detects a Python interpreter from `PYTHON_BIN`, `CONDA_PREFIX`, `python`, or `python3` instead of assuming `python` is on `PATH`.
- The post-training dataset loader no longer fans out to a 128-process initialization pool for a single local LeRobot repo; `--dataset-init-worker` can override the worker count when needed.
- The post-training trainer now normalizes floating-point batch tensors to `config.param_dtype` before the transformer forward pass, which avoids local `Float`/`BFloat16` dtype mismatches on Blackwell.
- The local RobotWin post-training config now overrides the transformer attention backend to `torch`, avoiding the `flex_attention` block-mask failure observed with `lingbot-va-base` on this machine.
- The clean base checkpoint `robbyant/lingbot-va-base` is now present locally under `checkpoints/lingbot-va-base` for post-training starts that should not inherit RoboTwin post-training weights.
- A local March 16, 2026 post-training smoke test showed that single-GPU full fine-tuning OOMs at optimizer-state initialization, while a 2-GPU run completed `num_steps=1` and saved `checkpoint_step_1` successfully.
- `agent-read/baseline/posttrain-data-v1.md` now records the verified 2-GPU baseline command, the failed `--batch-size 2` attempt, the safer accumulation alternative, and the direct checkpoint-eval workflow.
- The checkpoint-eval workflow now explicitly handles the real training artifact layout: `checkpoint_step_xxxx` usually saves only `transformer/`, so the server loads that transformer from `MODEL_PATH` while continuing to load `vae/`, `tokenizer/`, and `text_encoder/` from the base model root.
- The RoboTwin eval client entry now also prepends the local `lingbot-va` repo root to `sys.path`, so the documented absolute-path client command works even when launched from inside `RoboTwin-lingbot`.
- The local RoboTwin eval client now also propagates `expert_check` and `step_limit_override` overrides into the actual task path, which made it possible to obtain a first `place_can_basket` smoke eval result from the post-train `checkpoint_step_5000` on March 16, 2026.
- The first verified `place_can_basket` post-train smoke eval completed with `expert_check=false`, `step_limit_override=60`, and `test_num=1`; it produced a real `0/1` result plus metrics and video artifacts instead of crashing during startup.
- The latest post-training notes now also record that `NGPU=2 + batch_size=2` failed locally, and recommend `batch_size=1 + gradient_accumulation_steps=2` as the safer way to increase effective batch.
- A new local action-only DSRL baseline now exists under `wan_va/action_only_dsrl/`, with a dedicated training entry at `script/run_lingbot_action_only_dsrl.py` and config at `examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml`.
- The action-only DSRL path now injects steering through LingBot's initial action diffusion noise via `VA_Server.sample_actions(..., initial_noise=...)`, while leaving the original random-noise path intact when DSRL is disabled.
- March 16, 2026 validation confirmed:
  - `use_dsrl=false` still produces the original action path in mock mode
  - `use_dsrl=true` injects steering noise with shape `[1, 30, 2, 16, 1]`
  - local embodied-SAC metrics are emitted by the new trainer in mock mode
- Full RoboTwin online single-task training for the new DSRL entry is now runnable on this machine for RGB-based tasks. On March 16, 2026, a `click_bell` run completed one full online episode, logged SAC metrics at steps `2`, `3`, and `4`, and exited cleanly with `current_run_status: "finished_no_success"`.
- `pytorch3d` still is not installed on this machine, but the local `RoboTwin-lingbot/envs/camera/camera.py` now falls back to a CPU farthest-point sampler instead of terminating the process.
- V1 requirement, implementation, change-log, and environment docs now live under `agent-read/v1/`.
- The detailed implementation handoff for this feature is in `agent-read/v1/implementation_report_lingbot_action_only_dsrl.md`, with exact file diffs summarized in `agent-read/v1/change_log_lingbot_action_only_dsrl.md` and all environment edits logged in `agent-read/v1/env_change_log.md`.
- A Chinese gap-analysis note comparing the current implementation against the stricter `action_only_v1.4.md` requirements is now available in `agent-read/v1/action_only_v1.4_gap_zh.md`.
- A new static V2 package now exists under `wan_va/frozen_noise_dsrl/`, with a dedicated config at `examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml`, a dedicated training entry at `script/run_lingbot_frozen_noise_dsrl_v2.py`, and a static eval helper at `script/run_lingbot_frozen_noise_eval_v2.sh`.
- The full V2 bilingual handoff document set now lives under `agent-read/V2/`, and explicitly marks the current result as a static code/documentation integration that still requires runtime validation after migration.
- The V1 bilingual document set is now available under:
  - `agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_en.md`
  - `agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_zh.md`
  - `agent-read/v1/implementation_report_lingbot_action_only_dsrl_V1_en.md`
  - `agent-read/v1/implementation_report_lingbot_action_only_dsrl_V1_zh.md`
  - `agent-read/v1/change_log_lingbot_action_only_dsrl_V1_en.md`
  - `agent-read/v1/change_log_lingbot_action_only_dsrl_V1_zh.md`
  - `agent-read/v1/env_change_log_V1_en.md`
  - `agent-read/v1/env_change_log_V1_zh.md`
- The direct V1 action-only training command is documented in both `agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_en.md` and `agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_zh.md`.
- The V1 implementation report now also explains the current online RL budget (`max_episodes`, `max_action_chunks`, replay, warmup, update frequency) and the default WandB logging setup for action-only runs.
- The current default action-only task config is `demo_clean_large_d435`, which matches the `Large_D435` camera layout (`640x480`) used by the recollected data pipeline.
- March 16, 2026 regression validation after the action-only changes confirmed:
  - original LingBot RoboTwin eval still works with a fresh `click_bell` smoke run at `1/1`
  - original LingBot post-train still works with a fresh `num_steps=1` 2-GPU smoke run that saved `checkpoint_step_1`

## Current Evaluation Conclusions

- `click_bell` minimal eval was validated end-to-end earlier.
- `click_alarmclock` completed `10/10` successfully under the current `lingbot-va + RoboTwin-lingbot` setup.
- A fresh `click_alarmclock` eval rerun on March 15, 2026 also completed `10/10`, now with success-tagged `eval_result` videos, a per-episode latent decode manifest, and decoded latent videos saved alongside the raw eval videos.
- `press_stapler` had reached `9/9` success in `res.json` before the user's interruption/abnormal close.
- Heavier tasks such as `turn_switch` are still runnable in principle, but are much slower on this machine because RoboTwin must fall back from `curobo` to `MPLib`.
