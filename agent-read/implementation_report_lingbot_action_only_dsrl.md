# LingBot Action-Only DSRL Implementation Report

## Goal

Implement a minimum viable `lingbot_action_only_dsrl` baseline for LingBot-VA:

- keep LingBot frozen
- use LingBot future generation as frozen context
- inject a learned steering signal into LingBot action sampling
- train only the steering actor / critic / alpha
- keep the default LingBot inference path unchanged when DSRL is disabled

## Scope Boundary

### Done

- priority-1 noise injection via LingBot initial action noise
- frozen LingBot wrapper
- compact steering actor and multi-Q critic
- minimal embodied-SAC style trainer
- dedicated config
- dedicated training entry
- environment change logging
- handoff documentation

### Not Done

- no planner-side guidance
- no future-latent value alignment
- no dual-flow optimization
- no LingBot backbone finetuning
- no RLinf repo modification
- no RoboTwin repo modification

## What Was Reused From RLinf

The RLinf Pi0 DSRL implementation was inspected first and reused at the design level:

- compact image/state encoders
- Gaussian steering actor
- multi-Q critic shape
- embodied-SAC style losses
- separation between frozen backbone and trainable lightweight heads

Because the user only authorized changes inside `lingbot-va`, the RLinf logic was reimplemented locally under `wan_va/action_only_dsrl/` instead of modifying `/home/zaijia001/vam/RLinf`.

## Architecture Overview

### LingBot side

- `VA_Server.sample_future_latents(...)`
  - runs the frozen video/future branch
  - leaves the transformer cache prepared for action decoding

- `VA_Server.sample_actions(..., initial_noise=...)`
  - runs the frozen action branch
  - uses `initial_noise` if provided
  - otherwise preserves the old random initialization path

### DSRL side

- `LingBotActionOnlyDsrlPolicy`
  - owns the frozen LingBot server
  - extracts compact RGB/state/future-summary features
  - samples steering noise from a Gaussian actor
  - reshapes the steering vector into the exact LingBot action-noise tensor

- `ActionOnlyDsrlTrainer`
  - owns replay buffer, optimizers, target critic, and alpha temperature
  - performs embodied-SAC style actor / critic / alpha updates

## Step-by-Step Data / Control Flow

### `use_dsrl = false`

1. reset LingBot with prompt
2. call `sample_future_latents(...)`
3. call `sample_actions(..., initial_noise=None)`
4. LingBot action branch starts from its original random noise
5. return postprocessed action chunk

### `use_dsrl = true`

1. reset LingBot with prompt
2. call `sample_future_latents(...)`
3. mean-pool the predicted future latent into a compact summary
4. encode RGB / state / future-summary with lightweight trainable encoders
5. steering actor samples `steer_noise`
6. reshape `steer_noise` to `[1, action_dim, frame_chunk, action_per_frame, 1]`
7. call `sample_actions(..., initial_noise=steer_noise)`
8. LingBot action branch decodes actions from the injected initial noise
9. execute action chunk in RoboTwin
10. use reward for actor / critic / alpha updates

## File-Level Explanation

### `wan_va/wan_va_server.py`

- before:
  - `_infer(...)` inlined both future-latent generation and action sampling
  - no explicit action-noise injection interface existed

- after:
  - `sample_future_latents(...)` handles the frozen future branch
  - `sample_actions(...)` handles the frozen action branch
  - `_prepare_initial_action_noise(...)` validates and reshapes external steering noise
  - `_infer(...)` composes the two paths and remains backward compatible

### `wan_va/action_only_dsrl/modules.py`

- local compact modules mirroring RLinf Pi0 DSRL style
- no LingBot parameters appear here

### `wan_va/action_only_dsrl/policy.py`

- wraps the frozen LingBot stack
- logs freeze / optimizer scope
- exports `act(...)` with `use_dsrl=true/false`
- trainer update logs SAC metrics

### `wan_va/action_only_dsrl/robotwin_env.py`

- imports RoboTwin at runtime from `ROBOTWIN_ROOT`
- reuses task config and observation formatting
- does not modify RoboTwin itself

### `script/run_lingbot_action_only_dsrl.py`

- reads YAML config
- runs mock validation first
- then attempts RoboTwin single-task startup
- if RoboTwin fails, logs the blocker cleanly instead of crashing without context

## Config Explanation

Main config: `examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml`

- `policy.use_dsrl`
  - enables or disables steering

- `policy.dsrl_mode`
  - set to `action_only`

- `policy.freeze_backbone`
  - documented intent flag
  - implementation currently freezes the entire LingBot stack

- `policy.dsrl_noise_injection_mode`
  - `initial_noise`

- `policy.dsrl_action_noise_dim`
  - must equal the flattened LingBot action-noise tensor
  - current RoboTwin config: `30 * 2 * 16 = 960`

- `algorithm.adv_type`
  - `embodied_sac`

- `algorithm.loss_type`
  - `embodied_sac`

- `algorithm.batch_size`
  - set to `1` for the local smoke path so a single mock transition already produces SAC metrics

## Optimizer Verification

The optimizer contains only:

- steering actor encoders + Gaussian actor
- critic encoders + multi-Q head
- alpha temperature parameter

The optimizer does not contain:

- LingBot transformer
- LingBot text encoder
- LingBot VAE

Observed startup report:

- frozen LingBot params: `11,474,471,674`
- trainable DSRL params: `6,387,402`
- actor params: `2,439,264`
- critic params: `3,948,138`
- `optimizes_lingbot: false`

## Validation Performed

### 1. `use_dsrl=false` path

Validated in mock mode.

Observed log:

- `injection_mode: "disabled"`
- `steer_noise_shape: null`
- `action_output_shape: [16, 2, 16]`

### 2. `use_dsrl=true` path

Validated in mock mode.

Observed log:

- `injection_mode: "initial_noise"`
- `steer_noise_shape: [1, 30, 2, 16, 1]`
- `action_output_shape: [16, 2, 16]`

### 3. SAC metrics visibility

Validated in mock mode.

Observed log:

- `train/critic_loss`
- `train/actor_loss`
- `train/alpha_loss`
- `train/alpha`
- `train/global_step`
- `train/replay_size`

### 4. RoboTwin single-task startup

Attempted with:

- task: `click_bell`
- config: `demo_clean`
- environment: `lingbot-va`

Current result:

- not fully runnable yet
- blocked in RoboTwin seed validation / setup because the local `pytorch3d` path cannot be built on this Blackwell host with the available CUDA 12.1 toolchain

## Known Limitations

- RoboTwin single-task runtime is still blocked by `pytorch3d` on this machine.
- The current trainer is local to `lingbot-va`; it mirrors RLinf DSRL style but does not register itself into the RLinf repo.
- The current implementation freezes the full LingBot stack instead of selectively freezing named backbone/future/action-decoder submodules.
- Mock validation proves the steering path and SAC updates, but it does not replace a full RoboTwin online training run.

## Next-Step Suggestions

1. Resolve the RoboTwin `pytorch3d` dependency on Blackwell, ideally with a toolchain that supports `sm_120`.
2. Once RoboTwin setup succeeds, rerun the new training entry and verify the first online transition plus SAC update.
3. If later authorized, register the same wrapper into RLinf proper instead of keeping the trainer local to `lingbot-va`.

## Minimal Reproduction

1. Repository path
   - `/home/zaijia001/vam/lingbot-va`

2. Conda environment
   - `conda activate lingbot-va`

3. Dependency checks
   - `python -c "import torch, diffusers, transformers"`
   - `python -c "import sapien, mplib, open3d, trimesh, toppra"`

4. Exact run command

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va
CUDA_VISIBLE_DEVICES=0 python script/run_lingbot_action_only_dsrl.py \
  --config examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml
```

5. Expected logs

- startup report with frozen / trainable param counts
- `validation/use_dsrl_false`
- `validation/use_dsrl_true`
- `validation/mock_sac_metrics`
- either RoboTwin episode logs or a clean `robowin/blocker` entry

6. What indicates success

- `validation/use_dsrl_true` shows `steer_noise_shape: [1, 30, 2, 16, 1]`
- `validation/mock_sac_metrics` shows actor / critic / alpha metrics
- no LingBot parameter group is reported as optimized

7. Common failure modes

- `ModuleNotFoundError` on RoboTwin packages
  - fix missing simulator/runtime packages in `lingbot-va`
- `missing pytorch3d`
  - current host blocker for full RoboTwin online run
- `Unsupported gpu architecture 'compute_120'`
  - PyTorch3D build is using a CUDA toolchain too old for Blackwell

## Final Status

- Core action-only DSRL implementation: complete
- Mock validation: complete
- RoboTwin online single-task training: blocked by local PyTorch3D / Blackwell environment incompatibility
