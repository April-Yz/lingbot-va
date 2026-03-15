# LingBot-VA Post-Training Data Pipeline V1

## Scope

This document records the path from RoboTwin data collection to LingBot-VA post-training, with the current plan centered on a recollected `Large_D435` version of:

- Raw task data: `/home/zaijia001/ssd/RoboTwin/data/place_can_basket/demo_clean`
- Output training bundle: `/home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean`

## 1. RoboTwin Raw Data Structure

The `collect_data.sh` pipeline in RoboTwin saves one task directory with these relevant parts:

- `data/episode*.hdf5`
  - per-frame RGB for `head_camera`, `left_camera`, `right_camera`, `front_camera`
  - per-frame `endpose/left_endpose`, `endpose/right_endpose`
  - per-frame `endpose/left_gripper`, `endpose/right_gripper`
  - per-frame joint-space data under `joint_action/*`
- `instructions/episode*.json`
  - natural-language instructions already instantiated for each episode
- `scene_info.json`
  - object/background placeholders used during generation
- `_traj_data/episode*.pkl`
  - planned joint paths used during data collection

For LingBot-VA post-training, the critical inputs are the HDF5 per-frame observations/actions and the per-episode instruction JSON.

The current converter is now intentionally strict about the raw camera size:

- `head_camera`: `480x640`
- `left_camera`: `480x640`
- `right_camera`: `480x640`

If the raw HDF5 contains a different size, the converter raises an error instead of adapting automatically.

## 2. How LingBot-VA Expects Training Data

The released post-training loader does not consume RoboTwin raw HDF5 directly. It expects a LeRobot dataset plus extra latent files:

1. LeRobot episode metadata and videos
2. `meta/episodes.jsonl` with `action_config`
3. `latents/` mirrored to the video layout
4. `empty_emb.pt` at the dataset bundle root

The training loader in `wan_va/dataset/lerobot_latent_dataset.py` uses:

- `action` from the LeRobot parquet data
- `action_config.start_frame/end_frame/action_text`
- `latents/chunk-xxx/<camera>/episode_{idx}_{start}_{end}.pth`
- `empty_emb.pt`

## 3. RoboTwin Raw Data -> LingBot Format

The new converter is:

- [`prepare_robotwin_posttrain.py`](/home/zaijia001/vam/lingbot-va/script/prepare_robotwin_posttrain.py)

It performs these steps:

1. Read each `episode*.hdf5`
2. Build a 16D action/state vector per frame:
   - left end-effector pose `7`
   - left gripper `1`
   - right end-effector pose `7`
   - right gripper `1`
3. Choose one language instruction per episode from `instructions/episode*.json`
4. Convert the episode into a local LeRobot repo with videos:
   - `head_camera -> observation.images.cam_high`
   - `left_camera -> observation.images.cam_left_wrist`
   - `right_camera -> observation.images.cam_right_wrist`
   - raw LeRobot video resolution is kept at `480x640` to match `Large_D435`
5. Patch `meta/episodes.jsonl` to add one full-episode `action_config`
6. Encode video latents with the Wan VAE
7. Encode the action text with the Wan text encoder
8. Save `empty_emb.pt`

## 4. Important Alignment Detail

LingBot-VA's training loader and Wan VAE encoder have slightly different frame-count constraints:

- training action length should be a multiple of `4`
- Wan VAE video encoding wants `num_frames = 1 (mod 4)`

The converter resolves this by:

1. padding state/action/video to the nearest upper multiple of `4`
2. encoding latents from the first `action_length - 3` frames

For a `255`-frame rollout, this becomes:

- LeRobot/action length: `256`
- latent video length: `253`

This is the combination that keeps both `AutoencoderKLWan.encode(...)` and `_action_post_process(...)` consistent.

## 5. Recommended Data Collection Command

This repository does not modify `/home/zaijia001/ssd/RoboTwin`, but the expected recollection flow is:

```bash
cd /home/zaijia001/ssd/RoboTwin
bash collect_data.sh place_can_basket demo_clean_large_d435 0
```

Important assumption for the recollected `demo_clean`:

- the task config uses `Large_D435` for the head/wrist cameras

If you prefer not to overwrite the old dataset, use a new task config name and then replace the paths below accordingly.

## 6. Actual Processing Command

```bash
cd /home/zaijia001/vam/lingbot-va
/home/zaijia001/ssd/miniconda3/envs/lingbot-va/bin/python script/prepare_robotwin_posttrain.py \
  --raw-dir /home/zaijia001/ssd/RoboTwin/data/place_can_basket/demo_clean \
  --bundle-dir /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --repo-id place_can_basket_demo_clean_lerobot \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
  --instruction-key seen \
  --instruction-index 0 \
  --overwrite

/home/zaijia001/ssd/miniconda3/envs/lingbot-va/bin/python script/prepare_robotwin_posttrain.py \
  --raw-dir /home/zaijia001/ssd/RoboTwin/data/place_can_basket/demo_clean_large_d435 \
  --bundle-dir /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --repo-id place_can_basket_demo_clean_lerobot \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
  --instruction-key seen \
  --instruction-index 0 \
  --overwrite

```

The generated bundle should look like:

```text
lingbot-posttrain-demo_clean/
├── README.md
├── empty_emb.pt
└── place_can_basket_demo_clean_lerobot/
    ├── data/
    ├── meta/
    ├── videos/
    └── latents/
```

The converter now assumes the recollected data already satisfies:

- raw camera frames are `480x640`
- camera names are still `head_camera`, `left_camera`, `right_camera`

The base model used for clean post-training should exist locally at:

- `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-base`

In this workspace it has already been downloaded from `robbyant/lingbot-va-base`.

## 7. Post-Training Command

`wan_va/train.py` now supports direct CLI overrides for local dataset paths. In this workspace the current recommendations are:

### 7.1 Verified Stable Command

This is the command shape that has already been validated locally. A 2-GPU smoke run completed `num_steps=1`, logged to WandB, and saved `checkpoint_step_1` successfully.

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=baseline_place_can_basket \
CUDA_VISIBLE_DEVICES=2,3 \
NGPU=2 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --dataset-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-base \
  --save-root /home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean \
  --enable-wandb true \
  --attn-mode torch \
  --dataset-init-worker 1 \
  --save-interval 5000
```

### 7.2 Higher-Utilization Attempt

If you want to use more of each GPU, the first thing to try is increasing the per-GPU batch size from `1` to `2`.

This was attempted locally and did not remain stable in this workspace. The 2-GPU `batch_size=2` run failed almost immediately after training started, so it should currently be treated as a known non-working setting rather than the recommended next step.

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=baseline_place_can_basket_bs2 \
CUDA_VISIBLE_DEVICES=2,3 \
NGPU=2 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --dataset-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-base \
  --save-root /home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean_bs2 \
  --enable-wandb true \
  --attn-mode torch \
  --dataset-init-worker 1 \
  --batch-size 2 \
  --save-interval 5000
```

Current status of this command:

- `NGPU=2 + batch_size=2` failed locally on March 16, 2026.
- This does not mean the codebase fundamentally rejects `batch_size=2`; it means that on this machine, with this model and current FSDP setup, the per-GPU activation footprint at `batch_size=2` was not stable enough to keep training running.

### 7.3 Recommended Way To Increase Effective Batch

If you want a larger effective batch without pushing per-GPU activation memory as hard as `batch_size=2`, the safer next step is to keep per-GPU batch size at `1` and increase gradient accumulation:

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=baseline_place_can_basket_accum2 \
CUDA_VISIBLE_DEVICES=2,3 \
NGPU=2 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --dataset-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-base \
  --save-root /home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean_accum2 \
  --enable-wandb true \
  --attn-mode torch \
  --dataset-init-worker 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --save-interval 5000
```

Why this is the safer scaling path:

- `batch_size` here is per GPU, not global.
- `NGPU=2 + batch_size=1` already gives a global batch of `2`.
- `NGPU=2 + batch_size=2` raises the per-GPU activation memory directly, which is why it is the first configuration to fail.
- `NGPU=2 + batch_size=1 + gradient_accumulation_steps=2` keeps the same per-GPU memory pattern as the stable run, but increases effective global batch to `4`.
- In short: accumulation grows effective batch through extra optimizer delay, while `batch_size=2` grows instantaneous memory pressure on every GPU.

Notes:

- `dataset-path` points to the bundle root, not directly to the LeRobot repo.
- The loader recursively finds `meta/info.json` under that root.
- The default base config still inherits RobotWin normalization and camera layout.
- `script/run_va_posttrain.sh` now preserves your existing WandB login state instead of overwriting `WANDB_*` with placeholders.
- If `WANDB_PROJECT` is unset, the launcher defaults it to `lingbot`.
- You can control the WandB run name with `WANDB_RUN_NAME`.
- The launcher auto-detects the interpreter in this order: `PYTHON_BIN`, `${CONDA_PREFIX}/bin/python`, `python`, then `python3`.
- The training loader now initializes LeRobot repos with a bounded worker count; for a single local repo it stays single-process instead of spawning a 128-process pool.
- The training path now casts floating-point batch tensors to the model parameter dtype before the forward pass, avoiding the `Float` vs `BFloat16` mismatch hit in the first local smoke run.
- The local post-training config now forces `attn_mode='torch'` by default so the base checkpoint does not try to use the `flex` backend that failed on this machine during training smoke tests.
- `batch_size` is per GPU, not global batch.
- The currently verified stable setting is `NGPU=2 + batch_size=1`.
- The attempted `NGPU=2 + batch_size=2` run failed locally and should not be treated as the default recommendation.
- The next safer scaling step is `batch_size=1 + gradient_accumulation_steps=2`.

Why `NGPU=2` is the current recommendation:

- A local `NGPU=1` smoke run reached the first optimizer step but OOMed when `AdamW` initialized optimizer state on a single 96 GB Blackwell GPU.
- A local `NGPU=2` smoke run completed `num_steps=1`, logged to WandB, and saved `checkpoint_step_1` successfully.
- The successful smoke command used:
  - `CUDA_VISIBLE_DEVICES=1,2`
  - `NGPU=2`
  - `--dataset-init-worker 1`
  - `--attn-mode torch`

Successful 2-GPU smoke artifacts:

- Save root: `/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean_smoke_v5_2gpu`
- Checkpoint: `/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean_smoke_v5_2gpu/checkpoints/checkpoint_step_1/transformer`
- WandB run: `baseline_place_can_basket_smoke_test_v5_2gpu`

If you do not activate `lingbot-va` first, you can also force the interpreter explicitly:

```bash
cd /home/zaijia001/vam/lingbot-va
PYTHON_BIN=/home/zaijia001/ssd/miniconda3/envs/lingbot-va/bin/python \
WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=baseline_place_can_basket \
CUDA_VISIBLE_DEVICES=2,3 \
NGPU=2 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --dataset-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-base \
  --save-root /home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean \
  --enable-wandb true \
  --attn-mode torch \
  --dataset-init-worker 1 \
  --save-interval 5000
```

## 8. What To Change For Another Task

For another RoboTwin single-task dataset, usually only these inputs change:

- `--raw-dir`
- `--bundle-dir`
- `--repo-id`
- `--instruction-key` / `--instruction-index`
- training `--save-root`

The action mapping stays valid as long as the raw HDF5 still provides:

- `endpose/left_endpose`
- `endpose/right_endpose`
- `endpose/left_gripper`
- `endpose/right_gripper`

## 9. Known Limitation

This pipeline uses the current local LingBot checkpoint path for VAE/text encoding:

- `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`

That is sufficient to produce training-ready latents and text embeddings. If you want a clean post-training run that starts from the pre-RoboTwin base model, use a local checkout of:

- `robbyant/lingbot-va-base`

and pass that path to `wan_va/train.py` via `--model-path`.
