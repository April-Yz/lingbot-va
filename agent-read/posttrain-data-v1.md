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
bash collect_data.sh place_can_basket demo_clean 0
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

## 7. Post-Training Command

`wan_va/train.py` now supports direct CLI overrides for local dataset paths. The minimal local training command is:

```bash
cd /home/zaijia001/vam/lingbot-va
NGPU=1 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --dataset-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --save-root /home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean \
  --enable-wandb false
```

Notes:

- `dataset-path` points to the bundle root, not directly to the LeRobot repo.
- The loader recursively finds `meta/info.json` under that root.
- The default base config still inherits RobotWin normalization and camera layout.

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
