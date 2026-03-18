# LingBot Frozen-Noise-DSRL V2 Run Guide

The commands below are expected commands for a future server that has the required runtime environment. They were not executed on the current server.

## 0. Two-Server Path Examples

- `servery`
  - `lingbot-va`: `/home/e230112/vam/lingbot-va`
  - `RoboTwin-lingbot`: `/home/e230112/vam/RoboTwin-lingbot`
  - `RLinf`: `/home/e230112/vam/RLinf`
- `serverd`
  - `lingbot-va`: `/home/zaijia001/vam/lingbot-va`
  - `RoboTwin-lingbot`: `/home/zaijia001/vam/RoboTwin-lingbot`
  - `RLinf`: `/home/zaijia001/vam/RLinf`

## 1. Expected Baseline Command

### baseline eval server

`servery`:

```bash
conda activate lingbot-va
cd /home/e230112/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_PATH=/home/e230112/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

`serverd`:

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_PATH=/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

### baseline eval client

`servery`:

```bash
conda activate RoboTwin-lingbot
cd /home/e230112/vam/RoboTwin-lingbot

PYTHONWARNINGS=ignore::UserWarning \
SAPIEN_RT_DENOISER=none \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/e230112/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name click_bell \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --model_tag baseline \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_official_posttrain_click_bell \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

`serverd`:

```bash
conda activate RoboTwin-lingbot
cd /home/zaijia001/vam/RoboTwin-lingbot

PYTHONWARNINGS=ignore::UserWarning \
SAPIEN_RT_DENOISER=none \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name click_bell \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --model_tag baseline \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_official_posttrain_click_bell \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

## 2. Expected Original Post-Train Command

`servery`:

```bash
conda activate lingbot-va
cd /home/e230112/vam/lingbot-va

WANDB_TEAM_NAME=haoyuan-lingbot \
WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=baseline_place_can_basket \
CUDA_VISIBLE_DEVICES=2,3 \
NGPU=2 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --dataset-path /home/e230112/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/e230112/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --model-path /home/e230112/vam/lingbot-va/checkpoints/lingbot-va-base \
  --save-root /home/e230112/vam/lingbot-va/train_out/place_can_basket_demo_clean \
  --enable-wandb true \
  --attn-mode torch \
  --dataset-init-worker 1 \
  --save-interval 5000
```

`serverd`:

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

WANDB_TEAM_NAME=haoyuan-lingbot \
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

## 3. Expected V2 Training Command

`servery`:

```bash
conda activate lingbot-va
cd /home/e230112/vam/lingbot-va

python script/run_lingbot_frozen_noise_dsrl_v2.py \
  --config examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml \
  --save-episode-videos true \
  --episode-video-subdir episode_videos \
  --wandb-video-upload true \
  --wandb-video-upload-interval 5
```

`serverd`:

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

python script/run_lingbot_frozen_noise_dsrl_v2.py \
  --config examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml \
  --save-episode-videos true \
  --episode-video-subdir episode_videos \
  --wandb-video-upload true \
  --wandb-video-upload-interval 5
```

These video-control flags live in the training entry script:

- `script/run_lingbot_frozen_noise_dsrl_v2.py`

Meaning:

- `--save-episode-videos`
  - whether to save one local rollout video for every episode
- `--episode-video-subdir`
  - local video directory under `runner.save_root`
- `--wandb-video-upload`
  - whether to upload selected rollout videos to WandB
- `--wandb-video-upload-interval`
  - upload one episode video to WandB every N completed episodes

Recommended pattern:

- save every episode locally
- upload to WandB only at a fixed interval

For example:

- `--save-episode-videos true`
- `--wandb-video-upload true`
- `--wandb-video-upload-interval 5`

This means every episode is kept locally, but only one video is uploaded to WandB every 5 episodes.

## 4. Expected V2 Eval Command

### server

`servery`:

```bash
conda activate lingbot-va
cd /home/e230112/vam/lingbot-va

MODEL_PATH=/home/e230112/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

`serverd`:

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

MODEL_PATH=/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

### client

`servery`:

```bash
conda activate RoboTwin-lingbot
cd /home/e230112/vam/RoboTwin-lingbot

PYTHONWARNINGS=ignore::UserWarning \
SAPIEN_RT_DENOISER=none \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/e230112/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name click_bell \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --model_tag frozen-noise-v2 \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_frozen_noise_v2_click_bell \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

`serverd`:

```bash
conda activate RoboTwin-lingbot
cd /home/zaijia001/vam/RoboTwin-lingbot

PYTHONWARNINGS=ignore::UserWarning \
SAPIEN_RT_DENOISER=none \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name click_bell \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --model_tag frozen-noise-v2 \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_frozen_noise_v2_click_bell \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

## 5. Log Directories

- V2 train log directory: `/home/e230112/vam/lingbot-va/train_out/frozen_noise_dsrl_v2_click_bell`
- V2 train log file: `frozen_noise_dsrl_v2.log`
- baseline eval result directory: `/home/e230112/vam/RoboTwin-lingbot/results_official_posttrain_click_bell`
- V2 eval result directory: `/home/e230112/vam/RoboTwin-lingbot/results_frozen_noise_v2_click_bell`

## 6. Checkpoint Directories

- baseline post-train checkpoint root: `/home/e230112/vam/lingbot-va/train_out/place_can_basket_demo_clean`
- expected V2 save root: `/home/e230112/vam/lingbot-va/train_out/frozen_noise_dsrl_v2_click_bell`

## 7. Common Error Handling Suggestions

- path mismatch: check `robowin_root`, `lingbot_model_path`, and `save_root`
- port conflict: check the server default port and the client `--port`
- missing RoboTwin runtime packages: record first and investigate after migration; do not install on the current server
- GPU OOM: reduce episode / chunk / batch first, then reassess on a larger-memory machine
- when `MODEL_PATH` points to a training checkpoint root: confirm the server still resolves `transformer/` correctly

## 8. Explicit Statement

These commands were not executed on the current server. This document only records the expected run shape and does not claim that the pipeline has been run through.
