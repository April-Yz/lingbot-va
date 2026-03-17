# LingBot Frozen-Noise-DSRL V2 Run Guide

The commands below are expected commands for a future server that has the required runtime environment. They were not executed on the current server.

## 1. Expected Baseline Command

### baseline eval server

```bash
conda activate lingbot-va
cd /home/e230112/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_PATH=/home/e230112/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

### baseline eval client

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

## 2. Expected Original Post-Train Command

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

## 3. Expected V2 Training Command

```bash
conda activate lingbot-va
cd /home/e230112/vam/lingbot-va

python script/run_lingbot_frozen_noise_dsrl_v2.py \
  --config examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml
```

## 4. Expected V2 Eval Command

### server

```bash
conda activate lingbot-va
cd /home/e230112/vam/lingbot-va

MODEL_PATH=/home/e230112/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

### client

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
