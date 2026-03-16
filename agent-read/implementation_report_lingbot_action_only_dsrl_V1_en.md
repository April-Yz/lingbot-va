# LingBot Action-Only DSRL V1 Implementation Report

## Goal

Provide a reproducible V1 action-only DSRL baseline for LingBot-VA on RoboTwin, while preserving the original LingBot eval and post-train workflows.

## What Was Implemented

- priority-1 steering injection through `VA_Server.sample_actions(..., initial_noise=...)`
- frozen LingBot wrapper with trainable steering actor / critic / alpha only
- local RLinf-style compact actor / critic modules
- single-task training entry:
  - `script/run_lingbot_action_only_dsrl.py`
- single config:
  - `examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml`

## What Was Not Implemented

- no full-method future-latent guidance
- no planner-side value alignment
- no dual-flow optimization
- no LingBot finetuning
- no RLinf repo-native registration yet

## Main Runtime Path

1. reset LingBot with prompt
2. encode current observation
3. run frozen future branch
4. steering actor predicts flattened `steer_noise`
5. reshape noise to LingBot action-noise tensor
6. decode actions with frozen LingBot action branch
7. step RoboTwin
8. update actor / critic / alpha with embodied-SAC losses

## Verified V1 Training Command

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

CUDA_VISIBLE_DEVICES=1 \
python script/run_lingbot_action_only_dsrl.py \
  --config examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml
```

Default config values used by this command:

- task: `click_bell`
- task config: `demo_clean`
- model path: `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`
- save root: `/home/zaijia001/vam/lingbot-va/train_out/action_only_dsrl_click_bell`

## Validation Summary

### Action-Only Path

- `use_dsrl=false`: validated
- `use_dsrl=true`: validated
- online single-episode RoboTwin run: validated
- latest online status: `finished_no_success`

### Original LingBot Eval Regression

Validated on March 16, 2026.

Command shape used:

1. start server
```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va
CUDA_VISIBLE_DEVICES=1 bash evaluation/robotwin/launch_server.sh
```

2. run client smoke
```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va
PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python -m evaluation.robotwin.eval_polict_client_openpi \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name click_bell \
  --task_config demo_clean \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_regression_eval \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

Observed result:

- task: `click_bell`
- metric file: `/home/zaijia001/vam/RoboTwin-lingbot/results_regression_eval/stseed-10000/metrics/click_bell/res.json`
- result: `1 / 1`, `succ_rate = 1.0`

### Original LingBot Post-Train Regression

Validated on March 16, 2026.

Command used:

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

MASTER_PORT=29621 \
CUDA_VISIBLE_DEVICES=0,1 \
NGPU=2 CONFIG_NAME=robotwin_train \
bash script/run_va_posttrain.sh \
  --dataset-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-base \
  --save-root /home/zaijia001/vam/lingbot-va/train_out/posttrain_regression_smoke \
  --enable-wandb false \
  --attn-mode torch \
  --dataset-init-worker 1 \
  --num-steps 1 \
  --save-interval 1
```

Observed result:

- training reached `1 / 1` steps
- checkpoint saved successfully
- checkpoint path:
  - `/home/zaijia001/vam/lingbot-va/train_out/posttrain_regression_smoke/checkpoints/checkpoint_step_1/transformer/config.json`
  - `/home/zaijia001/vam/lingbot-va/train_out/posttrain_regression_smoke/checkpoints/checkpoint_step_1/transformer/diffusion_pytorch_model.safetensors`

## Minimal Reproduction

1. repo path
   - `/home/zaijia001/vam/lingbot-va`
2. environment
   - `conda activate lingbot-va`
3. dependency checks
   - `python -c "import torch, diffusers, transformers"`
   - `python -c "import sapien, mplib, open3d, trimesh, toppra, lxml"`
4. exact action-only run command
   - see the command in the `Verified V1 Training Command` section
5. expected logs
   - `validation/use_dsrl_false`
   - `validation/use_dsrl_true`
   - `train/metrics`
   - `final/report`
6. success indicator
   - process exits cleanly and writes `final/report`
7. common failure modes
   - missing RoboTwin-side runtime packages
   - port conflicts
   - `pytorch3d` unavailable on Blackwell with CUDA 12.1

## Current Limitation

- V1 proves the action-only training pipeline and regression compatibility
- V1 does not yet prove good task success rate
