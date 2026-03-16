# LingBot Command Index

This is a quick command lookup for the most commonly used local workflows in this workspace.

## 1. RoboTwin Raw Data -> Post-Train Bundle

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

/home/zaijia001/ssd/miniconda3/envs/lingbot-va/bin/python script/prepare_robotwin_posttrain.py \
  --raw-dir /home/zaijia001/ssd/RoboTwin/data/place_can_basket/demo_clean_large_d435 \
  --bundle-dir /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --repo-id place_can_basket_demo_clean_lerobot \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
  --instruction-key seen \
  --instruction-index 0 \
  --overwrite
```

Reference:

- `agent-read/baseline/posttrain-data-v1.md`

## 2. Baseline Post-Training

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

Reference:

- `agent-read/baseline/posttrain-data-v1.md`

## 3. Eval Official RoboTwin Post-Train Checkpoint

Server:

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_PATH=/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

Client:

```bash
conda activate RoboTwin-lingbot
cd /home/zaijia001/vam/RoboTwin-lingbot

PYTHONWARNINGS=ignore::UserWarning \
SAPIEN_RT_DENOISER=none \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name place_can_basket \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_official_posttrain_place_can_basket \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

Reference:

- `agent-read/baseline/lingbot-v0.md`

### 3.1 Fast Comparison Command For A Light Task

If you want to quickly verify that the official checkpoint path and websocket pipeline are still healthy, use a lighter task first:

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
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_official_posttrain_click_bell \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

This is the recommended first sanity check before spending time on a heavier task like `place_can_basket`.

### 3.2 Why A Heavy Task Can Look Much Slower

`place_can_basket` is much slower than `click_bell` in this workspace because:

- it uses a heavier dual-arm task script
- `expert_check=true` runs the task's expert seed-filter path before model eval
- local `curobo` is currently not usable, so RoboTwin falls back to `MPLib`

The current `RoboTwin-lingbot` environment has:

- PyTorch runtime: `2.9.0+cu128`
- torch CUDA runtime tag: `12.8`

But the actual CUDA compiler used by local Curobo JIT builds is still:

- `/home/zaijia001/ssd/cuda-12.1/bin/nvcc`

So the current blocker is not "the conda env has no CUDA 12.8 support". The blocker is that Curobo's extension build path is still using a CUDA 12.1 toolchain against a Blackwell machine and a newer PyTorch runtime, which leads to:

- ABI mismatch on prebuilt `.so` files
- failed local JIT rebuilds

As a result, heavy tasks currently run through the `MPLib` fallback and look much slower.

### 3.3 CUDA Device Numbering Note

If you launch the server with:

```bash
CUDA_VISIBLE_DEVICES=2 bash evaluation/robotwin/launch_server.sh
```

then PyTorch remaps that physical GPU into the process-local device list.

Inside the server process:

- physical GPU `2`
- becomes local `cuda:0`

So an OOM that says `GPU 0` does not mean the client or server fell back to physical GPU `0`. It usually means:

- the process is correctly restricted to physical GPU `2`
- but inside that isolated process view, PyTorch labels it as local `GPU 0`

The client does not choose the server GPU. `--port` only chooses which websocket server the client connects to.

### 3.4 OIDN Renderer Log Note

If the client prints lines like:

```text
[svulkan2] [error] OIDN Error: invalid handle
```

that is a renderer-side denoiser log from SAPIEN / svulkan2, not an action-model error.

In this workspace, the safer client launch pattern is:

- `SAPIEN_RT_DENOISER=none`

and you can also skip the startup render self-check entirely:

- `LINGBOT_SKIP_RENDER_TEST=1`

Recommended quiet client launch pattern:

```bash
PYTHONWARNINGS=ignore::UserWarning \
SAPIEN_RT_DENOISER=none \
LINGBOT_SKIP_RENDER_TEST=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python ...
```

## 4. Eval Local Post-Train Checkpoint

Server:

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_PATH=/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean/checkpoints/checkpoint_step_5000 \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

Client:

```bash
conda activate RoboTwin-lingbot
cd /home/zaijia001/vam/RoboTwin-lingbot

PYTHONWARNINGS=ignore::UserWarning \
SAPIEN_RT_DENOISER=none \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name place_can_basket \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_posttrain_eval_step5000 \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

Debug smoke version:

```bash
conda activate RoboTwin-lingbot
cd /home/zaijia001/vam/RoboTwin-lingbot

PYTHONWARNINGS=ignore::UserWarning \
SAPIEN_RT_DENOISER=none \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name place_can_basket \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_posttrain_eval_step5000_fix4 \
  --expert_check false \
  --step_limit_override 60 \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

Reference:

- `agent-read/baseline/posttrain-data-v1.md`
- `agent-read/baseline/debug-posttrain-eval-place_can_basket-v1.md`

## 5. Parallel Eval On Multiple Servers

If you want to evaluate two checkpoints or tasks at the same time, each server should use a different websocket port and a different torch distributed master port.

Example server A:

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
START_PORT=29058 \
MASTER_PORT=29068 \
MODEL_PATH=/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

Example server B:

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
START_PORT=29059 \
MASTER_PORT=29069 \
MODEL_PATH=/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean/checkpoints/checkpoint_step_5000 \
CUDA_VISIBLE_DEVICES=2 \
bash evaluation/robotwin/launch_server.sh
```

The matching clients must use the corresponding `--port`:

- server A -> `--port 29058`
- server B -> `--port 29059`

## 6. Action-Only V1 Training

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

WANDB_TEAM_NAME=haoyuan-lingbot \
WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=action_only_click_bell_v1 \
CUDA_VISIBLE_DEVICES=0 \
python script/run_lingbot_action_only_dsrl.py \
  --config examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml
```

Reference:

- `agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_zh.md`
- `agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_en.md`
