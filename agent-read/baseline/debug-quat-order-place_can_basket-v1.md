# Debug: Quaternion Order Smoke Test For `place_can_basket`

## Goal

This debug note records a temporary eval-only experiment for the locally trained:

- `checkpoint_step_10000`

The purpose was narrow:

- keep the same checkpoint
- keep the same task and seed
- only change how the eval client composes quaternion actions before sending them to RoboTwin

This was designed to test whether the visible pose/orientation issue could be explained purely by the eval-side `xyzw` vs `wxyz` interpretation.

## Temporary Change

The RoboTwin eval client now accepts:

- `--quat_order_mode legacy_xyzw`
- `--quat_order_mode robowin_wxyz`

in:

- `/home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py`

Behavior:

- `legacy_xyzw`: preserve the old logic
- `robowin_wxyz`: treat incoming RoboTwin pose quaternions as `wxyz`, convert to `xyzw` only for `scipy`, then convert the composed result back to `wxyz` before sending the action to RoboTwin

This is a temporary debug switch for eval only. It does not change the training bundle or the training data loader.

## Test Setup

Server:

- model: `/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean/checkpoints/checkpoint_step_10000`
- port: `29060`

Client settings shared by both runs:

- task: `place_can_basket`
- task config: `demo_clean_large_d435`
- seed: `0` which maps to start seed `10000`
- `expert_check=false`
- `step_limit_override=60`
- `test_num=1`

Renderer / environment workaround used during this test:

- `LINGBOT_SKIP_RENDER_TEST=1`
- `SAPIEN_RT_DENOISER=none`
- `CUDA_VISIBLE_DEVICES=3` for the RoboTwin client

The client was pinned to GPU 3 because an earlier attempt without that pin hit:

- `curobo` import OOM on the default CUDA device
- `RuntimeError: cannot create buffer`

## Commands Used

### Legacy eval behavior

```bash
conda activate RoboTwin-lingbot
cd /home/zaijia001/vam/RoboTwin-lingbot

CUDA_VISIBLE_DEVICES=3 \
PYTHONWARNINGS=ignore::UserWarning \
LINGBOT_SKIP_RENDER_TEST=1 \
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
  --model_tag ckpt10000-legacyquat \
  --quat_order_mode legacy_xyzw \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_posttrain_eval_step10000_legacyquat \
  --expert_check false \
  --step_limit_override 60 \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29060
```

### Temporary `wxyz` eval behavior

```bash
conda activate RoboTwin-lingbot
cd /home/zaijia001/vam/RoboTwin-lingbot

CUDA_VISIBLE_DEVICES=3 \
PYTHONWARNINGS=ignore::UserWarning \
LINGBOT_SKIP_RENDER_TEST=1 \
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
  --model_tag ckpt10000-wxyzquat \
  --quat_order_mode robowin_wxyz \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_posttrain_eval_step10000_wxyzquat \
  --expert_check false \
  --step_limit_override 60 \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29060
```

## Results

### Legacy result

- result file:
  - `/home/zaijia001/vam/RoboTwin-lingbot/eval_result/place_can_basket/ACT/demo_clean_large_d435/0/ckpt10000-legacyquat/2026-03-16 21:50:11/_result.txt`
- metrics:
  - `/home/zaijia001/vam/RoboTwin-lingbot/results_posttrain_eval_step10000_legacyquat/stseed-10000/metrics/place_can_basket/res.json`
- outcome:
  - `0/1`

### `robowin_wxyz` result

- result file:
  - `/home/zaijia001/vam/RoboTwin-lingbot/eval_result/place_can_basket/ACT/demo_clean_large_d435/0/ckpt10000-wxyzquat/2026-03-16 21:54:01/_result.txt`
- metrics:
  - `/home/zaijia001/vam/RoboTwin-lingbot/results_posttrain_eval_step10000_wxyzquat/stseed-10000/metrics/place_can_basket/res.json`
- outcome:
  - `0/1`

## Interpretation

This test did **not** show a one-seed success improvement from changing only the eval-side quaternion composition.

That means:

- simply swapping the quaternion handling inside eval did not immediately turn this `checkpoint_step_10000` run into a successful `place_can_basket` rollout

But this result does **not** clear the preprocessing bug.

Why not:

1. the checkpoint itself was trained on data likely built with the old convention mismatch
2. changing only eval-side composition cannot retroactively repair a model that learned from inconsistent rotation targets
3. so a `0/1` vs `0/1` tie here is still compatible with “the training data pipeline has a quaternion-order bug”

## Current Best Reading

High-confidence statement:

- the RoboTwin raw-data-to-LingBot pipeline still has a real quaternion convention risk

What this smoke test adds:

- the issue is probably not fixable by an eval-only patch alone

Most likely next proper validation:

1. patch the raw-data conversion and training-data loader to use one explicit quaternion convention
2. regenerate a small post-train bundle
3. retrain a small smoke checkpoint
4. compare that new checkpoint against the current one on the same seed/task
