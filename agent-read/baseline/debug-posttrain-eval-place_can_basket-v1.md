# Debug Record: Place Can Basket Post-Train Eval V1

## Scope

This note records the local March 16, 2026 debugging cycle for running a LingBot post-train checkpoint on RoboTwin `place_can_basket`.

## Goal

Produce the first real RoboTwin eval result from:

- checkpoint: `/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean/checkpoints/checkpoint_step_5000`
- task: `place_can_basket`
- task config: `demo_clean_large_d435`

## Issue 1: `MODEL_PATH` Could Not Be A Training Checkpoint Root

### Symptom

The LingBot server failed immediately with an error like:

- `.../checkpoint_step_5000/vae is not the path to a directory containing a config.json file`

### Root Cause

Training checkpoints such as `checkpoint_step_5000` only save:

- `transformer/`

They do not save:

- `vae/`
- `tokenizer/`
- `text_encoder/`

The original server logic treated `MODEL_PATH` as if it always pointed to a complete model root.

### Fix

`wan_va_server.py` now supports two model-path layouts:

- a full model root with `transformer/`, `vae/`, `tokenizer/`, `text_encoder/`
- a train checkpoint root with only `transformer/`

When the second layout is detected, the server loads:

- `transformer` from `MODEL_PATH`
- `vae/tokenizer/text_encoder` from the configured base-model root

## Issue 2: The Client Could Not Import `evaluation.robotwin.*`

### Symptom

The client failed immediately with:

- `ModuleNotFoundError: No module named 'evaluation'`

### Root Cause

The client was executed from inside `RoboTwin-lingbot`, but the script itself lives in `lingbot-va` and imports `evaluation.robotwin.*`. Python did not automatically treat the `lingbot-va` repo root as an import root.

### Fix

`eval_polict_client_openpi.py` now prepends:

- the local `lingbot-va` repo root
- `ROBOTWIN_ROOT`

to `sys.path` before importing project modules.

## Issue 3: `place_can_basket` Could Crash During Task Setup Or Planning

### Symptom A

During eval the task could abort with:

- `AssertionError: target_pose cannot be None for move action.`

### Root Cause A

`place_can_basket` eventually calls `grasp_actor(...)`. When no valid grasp pose is found, the old code still built a `move` action with `target_pose=None`, which turned a normal planning failure into a hard exception.

### Fix A

`RoboTwin-lingbot/envs/_base_task.py` now treats missing grasp poses as an ordinary planning failure:

- marks `self.plan_success = False`
- returns `None, []`

This lets task logic continue or fail normally instead of crashing the whole eval.

### Symptom B

When skipping expert-check, the task could lack prompt substitutions for `{A}`, `{B}`, `{a}`.

### Root Cause B

`place_can_basket.setup_demo(...)` initialized the scene but did not populate `self.info["info"]` until `play_once()` completed.

### Fix B

`RoboTwin-lingbot/envs/place_can_basket.py` now fills `self.info["info"]` during `setup_demo(...)`.

## Issue 4: Debug Overrides Existed But Did Not Fully Reach The Eval Path

### Symptom

`expert_check` and shortened step limits were not reliably affecting the real episode path.

### Root Cause

The client script had partial support for overrides, but they were not consistently propagated into task args and episode setup.

### Fix

`eval_polict_client_openpi.py` now:

- propagates `expert_check` into task args
- accepts string truthy/falsy values robustly
- propagates `step_limit_override`
- when `expert_check=false`, reads prompt metadata from `TASK_ENV.info`
- applies the step-limit override after `setup_demo(...)`
- fixes the latent manifest path generation

## Known Non-Fatal Warnings

These warnings were present during the successful smoke run and are not the direct blocker:

- `RequestsDependencyWarning`
- `clip_output` ignored in Wan VAE config
- `curobo` import or JIT-build failure on Blackwell with CUDA 12.1
- missing `pytorch3d`, followed by CPU farthest-point fallback

They affect planner choice or log noise, but they did not stop the first result from being produced.

## First Completed Eval Result

The first verified result-producing smoke command used:

- `--expert_check false`
- `--step_limit_override 60`
- `--test_num 1`
- port `29058`

Result:

- task: `place_can_basket`
- seed: `10000`
- success count: `0/1`
- status: pipeline completed and produced artifacts

Artifacts:

- metrics: `/home/zaijia001/vam/RoboTwin-lingbot/results_posttrain_eval_step5000_fix4/stseed-10000/metrics/place_can_basket/res.json`
- summary: `/home/zaijia001/vam/RoboTwin-lingbot/eval_result/place_can_basket/ACT/demo_clean_large_d435/0/2026-03-16 15:11:37/_result.txt`
- rollout video: `/home/zaijia001/vam/RoboTwin-lingbot/results_posttrain_eval_step5000_fix4/stseed-10000/visualization/place_can_basket/`

## Practical Conclusion

As of March 16, 2026:

- the post-train checkpoint eval pipeline is now able to reach a real RoboTwin result on `place_can_basket`
- the first smoke result was a failure case (`0/1`), but it confirms the server-client-task pipeline now completes end-to-end
- further work should focus on task success, not on the earlier launch-time crashes
