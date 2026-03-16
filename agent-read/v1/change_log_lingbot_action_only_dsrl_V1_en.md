# LingBot Action-Only DSRL V1 Change Log

## Scope

This file is the V1 English change log for the action-only DSRL baseline.

## Modified Files

- `wan_va/wan_va_server.py`
  - added `sample_future_latents(...)`
  - added `sample_actions(..., initial_noise=...)`
  - preserved the original random-noise path when no external steering is provided

- `script/run_lingbot_action_only_dsrl.py`
  - added a standalone single-task training entry
  - added mock validation logging
  - added final status reporting for blocked / finished-no-success / finished-success runs

- `/home/zaijia001/vam/RoboTwin-lingbot/envs/camera/camera.py`
  - replaced the old `missing pytorch3d -> exit()` fallback with a CPU farthest-point sampler

## Added Files

- `wan_va/action_only_dsrl/__init__.py`
- `wan_va/action_only_dsrl/modules.py`
- `wan_va/action_only_dsrl/policy.py`
- `wan_va/action_only_dsrl/robotwin_env.py`
- `examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml`

## Backward Compatibility

- `use_dsrl=false` preserves the original LingBot action generation path
- original LingBot eval code path is preserved
- original LingBot post-train code path is preserved at the interface level; regression smoke validation is tracked separately in the implementation report
