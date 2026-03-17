# LingBot Frozen-Noise-DSRL V2 Reproduction Checklist

This checklist is only for validation after migration to a server with the required runtime environment. It was not executed on the current server.

## 1. Baseline Health Check

1. Confirm the baseline server starts correctly
2. Confirm the baseline client completes a single-task smoke eval
3. Confirm `metrics`, `video`, and `result` artifacts are written
4. Confirm the V2 integration does not change baseline default behavior

## 2. Original Post-Train Health Check

1. Confirm `script/run_va_posttrain.sh` still starts
2. Confirm the dataset path, empty-embedding path, and base-model path are valid
3. Confirm the checkpoint directory is written correctly
4. Confirm the original post-train path does not depend on the V2 modules

## 3. V2 Health Check

1. Confirm `examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml` resolves correctly
2. Confirm `run_lingbot_frozen_noise_dsrl_v2.py` loads the config correctly
3. Confirm the startup report shows:
   - `freeze_future_video_module=true`
   - `freeze_inverse_or_action_flow=true`
   - `train_noise_policy_only=true`
   - `future_latent_as_condition=true`
4. Confirm future latents enter the V2 noise-policy conditioning path
5. Confirm action noise is injected through `sample_actions(..., initial_noise=...)`
6. Confirm only the new V2 modules receive training updates
7. Confirm V2 log and checkpoint naming is clear

## 4. V2 Eval Health Check

1. Confirm the server path points to the intended checkpoint
2. Confirm the client uses `model_tag=frozen-noise-v2`
3. Confirm the V2 result directory is separate from baseline
4. Confirm the V2 eval path does not overwrite baseline eval outputs

## 5. Environment-Boundary Check

1. Confirm environment changes remain restricted to the LingBot and RLInf target scopes
2. Confirm system Python / CUDA / global pip remain untouched
3. Confirm no unrelated conda environment was modified
4. If later environment changes are required, log them explicitly

## 6. Current Statement

This checklist is for post-migration validation only. None of the checks above were executed on the current server.
