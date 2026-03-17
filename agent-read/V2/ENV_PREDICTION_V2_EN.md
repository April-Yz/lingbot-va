# LingBot Frozen-Noise-DSRL V2 Environment Prediction

## 1. Why Real Runtime Validation Cannot Be Done Now

- the current server does not provide a guaranteed full runtime environment
- the task explicitly forbids running training, evaluation, and environment tests
- the task explicitly forbids installing, upgrading, or downgrading dependencies

## 2. What Was Changed Statically

- added the new `wan_va/frozen_noise_dsrl/` V2 package
- added the V2 config
- added the V2 training entrypoint
- added the V2 static eval entry
- added the bilingual V2 documentation set
- updated the repo README and CHANGELOG

## 3. What Must Wait For Post-Migration Validation

- whether the LingBot checkpoint loads correctly
- whether RoboTwin initializes correctly
- whether the runtime future-latent shape matches the V2 config
- whether V2 noise-policy updates are stable in real training
- whether the V2 eval path produces complete outputs

## 4. Most Likely Environment Problems

### Issue A: Missing Or Incompatible RoboTwin Runtime Dependencies

- impacted modules:
  - `wan_va.action_only_dsrl.robotwin_env`
  - `script/run_lingbot_frozen_noise_dsrl_v2.py`
  - eval client / server
- likely effect:
  - environment startup failure
  - task reset failure
  - rendering or physics import failure
- suggested triage:
  1. check `sapien`, `mplib`, `open3d`, `trimesh`, `toppra`
  2. check RoboTwin task config and camera config
  3. record the issue first instead of broadly upgrading the environment

### Issue B: GPU / CUDA / PyTorch Mismatch

- impacted modules:
  - LingBot server
  - future-latent sampling
  - action sampling
- likely effect:
  - CUDA startup errors
  - OOM
  - kernel / attention-backend failures
- suggested triage:
  1. force `attn_mode='torch'` first
  2. verify CUDA version and PyTorch wheel compatibility
  3. then check whether GPU memory is sufficient

### Issue C: Checkpoint Layout Mismatch Against Server Expectations

- impacted modules:
  - `launch_server.sh`
  - `VA_Server`
- likely effect:
  - missing `config.json`
  - missing or unresolved `transformer/` subdirectory
- suggested triage:
  1. confirm whether `MODEL_PATH` points to a full model root or a training checkpoint root
  2. confirm `transformer/` exists
  3. confirm the base model root still provides `vae/`, `tokenizer/`, and `text_encoder/`

### Issue D: Future-Latent And Action-Noise Dimension Mismatch

- impacted modules:
  - `LingBotFrozenNoiseV2Policy`
  - V2 actor / critic path
- likely effect:
  - `dsrl_action_noise_dim` validation failure
  - future-summary encoding failure
- suggested triage:
  1. inspect `VA_Server._action_noise_shape()`
  2. inspect `dsrl_action_noise_dim`
  3. inspect `dsrl_future_summary_input_dim`
  4. confirm the selected checkpoint matches the config

### Issue E: WandB Or Log Directory Not Writable

- impacted modules:
  - `run_lingbot_frozen_noise_dsrl_v2.py`
- likely effect:
  - startup interruption
  - log write failure
- suggested triage:
  1. disable WandB or switch to offline mode first
  2. confirm `save_root` is writable
  3. confirm sufficient disk space on the target machine

## 5. Which Statements Mean “Code Updated But Not Runtime-Validated”

- the V2 code structure is in place
- the V2 config is in place
- the V2 documentation set is in place
- the V2 future-latent conditioning path is in place
- the V2 action-noise injection path is in place

None of the above means the pipeline has been run through.

## 6. Explicit Wording Restriction

The current result must not be described as:

- fully run through
- validated runnable
- training passed
- evaluation passed

The current result may only be described as:

- static code integration completed
- not yet validated in a real runtime environment
- requires post-migration runtime confirmation
