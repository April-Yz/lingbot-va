# LingBot Action-Only DSRL Change Log

## Scope

This change set implements the minimum viable `lingbot_action_only_dsrl` baseline requested in `agent-read/action_only_v1.md`, while keeping all code changes inside `/home/zaijia001/vam/lingbot-va`.

## Changed Files

### Modified

- `wan_va/wan_va_server.py`
  - Why: LingBot needed a priority-1 steering injection interface.
  - What changed:
    - added `_action_noise_shape(...)`
    - added `_prepare_initial_action_noise(...)`
    - split `_infer(...)` into `sample_future_latents(...)` and `sample_actions(...)`
    - `_infer(...)` now composes those two methods and accepts `obs["initial_noise"]`
  - Backward compatibility: preserved. If no `initial_noise` is passed, the old `torch.randn(...)` path is still used.

- `script/run_lingbot_action_only_dsrl.py`
  - Why: needed a minimal runnable entry for the new baseline.
  - What changed:
    - added a standalone YAML-driven training entry
    - added mock validation before RoboTwin startup
    - added explicit logging for `use_dsrl=false`, `use_dsrl=true`, optimizer setup, and SAC metrics
    - added graceful reporting when RoboTwin startup is blocked by environment/runtime issues
  - Backward compatibility: additive only.

- `agent-read/README.md`
  - Why: project overview must remain current.
  - What changed: added the new action-only DSRL baseline status, validation result, and the current RoboTwin blocker.

- `agent-read/CHANGELOG.md`
  - Why: record the implementation and validation outcome.
  - What changed: added a 2026-03-16 entry for the new baseline, mock validation, and the current RoboTwin limitation.

### Added

- `wan_va/action_only_dsrl/__init__.py`
  - Why: package entry for the new baseline code.

- `wan_va/action_only_dsrl/modules.py`
  - Why: local RLinf-style compact actor/critic building blocks and replay buffer.
  - What changed:
    - `LightweightImageEncoder64`
    - `CompactStateEncoder`
    - `CompactFutureEncoder`
    - `GaussianPolicy`
    - `CompactMultiQHead`
    - `AlphaTemperature`
    - `SimpleReplayBuffer`

- `wan_va/action_only_dsrl/policy.py`
  - Why: LingBot wrapper and minimal SAC trainer.
  - What changed:
    - `LingBotActionOnlyDsrlPolicy`
    - `ActionOnlyDsrlTrainer`
    - freeze reporting and optimizer-group reporting
    - steering noise reshape and future-summary extraction

- `wan_va/action_only_dsrl/robotwin_env.py`
  - Why: RoboTwin environment setup had to be reused without modifying RoboTwin itself.
  - What changed:
    - RoboTwin bootstrap and config loading
    - formatted observation helpers
    - action execution helper
    - reduced seed validation that can skip the expensive expert-demo requirement

- `examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml`
  - Why: dedicated config requested by the spec.
  - What changed:
    - `use_dsrl`
    - `dsrl_mode: action_only`
    - `freeze_backbone`
    - `freeze_future_branch`
    - `freeze_action_decoder`
    - `dsrl_noise_injection_mode`
    - `dsrl_action_noise_dim`
    - `dsrl_num_q_heads`
    - `dsrl_image_latent_dim`
    - `dsrl_state_latent_dim`
    - `dsrl_future_latent_dim`
    - `dsrl_hidden_dims`
    - `algorithm: embodied_sac`

- `agent-read/env_change_log.md`
  - Why: required by the implementation brief for all environment changes.

- `agent-read/implementation_report_lingbot_action_only_dsrl.md`
  - Why: required handoff-quality implementation report.

- `agent-read/V1.2.md`
  - Why: versioned project milestone note for this feature branch.

## Implementation Status

- Steering injection: implemented via `initial_noise`.
- LingBot freeze policy: implemented by freezing the whole LingBot stack (`transformer`, `vae`, `text_encoder`), which is stricter than freezing only the backbone/future/action decoder subsets.
- Steering actor / critic: implemented locally in RLinf-style compact form.
- Optimizer scope: only steering actor / critic / alpha are optimized.
- Backward compatibility with DSRL disabled: preserved and validated in mock mode.
- RoboTwin single-task startup: still blocked on the local `pytorch3d` / Blackwell toolchain mismatch during RoboTwin seed validation.
