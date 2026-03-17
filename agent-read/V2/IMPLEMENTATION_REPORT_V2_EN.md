# LingBot Frozen-Noise-DSRL V2 Static Implementation Report

## 1. Core Experiment Goal

V2 keeps LingBot as a frozen inference backbone:

- generate future latents with `VA_Server.sample_future_latents(...)`
- feed those future latents into a new noise policy as condition
- replace LingBot's default random action noise with the learned noise policy output
- update only the new noise policy and critic modules, not the LingBot backbone

This delivery is a static code-integration version and has not been executed in a real environment.

## 2. LingBot's Role In V2

LingBot serves two frozen roles in V2:

- frozen future latent proposer
- frozen action decoder / inverse-action path

The implementation does not restructure the LingBot core network. V2 directly reuses the minimum existing interfaces in [wan_va_server.py](/home/e230112/vam/lingbot-va/wan_va/wan_va_server.py):

- `sample_future_latents(...)`
- `sample_actions(..., initial_noise=...)`
- `_reset(...)`
- `_compute_kv_cache(...)`

## 3. RLInf's Role In V2

V2 does not directly wire `/home/e230112/vam/RLinf` into the runtime entrypoints. Instead, it borrows RLInf training ideas rather than rewriting LingBot into an RLInf-native model:

- `embodied_sac` naming and actor-critic structure
- replay buffer + target critic + alpha temperature update flow
- separate actor / critic optimizers
- soft target updates

These mechanisms remain locally implemented inside `lingbot-va` to keep LingBot minimally changed.

## 4. What V2 Reuses From V1

V2 reuses the following V1 patterns:

- separate experiment package and separate entrypoint
- frozen-forward reuse through `VA_Server`
- additive YAML config layout that coexists with baseline / original post-train
- training script, logging, mock-validation, and documentation structure
- local compact RLInf-style actor / critic implementation pattern

Key V2 differences from V1:

- V1 focused on action-only steering
- V2 explicitly centers on future-latent-conditioned noise control
- V2 documents the future-latent generation point, detach point, and action-noise replacement point as an explicit contract
- V2 config explicitly includes `freeze_future_video_module`, `freeze_inverse_or_action_flow`, `train_noise_policy_only`, `future_latent_as_condition`, and `use_rlinf_or_v1_style_training`

## 5. Where Future Latents Are Generated

Future latents are generated in [wan_va_server.py](/home/e230112/vam/lingbot-va/wan_va/wan_va_server.py) inside `VA_Server.sample_future_latents(...)`.

V2 calls that interface in [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py) inside `LingBotFrozenNoiseV2Policy.act(...)`.

## 6. Where Future Latents Are Detached / Stop-Grad

The stop-gradient point is in [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py) inside `build_step_batch(...)`:

- `future_condition = future_latents.detach() if self.detach_future_latent else future_latents`

Only the detached future-latent summary is passed into the new future encoder.

## 7. Noise Policy Inputs And Outputs

Inputs:

- image-history encoding
- state-vector encoding
- future-latent summary encoding

Output:

- a flat action-noise code that matches LingBot's action diffusion noise shape

Main modules:

- [modules.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/modules.py)
- [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py)

## 8. Where The Original Action Noise Is Replaced

The replacement path is in [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py) inside `act(...)`:

1. `noise_policy.sample(...)` produces `steer_noise_flat`
2. `self._reshape_noise(...)` maps it back to LingBot's action-noise tensor
3. `self.server.sample_actions(..., initial_noise=steer_noise)` injects it into the frozen LingBot decoder path

So V2 does not rewrite LingBot's action decoder. It only swaps the default random initial noise for a learned controller output.

## 9. Why The LingBot Backbone Is Frozen And How It Is Frozen

Why:

- V2 is meant to be an external noise controller, not a LingBot backbone rewrite
- keeping the future proposer and action decoder frozen preserves baseline behavior
- the optimization scope stays focused on the new V2 modules

How:

The freezing logic is implemented in [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py) inside `_freeze_lingbot(...)`:

- `self.server.transformer.requires_grad_(False)`
- `self.server.vae.requires_grad_(False)`
- `self.server.text_encoder.requires_grad_(False)`

The current LingBot transformer serves both the future and action sides, so this static V2 implementation freezes the whole loaded backbone. That is stricter than freezing only a sub-branch, but it preserves default behavior.

## 10. Which Parameters Update During Training

Only the new V2 modules are trainable:

- `actor_image_encoder`
- `actor_state_encoder`
- `actor_future_encoder`
- `noise_policy`
- `critic_image_encoder`
- `critic_state_encoder`
- `critic_future_encoder`
- `q_head`
- `alpha_temperature`

LingBot `transformer`, `vae`, and `text_encoder` remain frozen.

## 11. Whether RLInf Was Directly Integrated

No direct RLInf runtime integration was added.

Borrowed from RLInf:

- SAC/DSRL-style training organization
- `adv_type: embodied_sac`
- `loss_type: embodied_sac`
- replay buffer
- separate actor / critic / alpha updates
- soft target critic updates

Borrowed from V1:

- directory layout
- local compact module strategy
- standalone runner + config + documentation handoff pattern

This is the more pragmatic choice for the current task because the brief requires minimal LingBot changes and the current server is not suitable for heavier cross-repo runtime validation.

## 12. Why Original LingBot Eval / Post-Train Remain Compatible

V2 is additive:

- new package: `wan_va/frozen_noise_dsrl/`
- new training entry: `script/run_lingbot_frozen_noise_dsrl_v2.py`
- new static eval entry: `script/run_lingbot_frozen_noise_eval_v2.sh`
- new config: `examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml`

The baseline, post-train, and V1 entrypoints are not replaced, and LingBot's default behavior without `initial_noise` is unchanged.

## 13. Files Changed

Added:

- [__init__.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/__init__.py)
- [modules.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/modules.py)
- [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py)
- [run_lingbot_frozen_noise_dsrl_v2.py](/home/e230112/vam/lingbot-va/script/run_lingbot_frozen_noise_dsrl_v2.py)
- [run_lingbot_frozen_noise_eval_v2.sh](/home/e230112/vam/lingbot-va/script/run_lingbot_frozen_noise_eval_v2.sh)
- [robotwin_lingbot_frozen_noise_dsrl_v2.yaml](/home/e230112/vam/lingbot-va/examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml)
- the full bilingual `agent-read/V2/` document set

Modified:

- [README.md](/home/e230112/vam/lingbot-va/agent-read/README.md)
- [CHANGELOG.md](/home/e230112/vam/lingbot-va/agent-read/CHANGELOG.md)

## 14. What Is Still Static And Unverified

The following are static integrations only:

- V2 trainer entrypoint
- V2 future-latent-conditioned noise policy
- V2 critic / alpha update path
- V2 YAML config
- V2 static eval entry script
- V2 run guide and reproduction checklist
- environment-risk prediction

The following were not executed:

- training
- evaluation
- environment tests
- dependency installation
- runtime integration checks
