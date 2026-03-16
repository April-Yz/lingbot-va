You are working on a targeted implementation task in an existing robotics codebase.

Task name:
LingBot Action-Only DSRL Baseline V1

## Goal

Implement a minimal viable action-only DSRL baseline for LingBot-VA using RLinf’s existing DSRL / embodied_sac training pipeline on RoboTwin.

This is NOT the full method.
Do NOT implement future-latent guidance, dual-flow optimization, planner-side value alignment, or any full-method extensions.

The only target is:

- freeze LingBot-VA backbone
- freeze LingBot-VA future branch
- freeze LingBot-VA action decoder main body
- add a lightweight steering actor + critic using RLinf’s existing DSRL style
- inject the steering signal into LingBot’s action sampling path
- train only the steering actor / critic
- keep original LingBot eval working
- keep original LingBot post-train working
- preserve backward compatibility when DSRL is disabled

## Documentation Directory and Versioning

All new task, implementation, environment, and change-tracking documents must be placed under:

agent-read/

All files for this iteration must include the suffix:
V1

Examples:
- agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_en.md
- agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_zh.md
- agent-read/v1/implementation_report_lingbot_action_only_dsrl_V1_en.md
- agent-read/v1/implementation_report_lingbot_action_only_dsrl_V1_zh.md
- agent-read/v1/change_log_lingbot_action_only_dsrl_V1_en.md
- agent-read/v1/change_log_lingbot_action_only_dsrl_V1_zh.md
- agent-read/v1/env_change_log_V1_en.md
- agent-read/v1/env_change_log_V1_zh.md

## Bilingual Documentation Requirement

All required documentation must be produced in both English and Chinese.

The English and Chinese versions must stay synchronized.
If one version is updated, the other must also be updated accordingly.

Do not produce English-only or Chinese-only implementation records.

## Strict Scope

You must only implement the minimum changes necessary for an action-only DSRL baseline.

Do NOT:
- modify RoboTwin task logic
- modify RoboTwin rewards
- rewrite RLinf SAC
- introduce PPO / GRPO / new RL algorithms
- finetune LingBot backbone
- finetune LingBot future branch
- finetune LingBot action decoder main body
- add full-method logic
- perform unrelated cleanup / refactor / formatting-only changes

If a change is not necessary for action-only DSRL, do not make it.

## Backward Compatibility Requirement

You must preserve all original non-DSRL workflows.

Specifically:
- original LingBot eval must still run correctly
- original LingBot post-train must still run correctly
- original inference behavior must remain compatible when DSRL is disabled
- use_dsrl = false must preserve original behavior

This is a hard requirement.

## Environment Constraints

Preferred working environments:
1. RoboTwin-lingbot
2. the existing LingBot / RLinf conda environment already configured for this project

Default rule:
- perform implementation and validation inside these existing isolated conda environments
- do not pollute the system Python environment
- do not install packages globally
- do not modify unrelated shared environments

If the RLinf environment is broken or has dependency issues:
- you may perform diagnostic fixes in an isolated conda environment or another clearly isolated environment
- but every such action must be recorded
- and the final reproducible validation must still be completed in the designated project environments

All environment-related actions must be recorded in:
- agent-read/v1/env_change_log_V1_en.md
- agent-read/v1/env_change_log_V1_zh.md

For each environment change, record:
- datetime
- conda environment name
- command executed
- what was installed or changed
- why it was needed
- whether it succeeded
- how it was verified

## Functional Requirement

Implement a new mode:
lingbot_action_only_dsrl_V1

Expected per-step behavior:

1. receive RoboTwin observation
2. encode observation / language / state with LingBot
3. generate future latent or action-conditioning context using the frozen LingBot future branch
4. produce steer_noise (or steering_embedding) with a lightweight RLinf SAC steering actor
5. inject this steering signal into LingBot action sampling
6. produce action with frozen LingBot action decoder
7. step RoboTwin environment
8. use environment reward to train steering actor + critic with RLinf embodied_sac

## Injection Priority

You must search for and implement the steering injection point in this priority order:

Priority 1:
Expose an action sampling initial noise interface, e.g.

sample_actions(
    obs,
    future_latent=None,
    initial_noise=None,
    ...
)

Behavior:
- if initial_noise is provided, use it
- otherwise keep original random initialization

Priority 2:
If explicit initial noise is not available, implement:

sample_actions(
    obs,
    future_latent=None,
    steering_embedding=None,
    ...
)

and inject steering_embedding into the action decoder conditioning path.

Priority 3:
Only if both options above are impossible, implement residual steering on the action field.
Do not choose Priority 3 unless 1 and 2 are truly infeasible.

## Reuse Requirement

You must maximize reuse of RLinf’s existing Pi0 DSRL implementation.

First inspect and reuse:
- existing use_dsrl logic
- embodied_sac pipeline
- DSRL actor / critic wiring
- replay buffer / learner / evaluator / logger path
- noise injection path used for Pi0 DSRL

Your implementation should mirror the existing RLinf DSRL style as closely as possible.

## Code Changes Required

### 1. LingBot DSRL wrapper
Add a policy/model wrapper for LingBot action-only DSRL, for example:
- LingBotDsrlPolicy
or
- LingBotActionOnlyDsrlPolicy

It must:
- load LingBot checkpoint
- expose a clean RLinf-compatible act/forward interface
- support use_dsrl = true/false
- support dsrl_mode = action_only
- freeze LingBot modules according to config
- call the steering actor only when DSRL is enabled
- preserve original behavior when DSRL is disabled

### 2. LingBot action sampling API
Modify LingBot minimally to expose a steering injection interface.

Preferred:
- initial_noise

Fallback:
- steering_embedding

Default inference behavior must remain unchanged when no steering signal is passed.

The original eval and post-train paths must remain usable.

### 3. Steering actor
Add a lightweight steering actor, following RLinf DSRL style.

Suggested inputs:
- current RGB features
- robot state / proprio
- optional future latent summary

Suggested output:
- steer_noise matching action sampling noise shape

Keep this module lightweight.

### 4. Critic
Reuse or extend RLinf’s existing critic style.

Suggested target form:
Q(obs, state, future_latent_summary, steer_noise)

Use the existing multi-Q-head style where possible.

### 5. Config
Add a dedicated config for this mode, for example:
examples/embodiment/config/robotwin_lingbot_action_only_dsrl_V1.yaml

The config must clearly include:
- use_dsrl
- dsrl_mode: action_only
- freeze_backbone
- freeze_future_branch
- freeze_action_decoder
- dsrl_noise_injection_mode
- dsrl_action_noise_dim
- dsrl_num_q_heads
- dsrl_image_latent_dim
- dsrl_state_latent_dim
- dsrl_hidden_dims
- algorithm: embodied_sac

Every new config field must be documented in both English and Chinese documentation.

## Optimizer Constraint

The optimizer must contain only:
- steering actor parameters
- critic parameters
- SAC temperature / alpha parameters if needed

The optimizer must NOT contain:
- LingBot backbone parameters
- LingBot future branch parameters
- LingBot action decoder main-body parameters

You must verify and document this.

## Logging and Validation

Add sufficient logs to verify behavior, including at least:
- use_dsrl enabled/disabled
- dsrl_mode
- injection mode
- steer_noise shape
- action output shape
- number of frozen LingBot parameters
- number of trainable parameters
- optimizer parameter groups
- SAC training metrics

You must validate at least:
1. use_dsrl = false preserves original LingBot inference path
2. use_dsrl = true actually injects steering into action sampling
3. LingBot backbone/future/action decoder are not optimized
4. RoboTwin single-task training starts successfully
5. SAC metrics are visible in logs
6. original LingBot eval still works
7. original LingBot post-train still works

## Deliverables

You must produce both code and documentation.

### Required code deliverables
- working implementation
- config file
- any required wrapper / registration changes
- minimal runnable single-task training entry

### Required documentation deliverables

1. agent-read/v1/change_log_lingbot_action_only_dsrl_V1_en.md
2. agent-read/v1/change_log_lingbot_action_only_dsrl_V1_zh.md

Document:
- every changed file
- whether it was added/modified/deleted
- why it was changed
- what exactly changed
- whether behavior remains backward compatible

3. agent-read/v1/env_change_log_V1_en.md
4. agent-read/v1/env_change_log_V1_zh.md

Document every environment-related action.

5. agent-read/v1/implementation_report_lingbot_action_only_dsrl_V1_en.md
6. agent-read/v1/implementation_report_lingbot_action_only_dsrl_V1_zh.md

These documents must be readable by both humans and agents and must be sufficient for reproduction and future handoff.

They must include:
- implementation goal
- scope boundaries
- what was done
- what was not done
- architecture overview
- step-by-step data/control flow
- file-level explanation
- config explanation
- run instructions
- validation steps
- known limitations
- next-step suggestions
- minimal reproduction steps

7. agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_en.md
8. agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_zh.md

These should reflect the final synchronized task specification.

## Minimal Reproduction Section Requirement

The implementation report must include a strict minimal reproduction section with:
1. repository path
2. conda environment to activate
3. dependency checks
4. exact run command
5. expected logs
6. what indicates success
7. common failure modes and debugging hints

Another agent or developer should be able to continue the work without reading chat history.

## Workflow Order

Follow this order strictly:

1. inspect RLinf Pi0 DSRL implementation and summarize what will be reused
2. inspect LingBot action sampling path and determine whether initial_noise can be injected
3. implement wrapper and interface passthrough
4. connect steering actor and critic
5. add config and run entry
6. validate at least one RoboTwin single-task run
7. validate original LingBot eval still works
8. validate original LingBot post-train still works
9. write bilingual logs and implementation documentation

## Output Style

When you finish, provide a concise implementation summary including:
- files changed
- files added
- environment changes
- commands executed
- current run status
- whether single-task training starts
- whether original eval still works
- whether original post-train still works
- known issues or limitations

Do not be vague.
Avoid phrases like:
- “made some changes”
- “fixed several issues”
- “updated config”
- “added related support”

Be specific:
- class/function/config names
- exact behavior before/after
- reason for each change

## Final instruction

Apply the minimum necessary code changes to implement a reproducible, well-documented, bilingual, action-only DSRL baseline V1 for LingBot-VA on RLinf + RoboTwin, while preserving original LingBot eval and post-train functionality and avoiding environment pollution.
