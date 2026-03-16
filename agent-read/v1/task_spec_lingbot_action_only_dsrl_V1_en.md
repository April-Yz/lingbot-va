# LingBot Action-Only DSRL V1 Task Spec

## Goal

Implement a minimum viable `lingbot_action_only_dsrl` baseline for LingBot-VA on RoboTwin, following RLinf DSRL / embodied-SAC design as closely as possible without implementing the full method.

## In Scope

- keep LingBot frozen for this baseline
- use the frozen future branch as action-generation context
- add a lightweight steering actor
- add or reuse a critic
- inject steering into LingBot action sampling
- train only steering actor / critic / alpha
- preserve original LingBot behavior when DSRL is disabled
- preserve original LingBot eval and post-train workflows

## Out of Scope

- no full-method future-latent guidance
- no dual-flow optimization
- no planner-side value alignment
- no LingBot backbone finetuning
- no future-branch finetuning
- no action-decoder main-body finetuning
- no RoboTwin task/reward modification
- no PPO / GRPO / unrelated RL algorithm work

## Required Behavior

Per step:

1. receive RoboTwin observation
2. encode observation / language / state with LingBot
3. obtain frozen future context
4. steering actor predicts `steer_noise`
5. inject `steer_noise` into LingBot action sampling
6. frozen LingBot action branch decodes action
7. step RoboTwin
8. use reward for embodied-SAC updates of actor / critic / alpha

## Preferred Injection Priority

1. `sample_actions(..., initial_noise=...)`
2. `sample_actions(..., steering_embedding=...)`
3. residual steering on final action only if 1 and 2 are infeasible

## Required Deliverables

- working code path
- config file
- minimal runnable single-task training entry
- bilingual task spec, implementation report, change log, and environment log

## Verified Minimal Training Command

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

WANDB_TEAM_NAME=haoyuan-lingbot \
WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=action_only_click_bell_v1 \
CUDA_VISIBLE_DEVICES=1 \
python script/run_lingbot_action_only_dsrl.py \
  --config examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml
```

This command uses the current checked-in default config:

- task: `click_bell`
- task config: `demo_clean_large_d435`
- save root: `/home/zaijia001/vam/lingbot-va/train_out/action_only_dsrl_click_bell`
- model path: `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`

## Current V1 Status

- action-only steering injection is implemented through `initial_noise`
- `use_dsrl=false` and `use_dsrl=true` are both validated
- RoboTwin online single-episode execution is validated
- current online success rate is not yet the target; V1 proves the pipeline, not final task performance
