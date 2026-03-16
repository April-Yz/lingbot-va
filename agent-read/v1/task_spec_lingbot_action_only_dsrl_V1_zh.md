# LingBot Action-Only DSRL V1 任务规格

## 目标

在 RoboTwin 上为 LingBot-VA 实现一个最小可行的 `lingbot_action_only_dsrl` 基线，尽量贴近 RLinf 的 DSRL / embodied-SAC 设计，但不实现 full method。

## 范围内

- 在这个基线中保持 LingBot 冻结
- 使用冻结 future branch 提供动作生成上下文
- 新增一个轻量 steering actor
- 新增或复用一个 critic
- 将 steering 信号注入 LingBot action sampling
- 只训练 steering actor / critic / alpha
- 在关闭 DSRL 时保持原始 LingBot 行为
- 保持原始 LingBot eval 与 post-train 流程可用

## 范围外

- 不做 full method 的 future-latent guidance
- 不做 dual-flow optimization
- 不做 planner-side value alignment
- 不做 LingBot backbone finetune
- 不做 future branch finetune
- 不做 action decoder 主体 finetune
- 不修改 RoboTwin task / reward
- 不引入 PPO / GRPO / 其他无关 RL 算法

## 必要行为

每个 step 的标准路径：

1. 接收 RoboTwin observation
2. 用 LingBot 编码 observation / language / state
3. 获取冻结 future branch 的上下文
4. steering actor 预测 `steer_noise`
5. 将 `steer_noise` 注入 LingBot action sampling
6. 冻结的 LingBot action branch 解码动作
7. 在 RoboTwin 中执行动作
8. 用环境 reward 对 actor / critic / alpha 做 embodied-SAC 更新

## 注入优先级

1. `sample_actions(..., initial_noise=...)`
2. `sample_actions(..., steering_embedding=...)`
3. 只有前两者都不可行时才做 residual action steering

## 必需交付

- 可运行代码路径
- 配置文件
- 最小单任务训练入口
- 双语 task spec / implementation report / change log / env log

## 当前可直接使用的最小训练命令

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

这条命令使用当前仓库默认配置：

- task: `click_bell`
- task config: `demo_clean_large_d435`
- save root: `/home/zaijia001/vam/lingbot-va/train_out/action_only_dsrl_click_bell`
- model path: `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`

## 当前 V1 状态

- action-only steering 注入已经通过 `initial_noise` 实现
- `use_dsrl=false` 与 `use_dsrl=true` 都已验证
- RoboTwin 在线单 episode 执行已经验证通过
- 当前在线 success rate 还不是目标，V1 证明的是 pipeline 已打通，而不是最终任务性能
