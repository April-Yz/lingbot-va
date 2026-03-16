# LingBot Action-Only DSRL V1 实现报告

## 目标

提供一个可复现的 LingBot-VA + RoboTwin action-only DSRL V1 基线，并保持原始 LingBot eval 与 post-train 流程可用。

## 已实现内容

- 通过 `VA_Server.sample_actions(..., initial_noise=...)` 实现 priority-1 steering 注入
- 冻结 LingBot，并且只训练 steering actor / critic / alpha
- 本地实现 RLinf 风格的轻量 actor / critic 模块
- 单任务训练入口：
  - `script/run_lingbot_action_only_dsrl.py`
- 单一配置文件：
  - `examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml`

## 未实现内容

- 不包含 full method 的 future-latent guidance
- 不包含 planner-side value alignment
- 不包含 dual-flow optimization
- 不包含 LingBot finetuning
- 还没有并入 RLinf 原生注册路径

## 主运行路径

1. 用 prompt reset LingBot
2. 编码当前 observation
3. 运行冻结的 future branch
4. steering actor 预测扁平化的 `steer_noise`
5. 将噪声 reshape 成 LingBot action-noise tensor
6. 由冻结的 LingBot action branch 解码动作
7. 在 RoboTwin 中执行动作
8. 用 embodied-SAC loss 更新 actor / critic / alpha

## 当前可直接使用的 V1 训练命令

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

WANDB_TEAM_NAME=haoyuan-lingbot \
WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=action_only_click_bell_v1 \
CUDA_VISIBLE_DEVICES=0 \
python script/run_lingbot_action_only_dsrl.py \
  --config examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml
```

这条命令对应的默认配置是：

- task: `click_bell`
- task config: `demo_clean_large_d435`
- model path: `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`
- save root: `/home/zaijia001/vam/lingbot-va/train_out/action_only_dsrl_click_bell`
- wandb entity: `haoyuan-lingbot`
- wandb project: `lingbot`
- wandb run name: `action_only_click_bell_v1`

## 当前 RL 训练配置说明

当前这版 V1 不是按“训练多少个 gradient steps”来定义时长，而是按“最多跑多少个 online episode、每个 episode 最多采多少个 action chunk”来控制。

配置文件：

- [robotwin_lingbot_action_only_dsrl.yaml](/home/zaijia001/vam/lingbot-va/examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml)

当前默认值：

- `runner.seed = 10000`
- `runner.max_episodes = 1`
- `runner.max_action_chunks = 2`
- `algorithm.replay_size = 4096`
- `algorithm.batch_size = 1`
- `algorithm.warmup_steps = 1`
- `algorithm.updates_per_step = 1`
- `algorithm.gamma = 0.99`
- `algorithm.tau = 0.005`
- `algorithm.actor_lr = 1e-4`
- `algorithm.critic_lr = 1e-4`
- `algorithm.alpha_lr = 1e-4`
- `algorithm.initial_alpha = 0.01`
- `algorithm.target_entropy = -240.0`

这些参数的含义可以直接记成：

- `max_episodes`
  - 训练最多跑多少个 online episode
- `max_action_chunks`
  - 每个 episode 最多向 RoboTwin 发多少次 action chunk
- `warmup_steps`
  - replay buffer 至少积累多少步后才开始做 SAC 更新
- `updates_per_step`
  - 每次环境交互后做多少次参数更新
- `batch_size`
  - 每次从 replay buffer 采样多少条 transition
- `replay_size`
  - replay buffer 最大容量

所以当前默认配置本质上是一个非常小的 smoke-train：

- 最多 `1` 个 episode
- 每个 episode 最多 `2` 个 action chunks
- warmup 只要 `1` 步
- 每步只更新 `1` 次

它的目标主要是验证：

- action-only pipeline 能启动
- DSRL actor / critic / alpha 能反向传播
- 在线 RoboTwin 交互能打通

而不是追求任务成功率。

## WandB 上报内容

当前 action-only 训练脚本已经支持 WandB，并默认写到：

- entity: `haoyuan-lingbot`
- project: `lingbot`

默认会上报：

- 完整 YAML config
- `startup/report`
- `validation/mock_sac_metrics`
- `train/critic_loss`
- `train/actor_loss`
- `train/alpha_loss`
- `train/alpha`
- `train/global_step`
- `train/replay_size`
- `episode/return`
- `episode/successes`
- `final/report`

## 验证总结

### Action-Only 路径

- `use_dsrl=false`：已验证
- `use_dsrl=true`：已验证
- RoboTwin 在线单 episode：已验证
- 最近一次在线状态：`finished_no_success`
- 当前日志里看到的：
  - `curobo.types` 缺失
  - `missing pytorch3d`
  - `Vulkan ICD warning`
  都不是这条链路的直接致命错误；当前实现会回退到 `MPLib` 和 CPU farthest-point sampler，所以只要后面继续推进到 `Reset.`、`train/metrics`、`final/report`，这类 warning 可以先视为已知噪声。

### 原始 LingBot Eval 回归验证

验证时间：2026-03-16

使用命令：

1. 启动 server
```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va
CUDA_VISIBLE_DEVICES=1 bash evaluation/robotwin/launch_server.sh
```

2. 启动 client smoke
```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va
PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python -m evaluation.robotwin.eval_polict_client_openpi \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name click_bell \
  --task_config demo_clean \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_regression_eval \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

观察结果：

- task: `click_bell`
- 指标文件：`/home/zaijia001/vam/RoboTwin-lingbot/results_regression_eval/stseed-10000/metrics/click_bell/res.json`
- 结果：`1 / 1`，`succ_rate = 1.0`

说明：

- 这次回归验证使用的是原始 `demo_clean` eval 路径，用来确认向后兼容性
- 当前 action-only 默认配置已经切到 `demo_clean_large_d435`，这样和你重新采集的 `Large_D435` 数据保持一致

### 原始 LingBot Post-Train 回归验证

验证时间：2026-03-16

使用命令：

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

MASTER_PORT=29621 \
CUDA_VISIBLE_DEVICES=0,1 \
NGPU=2 CONFIG_NAME=robotwin_train \
bash script/run_va_posttrain.sh \
  --dataset-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-base \
  --save-root /home/zaijia001/vam/lingbot-va/train_out/posttrain_regression_smoke \
  --enable-wandb false \
  --attn-mode torch \
  --dataset-init-worker 1 \
  --num-steps 1 \
  --save-interval 1
```

观察结果：

- 训练成功跑到 `1 / 1` steps
- checkpoint 成功保存
- checkpoint 路径：
  - `/home/zaijia001/vam/lingbot-va/train_out/posttrain_regression_smoke/checkpoints/checkpoint_step_1/transformer/config.json`
  - `/home/zaijia001/vam/lingbot-va/train_out/posttrain_regression_smoke/checkpoints/checkpoint_step_1/transformer/diffusion_pytorch_model.safetensors`

## 最小复现

1. 仓库路径
   - `/home/zaijia001/vam/lingbot-va`
2. 环境
   - `conda activate lingbot-va`
3. 依赖检查
   - `python -c "import torch, diffusers, transformers"`
   - `python -c "import sapien, mplib, open3d, trimesh, toppra, lxml"`
4. 精确 action-only 运行命令
   - 见上面的 `V1 训练命令`
5. 预期日志
   - `validation/use_dsrl_false`
   - `validation/use_dsrl_true`
   - `train/metrics`
   - `final/report`
6. 成功判据
   - 进程正常退出并写出 `final/report`
7. 常见失败模式
   - 缺少 RoboTwin 侧运行依赖
   - 端口冲突
   - Blackwell + CUDA 12.1 下 `pytorch3d` 不可用

## 当前限制

- V1 证明了 action-only 训练链路和回归兼容性
- V1 还没有证明任务成功率已经足够好

## Post-Train Eval 现状补充

`place_can_basket` 这次 baseline post-train eval 的失败，不是 checkpoint 推理错误，而是 RoboTwin 任务本身在 expert-check 阶段先失败了。

关键原因在 [eval_polict_client_openpi.py](/home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py) 的 `eval_policy(...)`：

1. client 在真正调用模型之前，会先跑一次：
   - `TASK_ENV.setup_demo(...)`
   - `TASK_ENV.play_once()`
2. 这一步是 RoboTwin 自己的 scripted expert-check，不依赖 LingBot 预测动作。
3. 这次报错栈里：
   - `left arm planning failed (IK Failed! Cannot find valid solution.)`
   - `target_pose cannot be None for move action.`
4. 对应的是 [place_can_basket.py](/home/zaijia001/vam/RoboTwin-lingbot/envs/place_can_basket.py) 里 `self.grasp_actor(self.basket, ...)` 在 fallback 抓篮子阶段没有拿到有效的 IK pose。

所以当前结论是：

- `place_can_basket` 这次失败发生在 policy 真正介入之前
- 它是 RoboTwin 环境或任务 planning 的问题
- 不能把它当成 post-train checkpoint 本身失效的证据
