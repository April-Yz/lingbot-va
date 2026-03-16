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

CUDA_VISIBLE_DEVICES=1 \
python script/run_lingbot_action_only_dsrl.py \
  --config examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml
```

这条命令对应的默认配置是：

- task: `click_bell`
- task config: `demo_clean`
- model path: `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`
- save root: `/home/zaijia001/vam/lingbot-va/train_out/action_only_dsrl_click_bell`

## 验证总结

### Action-Only 路径

- `use_dsrl=false`：已验证
- `use_dsrl=true`：已验证
- RoboTwin 在线单 episode：已验证
- 最近一次在线状态：`finished_no_success`

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
