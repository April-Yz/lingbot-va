# LingBot Frozen-Noise-DSRL V2 复现检查清单

以下清单只用于迁移到具备完整环境的服务器后做验证。当前服务器未执行这些检查。

## 1. baseline 正常检查

1. 确认 baseline server 能正常启动
2. 确认 baseline client 能完成单任务 smoke eval
3. 确认产出 `metrics`、`video`、`result` 文件
4. 确认 baseline 路径未因 V2 代码接入而改变默认行为

## 2. original post-train 正常检查

1. 确认 `script/run_va_posttrain.sh` 仍可启动
2. 确认 dataset 路径、empty emb 路径、base model 路径有效
3. 确认 checkpoint 目录正常写出
4. 确认原 post-train 不依赖 V2 新模块

## 3. V2 正常检查

1. 确认 `examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml` 路径有效
2. 确认 `run_lingbot_frozen_noise_dsrl_v2.py` 能正确读取配置
3. 确认 startup report 中冻结标志为：
   - `freeze_future_video_module=true`
   - `freeze_inverse_or_action_flow=true`
   - `train_noise_policy_only=true`
   - `future_latent_as_condition=true`
4. 确认 future latent 可以进入 V2 noise policy 条件分支
5. 确认 action noise 被注入 `sample_actions(..., initial_noise=...)`
6. 确认训练更新只落在 V2 新模块
7. 确认 V2 训练日志和 checkpoint 命名清晰

## 4. V2 eval 正常检查

1. 确认 server 路径指向正确 checkpoint
2. 确认 client `model_tag=frozen-noise-v2`
3. 确认结果目录与 baseline 分离
4. 确认 V2 eval 分支不覆盖 baseline eval 结果

## 5. 环境修改边界检查

1. 确认环境改动严格限制在 LingBot 与 RLInf 目标环境范围
2. 确认未污染系统 Python / CUDA / 全局 pip
3. 确认没有对无关 conda 环境做改动
4. 若后续发生环境改动，必须补写日志

## 6. 当前声明

本清单仅供迁移后验证使用，当前服务器未执行上述检查。
