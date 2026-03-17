# LingBot Frozen-Noise-DSRL V2 环境问题预测

## 1. 当前为什么不能做真实运行验证

- 当前服务器未提供完整可运行环境保证
- 本次任务明确禁止运行训练、评测和环境测试
- 本次任务明确禁止安装、升级或降级依赖

## 2. 哪些部分属于静态代码修改

- 新增 `wan_va/frozen_noise_dsrl/` V2 模块
- 新增 V2 配置
- 新增 V2 训练入口
- 新增 V2 静态评测入口
- 新增 V2 中英文文档
- 更新仓库 README 和 CHANGELOG

## 3. 哪些部分必须迁移后再验证

- LingBot checkpoint 实际加载是否正常
- RoboTwin 环境初始化是否正常
- future latent 实际 shape 是否与 V2 配置匹配
- V2 noise policy 的实际训练更新是否稳定
- V2 eval 路径是否能产出完整结果

## 4. 预测最可能出现的环境问题

### 问题 A：RoboTwin 运行时依赖缺失或版本不兼容

- 影响模块：
  - `wan_va.action_only_dsrl.robotwin_env`
  - `script/run_lingbot_frozen_noise_dsrl_v2.py`
  - eval client / server
- 典型影响：
  - 环境启动失败
  - task reset 失败
  - 渲染或物理模块导入失败
- 建议排查：
  1. 先检查 `sapien`、`mplib`、`open3d`、`trimesh`、`toppra`
  2. 再检查 RoboTwin task config 与相机配置
  3. 保持问题记录，不要先大范围升级环境

### 问题 B：GPU / CUDA / PyTorch 组合不兼容

- 影响模块：
  - LingBot server
  - future latent 采样
  - action sampling
- 典型影响：
  - 启动时报 CUDA 错误
  - 显存不足
  - kernel / attention backend 失败
- 建议排查：
  1. 先确认 `attn_mode='torch'`
  2. 再检查 CUDA 版本和 PyTorch wheel 是否匹配
  3. 若仍失败，再检查 GPU 显存是否足够

### 问题 C：checkpoint 路径布局与 server 预期不一致

- 影响模块：
  - `launch_server.sh`
  - `VA_Server`
- 典型影响：
  - `config.json` 找不到
  - `transformer/` 子目录无法识别
- 建议排查：
  1. 确认 `MODEL_PATH` 指向完整模型根目录还是训练 checkpoint 根目录
  2. 确认 `transformer/` 是否存在
  3. 确认 base model 根目录仍可提供 `vae/`、`tokenizer/`、`text_encoder/`

### 问题 D：future latent 与 action noise 维度不匹配

- 影响模块：
  - `LingBotFrozenNoiseV2Policy`
  - V2 actor / critic
- 典型影响：
  - `dsrl_action_noise_dim` 校验失败
  - future summary 编码失败
- 建议排查：
  1. 检查 `VA_Server._action_noise_shape()`
  2. 检查 `dsrl_action_noise_dim`
  3. 检查 `dsrl_future_summary_input_dim`
  4. 确认所用 checkpoint 与 config 对应

### 问题 E：WandB 或日志目录不可写

- 影响模块：
  - `run_lingbot_frozen_noise_dsrl_v2.py`
- 典型影响：
  - 启动中断
  - 日志写入失败
- 建议排查：
  1. 先禁用 WandB 或改为离线模式
  2. 确认 `save_root` 可写
  3. 确认训练机磁盘空间足够

## 5. 哪些结论属于“已完成代码修改但未运行验证”

- V2 代码结构已完成
- V2 配置已完成
- V2 文档已完成
- V2 future latent conditioning 接口已完成
- V2 action noise 注入路径已完成

以上都不代表已经跑通。

## 6. 明确禁止性表述

当前不得表述为：

- 已跑通
- 已验证可运行
- 训练通过
- 评测通过

当前只能表述为：

- 已完成静态代码接入
- 尚未在真实环境执行验证
- 需要迁移后再做运行确认
