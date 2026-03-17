# LingBot Frozen-Noise-DSRL V2 静态实现报告

## 1. V2 核心实验目标

V2 的目标是把 LingBot 作为冻结推理 backbone 使用：

- 用 `VA_Server.sample_future_latents(...)` 生成 future latent
- 用 future latent 作为条件输入新的 noise policy
- 用 noise policy 输出的 `action noise` 替换 LingBot action path 里的默认随机 noise
- 训练范围只包含新的 noise policy 和 critic，不更新 LingBot backbone

本次交付属于静态代码接入版本，尚未在真实环境执行验证。

## 2. LingBot 在 V2 中的角色

LingBot 在 V2 中承担两个冻结角色：

- 冻结的 future latent proposer
- 冻结的 action decoder / inverse-action path

本次实现没有重构 LingBot 主体网络结构。V2 直接复用 [wan_va_server.py](/home/e230112/vam/lingbot-va/wan_va/wan_va_server.py) 已有的最小接口：

- `sample_future_latents(...)`
- `sample_actions(..., initial_noise=...)`
- `_reset(...)`
- `_compute_kv_cache(...)`

## 3. RLInf 在 V2 中的角色

本次没有直接把 `/home/e230112/vam/RLinf` 作为运行时依赖接入到 V2 入口中。V2 借鉴的是 RLInf 的训练组织思想，而不是改写 LingBot 主体：

- `embodied_sac` 命名和 actor-critic 组织方式
- replay buffer + target critic + alpha temperature 的更新结构
- actor / critic 分离优化器
- soft target update

这些机制继续以本地轻量实现的方式保留在 `lingbot-va` 仓库内部，避免把 LingBot 改写成 RLInf 原生模型。

## 4. V2 借鉴了 V1 的哪些部分

V2 主要借鉴了 V1 的以下内容：

- 独立实验包和独立入口的组织方式
- 通过 `VA_Server` 复用 LingBot frozen forward 的工程方式
- 与 baseline / original post-train 并行存在的独立 YAML 配置
- 训练脚本、日志、mock 验证、文档分层的写法
- 本地 RLinf-style 紧凑 actor / critic 实现思路

V2 与 V1 的关键差异是：

- V1 侧重 action-only steering
- V2 明确强调 future latent conditioned noise policy
- V2 文档中把 future latent 的生成、detach 和 action noise 替换点写成显式合同
- V2 配置显式包含 `freeze_future_video_module`、`freeze_inverse_or_action_flow`、`train_noise_policy_only`、`future_latent_as_condition`、`use_rlinf_or_v1_style_training`

## 5. future latent 在哪里生成

future latent 在 [wan_va_server.py](/home/e230112/vam/lingbot-va/wan_va/wan_va_server.py) 的 `VA_Server.sample_future_latents(...)` 中生成。

V2 在 [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py) 的 `LingBotFrozenNoiseV2Policy.act(...)` 中调用该接口。

## 6. future latent 在哪里 detach / stop-grad

future latent 的 stop-grad 发生在 [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py) 的 `build_step_batch(...)`：

- `future_condition = future_latents.detach() if self.detach_future_latent else future_latents`

随后只对 detach 后的 future latent 做 summary pooling，并送入新的 future encoder。

## 7. noise policy 的输入和输出

输入：

- 图像 history 编码
- 状态向量编码
- future latent summary 编码

输出：

- 与 LingBot action diffusion 初始 noise 同形状的扁平 action noise code

具体模块位于：

- [modules.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/modules.py)
- [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py)

## 8. 原始 action noise 在哪里被替换

原始随机 noise 的替换点在 [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py) 的 `act(...)`：

1. `noise_policy.sample(...)` 生成 `steer_noise_flat`
2. `self._reshape_noise(...)` 转回 LingBot action noise tensor
3. 调用 `self.server.sample_actions(..., initial_noise=steer_noise)`

因此，V2 没有改写 LingBot action decoder 主体，只是把默认随机初始 noise 改成了可学习 noise policy 输出。

## 9. 为什么 LingBot backbone 要冻结，以及如何冻结

冻结原因：

- 保持 V2 的定位是外挂 noise controller，而不是修改 LingBot 主体
- 避免未来 latent proposer 与 action decoder 一起漂移，破坏 baseline 路径
- 使训练更新范围集中在 V2 新模块

冻结实现位于 [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py) 的 `_freeze_lingbot(...)`：

- `self.server.transformer.requires_grad_(False)`
- `self.server.vae.requires_grad_(False)`
- `self.server.text_encoder.requires_grad_(False)`

当前 LingBot 的 transformer 同时承担 future / action 两侧逻辑，所以本次静态实现采用“整套已加载 backbone 全冻结”，这比只冻结子分支更严格，但不改变原默认行为。

## 10. 实际训练时哪些参数会更新

训练更新范围仅限 V2 新模块：

- `actor_image_encoder`
- `actor_state_encoder`
- `actor_future_encoder`
- `noise_policy`
- `critic_image_encoder`
- `critic_state_encoder`
- `critic_future_encoder`
- `q_head`
- `alpha_temperature`

LingBot 的 `transformer`、`vae`、`text_encoder` 不参与更新。

## 11. 是否直接集成了 RLInf

没有直接把 RLInf 代码作为运行时依赖接入。

借鉴内容：

- SAC/DSRL 风格的训练组织
- `adv_type: embodied_sac`
- `loss_type: embodied_sac`
- replay buffer
- actor / critic / alpha 的分离优化
- target critic 软更新

借鉴 V1 的内容：

- 目录组织
- 本地 compact 模块实现方式
- standalone runner + config + docs 的交付风格

这样做更合理，因为当前任务要求尽量少改 LingBot 主体，并且当前服务器不适合做更重的跨仓库运行时耦合验证。

## 12. 原有 LingBot eval / post-train 为什么还能正常运行

V2 采用加法式接入：

- 新包：`wan_va/frozen_noise_dsrl/`
- 新训练入口：`script/run_lingbot_frozen_noise_dsrl_v2.py`
- 新静态评测入口：`script/run_lingbot_frozen_noise_eval_v2.sh`
- 新配置：`examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml`

原有 baseline、post-train、V1 入口文件没有被替换，也没有修改 LingBot 默认无 `initial_noise` 时的行为。

## 13. 具体修改了哪些文件

新增：

- [__init__.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/__init__.py)
- [modules.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/modules.py)
- [policy.py](/home/e230112/vam/lingbot-va/wan_va/frozen_noise_dsrl/policy.py)
- [run_lingbot_frozen_noise_dsrl_v2.py](/home/e230112/vam/lingbot-va/script/run_lingbot_frozen_noise_dsrl_v2.py)
- [run_lingbot_frozen_noise_eval_v2.sh](/home/e230112/vam/lingbot-va/script/run_lingbot_frozen_noise_eval_v2.sh)
- [robotwin_lingbot_frozen_noise_dsrl_v2.yaml](/home/e230112/vam/lingbot-va/examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml)
- `agent-read/V2/` 下全部 V2 中英文文档

修改：

- [README.md](/home/e230112/vam/lingbot-va/agent-read/README.md)
- [CHANGELOG.md](/home/e230112/vam/lingbot-va/agent-read/CHANGELOG.md)

## 14. 当前哪些内容仅完成静态修改，尚未运行验证

以下内容当前都只完成了静态接入：

- V2 trainer 入口
- V2 future-latent-conditioned noise policy
- V2 critic / alpha 更新逻辑
- V2 YAML 配置
- V2 eval 静态入口脚本
- V2 运行指南与复现清单
- 环境风险预测

当前没有执行：

- 训练
- 评测
- 环境测试
- 依赖安装
- 任何运行时联调
