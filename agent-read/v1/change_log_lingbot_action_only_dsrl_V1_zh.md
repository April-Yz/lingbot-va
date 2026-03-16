# LingBot Action-Only DSRL V1 变更记录

## 范围

这是 action-only DSRL 基线的 V1 中文变更记录。

## 已修改文件

- `wan_va/wan_va_server.py`
  - 新增 `sample_future_latents(...)`
  - 新增 `sample_actions(..., initial_noise=...)`
  - 在未传 steering 时保留原始随机噪声路径

- `script/run_lingbot_action_only_dsrl.py`
  - 新增独立的单任务训练入口
  - 新增 mock validation 日志
  - 新增 blocked / finished_no_success / finished_success 状态输出

- `/home/zaijia001/vam/RoboTwin-lingbot/envs/camera/camera.py`
  - 将旧的 `missing pytorch3d -> exit()` fallback 改为 CPU farthest-point sampler

## 已新增文件

- `wan_va/action_only_dsrl/__init__.py`
- `wan_va/action_only_dsrl/modules.py`
- `wan_va/action_only_dsrl/policy.py`
- `wan_va/action_only_dsrl/robotwin_env.py`
- `examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml`

## 向后兼容性

- `use_dsrl=false` 时保留原始 LingBot 动作生成路径
- 原始 LingBot eval 代码路径保留
- 原始 LingBot post-train 接口层面保持兼容；回归 smoke 验证结果单独记录在 implementation report 中
