# LingBot Frozen-Noise-DSRL V2 变更日志

## 变更范围

本次变更只覆盖 V2 静态代码接入与 V2 文档整理，不包含真实训练、真实评测、环境安装、环境升级或环境测试。

## 新增文件

### `wan_va/frozen_noise_dsrl/__init__.py`

- 目的：提供 V2 模块导出入口
- 影响范围：仅 V2 训练入口
- 是否影响原流程：否
- 状态：仅静态接入，未运行验证

### `wan_va/frozen_noise_dsrl/modules.py`

- 目的：新增 future-latent-conditioned noise policy 相关模块和 critic 包装
- 影响范围：仅 V2
- 是否影响原流程：否
- 状态：仅静态接入，未运行验证

### `wan_va/frozen_noise_dsrl/policy.py`

- 目的：新增冻结 LingBot backbone 的 V2 policy / trainer
- 影响范围：仅 V2
- 是否影响原流程：否
- 状态：仅静态接入，未运行验证

### `script/run_lingbot_frozen_noise_dsrl_v2.py`

- 目的：提供 V2 独立训练入口
- 影响范围：仅 V2
- 是否影响原流程：否
- 状态：仅静态接入，未运行验证

### `script/run_lingbot_frozen_noise_eval_v2.sh`

- 目的：提供 V2 独立静态评测入口说明
- 影响范围：仅 V2
- 是否影响原流程：否
- 状态：仅静态接入，未运行验证

### `examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml`

- 目的：提供 V2 独立配置
- 影响范围：仅 V2
- 是否影响原流程：否
- 状态：仅静态接入，未运行验证

### `agent-read/V2/IMPLEMENTATION_REPORT_V2_CN.md`

- 目的：记录 V2 中文实现说明
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档

### `agent-read/V2/IMPLEMENTATION_REPORT_V2_EN.md`

- 目的：记录 V2 英文实现说明
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档

### `agent-read/V2/CHANGELOG_V2_CN.md`

- 目的：记录 V2 中文文件级变更
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档

### `agent-read/V2/CHANGELOG_V2_EN.md`

- 目的：记录 V2 英文文件级变更
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档

### `agent-read/V2/ENVIRONMENT_RECORD_V2_CN.md`

- 目的：记录 V2 中文环境范围和未改动状态
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档

### `agent-read/V2/ENVIRONMENT_RECORD_V2_EN.md`

- 目的：记录 V2 英文环境范围和未改动状态
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档

### `agent-read/V2/RUN_GUIDE_V2_CN.md`

- 目的：记录 V2 中文预期运行命令
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档，命令未执行

### `agent-read/V2/RUN_GUIDE_V2_EN.md`

- 目的：记录 V2 英文预期运行命令
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档，命令未执行

### `agent-read/V2/REPRO_CHECKLIST_V2_CN.md`

- 目的：记录迁移后验证清单中文版本
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档，未执行

### `agent-read/V2/REPRO_CHECKLIST_V2_EN.md`

- 目的：记录迁移后验证清单英文版本
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档，未执行

### `agent-read/V2/ENV_PREDICTION_V2_CN.md`

- 目的：记录环境风险预测中文版本
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档

### `agent-read/V2/ENV_PREDICTION_V2_EN.md`

- 目的：记录环境风险预测英文版本
- 影响范围：文档
- 是否影响原流程：否
- 状态：静态文档

## 修改文件

### `agent-read/README.md`

- 目的：把 V2 静态接入结果写入总览
- 影响范围：项目说明
- 是否影响原流程：否
- 状态：文档更新，未运行验证

### `agent-read/CHANGELOG.md`

- 目的：把 2026-03-18 的 V2 静态改动写入仓库总日志
- 影响范围：项目日志
- 是否影响原流程：否
- 状态：文档更新，未运行验证

## 删除文件

- 无

## 兼容性说明

- 原 baseline 流程未替换
- 原 original post-train 流程未替换
- 原 V1 流程未替换
- 当前结果属于静态接入版本，尚未在真实环境执行验证
