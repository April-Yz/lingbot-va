# LingBot Action-Only DSRL V1 环境记录

## 已授权环境

- `lingbot-va`
- `RoboTwin-lingbot`

## 已记录环境改动

### 2026-03-16 | `lingbot-va`

- 安装：`sapien==3.0.0b1`、`mplib==0.2.1`
- 安装：`setuptools<81`
- 安装：`transforms3d==0.4.2`
- 安装：`open3d==0.18.0`、`trimesh==4.4.3`、`zarr`、`openai`、`moviepy`、`azure==4.0.0`、`azure-ai-inference`、`pyglet<2`
- 安装：`toppra`
- 安装：`lxml`
- 尝试但失败：从 upstream stable 分支编译 `pytorch3d`

## 原因

这些包是让 LingBot 侧 action-only 训练入口能够在已授权环境内导入并运行 RoboTwin 组件所必需的。

## 验证方式

- 安装 `lxml` 后，RoboTwin setup 已能返回 observation
- action-only 在线 episode 已能完整跑通

## 说明

- 这台机器目前仍然无法安装 `pytorch3d`，原因是现有 CUDA 12.1 工具链无法为 Blackwell `sm_120` 构建
- 但 RoboTwin 相机路径已经加了 CPU fallback，因此它不再是当前 RGB-based 验证链路的硬阻塞
