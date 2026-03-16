# LingBot-VA Post-Training Data Pipeline V1 中文版

这份文档是 [posttrain-data-v1.md](/home/zaijia001/vam/lingbot-va/agent-read/baseline/posttrain-data-v1.md) 的中文同步版，记录 RoboTwin 原始数据如何整理成 LingBot-VA post-train 可直接消费的格式，以及当前本机已经验证过的训练与 eval 方法。

## 1. 范围

当前记录的重点任务是：

- 原始任务数据：`/home/zaijia001/ssd/RoboTwin/data/place_can_basket/demo_clean_large_d435`
- 输出训练 bundle：`/home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean`

## 2. RoboTwin 原始数据结构

`collect_data.sh` 生成的数据目录里，后续 post-train 真正会用到的部分主要有：

- `data/episode*.hdf5`
  - 每帧 RGB：`head_camera`、`left_camera`、`right_camera`、`front_camera`
  - 每帧末端位姿：`endpose/left_endpose`、`endpose/right_endpose`
  - 每帧夹爪状态：`endpose/left_gripper`、`endpose/right_gripper`
  - 每帧关节空间数据：`joint_action/*`
- `instructions/episode*.json`
  - 每条 episode 已经实例化好的语言指令
- `scene_info.json`
- `_traj_data/episode*.pkl`

对 LingBot-VA post-train 来说，关键输入是：

- HDF5 里的逐帧观测和动作
- 每条 episode 对应的语言指令

当前转换脚本对原始相机尺寸是严格校验的，要求：

- `head_camera`: `480x640`
- `left_camera`: `480x640`
- `right_camera`: `480x640`

如果原始 HDF5 不是这个尺寸，脚本会直接报错，不再自动适配旧数据。

## 3. LingBot-VA 训练实际需要什么格式

发布版 post-train loader 不能直接吃 RoboTwin 原始 HDF5。它需要的是：

1. 一个本地 LeRobot 数据集
2. `meta/episodes.jsonl` 里的 `action_config`
3. 与视频布局对应的 `latents/`
4. bundle 根目录下的 `empty_emb.pt`

训练 loader 在 [lerobot_latent_dataset.py](/home/zaijia001/vam/lingbot-va/wan_va/dataset/lerobot_latent_dataset.py) 里，实际会读取：

- LeRobot parquet 里的 `action`
- `action_config.start_frame/end_frame/action_text`
- `latents/chunk-xxx/<camera>/episode_{idx}_{start}_{end}.pth`
- `empty_emb.pt`

## 4. RoboTwin 原始数据如何变成 LingBot 格式

当前使用的转换脚本是：

- [prepare_robotwin_posttrain.py](/home/zaijia001/vam/lingbot-va/script/prepare_robotwin_posttrain.py)

它做的事情是：

1. 读取每个 `episode*.hdf5`
2. 构造每帧 16 维动作或状态：
   - 左臂末端位姿 `7`
   - 左夹爪 `1`
   - 右臂末端位姿 `7`
   - 右夹爪 `1`
3. 从 `instructions/episode*.json` 里为每条 episode 选一条语言指令
4. 转成一个本地 LeRobot repo，并写出视频：
   - `head_camera -> observation.images.cam_high`
   - `left_camera -> observation.images.cam_left_wrist`
   - `right_camera -> observation.images.cam_right_wrist`
   - 视频分辨率保持 `480x640`，与 `Large_D435` 一致
5. 修改 `meta/episodes.jsonl`，为整条 episode 加上一个 full-episode `action_config`
6. 用 Wan VAE 编码视频 latent
7. 用 Wan text encoder 编码动作文本
8. 保存 `empty_emb.pt`

## 5. 一个重要对齐细节

LingBot-VA 训练 loader 和 Wan VAE 编码对帧数的约束不完全一样：

- 动作长度希望是 `4` 的倍数
- Wan VAE 视频编码希望 `num_frames = 1 (mod 4)`

当前脚本的解决方式是：

1. 先把 state、action、video pad 到最近的上取整 `4` 的倍数
2. 真正给 latent 编码时只取 `action_length - 3` 帧

例如：

- 如果原始 rollout 是 `255` 帧
- LeRobot 或 action 长度会变成 `256`
- latent video 长度会变成 `253`

这个组合可以同时满足：

- `AutoencoderKLWan.encode(...)`
- `_action_post_process(...)`

## 6. 推荐的数据采集命令

这个仓库不会去改 `/home/zaijia001/ssd/RoboTwin` 本体，但当前预期的数据采集方式是：

```bash
cd /home/zaijia001/ssd/RoboTwin
bash collect_data.sh place_can_basket demo_clean_large_d435 0
```

这里默认假设：

- 任务配置里 head 和 wrist 相机都用了 `Large_D435`

如果你不想覆盖旧数据，可以换一个新的 task config 名称，然后把下面路径一并改掉。

## 7. 当前实际处理命令

```bash
cd /home/zaijia001/vam/lingbot-va
/home/zaijia001/ssd/miniconda3/envs/lingbot-va/bin/python script/prepare_robotwin_posttrain.py \
  --raw-dir /home/zaijia001/ssd/RoboTwin/data/place_can_basket/demo_clean_large_d435 \
  --bundle-dir /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --repo-id place_can_basket_demo_clean_lerobot \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
  --instruction-key seen \
  --instruction-index 0 \
  --overwrite
```

生成后的 bundle 结构应该类似：

```text
lingbot-posttrain-demo_clean/
├── README.md
├── empty_emb.pt
└── place_can_basket_demo_clean_lerobot/
    ├── data/
    ├── meta/
    ├── videos/
    └── latents/
```

当前默认假设：

- 原始相机帧已经是 `480x640`
- 相机名仍然是 `head_camera`、`left_camera`、`right_camera`

## 8. 当前基线训练命令

### 8.1 已验证稳定可跑的命令

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

WANDB_TEAM_NAME=haoyuan-lingbot \
WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=baseline_place_can_basket \
CUDA_VISIBLE_DEVICES=2,3 \
NGPU=2 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --dataset-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-base \
  --save-root /home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean \
  --enable-wandb true \
  --attn-mode torch \
  --dataset-init-worker 1 \
  --save-interval 5000
```

这条命令对应的是当前已经验证过的 2 卡 `bs=1` baseline。

### 8.2 更高利用率的尝试

如果你想把每张卡吃得更满，最直接的想法是把每卡 batch 从 `1` 提到 `2`：

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

WANDB_TEAM_NAME=haoyuan-lingbot \
WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=baseline_place_can_basket_bs2 \
CUDA_VISIBLE_DEVICES=2,3 \
NGPU=2 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --dataset-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-base \
  --save-root /home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean_bs2 \
  --enable-wandb true \
  --attn-mode torch \
  --dataset-init-worker 1 \
  --batch-size 2 \
  --save-interval 5000
```

但这条命令在这台机器上已经验证过不稳定：

- `NGPU=2 + batch_size=2` 在 2026-03-16 的本地测试里失败了

这不代表代码永远不支持 `batch_size=2`，而是说明：

- 当前机器
- 当前模型
- 当前 FSDP 配置

这三个条件叠在一起时，每卡 `batch_size=2` 的激活显存压力已经不稳定。

### 8.3 更推荐的扩有效 batch 方式

如果你想提高有效 batch，但不想把每卡瞬时显存压力直接抬高到 `bs=2`，更稳的方式是：

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

WANDB_TEAM_NAME=haoyuan-lingbot \
WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=baseline_place_can_basket_accum2 \
CUDA_VISIBLE_DEVICES=2,3 \
NGPU=2 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --dataset-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/zaijia001/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --model-path /home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-base \
  --save-root /home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean_accum2 \
  --enable-wandb true \
  --attn-mode torch \
  --dataset-init-worker 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --save-interval 5000
```

为什么这个更稳：

- 这里的 `batch_size` 是每张卡，不是全局 batch
- `NGPU=2 + batch_size=1` 已经有全局 batch `2`
- `NGPU=2 + batch_size=2` 会直接抬高每张卡的瞬时激活显存，所以最容易先炸
- `NGPU=2 + batch_size=1 + gradient_accumulation_steps=2` 保持了稳定配置的每卡显存模式，但把有效全局 batch 提到了 `4`

简单说：

- `bs=2` 是增加瞬时压力
- accumulation 是增加延迟更新后的有效 batch

## 9. 为什么当前推荐 2 卡

原因很直接：

- 单卡 `NGPU=1` 的 smoke test 已经跑到第一次 `optimizer.step()`
- 但在 `AdamW` 初始化 optimizer state 时 OOM 了

而 2 卡的 smoke test 已经验证：

- `num_steps=1` 能完成
- WandB 能正常记录
- `checkpoint_step_1` 能成功写出

## 10. 如何评测一个 post-train checkpoint

假设你已经用上面的 baseline 命令跑出了：

- `/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean/checkpoints/checkpoint_step_5000`

那么 RoboTwin eval 不需要再额外转格式，直接把 LingBot server 指到这个 checkpoint 就行。

这里有一个关键更正：

- 像 `checkpoint_step_5000` 这样的训练 checkpoint，通常只保存了 `transformer/`
- 它不是一个包含 `vae/`、`tokenizer/`、`text_encoder/` 的完整模型目录
- 当前本地 server 已经做了兼容：
  - `transformer/` 从 `MODEL_PATH` 指定的 checkpoint 里加载
  - `vae/`、`tokenizer/`、`text_encoder/` 继续从 `va_robotwin_cfg.py` 里的基础模型目录加载

### 10.1 用 post-train checkpoint 启动 server

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

MODEL_PATH=/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean/checkpoints/checkpoint_step_5000 \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

这里要注意：

- `MODEL_PATH` 现在可以指向两类目录：
  - 一个完整模型根目录，下面同时有 `transformer/`、`vae/`、`tokenizer/`、`text_encoder/`
  - 或者一个训练 checkpoint 根目录，只要里面有 `transformer/` 也可以
- 本地 launcher 默认端口仍然是 `29056`

### 10.2 在 RoboTwin 侧跑 eval

另开一个 shell：

```bash
conda activate RoboTwin-lingbot
cd /home/zaijia001/vam/RoboTwin-lingbot

PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name place_can_basket \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_posttrain_eval_step5000 \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

这条命令做的事情是：

- RoboTwin 侧保持在 `RoboTwin-lingbot` 环境里
- 通过 websocket 调用 LingBot server
- 输出 metrics 和 rollout 视频到 `/home/zaijia001/vam/RoboTwin-lingbot/results_posttrain_eval_step5000`

### 10.3 如何扩大量评测

- smoke test：`--test_num 1`
- 小规模成功率估计：`--test_num 10`
- 更正式的单任务估计：`--test_num 100`

如果换任务：

- 改 `--task_name`
- 如果也是 `Large_D435` 采集出来的数据，`--task_config` 继续用 `demo_clean_large_d435`

## 11. 已知限制

当前流程里，数据处理阶段使用的是本地 LingBot checkpoint 路径来做 VAE 和 text encoding：

- `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`

这足够产出训练可用的 latent 和 text embedding。  
如果你想从 RoboTwin post-train 之前的干净基础模型开始训，训练命令的 `--model-path` 应该指向：

- `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-base`
