# LingBot 命令索引

这份文档只做一件事：把当前工作区里最常用的数据处理、训练和 eval 命令集中到一个地方，方便快速查找。

## 1. RoboTwin 原始数据转 Post-Train Bundle

```bash
conda activate lingbot-va
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

对应详细说明：

- `agent-read/baseline/posttrain-data-v1_ZH.md`

## 2. Baseline Post-Train 训练

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

对应详细说明：

- `agent-read/baseline/posttrain-data-v1_ZH.md`

## 3. 评测官方 RoboTwin Post-Train Checkpoint

Server：

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_PATH=/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=2 \
bash evaluation/robotwin/launch_server.sh
```

Client：

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
  --save_root ./results_official_posttrain_place_can_basket \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

对应详细说明：

- `agent-read/baseline/lingbot-v0.md`

### 3.1 轻任务的快速对照命令

如果你想先快速确认“官方 checkpoint 路径 + websocket eval 链路”还健康，建议先跑轻一点的任务：

```bash
conda activate RoboTwin-lingbot
cd /home/zaijia001/vam/RoboTwin-lingbot

PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name click_bell \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_official_posttrain_click_bell \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

这条命令适合先做 sanity check，再决定要不要继续花时间压 `place_can_basket` 这种重任务。

### 3.2 为什么重任务会明显更慢

在当前工作区里，`place_can_basket` 比 `click_bell` 慢很多，主要原因是：

- 任务本身是更重的双臂任务
- `expert_check=true` 时，会先跑任务自己的 expert seed 过滤路径
- 当前本机 `curobo` 还不可用，RoboTwin 会回退到 `MPLib`

当前 `RoboTwin-lingbot` 环境本身的运行时其实是：

- PyTorch：`2.9.0+cu128`
- torch 的 CUDA runtime tag：`12.8`

但本地 Curobo 真正拿来 JIT 编译扩展时调用的 `nvcc` 仍然是：

- `/home/zaijia001/ssd/cuda-12.1/bin/nvcc`

所以现在的问题不是“conda 环境完全没有 CUDA 12.8”。真正的问题是：

- Curobo 的扩展编译链路还在走 CUDA 12.1 toolchain
- 机器是 Blackwell
- PyTorch runtime 又更高

这会同时导致：

- 旧 `.so` 的 ABI 不匹配
- 本地 JIT rebuild 失败

因此当前重任务只能走 `MPLib` fallback，看起来就会慢很多。

### 3.3 CUDA 编号为什么会显示成 GPU 0

如果你这样启动 server：

```bash
CUDA_VISIBLE_DEVICES=2 bash evaluation/robotwin/launch_server.sh
```

那么 PyTorch 会把“物理 GPU 2”重新映射成这个进程自己看到的本地设备列表。

所以在 server 进程内部：

- 物理 GPU `2`
- 会变成进程里的本地 `cuda:0`

这就是为什么你明明指定了 `CUDA_VISIBLE_DEVICES=2`，但 OOM 日志里仍然会写 `GPU 0`。

这并不表示 client 把 server 拉回了物理 `GPU 0`。

client 端的 `--port` 只决定它连哪个 websocket server，不决定 server 用哪张卡。

所以如果你看到“GPU 0 只剩几十 MiB”的 OOM，正确理解应该是：

- 你选中的那张物理卡已经很满了
- 只是在进程内部，它被编号成了本地 `GPU 0`

## 4. 评测本地继续训练出来的 Post-Train Checkpoint

Server：

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_PATH=/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean/checkpoints/checkpoint_step_5000 \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

Client：

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

Debug smoke 版本：

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
  --save_root ./results_posttrain_eval_step5000_fix4 \
  --expert_check false \
  --step_limit_override 60 \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

对应详细说明：

- `agent-read/baseline/posttrain-data-v1_ZH.md`
- `agent-read/baseline/debug-posttrain-eval-place_can_basket-v1_ZH.md`

## 5. 并行评测时如何改端口

如果你想同时评测两组不同 checkpoint 或不同任务，每个 server 都应该用不同的：

- websocket 端口
- torch distributed 的 master port

例如 server A：

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
START_PORT=29058 \
MASTER_PORT=29068 \
MODEL_PATH=/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

例如 server B：

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
START_PORT=29059 \
MASTER_PORT=29069 \
MODEL_PATH=/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean/checkpoints/checkpoint_step_5000 \
CUDA_VISIBLE_DEVICES=2 \
bash evaluation/robotwin/launch_server.sh
```

对应 client 端要用匹配的 `--port`：

- server A -> `--port 29058`
- server B -> `--port 29059`

## 6. Action-Only V1 训练

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

对应详细说明：

- `agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_zh.md`
- `agent-read/v1/task_spec_lingbot_action_only_dsrl_V1_en.md`
