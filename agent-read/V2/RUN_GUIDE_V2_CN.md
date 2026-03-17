# LingBot Frozen-Noise-DSRL V2 运行指南

以下命令是迁移到具备完整环境的服务器后建议使用的预期命令。当前服务器未执行这些命令。

## 1. baseline 预期运行命令

### baseline eval server

```bash
conda activate lingbot-va
cd /home/e230112/vam/lingbot-va

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_PATH=/home/e230112/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

### baseline eval client

```bash
conda activate RoboTwin-lingbot
cd /home/e230112/vam/RoboTwin-lingbot

PYTHONWARNINGS=ignore::UserWarning \
SAPIEN_RT_DENOISER=none \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/e230112/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name click_bell \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --model_tag baseline \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_official_posttrain_click_bell \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

## 2. original post-train 预期运行命令

```bash
conda activate lingbot-va
cd /home/e230112/vam/lingbot-va

WANDB_TEAM_NAME=haoyuan-lingbot \
WANDB_PROJECT=lingbot \
WANDB_RUN_NAME=baseline_place_can_basket \
CUDA_VISIBLE_DEVICES=2,3 \
NGPU=2 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \
  --dataset-path /home/e230112/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean \
  --empty-emb-path /home/e230112/ssd/RoboTwin/data/place_can_basket/lingbot-posttrain-demo_clean/empty_emb.pt \
  --model-path /home/e230112/vam/lingbot-va/checkpoints/lingbot-va-base \
  --save-root /home/e230112/vam/lingbot-va/train_out/place_can_basket_demo_clean \
  --enable-wandb true \
  --attn-mode torch \
  --dataset-init-worker 1 \
  --save-interval 5000
```

## 3. V2 预期训练命令

```bash
conda activate lingbot-va
cd /home/e230112/vam/lingbot-va

python script/run_lingbot_frozen_noise_dsrl_v2.py \
  --config examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml
```

## 4. V2 eval 预期运行命令

### server

```bash
conda activate lingbot-va
cd /home/e230112/vam/lingbot-va

MODEL_PATH=/home/e230112/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh
```

### client

```bash
conda activate RoboTwin-lingbot
cd /home/e230112/vam/RoboTwin-lingbot

PYTHONWARNINGS=ignore::UserWarning \
SAPIEN_RT_DENOISER=none \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/e230112/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name click_bell \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --model_tag frozen-noise-v2 \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_frozen_noise_v2_click_bell \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29056
```

## 5. 日志目录

- V2 训练日志目录：`/home/e230112/vam/lingbot-va/train_out/frozen_noise_dsrl_v2_click_bell`
- V2 训练日志文件：`frozen_noise_dsrl_v2.log`
- baseline eval 结果目录：`/home/e230112/vam/RoboTwin-lingbot/results_official_posttrain_click_bell`
- V2 eval 结果目录：`/home/e230112/vam/RoboTwin-lingbot/results_frozen_noise_v2_click_bell`

## 6. checkpoint 目录

- baseline post-train checkpoint 根目录：`/home/e230112/vam/lingbot-va/train_out/place_can_basket_demo_clean`
- V2 预期 save root：`/home/e230112/vam/lingbot-va/train_out/frozen_noise_dsrl_v2_click_bell`

## 7. 常见错误处理建议

- 路径不一致：先检查 `robowin_root`、`lingbot_model_path`、`save_root`
- 端口冲突：检查 `launch_server.sh` 默认端口和 client `--port`
- 缺少 RoboTwin 运行时依赖：只记录并在迁移后排查，不要在当前服务器安装
- GPU 显存不足：先降低 episode / chunk / batch，再看是否需要迁移到更大显存机器
- `MODEL_PATH` 指向 checkpoint 根目录时：确认 server 路径是否仍能正确找到 `transformer/`

## 8. 明确声明

以上命令当前未在本服务器执行。本文档只记录预期运行方式，不代表已经跑通。
