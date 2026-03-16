# Debug：`place_can_basket` 四元数顺序临时测试

## 目标

这份 debug 记录的是一个只改 eval、不改训练数据的临时实验，目标 checkpoint 是：

- `checkpoint_step_10000`

实验目的很单纯：

- checkpoint 不变
- 任务和 seed 不变
- 只改 eval client 在把动作发给 RoboTwin 前的四元数组合方式

这样可以验证：

- “看起来朝向明显不对”这个现象，是否能仅靠 eval 端把 `xyzw` / `wxyz` 调整过来就立刻改善

## 临时开关

RoboTwin eval client 现在支持：

- `--quat_order_mode legacy_xyzw`
- `--quat_order_mode robowin_wxyz`

对应文件：

- `/home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py`

含义：

- `legacy_xyzw`：保持旧逻辑不变
- `robowin_wxyz`：把 RoboTwin 进来的四元数按 `wxyz` 解释，只在 `scipy` 计算时临时转成 `xyzw`，组合完成后再转回 `wxyz` 发给 RoboTwin

这只是一个 eval 侧的临时 debug 开关，不会改变训练 bundle，也不会改变训练数据加载逻辑。

## 测试设置

Server：

- 模型：`/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean/checkpoints/checkpoint_step_10000`
- 端口：`29060`

两次 client 共用的设置：

- task：`place_can_basket`
- task config：`demo_clean_large_d435`
- seed：`0`，对应起始 seed `10000`
- `expert_check=false`
- `step_limit_override=60`
- `test_num=1`

这次为了避开渲染/显存噪声，还固定用了：

- `LINGBOT_SKIP_RENDER_TEST=1`
- `SAPIEN_RT_DENOISER=none`
- RoboTwin client 固定到 `CUDA_VISIBLE_DEVICES=3`

这里有一个中间插曲：

- 第一次没固定 client 显卡时，`curobo` 导入阶段先 OOM
- 随后相机渲染又报了 `RuntimeError: cannot create buffer`

所以后面的有效对照都是在 `CUDA_VISIBLE_DEVICES=3` 下完成的。

## 实际命令

### 旧逻辑

```bash
conda activate RoboTwin-lingbot
cd /home/zaijia001/vam/RoboTwin-lingbot

CUDA_VISIBLE_DEVICES=3 \
PYTHONWARNINGS=ignore::UserWarning \
LINGBOT_SKIP_RENDER_TEST=1 \
SAPIEN_RT_DENOISER=none \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name place_can_basket \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --model_tag ckpt10000-legacyquat \
  --quat_order_mode legacy_xyzw \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_posttrain_eval_step10000_legacyquat \
  --expert_check false \
  --step_limit_override 60 \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29060
```

### 临时 `wxyz` 测试逻辑

```bash
conda activate RoboTwin-lingbot
cd /home/zaijia001/vam/RoboTwin-lingbot

CUDA_VISIBLE_DEVICES=3 \
PYTHONWARNINGS=ignore::UserWarning \
LINGBOT_SKIP_RENDER_TEST=1 \
SAPIEN_RT_DENOISER=none \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python /home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name place_can_basket \
  --task_config demo_clean_large_d435 \
  --train_config_name 0 \
  --model_name 0 \
  --ckpt_setting 0 \
  --model_tag ckpt10000-wxyzquat \
  --quat_order_mode robowin_wxyz \
  --seed 0 \
  --policy_name ACT \
  --save_root ./results_posttrain_eval_step10000_wxyzquat \
  --expert_check false \
  --step_limit_override 60 \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num 1 \
  --port 29060
```

## 结果

### 旧逻辑结果

- 结果文件：
  - `/home/zaijia001/vam/RoboTwin-lingbot/eval_result/place_can_basket/ACT/demo_clean_large_d435/0/ckpt10000-legacyquat/2026-03-16 21:50:11/_result.txt`
- metrics：
  - `/home/zaijia001/vam/RoboTwin-lingbot/results_posttrain_eval_step10000_legacyquat/stseed-10000/metrics/place_can_basket/res.json`
- 结果：
  - `0/1`

### `robowin_wxyz` 结果

- 结果文件：
  - `/home/zaijia001/vam/RoboTwin-lingbot/eval_result/place_can_basket/ACT/demo_clean_large_d435/0/ckpt10000-wxyzquat/2026-03-16 21:54:01/_result.txt`
- metrics：
  - `/home/zaijia001/vam/RoboTwin-lingbot/results_posttrain_eval_step10000_wxyzquat/stseed-10000/metrics/place_can_basket/res.json`
- 结果：
  - `0/1`

## 解释

这次临时测试没有表现出“只改 eval 端四元数组合方式就能立刻把结果救回来”。

也就是说：

- 单独在 eval 里切换到 `robowin_wxyz`，并没有让这个 `checkpoint_step_10000` 在这个 seed 下从失败变成功

但这**不代表**前面的数据处理怀疑被排除了。

原因是：

1. 这个 checkpoint 本身就是在旧数据处理链路下训练出来的
2. 如果训练目标里的旋转本来就有顺序问题，只改 eval 不会自动把模型学到的内容纠正回来
3. 所以这里出现 `0/1` 对 `0/1`，仍然完全可能说明“训练数据链路里确实有四元数顺序 bug”

## 当前更合理的判断

高置信度结论：

- RoboTwin 原始数据到 LingBot 的处理链仍然有真实的 quaternion convention 风险

这次 smoke test 新增的结论是：

- 这个问题大概率不是只靠 eval 端临时补丁就能修好的

更有价值的下一步应该是：

1. 先把原始数据转换和训练数据加载里的四元数约定统一
2. 重新生成一小批 bundle
3. 训一个小的 smoke checkpoint
4. 再和当前 `checkpoint_step_10000` 做同 seed 对照
