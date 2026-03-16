# LingBot V0

这份文档总结当前 `/home/zaijia001/vam/lingbot-va` 在本机的实际可运行状态，重点记录 RoboTwin 评测链路、模型输入输出结构、latent 的位置，以及后续修改时最容易踩坑的点。

## 1. 当前结论

- `lingbot-va` conda 环境已经可用，当前按推理模式运行时建议使用 `attn_mode="torch"`。
- 当前 RoboTwin 评测默认接到 `/home/zaijia001/vam/RoboTwin-lingbot`。
- 当前实际测试使用的模型是作者已经做过 RoboTwin post-train 的权重：
  - `robbyant/lingbot-va-posttrain-robotwin`
  - https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin
- 如果你要看 RoboTwin post-train 之前的基础模型，应该用：
  - `robbyant/lingbot-va-base`
  - https://huggingface.co/robbyant/lingbot-va-base
- 本地 `va_robotwin_cfg.py` 当前指向的是 post-train RoboTwin 权重，而不是 base 权重。

## 2. 当前已经验证到什么程度

- 最小 smoke test 之前已经跑通过 `click_bell`。
- 这次进一步验证里，`click_alarmclock` 已经跑到 `10/10` 成功。
- `press_stapler` 在会话被异常中断前已经写出 `9/9` 成功；它没有完成原计划的第 10 次，所以这项只能算“已连续成功 9 次，但本轮未完整收尾”。
- 当前这台机器上 `curobo` 不能稳定用，RoboTwin 会回退到 `MPLib`。这会让重任务变慢，但不会阻止基础评测跑通。

## 3. 当前测试用的到底是不是 post-train 模型

是。

当前 RoboTwin eval 用的是：

- 配置文件：`wan_va/configs/va_robotwin_cfg.py`
- 当前路径：`/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`

也就是作者给 RoboTwin post-training 之后的专用模型，不是 pretrain/base 模型。

如果你想切回 RoboTwin post-train 之前的版本，应该把 `wan22_pretrained_model_name_or_path` 改到：

- `https://huggingface.co/robbyant/lingbot-va-base`

同时确认该模型目录下 `transformer/config.json` 的 `attn_mode` 仍然是推理可用的 `"torch"` 或 `"flashattn"`，不要保留训练用的 `"flex"`。

## 3.1 如果要直接用官方 RoboTwin post-train checkpoint 测 `place_can_basket`

如果你要测的是作者官方给的 RoboTwin post-train 权重，也就是：

- `/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`

那么最直接的做法是显式起一个 server，再从 RoboTwin client 连过去。

### server

```bash
conda activate lingbot-va
cd /home/zaijia001/vam/lingbot-va

MODEL_PATH=/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
CUDA_VISIBLE_DEVICES=1 \
bash evaluation/robotwin/launch_server.sh


  conda activate lingbot-va
  cd /home/zaijia001/vam/lingbot-va
  START_PORT=29058 \
  MASTER_PORT=29068 \
  MODEL_PATH=/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
  CUDA_VISIBLE_DEVICES=3 \
  bash evaluation/robotwin/launch_server.sh



```

### client

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

说明：

- 这条命令测的是“官方 RoboTwin post-train checkpoint”，不是你本地继续 post-train 出来的 `checkpoint_step_xxxx`
- 如果只是先看链路是否通，`--test_num 1` 就够
- 如果这个任务的上游 seed 过滤太脆，也可以临时改成 debug smoke 版本：
  - `--expert_check false`
  - `--step_limit_override 60`
- 输出会落在：
  - `RoboTwin-lingbot/results_official_posttrain_place_can_basket/`
  - `RoboTwin-lingbot/eval_result/place_can_basket/...`

## 4. 评测时模型的外部输入是什么

RoboTwin client 真正发给 LingBot server 的输入在 `evaluation/robotwin/eval_polict_client_openpi.py` 里组织，核心入口是 `format_obs(...)`。

当前外部输入是一个字典，主要字段如下：

```python
{
    "observation.images.cam_high": <head_camera rgb, HxWx3, uint8>,
    "observation.images.cam_left_wrist": <left_camera rgb, HxWx3, uint8>,
    "observation.images.cam_right_wrist": <right_camera rgb, HxWx3, uint8>,
    "observation.state": <joint_action["vector"]>,
    "task": <language instruction string>,
}
```

其中：

- 三路图像都来自 RoboTwin 的 `TASK_ENV.get_obs()`。
- `observation.state` 来自 RoboTwin 的 `joint_action["vector"]`，也就是 `left_jointstate + right_jointstate` 拼起来的关节状态向量。
- `task` 是自然语言指令，不是模板 ID。

在 websocket 协议层，client 会发三类请求：

1. `reset`
   - `{"reset": True, "prompt": prompt}`
   - 用来清空 server 端缓存并重新编码 prompt。
2. 普通推理
   - `{"obs": first_obs, "prompt": prompt, ...}`
   - 用当前观测生成一段 action chunk。
3. KV cache 更新
   - `{"obs": key_frame_list, "compute_kv_cache": True, "state": action, ...}`
   - 把刚执行完的一段观测和动作回灌进 transformer cache，为下一段 chunk 做条件化。

## 5. 模型内部是怎么处理这些输入的

服务端逻辑在 `wan_va/wan_va_server.py`。

### 5.1 图像到 latent

真正把图像变成 latent 的地方是 `_encode_obs(...)`。

处理顺序是：

1. 读入三路相机图像。
2. 按 RoboTwin T-shape 布局缩放：
   - `cam_high` 缩到 `256 x 320`
   - 两个 wrist camera 各缩到 `128 x 160`
3. 把像素从 `[0, 255]` 映射到 `[-1, 1]`。
4. 用 streaming VAE 编码：
   - 高位相机走 `self.streaming_vae.encode_chunk(...)`
   - 两个 wrist 相机走 `self.streaming_vae_half.encode_chunk(...)`
5. 把左右 wrist latent 和 high-camera latent 拼成 RoboTwin 的 T-shape latent。
6. 把 VAE 输出按通道一分为二，得到 `mu` 和 `logvar`。
7. 用 `vae.config.latents_mean` 和 `vae.config.latents_std` 对 `mu` 做归一化。
8. 得到 `video_latent`，这就是后面 transformer 使用的视觉条件 latent。

这里最关键的一点是：当前 websocket eval 链路里，图像不是直接喂 transformer，而是先过 VAE 变成 latent，再和 action token 一起进入 transformer。

### 5.2 文本

prompt 走 `encode_prompt(...)`：

- tokenizer 在 `tokenizer/`
- text encoder 在 `text_encoder/`
- 输出是 T5 风格的 text embeddings

这些 embedding 会同时条件化视频分支和动作分支。

### 5.3 动作内部表示

RoboTwin 配置里：

- `frame_chunk_size = 2`
- `action_dim = 30`
- `action_per_frame = 16`

所以模型内部生成的动作张量形状是：

```python
[B, 30, 2, 16, 1]
```

也就是：

- 30 个内部 action channel
- 每次推理 2 个 frame chunk
- 每个 frame chunk 里 16 个 action slot

注意：这 30 维是模型内部表示，不是最后直接执行到 RoboTwin 里的维度。

## 6. 模型的外部输出是什么

`_infer(...)` 的返回值其实是：

```python
return actions, latents
```

但 websocket 模式下，`infer(...)` 只把 `action` 返回给 client：

```python
return dict(action=action)
```

所以当前 RoboTwin eval 里，client 真正收到的是 action，不会直接收到 latent。

### 6.1 对 client 暴露的 action 形状

在 `postprocess_action(...)` 里，服务端会：

1. 把内部动作从 `[-1, 1]` 反归一化回真实数值范围。
2. 只保留 `used_action_channel_ids` 选中的通道。

当前 RoboTwin 配置对外暴露的是 16 维动作，返回形状是：

```python
[16, 2, 16]
```

16 个通道的语义顺序是：

```text
left_xyz(3),
left_quat(4),
left_gripper(1),
right_xyz(3),
right_quat(4),
right_gripper(1)
```

也就是：

```text
[left_x, left_y, left_z, left_q0, left_q1, left_q2, left_q3, left_gripper,
 right_x, right_y, right_z, right_q0, right_q1, right_q2, right_q3, right_gripper]
```

这里的 16 维是当前 RoboTwin 评测真正消费的动作接口。

## 7. client 收到 action 之后还做了什么中间处理

这一步很重要，因为 RoboTwin 里执行的不是“模型原样输出”。

在 `eval_polict_client_openpi.py` 里：

1. client 先读取 episode 初始的双臂末端位姿 `inint_eef_pose`。
2. 如果模型输出是 16 维：
   - 走 `add_init_pose(...)`
   - 把模型输出解释成“相对初始位姿的增量”
   - 和初始末端位姿合成
   - 再对 quaternion 做归一化
3. 合成后的结果才通过 `TASK_ENV.take_action(..., action_type='ee')` 执行到 RoboTwin。

所以当前链路里，模型输出的 16 维更接近“相对双臂末端位姿动作”，不是已经可直接执行的世界系绝对 pose。

### 7.1 一个容易改坏的点

这里混用了两套 quaternion 约定，后续你如果改 action 表达，需要特别小心：

- `evaluation/robotwin/geometry.py` 里的 `euler2quat(...)` 是按 `wxyz` 习惯来的。
- `scipy.spatial.transform.Rotation.from_quat(...)` 要求的是 `xyzw`。
- 当前 16 维路径主要走 quaternion 直接合成，没有经过 14 维 Euler 转换分支，所以问题没在现有 smoke/eval 中直接炸出来。

如果你后续要改成 Euler、改 quaternion 顺序、或者重新定义 action channel，先统一这件事，不然很容易得到“能跑但姿态不对”的结果。

## 8. latent 在哪里

你记得的没错，这个 VA 模型中间确实有 latent，而且不止一类。

### 8.1 观测编码 latent

位置：

- `wan_va/wan_va_server.py`
- 函数：`_encode_obs(...)`

它把三路图像编码成 `video_latent`，这是“输入观测对应的 VAE latent”。

### 8.2 模型生成的 rollout latent

位置：

- `wan_va/wan_va_server.py`
- 函数：`_infer(...)`

这里会先采样一个随机 latent tensor：

```python
latents = torch.randn(1, 48, frame_chunk_size, latent_height, latent_width, ...)
```

然后经过视频扩散去噪循环，把它变成模型预测的视频 latent。

这个 `latents` 会和 `actions` 一起从 `_infer(...)` 返回，但 websocket `infer(...)` 目前把它丢掉了，没有发回 client。

### 8.3 latent 落盘位置

虽然 websocket 没把 latent 回传给 client，但 server 端会异步保存：

- `visualization/real/<exp_name>/latents_<frame_st_id>.pt`
- `visualization/real/<exp_name>/actions_<frame_st_id>.pt`
- `visualization/real/<exp_name>/obs_data_<frame_st_id>.pt`

其中：

- `latents_*.pt` 是视频 latent
- `actions_*.pt` 是内部动作张量
- `obs_data_*.pt` 是用于更新 KV cache 的观测片段

这就是你后续要抓中间表示时最直接的入口。

如果你后面想把 latent 明确暴露给 client，最小改动点就在 `wan_va_server.py` 的 `infer(...)`，因为 `_infer(...)` 已经拿到了 `latents`。

## 9. 为什么会有 KV cache 这一段

当前 LingBot-VA 不是每一步都从零开始看整段历史。

它的做法是：

1. 首段观测先编码为 `init_latent`。
2. 生成一个 chunk 的 action 和视频 latent。
3. client 执行这段 action，并在关键帧处收集新的观测。
4. client 把这些关键帧观测和刚才的 action 再发回 server。
5. server 在 `_compute_kv_cache(...)` 里把：
   - 编码后的新观测 latent
   - `preprocess_action(...)` 处理过的动作 token
   一起写进 transformer cache。
6. 下一轮 chunk 推理基于这个 cache 继续自回归。

所以当前推理链路的核心不是单步 policy，而是“分块生成 + cache 续写”。

## 10. 你后续如果要改模型，最值得先看哪几个点

如果你的目标是继续改 RoboTwin eval，优先看：

1. `wan_va/configs/va_robotwin_cfg.py`
   - checkpoint 路径
   - camera key
   - action channel 选择
   - chunk 大小和 action slot 数

2. `wan_va/wan_va_server.py`
   - `_encode_obs(...)`
   - `preprocess_action(...)`
   - `postprocess_action(...)`
   - `_infer(...)`
   - `_compute_kv_cache(...)`

3. `evaluation/robotwin/eval_polict_client_openpi.py`
   - `format_obs(...)`
   - `add_init_pose(...)`
   - client 端 action 执行逻辑

4. `/home/zaijia001/vam/RoboTwin-lingbot/envs/_base_task.py`
   - `get_obs()`
   - `joint_action["vector"]`
   - `endpose`

## 11. 目前这份项目最重要的一句话总结

当前这套 RoboTwin eval 不是“图像直接到动作”的简单 policy，而是：

三路图像和语言先编码成 latent 与 text embedding，transformer 按 chunk 同时生成视频 latent 和动作，server 只把裁剪后的 16 维相对双臂末端动作返回给 client，client 再把它和初始末端位姿合成后执行到 RoboTwin。
