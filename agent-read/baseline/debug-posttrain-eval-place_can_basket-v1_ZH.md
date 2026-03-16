# Debug 记录：Place Can Basket Post-Train Eval V1

## 范围

这份文档记录 2026-03-16 本地把 LingBot post-train checkpoint 跑到 RoboTwin `place_can_basket` 时的调试过程。

## 目标

要真正跑出第一条 RoboTwin eval 结果，目标 checkpoint 和任务是：

- checkpoint：`/home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean/checkpoints/checkpoint_step_5000`
- task：`place_can_basket`
- task config：`demo_clean_large_d435`

## 问题 1：`MODEL_PATH` 不能直接当完整模型目录

### 现象

LingBot server 一启动就报错，类似：

- `.../checkpoint_step_5000/vae is not the path to a directory containing a config.json file`

### 原因

像 `checkpoint_step_5000` 这样的训练产物通常只保存了：

- `transformer/`

不会保存：

- `vae/`
- `tokenizer/`
- `text_encoder/`

原来的 server 逻辑默认把 `MODEL_PATH` 当成完整模型根目录来处理，所以会去找一个并不存在的 `vae/`。

### 解决

现在 `wan_va_server.py` 已经支持两种 `MODEL_PATH`：

- 完整模型根目录：同时有 `transformer/`、`vae/`、`tokenizer/`、`text_encoder/`
- 训练 checkpoint 根目录：只有 `transformer/`

如果检测到第二种情况，server 会：

- 从 `MODEL_PATH` 里加载 `transformer`
- 从基础模型目录继续加载 `vae/tokenizer/text_encoder`

## 问题 2：client 直接找不到 `evaluation.robotwin.*`

### 现象

client 一启动就报：

- `ModuleNotFoundError: No module named 'evaluation'`

### 原因

你是在 `RoboTwin-lingbot` 目录里执行 `lingbot-va` 里的脚本，但这个脚本自己又 import `evaluation.robotwin.*`。这时 Python 不会自动把 `lingbot-va` 根目录当成 import 根。

### 解决

现在 `eval_polict_client_openpi.py` 会主动把下面两个路径加进 `sys.path`：

- 本地 `lingbot-va` 仓库根目录
- `ROBOTWIN_ROOT`

所以文档里的绝对路径启动方式现在可以直接用。

## 问题 3：`place_can_basket` 在任务初始化或规划阶段会硬崩

### 现象 A

运行过程中可能直接报：

- `AssertionError: target_pose cannot be None for move action.`

### 原因 A

`place_can_basket` 内部会调用 `grasp_actor(...)`。如果没有找到有效抓取位姿，旧逻辑仍然会构造一个 `target_pose=None` 的 `move` 动作，把普通 planning fail 变成硬异常。

### 解决 A

现在 `RoboTwin-lingbot/envs/_base_task.py` 会把这种情况当成普通规划失败处理：

- 设置 `self.plan_success = False`
- 返回 `None, []`

这样任务会继续按失败路径处理，而不是直接把整个 eval 打崩。

### 现象 B

当跳过 expert-check 时，prompt 里的 `{A}`、`{B}`、`{a}` 可能没有被填好。

### 原因 B

`place_can_basket.setup_demo(...)` 只初始化了场景，但要等到 `play_once()` 结束后才写 `self.info["info"]`。

### 解决 B

现在 `RoboTwin-lingbot/envs/place_can_basket.py` 会在 `setup_demo(...)` 阶段就把 `self.info["info"]` 填好。

## 问题 4：debug 覆盖参数之前没有真正传进 eval 主路径

### 现象

`expert_check` 和缩短 step 限制这些覆盖参数，之前没有稳定地作用到真实 episode 执行路径。

### 原因

client 脚本虽然零散支持了一部分 override，但没有把它们一致地传到 task args 和 episode setup。

### 解决

现在 `eval_polict_client_openpi.py` 已经补齐：

- 把 `expert_check` 传入 task args
- 正确解析字符串形式的 true/false
- 把 `step_limit_override` 传入 task args
- 当 `expert_check=false` 时，直接从 `TASK_ENV.info` 里取 prompt 元数据
- 在 `setup_demo(...)` 后应用 `step_limit_override`
- 修掉 latent manifest 的路径拼接 bug

## 当前这些 warning 不是致命错误

下面这些 warning 在成功跑出 smoke 结果时也存在，它们不是这次的直接阻塞点：

- `RequestsDependencyWarning`
- Wan VAE 里 `clip_output` 被忽略
- Blackwell + CUDA 12.1 下 `curobo` import 或 JIT build 失败
- 缺少 `pytorch3d`，然后回退到 CPU farthest-point sampler

它们会影响 planner 选择或只是日志噪声，但没有阻止第一条结果生成。

## 第一条完整跑出的 eval 结果

第一次真正跑出结果时，用的是一条 debug smoke 命令，核心参数是：

- `--expert_check false`
- `--step_limit_override 60`
- `--test_num 1`
- `port 29058`

结果是：

- task：`place_can_basket`
- seed：`10000`
- success：`0/1`
- 状态：pipeline 已完整跑完并写出产物

主要产物路径：

- metrics：`/home/zaijia001/vam/RoboTwin-lingbot/results_posttrain_eval_step5000_fix4/stseed-10000/metrics/place_can_basket/res.json`
- summary：`/home/zaijia001/vam/RoboTwin-lingbot/eval_result/place_can_basket/ACT/demo_clean_large_d435/0/2026-03-16 15:11:37/_result.txt`
- rollout 视频：`/home/zaijia001/vam/RoboTwin-lingbot/results_posttrain_eval_step5000_fix4/stseed-10000/visualization/place_can_basket/`

## 当前结论

截至 2026-03-16：

- post-train checkpoint 的 `place_can_basket` eval 现在已经能真正跑出 RoboTwin 结果
- 第一条 smoke 结果虽然是失败样本 `0/1`，但已经证明 server-client-task 这条链路可以完整跑通
- 后续重点应该从“启动就崩”转成“为什么任务没有成功”
