# Eval Test Decoder V1 中文版

这份文档是 [eval-test-decoder-v1.md](/home/zaijia001/vam/lingbot-va/agent-read/baseline/eval-test-decoder-v1.md) 的中文同步版，记录 2026-03-15 在当前工作区完成的一次完整 RoboTwin 评测与 latent 解码实验。目标是把三件事一次跑通：

1. 真正测一轮任务成功率。
2. `eval_result` 里的原始评测视频不再只叫 `episode*.mp4`，而是直接带上成功结果。
3. 把 server 端保存的 `latents_*.pt` 解码成视频，也保存到同一个 `eval_result` 目录里。

## 1. 本次实验选择

- 任务：`click_alarmclock`
- 原因：这个任务在当前机器和当前 checkpoint 下已经被证明是最稳定的一批任务之一，适合先验证“成功率 + artifact 命名 + latent decoder”整条链路。

## 2. 本次使用的模型和环境

- LingBot checkpoint：
  - `robbyant/lingbot-va-posttrain-robotwin`
  - 本地路径：`/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`
- LingBot 环境：
  - conda env: `lingbot-va`
- RoboTwin 环境：
  - worktree: `/home/zaijia001/vam/RoboTwin-lingbot`
  - conda env: `RoboTwin-lingbot`

## 3. 本次实际结果

- 评测任务：`click_alarmclock`
- 评测轮数：`10`
- 成功数：`10`
- 成功率：`1.0`

指标文件：

- `/home/zaijia001/vam/RoboTwin-lingbot/eval-test-decoder/stseed-10000/metrics/click_alarmclock/res.json`

本次对应的 `eval_result` 目录：

- `/home/zaijia001/vam/RoboTwin-lingbot/eval_result/click_alarmclock/ACT/demo_clean/0/2026-03-15 15:27:35`

## 4. 现在 `eval_result` 里有什么

本次目录下现在同时有三类核心产物：

1. 原始 RoboTwin episode 视频
   - 例如：
     - `episode000-seed10000-success.mp4`
     - `episode009-seed10013-success.mp4`
2. latent 解码后的视频
   - 例如：
     - `latent-decode-episode001-seed10000-success.mp4`
     - `latent-decode-episode010-seed10013-success.mp4`
3. 两个索引文件
   - `latent_decode_manifest.json`
   - `latent_decode_results.json`

现在的命名规则已经不再是只有 `episode0.mp4` 这种匿名文件名，而是把：

- episode 序号
- seed
- success / fail

都直接放进文件名里。

## 5. manifest 和 decoder 现在怎么工作

### 5.1 manifest

在每个 episode 结束时，client 会把下面这些信息写进：

- `/home/zaijia001/vam/RoboTwin-lingbot/eval_result/click_alarmclock/ACT/demo_clean/0/2026-03-15 15:27:35/latent_decode_manifest.json`

每条记录包括：

- `episode_idx`
- `seed`
- `success`
- `prompt`
- `server_exp_save_root`
- `eval_video_path`
- `comparison_video_path`
- `latent_decode_video_path`

其中最关键的是 `server_exp_save_root`，它把每个 episode 对应到 LingBot server 侧的 latent 保存目录。

### 5.2 latent decoder

新增脚本：

- [decode_saved_latents.py](/home/zaijia001/vam/lingbot-va/evaluation/robotwin/decode_saved_latents.py)

它的工作方式是：

1. 读取 `latent_decode_manifest.json`
2. 对每个 episode 找到对应的 `server_exp_save_root`
3. 读取该目录下的 `latents_0.pt`, `latents_2.pt`, `latents_4.pt`, ...
4. 按时间顺序在时间维拼接这些 latent
5. 用同一个 checkpoint 下的 `vae/` 进行解码
6. 把视频保存回 `latent-decode-episodeXXX-seedYYYY-success.mp4`
7. 最后再写一个 `latent_decode_results.json`

### 5.3 从 eval latent 到可视化视频，一共经过几步

如果只看“视频 latent 从 eval 中间结果变成现在这个 mp4 可视化”这条链，当前实际是 6 步：

1. server 在每个 episode 开始时 `reset`
   - [wan_va_server.py](/home/zaijia001/vam/lingbot-va/wan_va/wan_va_server.py) 会创建一个新的 `exp_save_root`
   - 这个目录现在会通过 reset 响应回传给 client
2. eval 过程中每个 chunk 推理结束时保存 latent
   - `wan_va_server.py::_infer(...)` 会把当轮生成的视频 latent 保存成：
     - `latents_0.pt`
     - `latents_2.pt`
     - `latents_4.pt`
     - ...
   - 这些文件就在对应 episode 的 `server_exp_save_root` 下
3. client 在 episode 结束时把 latent 目录记进 manifest
   - [eval_polict_client_openpi.py](/home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py) 会把这个 episode 的：
     - `server_exp_save_root`
     - `success`
     - `seed`
     - `latent_decode_video_path`
   - 一起写入 `latent_decode_manifest.json`
4. 离线 decoder 读取 manifest
   - `decode_saved_latents.py` 不直接参与实时 eval
   - 它是等 eval 跑完后，再读取 `latent_decode_manifest.json`
5. decoder 把多个 `latents_*.pt` 沿时间维拼起来，并喂给 VAE
   - 它先按 `latents_0.pt`, `latents_2.pt`, `latents_4.pt` 的顺序排序
   - 用 `torch.cat(..., dim=2)` 在时间维拼接
   - 再用 checkpoint 里的 `vae/` 做两步处理：
     - 按 `latents_mean / latents_std` 反归一化
     - 调 `vae.decode(...)` 还原成视频帧
6. decoder 把帧序列导出成 mp4
   - 最终保存成：
     - `latent-decode-episode001-seed10000-success.mp4`
     - 这类文件
   - 同时把结果汇总写进 `latent_decode_results.json`

所以如果你后面要改这条链，最关键的 3 个落点就是：

- `wan_va_server.py::_infer(...)`
  - 决定 latent 何时保存、保存成什么张量
- `eval_polict_client_openpi.py`
  - 决定 episode 和 latent 目录怎么建立映射
- `decode_saved_latents.py`
  - 决定 latent 如何拼接、如何 decode、如何导出视频

## 6. 这次为了打通链路做了哪些代码改动

### 6.1 server 侧

文件：

- [wan_va_server.py](/home/zaijia001/vam/lingbot-va/wan_va/wan_va_server.py)

改动：

- `reset` 响应现在会返回：
  - `exp_name`
  - `exp_save_root`
- `exp_save_root` 现在返回绝对路径，避免 decoder 后续拿错相对目录。

### 6.2 client 侧

文件：

- [eval_polict_client_openpi.py](/home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py)

改动：

- 新增 success/fail 命名逻辑，不再保留匿名 `episode{n}.mp4`
- 每个 episode 会写入 manifest 记录
- `comparison_video_path`、`eval_video_path`、`latent_decode_video_path` 现在都按绝对路径或稳定基准记录
- `_result.txt` 里也会写 manifest 路径

### 6.3 离线 decoder

文件：

- [decode_saved_latents.py](/home/zaijia001/vam/lingbot-va/evaluation/robotwin/decode_saved_latents.py)

作用：

- 把 server 端保存的 `latents_*.pt` 离线解码成 mp4
- 不侵入实时 websocket eval 主循环

## 7. 本次实验里看到的运行时现象

- `curobo` 仍然不可用，当前 RoboTwin 继续回退到 `MPLib`
- 运行中仍有大量：
  - `OIDN Error`
  - `TOPP / FailUncontrollable`
- 这些 warning 会拖慢速度，也会让日志很脏，但这次没有阻止 `click_alarmclock` 的 10 次评测全部成功

## 8. 这次实验的一个实际坑

第一次跑 decoder 时，manifest 里的 `latent_decode_video_path` 是相对 `RoboTwin-lingbot` 根目录写的，而 decoder 是在 `lingbot-va` 根目录执行的，所以解码视频最开始被写到了错误的：

- `/home/zaijia001/vam/lingbot-va/eval_result/...`

后来已经修正：

- server 返回绝对 `exp_save_root`
- client 记录绝对路径或可确定基准的 artifact 路径
- 已经把本次 10 个 latent decode 视频移动回正确目录：
  - `/home/zaijia001/vam/RoboTwin-lingbot/eval_result/click_alarmclock/ACT/demo_clean/0/2026-03-15 15:27:35`

## 9. 本次实验的最终结论

当前这套 `lingbot-va + RoboTwin-lingbot` 工作流已经不仅能跑 RoboTwin eval，而且已经能在一次完整评测中同时产出：

- 成功率统计
- 带 success/fail 标记的原始 eval 视频
- 每个 episode 对应的 latent manifest
- 每个 episode 对应的 latent 解码视频

对后续继续改模型、改 latent 输出、或者想做“模型想象视频 vs 真实执行视频”的并排分析，这一版链路已经够用了。
