给 Codex 的明确需求文档
0. 任务背景

当前已经具备以下前提：

已配置好 RLinf 中 DSRL for Pi0 的运行环境

已配置好 LingBot 的运行环境

当前目标是在 LingBot + RLinf + RoboTwin 上实现一个 action-only DSRL baseline

这是一个最小可行实现，不是 full method，不做 future latent guidance，不做 dual-flow outcome alignment，不做大规模结构重写

本次任务只允许实现：

LingBot-VA 上的 action-only DSRL：冻结 LingBot 主干与 future branch，仅通过 RLinf 的 SAC steering actor 在 action sampling 入口注入 latent noise / steering embedding 来影响动作生成。

1. 严格目标
1.1 本次必须完成的目标

实现一个新的训练/推理路径，使得：

使用 RLinf 的 embodied_sac / DSRL 风格训练逻辑

使用 LingBot 作为基础策略

使用 RoboTwin 作为环境

冻结 LingBot 主体

冻结 LingBot future branch

冻结 LingBot action decoder 主体

新增一个轻量 steering actor

新增 / 复用一个 critic

steering actor 输出 steer_noise 或 steering_embedding

将其注入 LingBot 的 action sampling 入口

仅训练 steering actor / critic，不训练 LingBot 主干

1.2 本次明确不做的事

严禁实现以下内容：

不做 full method

不做 future latent critic

不做 future branch 的 value alignment

不做 dual-flow training

不做 LingBot backbone finetune

不做 action decoder 主体 finetune

不改 RoboTwin 环境逻辑

不重写 RLinf SAC 算法主体

不引入 PPO / GRPO / 其他新算法

不进行“顺手优化”式大规模代码清理或重构

如果某项改动不属于“为了实现 LingBot action-only DSRL 所必需”，则不要改。

2. 允许修改的环境范围
2.1 只能在以下 conda 环境中工作

只允许在下面两个环境内安装、修改、测试和记录：

RoboTwin-lingbot

这是 RoboTwin 相关环境

所有与 RoboTwin、LingBot 运行联调相关的操作，只能在这个环境中进行

LingBot / RLinf 对应环境

即当前已经配置好的 LingBot 与 RLinf 所使用的环境

所有与 RLinf DSRL 接口、LingBot 包装、训练脚本有关的改动，只能在这个环境中进行

2.2 禁止的环境行为

严禁：

新建额外 conda 环境

在其他环境中试跑并不记录

修改系统 Python 环境

全局 pip install 后不记录

在用户未说明的第三个环境中临时修 bug

不说明环境差异地混用多个环境

2.3 环境留痕要求

每次涉及环境变更，必须记录：

环境名称

执行时间

安装或修改了什么

修改原因

安装命令

安装结果

是否成功验证

必须输出一个环境记录文件，例如：

agent-read/v1/env_change_log.md

3. 代码改动边界
3.1 允许修改的代码范围

只允许修改与以下目标直接相关的代码：

A. RLinf 侧

DSRL policy wrapper 接口

LingBot policy wrapper

配置文件

launch / run 脚本

必要的模型注册逻辑

必要的日志输出逻辑

B. LingBot 侧

action sampling 入口暴露

为 action sampling 增加 initial_noise 或 steering_embedding 外部接口

允许从 wrapper 传入 steering 信号

不改变默认推理行为前提下的最小改造

C. 文档与记录

改动说明

环境说明

运行说明

复现说明

文件级变更记录

3.2 禁止修改的范围

严禁无关修改，例如：

修改 RoboTwin task 定义

修改 RoboTwin reward 机制

修改与本任务无关的 RLinf 算法实现

改 LingBot 的训练范式

改数据集预处理逻辑

做大范围 formatter / import 排序 / 重命名清理

改 unrelated README 内容而不说明

4. 实现要求：功能层面
4.1 新增功能名

新增一个明确的功能模式：

lingbot_action_only_dsrl

或者同等清晰命名。

要求从配置级别可控：

use_dsrl: true

dsrl_mode: action_only

4.2 行为定义

每个 step 的标准执行流程必须是：

接收 RoboTwin 当前 observation

使用 LingBot 编码 observation / language / state

使用 冻结的 LingBot future branch 生成 future latent 或 action conditioning context

steering actor 根据 observation 特征输出 steer_noise

将 steer_noise 注入 LingBot action sampling 入口

冻结的 LingBot action decoder 输出动作

动作送入 RoboTwin 环境

环境返回 reward

RLinf SAC 更新 steering actor 和 critic

4.3 注入点要求

必须优先查找并实现下面的优先级：

第一优先级

action sampling 的初始噪声接口

目标接口示例：

sample_actions(
    obs,
    future_latent=None,
    initial_noise=None,
    ...
)

要求：

如果传入 initial_noise，则使用它

如果未传入，则保持原始随机初始化逻辑

第二优先级

如果 LingBot 没有显式初始噪声入口，则实现：

sample_actions(
    obs,
    future_latent=None,
    steering_embedding=None,
    ...
)

并将 steering_embedding 拼接到 action decoder 的 conditioning inputs

第三优先级

如果上述都做不到，才允许实现 action field residual steering。
但除非前两种都不可行，否则不要选第三种。

5. 实现要求：训练层面
5.1 优化器约束

优化器中只允许包含：

steering actor 参数

critic 参数

temperature / alpha 参数（如果 SAC 实现需要）

禁止将以下参数加入 optimizer：

LingBot backbone

LingBot future branch

LingBot action decoder 主体

5.2 freeze 要求

必须明确提供 freeze 逻辑，并可在配置中控制：

freeze_backbone: true

freeze_future_branch: true

freeze_action_decoder: true

默认全部为 true。

5.3 算法要求

必须复用 RLinf 现有 embodied_sac / DSRL 风格训练链路。
不要另起一套 RL 训练逻辑。

6. 配置要求

必须新增一份单独配置文件，例如：

examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml

配置必须清晰，至少包含：

actor:
  model:
    lingbot:
      use_dsrl: true
      dsrl_mode: action_only
      freeze_backbone: true
      freeze_future_branch: true
      freeze_action_decoder: true
      dsrl_noise_injection_mode: initial_noise
      dsrl_state_dim: <robot_state_dim>
      dsrl_action_noise_dim: <noise_dim>
      dsrl_num_q_heads: 10
      dsrl_image_latent_dim: 64
      dsrl_state_latent_dim: 64
      dsrl_hidden_dims: [128, 128, 128]
      dsrl_use_future_latent_summary: true

algorithm:
  adv_type: embodied_sac
  loss_type: embodied_sac

要求：

所有新增字段都有注释

不允许隐藏 magic number 而不解释

所有 shape 相关字段必须可追踪来源

7. 交付物要求

本次任务完成后，必须交付以下内容。

7.1 代码实现

必须完成实际代码改动，并保证能跑通至少一个 RoboTwin 单任务训练入口。

7.2 改动记录文档

必须新增：

agent-read/v1/change_log_lingbot_action_only_dsrl.md

文档中必须写清楚：

A. 改了哪些文件

每个文件一项，格式类似：

文件路径

改动类型：新增 / 修改 / 删除

改动目的

核心改动点

是否影响原有行为

是否向后兼容

B. 每个改动为什么需要

不能只写“为了支持功能”。
要写具体原因，比如：

为了把 RLinf SAC actor 输出接到 LingBot action sampling 初始噪声

为了保证不改动 LingBot 默认推理路径

为了让 use_dsrl=False 时保持原行为

C. 改动后行为是什么

明确写出：

默认行为

DSRL 打开时行为

action-only 模式行为

7.3 环境改动记录

必须新增：

agent-read/v1/env_change_log.md

格式要求至少包含：

记录模板

日期时间

conda 环境名

执行命令

安装/修改内容

原因

是否成功

验证方法

备注

例如：

## 2026-03-16 14:30
- 环境：RoboTwin-lingbot
- 操作：pip install xxx==x.x.x
- 原因：LingBot wrapper 依赖缺失
- 结果：成功
- 验证：python -c "import xxx"
- 备注：无
7.4 实现说明文档

必须新增一份 agent 和人都能读懂、能复现 的实现说明：

agent-read/v1/implementation_report_lingbot_action_only_dsrl.md

这份文档必须同时满足：

人能读懂

agent 能据此继续接手

能复现

能定位问题

必须包含以下章节：

1. 目标概述

一句话说明本次实现了什么。

2. 实现边界

明确写出：

做了什么

没做什么

为什么没做

3. 架构说明

解释：

RLinf 在哪里负责 RL

LingBot 在哪里负责 base policy

steering actor 接在哪里

critic 看什么输入

为什么这是 action-only

4. 数据流 / 调用流

按 step 写清楚：

observation 从哪来

经过哪些模块

steering 在哪一步产生

action 在哪一步被生成

reward 在哪一步回流

5. 文件级说明

逐个列出关键文件及其作用。

6. 配置说明

解释所有新增关键配置项的意义。

7. 如何运行

至少提供：

激活哪个 conda 环境

启动命令

训练命令

eval 命令

日志在哪里看

checkpoint 在哪里保存

8. 如何验证

说明：

如何验证 use_dsrl=False 时不影响原行为

如何验证 use_dsrl=True 时 steering 确实生效

如何验证 LingBot 参数没有被训练

如何验证 optimizer 只包含 actor/critic

9. 已知限制

明确写目前版本的限制，例如：

只支持 action-only

future latent 仍冻结

只在单任务上验证

还没有做 full method

10. 后续扩展建议

给后续 agent 或开发者的明确建议。

7.5 复现说明

必须在实现说明中增加一个单独章节：

“最小复现步骤”

要求写成严格顺序：

进入哪个仓库

激活哪个 conda 环境

检查哪些依赖

执行哪条命令

观察什么日志

看到什么现象说明成功

哪些报错最常见，怎么排查

要求做到：
另一个 agent 或人不看聊天记录，只看这份文档，也能复现。

8. 输出格式要求
8.1 代码提交之外，必须输出总结

完成后必须给出一份清晰总结，包括：

修改了哪些文件

新增了哪些文件

哪些环境有变更

哪些命令被执行过

训练入口是什么

目前是否跑通

已知问题是什么

8.2 所有说明必须具体

禁止出现这种模糊表述：

“做了一些调整”

“修复了若干问题”

“增加了相关功能”

“改了配置”

必须具体写明：

改了哪个类 / 函数 / 配置项

改动前是什么

改动后是什么

为什么这么改

9. 调试与验证要求

必须增加必要日志，至少包括：

是否启用了 use_dsrl

dsrl_mode

注入模式 initial_noise / steering_embedding

steer_noise shape

LingBot action 输出 shape

optimizer 参数统计

LingBot 冻结参数数量

可训练参数数量

并至少验证以下几点：

验证 1

use_dsrl=False 时，LingBot 推理路径仍然可用

验证 2

use_dsrl=True 时，steering actor 输出确实传入 action sampling

验证 3

LingBot backbone / future / action decoder 未进入 optimizer

验证 4

RoboTwin 单任务可启动训练

验证 5

日志中能看到 SAC 相关指标

10. 实施顺序要求

必须按下面顺序做，不要跳步：

第一步

梳理 RLinf Pi0 DSRL 的现有实现方式
输出一段简短说明，说明要复用哪些模块

第二步

定位 LingBot action sampling 接口
明确说明能否插入 initial_noise

第三步

实现 wrapper 和接口透传
保证默认行为不变

第四步

接入 steering actor / critic
保证只训练新增模块

第五步

补配置文件与运行脚本

第六步

单任务跑通

第七步

补全文档、环境记录、实现报告

11. 给 Codex 的强约束

请严格遵守以下要求：

最小改动原则

最大复用 RLinf 现有 Pi0 DSRL

不扩展到 full method

不改动无关模块

所有环境操作必须记录

所有文件改动必须记录

最终必须产出人和 agent 都能读懂的实现报告

只允许在指定 conda 环境中工作

如果发现必须超出范围改动，先在文档中明确记录原因和必要性

不要省略验证步骤

12. 给 Codex 的最终执行指令

请完成一个 LingBot + RLinf + RoboTwin 的 action-only DSRL 最小可行实现，满足以下条件：

仅修改必要代码

仅使用指定 conda 环境

所有环境改动留痕

所有代码改动留痕

最终输出可运行代码、配置、改动日志、环境日志、实现说明、最小复现步骤

保证另一个 agent 或开发者在不看聊天记录的情况下也能复现并继续开发
