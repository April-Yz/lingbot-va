# `action_only_v1.4.md` 与当前实现的差异说明

## 1. 结论

当前实现已经完成了一个**可运行的 action-only DSRL 最小基线**，并且已经满足以下核心技术目标：

- LingBot 已暴露 `initial_noise` 注入接口
- `use_dsrl=false` 与 `use_dsrl=true` 两条推理路径都能跑
- steering actor / critic / alpha 与 LingBot 参数已经分离
- RoboTwin 在线单任务 pipeline 已经可以完整执行一个 episode

但如果严格按照 [action_only_v1.4.md](/home/zaijia001/vam/lingbot-va/agent-read/action_only_v1.4.md) 这份新要求来衡量，**当前版本仍然不是“完全符合 V1.4 要求的最终版”**。

主要差异不在“算法主线”，而在以下几类：

- 文档命名和双语同步要求没有满足
- 与 RLinf 的复用程度不够，当前更接近“本地复刻一个最小 SAC/DSRL 风格实现”
- 部分硬性验证项还没有重新补齐
- 一些命名和交付物形式与 V1.4 约定不一致

## 2. 当前已经符合 V1.4 的部分

### 2.1 核心目标基本已达成

当前代码已经实现了 V1.4 要求中的主线功能：

- 冻结 LingBot 主体
- 使用冻结 future branch 作为 action 生成上下文
- 新增轻量 steering actor
- 新增 critic
- 将 steering 信号注入 LingBot action sampling
- 仅优化 actor / critic / alpha
- `use_dsrl=false` 时保留原始推理路径

对应实现位置：

- [wan_va_server.py](/home/zaijia001/vam/lingbot-va/wan_va/wan_va_server.py)
- [policy.py](/home/zaijia001/vam/lingbot-va/wan_va/action_only_dsrl/policy.py)
- [modules.py](/home/zaijia001/vam/lingbot-va/wan_va/action_only_dsrl/modules.py)
- [run_lingbot_action_only_dsrl.py](/home/zaijia001/vam/lingbot-va/script/run_lingbot_action_only_dsrl.py)
- [robotwin_lingbot_action_only_dsrl.yaml](/home/zaijia001/vam/lingbot-va/examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml)

### 2.2 注入优先级满足 Priority 1

V1.4 明确要求优先暴露：

- `sample_actions(..., initial_noise=None)`

当前实现已经按这个优先级完成，并不是 fallback 到 `steering_embedding` 或 residual action。

### 2.3 优化器约束已满足

当前实现已经明确做到：

- optimizer 不包含 LingBot 主体参数
- 只优化 steering actor / critic / alpha

这一点和 V1.4 的优化器约束是一致的。

### 2.4 单任务在线运行已经能启动并完成

V1.4 要求至少完成：

- RoboTwin single-task training starts successfully

当前状态已经超过“仅启动”：

- `click_bell` 的在线 episode 已经完整执行结束
- final status 为 `finished_no_success`

也就是说：

- pipeline 已经通
- 只是当前这一轮任务没有成功

## 3. 和 V1.4 的主要差异

## 3.1 最大差异：当前不是“真正复用 RLinf pipeline”，而是“本地实现一个 RLinf 风格版本”

这是最重要的差异。

V1.4 的表述非常明确：

- 要最大化复用 RLinf 现有 Pi0 DSRL 实现
- 优先复用 existing `use_dsrl` logic、embodied_sac pipeline、replay buffer、learner、logger、registration path

但当前实现的实际情况是：

- 没有直接把 LingBot 接到 `/home/zaijia001/vam/RLinf` 的现有训练管线里
- 而是在 `lingbot-va` 仓库内新建了一个本地 `wan_va/action_only_dsrl/`
- 本地实现了 compact modules、replay buffer、actor/critic、trainer
- 训练入口也是单独的 [run_lingbot_action_only_dsrl.py](/home/zaijia001/vam/lingbot-va/script/run_lingbot_action_only_dsrl.py)

所以从 V1.4 的严格口径看：

- 现在是“RLinf style”
- 不是“RLinf native integration”

这会导致后续仍需要改的地方包括：

- 接入 RLinf 原生 policy wrapper / registration
- 尽量复用 RLinf 自带 learner / logger / evaluator
- 减少本地重复实现的 trainer 逻辑

## 3.2 文档命名体系与 V1.4 不一致

V1.4 要求所有文档使用带 `V1` 后缀、并区分 `en/zh` 的文件名，例如：

- `task_spec_lingbot_action_only_dsrl_V1_en.md`
- `task_spec_lingbot_action_only_dsrl_V1_zh.md`
- `implementation_report_lingbot_action_only_dsrl_V1_en.md`
- `implementation_report_lingbot_action_only_dsrl_V1_zh.md`

当前实际文档命名是：

- [implementation_report_lingbot_action_only_dsrl.md](/home/zaijia001/vam/lingbot-va/agent-read/implementation_report_lingbot_action_only_dsrl.md)
- [change_log_lingbot_action_only_dsrl.md](/home/zaijia001/vam/lingbot-va/agent-read/change_log_lingbot_action_only_dsrl.md)
- [env_change_log.md](/home/zaijia001/vam/lingbot-va/agent-read/env_change_log.md)

所以严格来说，V1.4 的“命名规范”没有满足。

## 3.3 双语文档没有补全，也没有做到同步命名

V1.4 要求：

- 英文和中文文档都要有
- 必须成对存在
- 必须同步更新

当前实际情况：

- 主要实现文档基本还是英文单份
- 中文版并没有和英文版形成一一对应的同步文档对
- task spec / implementation report / change log / env log 的双语成套文件不存在

这部分是当前最明确、也最容易补齐的缺口。

## 3.4 缺少 V1.4 要求的最终 task spec 成品文档

V1.4 要求交付：

- `task_spec_lingbot_action_only_dsrl_V1_en.md`
- `task_spec_lingbot_action_only_dsrl_V1_zh.md`

当前仓库里虽然有：

- [action_only_v1.4.md](/home/zaijia001/vam/lingbot-va/agent-read/action_only_v1.4.md)
- [action_only_v1.md](/home/zaijia001/vam/lingbot-va/agent-read/action_only_v1.md)
- [action_only_v1_ZH.md](/home/zaijia001/vam/lingbot-va/agent-read/action_only_v1_ZH.md)

但它们更像“需求输入文档”，不是“最终同步后的 task spec 交付件”。

## 3.5 freeze 粒度和 V1.4 文字表述不完全一致

V1.4 的要求是：

- freeze backbone
- freeze future branch
- freeze action decoder main body

当前实现里，策略上采取的是**更强的冻结**：

- 直接冻结整个 LingBot stack
- 包括 `transformer`、`vae`、`text_encoder`

这在功能上没有违背“不要训练 LingBot 主体”的目标，但和 V1.4 的语义并不完全等价。

问题不在“冻结太多会不会错”，而在：

- 当前实现没有明确拆出“backbone / future branch / action decoder main body”的命名级别 freeze 控制
- 文档里也承认这是“更严格但更粗粒度”的冻结方式

如果要完全满足 V1.4，后续需要：

- 把 LingBot 内部模块边界定义得更清楚
- 明确哪些层属于 future branch
- 明确哪些层属于 action decoder main body
- 用更细粒度的 param group / freeze report 来证明

## 3.6 原始 eval 和 post-train 的“回归验证”没有按 V1.4 再次补跑

V1.4 明确要求在 action-only baseline 完成后，还要验证：

1. original LingBot eval still works
2. original LingBot post-train still works

当前事实是：

- 历史上这两个链路我们都跑通过
- 但在 action-only 改动完成后，没有重新做一次“回归验证”

所以这部分不能写成“已经满足”，只能写成：

- 理论上兼容性设计是保留了
- 但按 V1.4 的标准，缺少一次明确的 regression validation

这是一个明确待补项。

## 3.7 minimal reproduction 文档不够完整

V1.4 对 implementation report 的最小复现章节要求非常细，除了：

- repo path
- conda env
- exact run command

还要求包含：

- expected logs
- what indicates success
- common failure modes
- debugging hints

当前 [implementation_report_lingbot_action_only_dsrl.md](/home/zaijia001/vam/lingbot-va/agent-read/implementation_report_lingbot_action_only_dsrl.md) 虽然已经有 reproduction section，但离 V1.4 的“严格最小复现模板”还差一点，尤其是：

- 成功判据没有系统展开
- 常见失败模式和调试提示不够结构化

## 3.8 配置和模式命名不完全按 V1.4 版本化

V1.4 推荐并默认要求更明确的 V1 命名，例如：

- `lingbot_action_only_dsrl_V1`
- `robotwin_lingbot_action_only_dsrl_V1.yaml`

当前实际命名是：

- `lingbot_action_only_dsrl`
- [robotwin_lingbot_action_only_dsrl.yaml](/home/zaijia001/vam/lingbot-va/examples/embodiment/config/robotwin_lingbot_action_only_dsrl.yaml)

这不影响功能，但不满足 V1.4 更严格的版本化交付要求。

## 4. 当前最需要修改的地方

如果下一轮是“按 V1.4 重新收口”，我建议优先级如下。

### 优先级 A：先补文档和命名，不动算法

这是最小代价、最先该做的事情。

需要补：

- `task_spec_lingbot_action_only_dsrl_V1_en.md`
- `task_spec_lingbot_action_only_dsrl_V1_zh.md`
- `implementation_report_lingbot_action_only_dsrl_V1_en.md`
- `implementation_report_lingbot_action_only_dsrl_V1_zh.md`
- `change_log_lingbot_action_only_dsrl_V1_en.md`
- `change_log_lingbot_action_only_dsrl_V1_zh.md`
- `env_change_log_V1_en.md`
- `env_change_log_V1_zh.md`

并且要明确标注：

- 当前实现是“已跑通 online episode 的本地 V1 baseline”
- 哪些部分已满足
- 哪些部分只是暂时实现
- 哪些部分仍不符合 V1.4 的严格口径

### 优先级 B：补回归验证

这是 V1.4 的硬要求。

至少需要补两项：

- 原始 LingBot eval 再跑一次最小 smoke test
- 原始 LingBot post-train 再做一次最小 smoke test

这两项不一定需要大规模跑，但要有：

- 命令
- 日志
- 成功判据
- 文档记录

### 优先级 C：把“RLinf style”改成“RLinf native reuse”

这是最重要的功能性差异，但成本也最高。

如果真的要完全按 V1.4 收口，后续应优先检查：

- RLinf 原有 DSRL policy wrapper 的接入方式
- RLinf 原生 learner / logger / replay buffer 的复用可能性
- LingBot wrapper 是否可以注册进 RLinf 的现有配置与启动路径

如果最终仍无法原生复用，也至少要在文档里明确说明：

- 为什么不能直接接 RLinf 原生 pipeline
- 当前本地实现复刻了哪些 RLinf 逻辑
- 与原生 RLinf 管线相比还缺什么

### 优先级 D：细化 freeze 边界

如果后续还要继续贴近 V1.4 语义，需要把“冻结整个 LingBot”改成更有可解释性的结构化冻结：

- backbone
- future branch
- action decoder main body

并且输出更细的 freeze report。

## 5. 我对当前版本的判断

如果按“能不能跑最小基线”来判断：

- 当前版本已经成功

如果按“是否完全满足 `action_only_v1.4.md` 的交付规范”来判断：

- 当前版本还没有完全满足

最准确的描述应该是：

- **功能上：V1 baseline 已经跑通**
- **规范上：还不是严格符合 V1.4 文档和交付要求的最终版**

## 6. 建议的后续策略

如果你下一步不想立刻再改功能代码，最合理的顺序是：

1. 先把 V1.4 要求的双语文档和命名体系补齐
2. 再补原始 eval / post-train 的回归验证
3. 最后再决定是否要把当前本地 trainer 进一步并回 RLinf 原生 pipeline

这样风险最小，也最符合你现在“先别动功能”的要求。
