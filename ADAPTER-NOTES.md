# 作者原实现与新增适配层的边界

作者代码完全位于`vendor/evotool`。原算法仍负责初始四模块提示、Planner/Selector/Caller/Synthesizer调用、归责、定向变异、接收规则、种群选择、原success与progress评测；本次未改任何原文件。

新增部分：

1. `rsi_tools/server.py`：V100用TransformersFP16模型服务，提供本机OpenAI兼容接口；不用vLLM/BF16/FlashAttention2；JSON对象模式用LMFormatEnforcer。后端差异必须随结果报告。
2. `rsi_tools/tracing.py`：对作者LLMClient做运行时子类包装，复用原generate/JSON解析；只记录调用，并保留同一作业内的原缓存行为。一次作业独立缓存，避免配置间误复用。
3. `pilot.py`：显式开发集采集/短演化入口；不使用作者预先演化的theta_star作为初始提示，不向evolve传测试集，关闭周期测试评测，成本计入全部角色。
4. `rsi_tools/analysis.py`：基于作者严格调用匹配函数，额外离线统计匹配的gold调用数；这是诊断指标，不替换原奖励、不反馈给原演化器。
5. `configs/bfcl-split.json`：先冻结作者90/30/30划分，再按ID取开发子集，避免缩小n_train导致原测试题进入诊断。
6. `cloud_job.py`：单卡作业监督，绑定正式runner入口和时限；不下载模型、不自动启动外部计费资源。

H1/H2/H3均未作为新方法加入基线。这一版先获取可解释的基线轨迹。原论文数字、作者参考配置、未来V100实测结果必须分列。

限时或模型错误时保留已完成日志，标记budget_exhausted/runtime_error；不自动扩大预算、不改精度、不回退CPU，不把空输出当作成功。JSON语法约束也可能被token/时限截断，需检查结束原因。

正式GPU计划尚未登记或批准；本包提供了未修改的项目审批工具副本和计划草稿生成器，未伪造review/approval。日志与环境变量不是身份认证，仍须按项目流程由人核查。
