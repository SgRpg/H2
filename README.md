# EVOTOOL：V100小规模核验代码包

本包面向TeslaV100-SXM2-16GB、Linux、Python3.11。**作者代码已拉取，项目适配层已准备；目前只通过本地CPU合同检查，尚未在V100加载模型。** 不含模型权重，不进行参数训练，不代表原论文性能复现。

## 代码来源和用途

- `vendor/evotool/`：作者官方仓库原文件，固定commit为`6a82719ea6e22cf23f963078e2527094e50b47f6`，MIT。84个原文件的哈希见`baseline-manifest.json`。
- `rsi_tools/`、`pilot.py`等：本项目新增适配器、日志和诊断。显式新入口会调用作者的模块、归责、变异、选择及评测实现。
- 模型固定为`Qwen/Qwen3-4B`，revision为`1cfa9a7208912126459214e8b04321603b3df60c`。FP16、关闭thinking、单请求串行；JSON对象模式使用语法约束。Transformers服务是V100适配，不是作者vLLM后端。
- 作者4B参考测试不等于论文8B主结果；作者BFCL评测代码为语义重实现，不能据此宣称完整官方BFCL复现。

当前先采H2需要的反馈粒度轨迹，同时记录H1需要的预算和截断信息。**未实现创新方法、未修改奖励、未把合成CPU结果当作科研结果。**

## 1. 上传后先检查文件

将ZIP上传到DSW的`/mnt/workspace`并解压，进入本包目录：

```bash
cd /mnt/workspace/rsi_evotool_v100
/mnt/workspace/rsi-v100-env/bin/python -B doctor.py
```

Python路径对应前面用uv准备的独立3.11环境。若还没有该环境，先完成环境创建；不要用系统Python3.13直接安装本包依赖。

## 2. 安装云端依赖

下面脚本只在你手动执行时安装；打包和CPU自检不会调用它。不安装Ray、vLLM或FlashAttention2，不升级驱动。

```bash
bash scripts/install_v100.sh /mnt/workspace/rsi-v100-env/bin/python
```

脚本固定PyTorch2.6.0+cu124和模型库版本，并执行兼容性检查。需要看到CUDA可见及`sm_70`；这仍不等于模型已成功加载。依赖安装需要联网及磁盘空间，安装失败请保留原始报错。

## 3. CPU合同检查

```bash
/mnt/workspace/rsi-v100-env/bin/python -B -m unittest discover -s tests -v
```

检查原基线哈希、固定划分、参数类型/重复调用匹配、H2构造反例、预算中断、合成轨迹采集、短演化及本机HTTP协议。测试模型输出是明确标注的合成结果，不读取GPU或远端API。

## 4. 在云端单独下载模型

本包没有捆绑模型。确认磁盘空间后，显式执行：

```bash
/mnt/workspace/rsi-v100-env/bin/python -B download_model.py --dest models/Qwen3-4B
```

只从公开HuggingFace固定revision下载，不读取已存HF凭据；生成模型文件哈希清单。中断后可对同一路径续传。不会自动改用其他模型、镜像源或量化精度。

## 5. 首轮方案：3题采集，先不演化

`configs/collect-3.json`固定：作者150题示例的90/30/30划分、seed42；仅取train前3题；原初始提示；6步、8192上下文、单次最多512新token；最多80次模型调用、80000总token、采集阶段15分钟。加上模型加载，整个云端作业上限20分钟。

先打印计划，不会调用模型：

```bash
/mnt/workspace/rsi-v100-env/bin/python -B pilot.py --plan configs/collect-3.json --out outputs/collect-3
```

`configs/evolve-2.json`是后续备选：3题train、2题selection、2代作者原算法；默认不运行。首轮不同时跑多个方案，也不评测测试集。

## 6. 正式GPU入口与审批

本包包含项目原有工作流的未修改副本`workflow/tools/research.py`和`workflow/schema/schema.sql`，不复制本地SQLite或任何批准记录。

模型与云环境准备好后，可生成绑定当前云端路径的计划草稿：

```bash
/mnt/workspace/rsi-v100-env/bin/python -B prepare_cloud_plan.py --model-dir models/Qwen3-4B
```

草稿不会自动登记、放行idea评审或批准。正式运行前，按项目规则完成idea的创新性/可行性阶段审查、登记具体计划，再由你本人在交互终端执行`experiment-approve`并输入所示短语；Codex不得代输。实验入口为`cloud_job.py`，它要求正式runner设置的`ARAG_RUN_DIR`，负责启动本地模型服务、采集轨迹及退出后清理子进程。

**当前交付状态为代码准备完成、GPU计划待审。** 后续先核对环境/模型/数据，再登记首轮方案；不要直接运行作者`run.py`的全量默认配置。

## 7. 输出和解释

- `manifest.json`：配置、版本、数据划分、源码与模型信息。
- `model-calls.jsonl`：所有角色输入、输出、token、结束原因、错误及耗时。
- `episodes.jsonl`：原始计划、工具参数、反馈、最终判分、原奖励、缓存命中和截断标记。
- `blame.jsonl`：开发题归责和理由；`generations.jsonl`：短演化时的提示变更/候选情况。
- `h2-offline-diagnostic.json`：同一道题在不同候选下是否出现“原奖励相同但参数正确性不同”。
- `summary.json`、`output-hashes.json`：完整/中断状态、全部角色的已知成本和输出校验。

第一轮只有初始策略，通常没有足够的同题多候选来判定H2；它首先验证接口与轨迹。合成反例只证明可能有信号粒度差异；真实碰撞也不自动证明演化受损。H1的“计划被步数上限截断”标记不自动等于错误归责。

测试题不用于诊断、阈值选择、规则生成或方法筛选。固定全部配置后才能做同预算比较；更换max_steps、模型、上下文或精度要新进程、新计划和新输出目录，避免作者缓存键遗漏条件的影响。

## 8. 来源与验证边界

论文：[EVOTOOL，ACL2026](https://aclanthology.org/2026.acl-long.2016/)。代码：[作者官方仓库](https://github.com/SYang2000/ACL_2026_EvoTool)。模型：[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)。

作者仓库带BFCL转化示例；其样本来源和评测口径需要在正式实验前逐项核对。代码MIT不能自动替代第三方数据许可。当前只使用公开代码/示例作准备与功能检查，尚无GPU实验结果和创新性结论。

本地CPU检查环境、检查数量和源码保护结果见本包`VALIDATION.json`；具体新入口与原算法的差异见`ADAPTER-NOTES.md`。
