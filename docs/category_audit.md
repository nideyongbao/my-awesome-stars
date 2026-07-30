# 分类体系审计：从旧 39 类到 Vault 6+3

本审计依据 2026-07-30 的本地 Vault 结构，而不是通用 GitHub 热门项目分类。

## 证据摘要

- `Domain-模型行为与控制`：203 篇 Markdown，核心是模型架构、多模态和训练范式。
- `Domain-计算加速`：92 篇，核心是框架底座、算子、编译与硬件。
- `Domain-推理框架`：69 篇，核心是推理引擎、KV Cache/调度和推理优化。
- `Domain-训练框架`：38 篇，核心是分布式/预训练、SFT 和 RL 后训练。
- 数据、应用、工程、工作流和理论当前规模较小，不需要复制内部目录层级。
- Vault 没有独立的游戏 AI 或音频模型知识树。

因此研究 taxonomy 只保留 20 个主题子类（含 `Other`），另加 1 个
`Personal-Repositories` owner 标记。判断原则是：只有能形成稳定研究入口的方向
才保留子类；低频、相邻或工具性主题直接合并。

## 旧 39 类逐项审查

| # | 旧分类 | 结论 | 新子分类 |
|---:|---|---|---|
| 1 | AI-Sys-Framework | 与硬件/系统底座合并 | Compute-System-Framework |
| 2 | AI-Sys-Hardware | 与框架底座合并 | Compute-System-Framework |
| 3 | AI-Sys-Kernel | 保留核心方向 | Compute-Kernel-Operator |
| 4 | AI-Sys-Compiler | 保留核心方向 | Compute-Compiler-Runtime |
| 5 | AI-Sys-MLOps | 移入工程 Pillar | Engineering-Practice |
| 6 | AI-Sys-Training | 与分布式、预训练循环合并 | Training-Distributed-Pretrain |
| 7 | AI-Sys-FineTuning | 保留核心方向 | Training-SFT-PEFT |
| 8 | AI-Sys-RLHF | 扩展到 RLVR/后训练 | Training-RL-PostTraining |
| 9 | AI-Sys-Cluster | 合并到分布式训练 | Training-Distributed-Pretrain |
| 10 | AI-Data-Pipeline | 合并为单一数据主题 | Data-Systems |
| 11 | AI-Data-Synthetic | 合并为单一数据主题 | Data-Systems |
| 12 | AI-Data-Vector | 合并为单一数据主题 | Data-Systems |
| 13 | AI-Data-Dataset | 合并为单一数据主题 | Data-Systems |
| 14 | AI-Data-Crawl | 合并为单一数据主题 | Data-Systems |
| 15 | AI-Data-Labeling | 合并为单一数据主题 | Data-Systems |
| 16 | AI-Sys-Inference | 作为推理框架入口 | Inference-Engines-Serving |
| 17 | AI-Sys-Quantization | 合并到推理优化 | Inference-Optimization |
| 18 | AI-Algo-LLM | 与 Attention/MoE 等组件合并 | Model-Architecture-Components |
| 19 | AI-Algo-Multi | 统一为多模态/VLM | Model-Multimodal-VLM |
| 20 | AI-Algo-Vision | 删除独立分类 | Model-Multimodal-VLM |
| 21 | AI-Algo-Audio | 删除独立分类 | Model-Multimodal-VLM |
| 22 | AI-Algo-Robotics | 删除独立分类 | Model-Multimodal-VLM |
| 23 | AI-Algo-Game | 删除独立分类；现有条目是 RL 实现 | Training-RL-PostTraining |
| 24 | AI-App-Framework | 与 Agent/RAG 工作流合并 | App-Agent-RAG |
| 25 | AI-App-RAG | 与 Agent/Research 工作流合并 | App-Agent-RAG |
| 26 | AI-App-Agent | 与 RAG/编排合并 | App-Agent-RAG |
| 27 | AI-App-MCP | 保留当前重点 | App-Tools-MCP |
| 28 | AI-Algo-Theory | 合并到理论与复现 | Theory-Reproduction |
| 29 | Research-Paper | 合并到理论与复现 | Theory-Reproduction |
| 30 | Dev-Web-FullStack | 降级为工程支撑 | Engineering-Practice |
| 31 | Dev-Infra-Cloud | 合并到工程实践 | Engineering-Practice |
| 32 | Dev-DB-Storage | 合并到工程实践 | Engineering-Practice |
| 33 | Dev-Lang-Core | 合并到工作流工具 | Workflow-Knowledge-Tools |
| 34 | Dev-Sec | 合并到工程实践 | Engineering-Practice |
| 35 | AI-App-Coding | 保留当前重点 | Workflow-AI-Coding |
| 36 | Tools-Efficiency | 合并到知识与工具链 | Workflow-Knowledge-Tools |
| 37 | Tools-Media | 删除独立分类 | Workflow-Knowledge-Tools |
| 38 | CS-Education | 合并到理论与复现 | Theory-Reproduction |
| 39 | Uncategorized | 严格兜底并改名 | Other |

## 最终 21 个主题子类

| Vault 主分类 | 主题子类 |
|---|---|
| Domain-计算加速 | Compute-System-Framework、Compute-Kernel-Operator、Compute-Compiler-Runtime |
| Domain-训练框架 | Training-Distributed-Pretrain、Training-SFT-PEFT、Training-RL-PostTraining |
| Domain-推理框架 | Inference-Engines-Serving、Inference-Scheduling-KVCache、Inference-Optimization |
| Domain-数据系统 | Data-Systems |
| Domain-模型行为与控制 | Model-Architecture-Components、Model-Multimodal-VLM、Model-Reasoning-Alignment |
| Domain-应用系统 | App-Agent-RAG、App-Tools-MCP |
| Pillar-工程与实践 | Engineering-Practice |
| Pillar-工作流与工具链 | Workflow-AI-Coding、Workflow-Knowledge-Tools、Personal-Repositories |
| Pillar-理论与复现 | Theory-Reproduction |
| 其他 | Other |

现有缓存通过 `legacy_names` 从旧 39 类和中间版本分类迁移，不调用 LLM、不丢失条目。
`nideyongbao/*` 由 owner 规则归入 `Personal-Repositories`，原技术主题作为
`subject_*` 辅助字段保留；其他新增仓库由新 Prompt 从研究主题中选择，不能动态
创建新分类。
