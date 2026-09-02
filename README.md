# 🌟 My Awesome AI Stars

> 🤖 由 GitHub Actions 自动更新；主分类与本地 Vault 6+3 体系一致。

当前收录 **232** 个仓库。

分类设计与旧 39 类迁移记录见 [分类体系审计](docs/category_audit.md)。

## 目录

- [Domain-计算加速 (28)](#vault-compute-acceleration)
- [Domain-训练框架 (52)](#vault-training-frameworks)
- [Domain-推理框架 (14)](#vault-inference-frameworks)
- [Domain-数据系统 (8)](#vault-data-systems)
- [Domain-模型行为与控制 (20)](#vault-model-behavior-control)
- [Domain-应用系统 (43)](#vault-application-systems)
- [Pillar-工程与实践 (11)](#vault-engineering-practice)
- [Pillar-工作流与工具链 (33)](#vault-workflow-toolchain)
- [Pillar-理论与复现 (20)](#vault-theory-reproduction)
- [其他 (3)](#vault-other)

---

## <span id="vault-compute-acceleration">Domain-计算加速 (28)</span>

GPU、CUDA、Triton、算子、编译、通信、HPC 与性能原语

### <span id="topic-compute-system-framework">Compute-System-Framework (8)</span>

GPU 与 AI 芯片、互联通信、PyTorch/JAX 框架底座及其执行机制

| Project | Description | Stars | Language |
|---|---|---:|---|
| [huggingface/transformers](https://github.com/huggingface/transformers) | 🤗 Transformers: the model-definition framework for state-of-the-art machine learning models in text, vision, audio, and multimodal models, for both inference an | 164706 | Python |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | Tensors and Dynamic neural networks in Python with strong GPU acceleration | 102710 | Python |
| [jinbooooom/OriginDL](https://github.com/jinbooooom/OriginDL) | Implement a Pytorch-like DL library in C++ from scratch, step by step | 370 | C++ |
| [keith2018/TinyTorch](https://github.com/keith2018/TinyTorch) | A lightweight deep learning training framework implemented from scratch in C++, featuring a PyTorch-style API. | 214 | C++ |
| [ViperEkura/AstrAI](https://github.com/ViperEkura/AstrAI) | A lightweight Transformer training & inference framework | 107 | Python |
| [abcdabcd987/libfabric-efa-demo](https://github.com/abcdabcd987/libfabric-efa-demo) |  | 83 | C++ |
| [tigert1998/mytorch](https://github.com/tigert1998/mytorch) | A toy Python DL training library with PyTorch like API | 38 | Python |
| [Eclipse-Arrebol/CUDA_MATMUAL](https://github.com/Eclipse-Arrebol/CUDA_MATMUAL) |  | 2 | Cuda |

### <span id="topic-compute-kernel-operator">Compute-Kernel-Operator (20)</span>

CUDA、Triton、CUTLASS、GEMM、FlashAttention、MoE 与算子融合优化

| Project | Description | Stars | Language |
|---|---|---:|---|
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Fast and memory-efficient exact attention | 24829 | Python |
| [triton-lang/triton](https://github.com/triton-lang/triton) | Development repository for the Triton language and compiler | 20064 | MLIR |
| [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | FlashMLA: Efficient Multi-head Latent Attention Kernels | 12894 | C++ |
| [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA) | Modern CUDA Learn Notes with PyTorch for Beginners, 200+ CUDA Kernels, Tensor Cores, HGEMM, FA-2 MMA. | 11868 | Cuda |
| [facebookresearch/xformers](https://github.com/facebookresearch/xformers) | Hackable and optimized Transformers building blocks, supporting a composable construction. | 10545 | Python |
| [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | Efficient Triton Kernels for LLM Training | 6596 | Python |
| [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | 🚀 Efficient implementations for emerging model architectures | 5680 | Python |
| [stepfun-ai/Step-3.5-Flash](https://github.com/stepfun-ai/Step-3.5-Flash) | Fast, Sharp & Reliable Agentic Intelligence | 2072 | C++ |
| [RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel) | Autoresearch for GPU kernels. Give it any PyTorch model, go to sleep, wake up to optimized Triton kernels. | 1543 | Python |
| [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | FP16xINT4 LLM inference kernel that can achieve near-ideal ~4x speedups up to medium batchsizes of 16-32 tokens. | 1138 | Python |
| [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA) | high-performance linear attention kernel library built on TileLang | 680 | Python |
| [hustvl/MoDA](https://github.com/hustvl/MoDA) | An hardware-aware Efficient Implementation for "Mixture-of-Depths Attention". | 274 | Python |
| [open-lm-engine/coda-kernels](https://github.com/open-lm-engine/coda-kernels) | CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs | 249 | Python |
| [DefTruth/CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes) | 📚200+ Tensor/CUDA Cores Kernels, ⚡️flash-attn-mma, ⚡️hgemm with WMMA, MMA and CuTe (98%~100% TFLOPS of cuBLAS/FA2 🎉🎉). | 93 | Cuda |
| [TongmingLAIC/AKO4X](https://github.com/TongmingLAIC/AKO4X) | Agentic Kernel Optimization — advanced & eXtensible: a closed-loop, campaign-based multi-agent system for optimizing GPU kernels (benchmark-swappable; default f | 69 | Python |
| [jt-zhang/Sparse_Attention_API](https://github.com/jt-zhang/Sparse_Attention_API) |  | 66 | Python |
| [deciding/cutez](https://github.com/deciding/cutez) | CuTeDSL tutorials, tools, autotuner, profiler, etc. | 43 | Python |
| [keith2018/TinyFA](https://github.com/keith2018/TinyFA) | A lightweight, from-scratch Flash Attention CUDA implementation | 4 | Cuda |
| [liangyuwang/MetaProfiler](https://github.com/liangyuwang/MetaProfiler) | MetaProfiler is a lightweight, structure-agnostic operator-level profiler for PyTorch models that leverages MetaTensor execution to simulate and benchmark indiv | 2 | Python |
| [Pearblossom-M/flash-swiglu-mlp](https://github.com/Pearblossom-M/flash-swiglu-mlp) | High-performance fused SwiGLU MLP kernel in Triton, outperforming Liger kernel and torch.compile. | 1 | Python |

## <span id="vault-training-frameworks">Domain-训练框架 (52)</span>

预训练、SFT、RLHF/RLVR、优化器与分布式训练

### <span id="topic-training-distributed-pretrain">Training-Distributed-Pretrain (26)</span>

预训练、中训练、训练循环以及 ZeRO、FSDP、Megatron、Ray 和并行策略

| Project | Description | Stars | Language |
|---|---|---:|---|
| [jingyaogong/minimind](https://github.com/jingyaogong/minimind) | 🧠 Train a 64M-parameter LLM from scratch in just 2h! | 57177 | Python |
| [ray-project/ray](https://github.com/ray-project/ray) | Ray is an AI compute engine. Ray consists of a core distributed runtime and a set of AI Libraries for accelerating ML workloads. | 43680 | Python |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Ongoing research training transformer models at scale | 17704 | Python |
| [deepspeedai/DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples) | Example models using DeepSpeed | 6846 | Python |
| [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | NanoGPT (124M) in 90 seconds | 5725 | Python |
| [pytorch/torchtitan](https://github.com/pytorch/torchtitan) | A PyTorch native platform for training generative AI models | 5691 | Python |
| [PKU-Alignment/align-anything](https://github.com/PKU-Alignment/align-anything) | Align Anything: Training All-modality Model with Feedback | 4669 | Python |
| [huggingface/nanotron](https://github.com/huggingface/nanotron) | Minimalistic large language model 3D-parallelism training | 2805 | Python |
| [qibin0506/Cortex](https://github.com/qibin0506/Cortex) | 从零构建大模型：从预训练到RLHF的完整实践 | 2690 | Python |
| [OpenDCAI/DataFlex](https://github.com/OpenDCAI/DataFlex) | Data-centric LLM training with dynamic sample selection, domain mixture optimization, and example reweighting inside the LLaMA-Factory training loop. | 2394 | Python |
| [huggingface/picotron](https://github.com/huggingface/picotron) | Minimalistic 4D-parallelism distributed training framework for education purpose | 2294 | Python |
| [Open-Reasoner-Zero/Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) | Official Repo for Open-Reasoner-Zero | 2100 | Python |
| [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) | 🚀 Pytorch Distributed native training library for LLMs/VLMs with OOTB Hugging Face support | 893 | Python |
| [DLYuanGod/MegaTrain](https://github.com/DLYuanGod/MegaTrain) |  | 690 | Python |
| [stepfun-ai/SteptronOss](https://github.com/stepfun-ai/SteptronOss) | A lightweight, AI-native training framework for large language models. Designed for fast iteration, reproducible experiments, and modular configuration across S | 586 | Python |
| [Victorwz/Open-Qwen2VL](https://github.com/Victorwz/Open-Qwen2VL) | [COLM 2025] Open-Qwen2VL: Compute-Efficient Pre-Training of Fully-Open Multimodal LLMs on Academic Resources | 315 | Python |
| [liangyuwang/zo2](https://github.com/liangyuwang/zo2) | ZO2 (Zeroth-Order Offloading): Full Parameter Fine-Tuning 175B LLMs with 18GB GPU Memory [COLM2025] | 207 | Python |
| [MiroMindAI/MiroTrain](https://github.com/MiroMindAI/MiroTrain) | MiroTrain is an efficient and algorithm-first framework research agent. | 142 | Python |
| [nex-agi/NexRL](https://github.com/nex-agi/NexRL) | NexRL is an ultra-loosely-coupled LLM post-training framework. | 119 | Python |
| [liangyuwang/Tiny-FSDP](https://github.com/liangyuwang/Tiny-FSDP) | Tiny-FSDP, a minimalistic re-implementation of the PyTorch FSDP | 111 | Python |
| [wxhcore/bumblecore](https://github.com/wxhcore/bumblecore) | An LLM training framework built from the ground up, featuring a custom BumbleBee architecture and end-to-end support for multiple open-source models across Pret | 101 | Python |
| [CoinCheung/gdGPT](https://github.com/CoinCheung/gdGPT) | Train llm (bloom, llama, baichuan2-7b, chatglm3-6b) with deepspeed pipeline mode. Faster than zero/zero++/fsdp. | 97 | Python |
| [liangyuwang/Tiny-DeepSpeed](https://github.com/liangyuwang/Tiny-DeepSpeed) | Tiny-DeepSpeed, a minimalistic re-implementation of the DeepSpeed library | 52 | Python |
| [liangyuwang/Tiny-Megatron](https://github.com/liangyuwang/Tiny-Megatron) | Tiny-Megatron, a minimalistic re-implementation of the Megatron library | 32 | Python |
| [XU-YIJIE/hobo-llm-from-scratch](https://github.com/XU-YIJIE/hobo-llm-from-scratch) | From Llama to Deepseek, grpo/mtp implemented. With pt/sft/lora/qlora included | 30 | Python |
| [liangyuwang/Streaming-Dataloader](https://github.com/liangyuwang/Streaming-Dataloader) |  A memory-efficient streaming data loader designed for LLM pretraining under limited CPU and GPU memory constraints | 3 | Python |

### <span id="topic-training-sft-peft">Training-SFT-PEFT (10)</span>

SFT、LoRA、QLoRA、PEFT、Adapter、数据 Packing 与微调框架

| Project | Description | Stars | Language |
|---|---|---:|---|
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | Local UI to run and train LLMs and diffusion models. Supports GGUF, MLX, Qwen3.8, Kimi K3, MiniMax-H3, Gemma 4, FLUX and more. | 75442 | Python |
| [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) | Unified Efficient Fine-Tuning of 100+ LLMs & VLMs (ACL 2024) | 74511 | Python |
| [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | Use PEFT or Full-parameter to CPT/SFT/DPO/GRPO 600+ LLMs (Qwen3.6, DeepSeek-V4, GLM-5.1, InternLM3, Llama4, ...) and 300+ MLLMs (Qwen3-VL, Qwen3-Omni, InternVL3 | 15479 | Python |
| [imoneoi/openchat](https://github.com/imoneoi/openchat) | OpenChat: Advancing Open-source Language Models with Imperfect Data | 5491 | Python |
| [adonis-dym/memory_reduced_optimizer](https://github.com/adonis-dym/memory_reduced_optimizer) |  | 529 | Python |
| [nwind/llm-finetune](https://github.com/nwind/llm-finetune) | 大模型微调与部署指南 | 145 | HTML |
| [AI-Study-Han/Zero-Qwen-VL](https://github.com/AI-Study-Han/Zero-Qwen-VL) | 训练一个对中文支持更好的LLaVA模型，并开源训练代码和数据。 | 83 | Python |
| [qibin0506/llm_trainer](https://github.com/qibin0506/llm_trainer) |  | 67 | Python |
| [yafo-ai/y-trainer](https://github.com/yafo-ai/y-trainer) | y-trainerY-Trainer 是一个LLM模型微调训练框架。  📊 核心优势： 📉 精准对抗过拟合： 专门优化，有效解决SFT中的过拟合难题。  🧩 突破遗忘瓶颈： 无需依赖通用语料，即可卓越地保留模型的泛化能力，守住核心竞争力的同时实现专项提升！🏆 | 48 | Python |
| [liangyuwang/Tiny-transformers](https://github.com/liangyuwang/Tiny-transformers) |  | 3 | Python |

### <span id="topic-training-rl-posttraining">Training-RL-PostTraining (16)</span>

RLHF、RLVR、PPO、DPO、GRPO、奖励系统与后训练框架

| Project | Description | Stars | Language |
|---|---|---:|---|
| [verl-project/verl](https://github.com/verl-project/verl) | verl/HybridFlow: A Flexible and Efficient RL Post-Training Framework  | 23240 | Python |
| [huggingface/trl](https://github.com/huggingface/trl) | Train transformer language models with reinforcement learning. | 19195 | Python |
| [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | High-quality single file implementation of Deep Reinforcement Learning algorithms with research-friendly features (PPO, DQN, C51, DDPG, TD3, SAC, PPG) | 10350 | Python |
| [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | An Easy-to-use, Scalable and High-performance Agentic RL Framework based on Ray (PPO & DAPO & REINFORCE++ &  VLM & TIS & vLLM & Ray & Async  RL) | 9968 | Python |
| [THUDM/slime](https://github.com/THUDM/slime) | slime is an LLM post-training framework for RL Scaling. | 8343 | Python |
| [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) | Solve Visual Understanding with Reinforced VLMs | 6016 | Python |
| [areal-project/AReaL](https://github.com/areal-project/AReaL) | The RL Bridge for LLM-based Agent Applications. Made Simple & Flexible. | 5708 | Python |
| [hiyouga/EasyR1](https://github.com/hiyouga/EasyR1) | EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework based on veRL | 5139 | Python |
| [alibaba/ROLL](https://github.com/alibaba/ROLL) | An Efficient and User-Friendly Scaling Library for Reinforcement Learning with Large Language Models | 3378 | Python |
| [radixark/miles](https://github.com/radixark/miles) | Miles is an enterprise-facing reinforcement learning framework for LLM and VLM post-training, forked from and co-evolving with slime. | 2304 | Python |
| [ChenmienTan/RL2](https://github.com/ChenmienTan/RL2) |  | 1310 | Python |
| [NVIDIA-NeMo/labs-molt](https://github.com/NVIDIA-NeMo/labs-molt) | An agentic-first RL framework for research (9k lines). | 995 | Python |
| [Accio-Lab/Dressage](https://github.com/Accio-Lab/Dressage) | Scalable RL for Any Agent and Sandbox. | 656 | Python |
| [MiroMindAI/MiroRL](https://github.com/MiroMindAI/MiroRL) | MiroRL is  an MCP-first reinforcement learning framework for deep research agent. | 249 | Python |
| [Tencent/Wechat-YATT](https://github.com/Tencent/Wechat-YATT) |  | 82 | Python |
| [DeepLink-org/LightRFT](https://github.com/DeepLink-org/LightRFT) | LightRFT (Light Reinforcement Fine-Tuning) is an advanced reinforcement learning fine-tuning framework designed for Large Language Models (LLMs) and Vision-Lang | 19 | Python |

## <span id="vault-inference-frameworks">Domain-推理框架 (14)</span>

推理引擎、服务、调度、批处理、KV Cache 与推理优化

### <span id="topic-inference-engines-serving">Inference-Engines-Serving (13)</span>

vLLM、SGLang、TGI、TensorRT-LLM、llama.cpp 与推理服务架构

| Project | Description | Stars | Language |
|---|---|---:|---|
| [ollama/ollama](https://github.com/ollama/ollama) | Get up and running with Kimi-K2.6, GLM-5.2, MiniMax, DeepSeek, gpt-oss, Qwen, Gemma and other models. | 179923 | Go |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | LLM inference in C/C++ | 126695 | C++ |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | A high-throughput and memory-efficient inference and serving engine for LLMs | 90720 | Python |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | SGLang is a high-performance serving framework for large language models and multimodal models. | 33068 | Python |
| [liguodongiot/llm-action](https://github.com/liguodongiot/llm-action) | 本项目旨在分享大模型相关技术原理以及实战经验（大模型工程化、大模型应用落地） | 24992 | HTML |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | Nano vLLM | 15269 | Python |
| [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) | A compact implementation of SGLang, designed to demystify the complexities of modern LLM serving systems. | 4929 | Python |
| [CalvinXKY/InfraTech](https://github.com/CalvinXKY/InfraTech) | 分享AI Infra知识&代码练习：PyTorch、vLLM/SGLang、slime/vime框架入门⚡️、性能加速🚀、大模型基础🧠、AI软硬件🔧等 | 3764 | Jupyter Notebook |
| [naklecha/simple-llm](https://github.com/naklecha/simple-llm) | ~950 line, minimal, extensible LLM inference engine built from scratch. | 481 | Python |
| [slwang-ustc/nano-vllm-v1](https://github.com/slwang-ustc/nano-vllm-v1) | Nano vLLM with vLLM v1's request scheduling strategy and chunked prefill | 97 | Python |
| [difey/nano-vllm-v1](https://github.com/difey/nano-vllm-v1) | Nano vLLM v1 engine | 15 | N/A |
| [cosmoliu2002/nano-vllm-triton](https://github.com/cosmoliu2002/nano-vllm-triton) | Nano vLLM Triton | 14 | Python |
| [RealJosephus/radix-turn-aware-nano-vllm](https://github.com/RealJosephus/radix-turn-aware-nano-vllm) | Radix Tree KV Cache with Turn-Aware Growth | 11 | Python |

### <span id="topic-inference-optimization">Inference-Optimization (1)</span>

推测解码、分布式推理、PD 分离、量化、性能诊断、压测与回归

| Project | Description | Stars | Language |
|---|---|---:|---|
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | A unified library of SOTA model optimization techniques like quantization, distillation, pruning, neural architecture search, speculative decoding, etc. It comp | 3678 | Python |

## <span id="vault-data-systems">Domain-数据系统 (8)</span>

数据清洗、标注、合成、数据集、评测数据与数据管道

### <span id="topic-data-systems">Data-Systems (8)</span>

数据获取、清洗、去重、合成、标注、数据集、评测集、向量检索与索引

| Project | Description | Stars | Language |
|---|---|---:|---|
| [opendatalab/MinerU](https://github.com/opendatalab/MinerU) | Transforms complex documents like PDFs and Office docs into LLM-ready markdown/JSON for your Agentic workflows. | 78968 | Python |
| [awesomedata/awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets) | A topic-centric list of HQ open datasets. | 78759 | N/A |
| [NanmiCoder/MediaCrawler](https://github.com/NanmiCoder/MediaCrawler) | 小红书笔记 \| 评论爬虫、抖音视频 \| 评论爬虫、快手视频 \| 评论爬虫、B 站视频 ｜ 评论爬虫、微博帖子 ｜ 评论爬虫、百度贴吧帖子 ｜ 百度贴吧评论回复爬虫  \| 知乎问答文章｜评论爬虫 | 64288 | Python |
| [wechat-article/wechat-article-exporter](https://github.com/wechat-article/wechat-article-exporter) | 一款在线的 微信公众号文章批量下载 工具，支持导出阅读量与评论数据，无需搭建任何环境，可通过 在线网站 使用，支持 docker 私有化部署和 Cloudflare 部署。  支持下载各种文件格式，其中 HTML 格式可100%还原文章排版与样式。 | 12833 | TypeScript |
| [cv-cat/Spider_XHS](https://github.com/cv-cat/Spider_XHS) | 小红书爬虫数据采集，小红书全域运营解决方案 | 7488 | Python |
| [cwjcw/xhs_douyin_content](https://github.com/cwjcw/xhs_douyin_content) | 自动抓取抖音和小红书创作者中心里的每条笔记/视频的播放，完播，点击，播放时长，点赞，分享，评论，收藏，主页访问，粉丝增量等互动数据 | 302 | Python |
| [yhslgg-arch/url-reader](https://github.com/yhslgg-arch/url-reader) | 智能网页内容读取器 - Claude Code Skill，支持微信公众号、小红书、今日头条等中国主流平台 | 188 | Python |
| [OpenDCAI/Flash-MinerU](https://github.com/OpenDCAI/Flash-MinerU) | Ray-powered accelerator for MinerU, turning PDF → Markdown into a scalable, cluster-ready data infrastructure. 基于 Ray 的 MinerU 加速层，将 PDF → Markdown 构建为可扩展、面向集群的 | 70 | Python |

## <span id="vault-model-behavior-control">Domain-模型行为与控制 (20)</span>

模型架构、Attention、MoE、VLM、能力、对齐与可控性

### <span id="topic-model-architecture-components">Model-Architecture-Components (6)</span>

LLM 模型家族、Dense/MoE 架构、Attention、RoPE、Norm 与核心组件

| Project | Description | Stars | Language |
|---|---|---:|---|
| [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) | Kronos: A Foundation Model for the Language of Financial Markets | 38349 | Python |
| [wgwang/awesome-LLMs-In-China](https://github.com/wgwang/awesome-LLMs-In-China) | 中国大模型 | 6472 | N/A |
| [Duxiaoman-DI/XuanYuan](https://github.com/Duxiaoman-DI/XuanYuan) | 轩辕：度小满中文金融对话大模型 | 1326 | Python |
| [wdndev/tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh) | 从零实现一个小参数量中文大语言模型。 | 1080 | Python |
| [wdndev/llama3-from-scratch-zh](https://github.com/wdndev/llama3-from-scratch-zh) | 从零实现一个 llama3 中文版 | 1055 | Jupyter Notebook |
| [Emericen/tiny-qwen](https://github.com/Emericen/tiny-qwen) | A minimal PyTorch re-implementation of Qwen 3.8 | 443 | Python |

### <span id="topic-model-multimodal-vlm">Model-Multimodal-VLM (14)</span>

VLM、VLA、视觉、音频、视频、多模态生成与理解，不再拆分音频或游戏门类

| Project | Description | Stars | Language |
|---|---|---:|---|
| [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) | The most powerful and modular diffusion model GUI, api and backend with a graph/nodes interface. | 131085 | Python |
| [ATH-MaaS/Pixelle-Video](https://github.com/ATH-MaaS/Pixelle-Video) | 🚀 AI 全自动短视频引擎 \| AI Fully Automated Short Video Engine | 27629 | Python |
| [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | Qwen3-VL is the multimodal large language model series developed by Qwen team, Alibaba Cloud. | 19881 | Jupyter Notebook |
| [QuentinFuxa/WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | Real-time, local speech-to-text with streaming ASR, speaker diarization, translation, and OpenAI/Deepgram-compatible APIs. | 10986 | Python |
| [NVlabs/Sana](https://github.com/NVlabs/Sana) | SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer | 8918 | Python |
| [RanFeng/clipsketch-ai](https://github.com/RanFeng/clipsketch-ai) | 将视频瞬间转化为手绘故事 Turn Video Moments into Hand-Drawn Stories | 1841 | TypeScript |
| [MoonshotAI/Kimi-VL](https://github.com/MoonshotAI/Kimi-VL) | Kimi-VL: Mixture-of-Experts Vision-Language Model for Multimodal Reasoning, Long-Context Understanding, and Strong Agent Capabilities | 1226 | N/A |
| [TinyLLaVA/TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) | A Framework of Small-scale Large Multimodal Models | 1004 | Python |
| [hkproj/pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma) | Coding a Multimodal (Vision) Language Model from scratch in PyTorch with full explanation: https://www.youtube.com/watch?v=vAmKB7iPkWw | 631 | Python |
| [bytedance/tarsier](https://github.com/bytedance/tarsier) | Tarsier -- a family of large-scale video-language models, which is designed to generate high-quality video descriptions , together with good capability of gener | 548 | Python |
| [Renaissance-Mind/DrawAI](https://github.com/Renaissance-Mind/DrawAI) | Make raster image ediatble. 将生成图像（如GPT-Image-2或Nano Banana）或截图变为可编辑的格式，实现高质量PPT、论文图直出。 | 141 | Python |
| [TinyLoopX/RLLaVA](https://github.com/TinyLoopX/RLLaVA) | RLLaVA is a user-friendly framework for multi-modal RL research and optimized for resource-constrained teams. | 58 | Python |
| [forXuyx/Cinego](https://github.com/forXuyx/Cinego) | 🚀 轻量视频🎥 大模型🤖 | 23 | Python |
| [Layjins/Spider](https://github.com/Layjins/Spider) | Code for paper "Spider: Any-to-Many Multimodal LLM" | 16 | Python |

## <span id="vault-application-systems">Domain-应用系统 (43)</span>

RAG、MCP、工具调用、产品级 Agent 与模型应用工作流

### <span id="topic-app-agent-rag">App-Agent-RAG (35)</span>

Agentic Loop、规划、记忆、多 Agent、RAG、Research Agent 与工作流编排

| Project | Description | Stars | Language |
|---|---|---:|---|
| [open-webui/open-webui](https://github.com/open-webui/open-webui) | User-friendly AI Interface (Supports Ollama, OpenAI API, ...) | 150644 | Python |
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | The agent engineering platform. | 145463 | Python |
| [x1xhlol/system-prompts-and-models-of-ai-tools](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools) | FULL Augment Code, Claude Code, Cluely, CodeBuddy, Comet, Cursor, Devin AI, Junie, Kiro, Leap.new, Lovable, Manus, NotionAI, Orchids.app, Perplexity, Poke, Qode | 143284 | N/A |
| [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) | 100+ AI Agents, Agent Skills and RAG Apps - Free and Open Source. | 135648 | Python |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | AI agents running research on single-GPU nanochat training automatically | 95067 | Python |
| [lobehub/lobehub](https://github.com/lobehub/lobehub) | 🤯 LobeHub is your Chief Agent Operator, organizing your agents into 7×24 operations by hiring, scheduling, and reporting on your entire AI team. | 82150 | TypeScript |
| [datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents) | 📚 《从零开始构建智能体》——从零开始的智能体原理与实践教程 | 76175 | Python |
| [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) | Bash is all you need -  A nano claude code–like 「agent harness」, built from 0 to 1 | 75845 | Python |
| [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) | 🌟 The Multi-Agent Framework: First AI Software Company, Towards Natural Language Programming | 70157 | Python |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | The best ChatGPT that $100 can buy. | 57721 | Python |
| [microsoft/qlib](https://github.com/microsoft/qlib) | Qlib is an AI-oriented Quant investment platform that aims to use AI tech to empower Quant Research, from exploring ideas to implementing productions. Qlib supp | 48193 | Python |
| [danielmiessler/Fabric](https://github.com/danielmiessler/Fabric) | Fabric is an open-source framework for augmenting humans using AI. It provides a modular system for solving specific problems using a crowdsourced set of AI pro | 43716 | Go |
| [666ghj/BettaFish](https://github.com/666ghj/BettaFish) | 微舆：人人可用的多Agent舆情分析助手，打破信息茧房，还原舆情原貌，预测未来走向，辅助决策！从0实现，不依赖任何框架。 | 42128 | Python |
| [MadsLorentzen/ai-job-search](https://github.com/MadsLorentzen/ai-job-search) | The job search that runs on your machine. AI job application framework built on Claude Code: evaluate postings, tailor CVs, write cover letters, prep interviews | 39869 | Python |
| [chatchat-space/Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) | Langchain-Chatchat（原Langchain-ChatGLM）基于 Langchain 与 ChatGLM, Qwen 与 Llama 等语言模型的 RAG 与 Agent 应用 \| Langchain-Chatchat (formerly langchain-ChatGLM), local knowl | 38608 | Python |
| [lfnovo/open-notebook](https://github.com/lfnovo/open-notebook) | An Open Source implementation of Notebook LM with more flexibility and features | 38067 | TypeScript |
| [continuedev/continue](https://github.com/continuedev/continue) | open-source coding agent | 35728 | TypeScript |
| [JCodesMore/ai-website-cloner-template](https://github.com/JCodesMore/ai-website-cloner-template) | Clone any website with one command using AI coding agents | 33581 | JavaScript |
| [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) | Tongyi Deep Research, the Leading Open-source Deep Research Agent | 19901 | Python |
| [tradecatlabs/vibe-coding-cn](https://github.com/tradecatlabs/vibe-coding-cn) | Vibe Coding 从入门到精通教程｜AI 结对编程工作流｜Prompt、Skill、Workflow、上下文管理、codex实战指南 | 16068 | Python |
| [dataelement/bisheng](https://github.com/dataelement/bisheng) | BISHENG is an open LLM devops platform for next generation Enterprise AI applications. Powerful and comprehensive features include: GenAI workflow, RAG, Agent,  | 11920 | Python |
| [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch) | Claude Autoresearch Skill — Autonomous goal-directed iteration for Claude Code. Inspired by Karpathy's autoresearch. Modify → Verify → Keep/Discard → Repeat for | 5975 | Shell |
| [shareAI-lab/Kode-CLI](https://github.com/shareAI-lab/Kode-CLI) | Kode CLI — Design for post-human workflows. One unit agent for every human & computer task. | 5211 | TypeScript |
| [mangiucugna/json_repair](https://github.com/mangiucugna/json_repair) | Repair malformed JSON from LLMs, APIs, logs, and user input in Python. | 5087 | Python |
| [Anning01/AIMedia](https://github.com/Anning01/AIMedia) | AIMedia 是一款自动抓取热点，AI创作文章，自动发布的集成软件。支持头条，小红书，公众号等 | 2445 | Python |
| [RAIT-09/obsidian-agent-client](https://github.com/RAIT-09/obsidian-agent-client) | Bring AI agents into Obsidian via Agent Client Protocol (ACP), such as Claude Code, Codex and Gemini CLI. | 2382 | TypeScript |
| [leo-lilinxiao/codex-autoresearch](https://github.com/leo-lilinxiao/codex-autoresearch) | Codex Autoresearch Skill — A self-directed iterative system for Codex that continuously cycles through: modify, verify, retain or discard, and repeat indefinite | 2332 | Python |
| [MeetKai/functionary](https://github.com/MeetKai/functionary) | Chat language model that can use tools and interpret the results | 1595 | Python |
| [study8677/repobrain](https://github.com/study8677/repobrain) | 🧠 RepoBrain (formerly Antigravity) — Give your repo a brain. ChatGPT for your codebase: works in Claude Code, Cursor, Codex, Windsurf & more. | 1322 | Python |
| [china-qijizhifeng/agentic-harness-engineering](https://github.com/china-qijizhifeng/agentic-harness-engineering) | Official AHE code — Agentic Harness Engineering: observability-driven automatic evolution of coding-agent harnesses (concurrent w/ meta-harness). NexAU-AHE reac | 865 | Python |
| [PolarSeeker/OpenSeeker](https://github.com/PolarSeeker/OpenSeeker) | OpenSeeker: A search agent with open-source data and models | 773 | Python |
| [microsoft/Orchard](https://github.com/microsoft/Orchard) | Orchard: An Open-Source Agentic Modeling Framework | 499 | N/A |
| [vibesurf-ai/VibeSurf](https://github.com/vibesurf-ai/VibeSurf) | A powerful browser assistant for vibe surfing 一个开源的AI浏览器智能助手 | 482 | Python |
| [NLPJCL/SearchAgent-Zero](https://github.com/NLPJCL/SearchAgent-Zero) | SearchAgent-Zero: A Scalable Multi-Turn Search Agent RL Framework | 162 | Python |
| [chmod777john/github-hunter](https://github.com/chmod777john/github-hunter) | AI 发掘潜在的爆火项目 | 81 | Jupyter Notebook |

### <span id="topic-app-tools-mcp">App-Tools-MCP (8)</span>

工具调用、Function Calling、MCP Server、Schema、权限与安全边界

| Project | Description | Stars | Language |
|---|---|---:|---|
| [PDFMathTranslate/PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate) | [EMNLP 2025 Demo] PDF scientific paper translation with preserved formats - 基于 AI 完整保留排版的 PDF 文档全文双语翻译，支持 Google/DeepL/Ollama/OpenAI 等服务，提供 CLI/GUI/MCP/Docker/Z | 36589 | Python |
| [xpzouying/xiaohongshu-mcp](https://github.com/xpzouying/xiaohongshu-mcp) | MCP for xiaohongshu.com | 15594 | Go |
| [idosal/git-mcp](https://github.com/idosal/git-mcp) | Put an end to code hallucinations! GitMCP is a free, open-source, remote MCP server for any GitHub project | 8364 | TypeScript |
| [agent-infra/sandbox](https://github.com/agent-infra/sandbox) | All-in-One Sandbox for AI Agents that combines Browser, Shell, File, MCP and VSCode Server in a single Docker container. | 5829 | Python |
| [iFurySt/RedNote-MCP](https://github.com/iFurySt/RedNote-MCP) | 🚀MCP server for accessing RedNote(XiaoHongShu, xhs). | 1101 | TypeScript |
| [instavm/open-skills](https://github.com/instavm/open-skills) | OpenSkills: Run Claude Skills Locally using any LLM | 449 | Python |
| [AI-QL/chat-ui](https://github.com/AI-QL/chat-ui) | Single-File AI Chatbot UI with Multimodal & MCP Support: An All-in-One HTML File for a Streamlined Chatbot Conversational Interface | 95 | HTML |
| [jswortz/antigravity-claude-skills](https://github.com/jswortz/antigravity-claude-skills) |  | 8 | Python |

## <span id="vault-engineering-practice">Pillar-工程与实践 (11)</span>

部署、Profiling、MLOps、成本、稳定性、治理与工程案例

### <span id="topic-engineering-practice">Engineering-Practice (11)</span>

容器、云原生、数据库、部署、Profiling、MLOps、稳定性、安全与工程交付

| Project | Description | Stars | Language |
|---|---|---:|---|
| [DigitalPlatDev/FreeDomain](https://github.com/DigitalPlatDev/FreeDomain) | Free domain registration and practical DNS learning resources for everyone. | 197094 | Markdown |
| [slidevjs/slidev](https://github.com/slidevjs/slidev) | Presentation Slides for Developers | 48381 | TypeScript |
| [vnpy/vnpy](https://github.com/vnpy/vnpy) | 基于Python的开源量化交易平台开发框架 | 45055 | Python |
| [DayuanJiang/next-ai-draw-io](https://github.com/DayuanJiang/next-ai-draw-io) | A next.js web application that integrates AI capabilities with draw.io diagrams. This app allows you to create, modify, and enhance diagrams through natural lan | 35512 | TypeScript |
| [komi-store/komi-store](https://github.com/komi-store/komi-store) | 🩵 A free, open-source app store for developers' releases on GitHub, Codeberg & Forgejo — browse, discover, and install apps with one click. Formerly GitHub Stor | 18154 | Kotlin |
| [alshedivat/al-folio](https://github.com/alshedivat/al-folio) | A beautiful, simple, clean, and responsive Jekyll theme for academics | 16083 | HTML |
| [jnsahaj/tweakcn](https://github.com/jnsahaj/tweakcn) | A visual no-code theme editor for shadcn/ui components | 10339 | TypeScript |
| [gamosoft/NoteDiscovery](https://github.com/gamosoft/NoteDiscovery) | Your Self-Hosted Knowledge Base | 2776 | JavaScript |
| [dqbd/tiktokenizer](https://github.com/dqbd/tiktokenizer) | Online playground for OpenAPI tokenizers | 1680 | TypeScript |
| [hezhizheng/go-wxpush](https://github.com/hezhizheng/go-wxpush) | 极简且免费的微信消息推送服务 (基于golang) | 1658 | Go |
| [YangWang92/remote-perfetto](https://github.com/YangWang92/remote-perfetto) |  | 16 | Python |

## <span id="vault-workflow-toolchain">Pillar-工作流与工具链 (33)</span>

AI Coding、IDE、Git、CLI、自动化与知识管理工具

### <span id="topic-workflow-ai-coding">Workflow-AI-Coding (8)</span>

Code Agent、AI Coding、Skills、Harness、IDE 与人机协作开发流程

| Project | Description | Stars | Language |
|---|---|---:|---|
| [rtk-ai/rtk](https://github.com/rtk-ai/rtk) | CLI proxy that reduces LLM token consumption by 60-90% on common dev commands. Single Rust binary, zero dependencies | 78233 | Rust |
| [google-labs-code/design.md](https://github.com/google-labs-code/design.md) | A format specification for describing a visual identity to coding agents. DESIGN.md gives agents a persistent, structured understanding of a design system. | 27654 | TypeScript |
| [chenhg5/cc-connect](https://github.com/chenhg5/cc-connect) | Bridge local AI coding agents (Claude Code, Cursor, Gemini CLI, Codex) to messaging platforms (Feishu/Lark, DingTalk, Slack, Telegram, Discord, LINE, WeChat Wor | 15309 | Go |
| [1rgs/nanocode](https://github.com/1rgs/nanocode) | Minimal Claude Code alternative. Single Python file, zero dependencies, ~250 lines. | 2560 | Python |
| [CloudAI-X/claude-workflow-v2](https://github.com/CloudAI-X/claude-workflow-v2) | Universal Claude Code workflow plugin with agents, skills, hooks, and commands | 1413 | Python |
| [jokemon/antiPM-Workflow](https://github.com/jokemon/antiPM-Workflow) | A collection of Antigravity workflows for Product Managers. (产品经理专属的 Antigravity 工作流合集) | 29 | N/A |
| [ChenZiHong-Gavin/weekly-vibe-coding](https://github.com/ChenZiHong-Gavin/weekly-vibe-coding) | 用提示词实现一百个idea | 24 | TypeScript |
| [crazyCrabs/opencode_goal](https://github.com/crazyCrabs/opencode_goal) | goal command of opencode | 3 | TypeScript |

### <span id="topic-workflow-knowledge-tools">Workflow-Knowledge-Tools (18)</span>

Obsidian、研究工作流、自动化、Git、CLI、编程语言与内容处理工具

| Project | Description | Stars | Language |
|---|---|---:|---|
| [tw93/Mole](https://github.com/tw93/Mole) | 🐹 Clean, uninstall, analyze, optimize, and monitor your Mac. Free open-source CLI, plus a native Mac app. | 65726 | Shell |
| [psf/black](https://github.com/psf/black) | The uncompromising Python code formatter | 41830 | Python |
| [lbjlaq/Antigravity-Manager](https://github.com/lbjlaq/Antigravity-Manager) | Professional Antigravity Account Manager & Switcher. One-click seamless account switching for Antigravity Tools. Built with Tauri v2 + React (Rust).专业的 Antigrav | 30870 | Rust |
| [ourongxing/newsnow](https://github.com/ourongxing/newsnow) | Elegant reading of real-time and hottest news | 21552 | TypeScript |
| [githubnext/monaspace](https://github.com/githubnext/monaspace) | An innovative superfamily of fonts for code | 19603 | Shell |
| [rendercv/rendercv](https://github.com/rendercv/rendercv) | Resume builder for academics and engineers | 17471 | Python |
| [iamgio/quarkdown](https://github.com/iamgio/quarkdown) | 🪐 Markdown with superpowers: from ideas to papers, presentations, websites, books, and knowledge bases. | 16059 | Kotlin |
| [VERT-sh/VERT](https://github.com/VERT-sh/VERT) | The next-generation file converter. Open source, fully local* and free forever. | 15486 | Svelte |
| [dreammis/social-auto-upload](https://github.com/dreammis/social-auto-upload) | 自动化上传视频到社交媒体：抖音、小红书、视频号、tiktok、youtube、bilibili | 14721 | Python |
| [funstory-ai/BabelDOC](https://github.com/funstory-ai/BabelDOC) | Yet Another Document Translator | 9454 | Python |
| [Diorser/LiteMonitor](https://github.com/Diorser/LiteMonitor) | 一款轻量级、高度可定制的 Windows桌面和任务栏硬件性能监控工具，支持监测 CPU、GPU、内存、磁盘、网速、FPS 计数、插件扩展及内存清理。A lightweight, customizable hardware monitor for the Windows desktop & taskbar. Featur | 6213 | C# |
| [axtonliu/axton-obsidian-visual-skills](https://github.com/axtonliu/axton-obsidian-visual-skills) | Visual Skills Pack for Obsidian: generate Canvas, Excalidraw, and Mermaid diagrams from text with Claude Code | 3556 | N/A |
| [op7418/Youtube-clipper-skill](https://github.com/op7418/Youtube-clipper-skill) |  | 2171 | Python |
| [Lulzx/tinypdf](https://github.com/Lulzx/tinypdf) | Minimal PDF creation library. <400 LOC, zero dependencies, makes real PDFs. | 1912 | TypeScript |
| [OpenGithubs/github-daily-rank](https://github.com/OpenGithubs/github-daily-rank) | Github开源项目:每天📈飙升榜 top10,每天早上8:30更新 | 1197 | N/A |
| [PKM-er/awesome-obsidian-zh](https://github.com/PKM-er/awesome-obsidian-zh) | Obsidian 优秀中文插件、主题与资源 | 751 | N/A |
| [zimya/zhihu_obsidian](https://github.com/zimya/zhihu_obsidian) | Zhihu on Obsidian \| 知乎 Obsidian 插件 | 272 | TypeScript |
| [simwy/Side-Markdown](https://github.com/simwy/Side-Markdown) | A sleek edge-mounted Markdown editor—accessible yet non-intrusive. Full support for headings, lists & code blocks, real-time rendering, and seamless full-screen | 4 | TypeScript |

### <span id="topic-personal-repositories">Personal-Repositories (7)</span>

nideyongbao 名下的个人代码、实验、训练、部署、环境与数据资产

| Project | Description | Stars | Language |
|---|---|---:|---|
| [nideyongbao/my-depoly](https://github.com/nideyongbao/my-depoly) |  | 1 | Python |
| [nideyongbao/my-train](https://github.com/nideyongbao/my-train) |  | 1 | Python |
| [nideyongbao/nvidia-gpu-baseline](https://github.com/nideyongbao/nvidia-gpu-baseline) |  | 1 | Python |
| [nideyongbao/llm-lab-v3-code](https://github.com/nideyongbao/llm-lab-v3-code) |  | 1 | Python |
| [nideyongbao/llm-lab-v3-datasets](https://github.com/nideyongbao/llm-lab-v3-datasets) |  | 1 | Python |
| [nideyongbao/llm-lab-v3-env](https://github.com/nideyongbao/llm-lab-v3-env) |  | 1 | Shell |
| [nideyongbao/LightRFT](https://github.com/nideyongbao/LightRFT) | LightRFT (Light Reinforcement Fine-Tuning) is an advanced reinforcement learning fine-tuning framework designed for Large Language Models (LLMs) and Vision-Lang | 1 | N/A |

## <span id="vault-theory-reproduction">Pillar-理论与复现 (20)</span>

论文、理论、数学、复现实验与 Benchmark 解读

### <span id="topic-theory-reproduction">Theory-Reproduction (20)</span>

数学与优化理论、论文实现、复现实验、Benchmark、教程与系统化资料

| Project | Description | Stars | Language |
|---|---|---:|---|
| [harvard-edge/cs249r_book](https://github.com/harvard-edge/cs249r_book) | Machine Learning Systems | 28097 | Python |
| [WangRongsheng/awesome-LLM-resources](https://github.com/WangRongsheng/awesome-LLM-resources) | 🧑‍🚀 全世界最好的LLM资料总结（多模态生成、Agent、辅助编程、AI审稿、数据处理、模型训练、模型推理、o1 模型、MCP、小语言模型、视觉语言模型） \| Summary of the world's best LLM resources.  | 8892 | N/A |
| [itcharge/AlgoNote](https://github.com/itcharge/AlgoNote) | ⛽️「算法通关手册」：从零开始的「算法与数据结构」学习教程，200 道「算法面试热门题目」，1000+ 道「LeetCode 题目解析」，持续更新中！ | 7797 | Python |
| [zhaochenyang20/Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial) | My learning notes for ML SYS. | 7057 | HTML |
| [xlite-dev/Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) | 📚A curated list of Awesome LLM/VLM Inference Papers with Codes: Flash-Attention, Paged-Attention, WINT8/4, Parallelism, etc.🎉 | 5479 | Python |
| [changyeyu/LLM-RL-Visualized](https://github.com/changyeyu/LLM-RL-Visualized) | 🌟100+ 原创 LLM / RL 原理图📚，《大模型算法》作者巨献！💥（100+  LLM/RL Algorithm Maps ） | 4837 | Python |
| [walkinglabs/hands-on-modern-rl](https://github.com/walkinglabs/hands-on-modern-rl) | 🚀 An open-source, hands-on curriculum bridging the gap from basic RL concepts to LLM alignment, RLVR, and advanced Agentic systems.  | 4180 | Python |
| [ginobefun/BestBlogs](https://github.com/ginobefun/BestBlogs) | bestblogs.dev - 汇集顶级编程、人工智能、产品、科技文章，大语言模型摘要评分辅助阅读，探索编程和技术未来 | 4007 | TypeScript |
| [wyf3/llm_related](https://github.com/wyf3/llm_related) | 复现大模型相关算法及一些学习记录 | 3509 | Python |
| [CoinCheung/pytorch-loss](https://github.com/CoinCheung/pytorch-loss) | label-smooth, amsoftmax, partial-fc, focal-loss, triplet-loss, lovasz-softmax. Maybe useful  | 2252 | Python |
| [caomaolufei/AIInfraGuide](https://github.com/caomaolufei/AIInfraGuide) | AI Infra 全栈从0入门学习资料：https://caomaolufei.github.io/AIInfraGuide/ | 1924 | Astro |
| [thinkwee/AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL) | Awesome List for Agentic RL | 1830 | HTML |
| [jinbooooom/ai-infra-hpc](https://github.com/jinbooooom/ai-infra-hpc) | hpc 教程，包含集合通信(mpi、nccl)、cuda 编程、向量化 SIMD、RDMA 通信等 | 690 | Cuda |
| [datawhalechina/deep-learning-notes](https://github.com/datawhalechina/deep-learning-notes) | Personal deep learning study notes and tutorial-style notebooks | 663 | Python |
| [firechecking/CleanTransformer](https://github.com/firechecking/CleanTransformer) | an implementation of transformer, bert, gpt, and diffusion models for learning purposes | 158 | Python |
| [ChinmayK0607/heiretsu](https://github.com/ChinmayK0607/heiretsu) | minimal pytorch 4D parallelism | 73 | Python |
| [KylinC/Awesome-Awesome-LLM](https://github.com/KylinC/Awesome-Awesome-LLM) | awesome LLM papers！ 🚀 🚀 🚀 | 49 | N/A |
| [APRIL-AIGC/awesome-optimizer](https://github.com/APRIL-AIGC/awesome-optimizer) | Evolution of Optimization Methods: Algorithms, Scenarios, and Evaluations | 36 | Python |
| [jqlong17/attnres-toy-jupyter](https://github.com/jqlong17/attnres-toy-jupyter) | A beginner-friendly Jupyter toy reproduction of MoonshotAI Attention Residuals with Chinese explanations, exported figures, and an executable notebook. | 6 | Jupyter Notebook |
| [hanfang/chatgpt-usage-taxonomies](https://github.com/hanfang/chatgpt-usage-taxonomies) | Taxonomies and classification prompts from the 'How People Use ChatGPT' research paper (NBER Working Paper No. 34255) | 4 | N/A |

## <span id="vault-other">其他 (3)</span>

读完仍无法归入现有 6+3 分类的仓库

### <span id="topic-other">Other (3)</span>

不符合当前 AI 系统研究主题且无法可靠归入现有子类

| Project | Description | Stars | Language |
|---|---|---:|---|
| [TapXWorld/ChinaTextbook](https://github.com/TapXWorld/ChinaTextbook) | 所有小初高、大学PDF教材。 | 81259 | Roff |
| [mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) |  | 214 | Python |
| [codfish-zz/cn-trader](https://github.com/codfish-zz/cn-trader) | Python back testing system for trading strategies, based on backtrader and AkShare, customized for China market. | 30 | Python |
