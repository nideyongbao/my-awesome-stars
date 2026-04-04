# 🌟 My Awesome AI Stars

> 🤖 自动生成于 GitHub Actions, Powered by LLM.

## 目录
- [AI-Sys-Framework (深度学习框架底座, PyTorch, TensorFlow, JAX, MXNet) (3)](#ai-sys-framework)
- [AI-Sys-Kernel (高性能算子与底层优化, FlashAttention, CUTLASS, Triton) (13)](#ai-sys-kernel)
- [AI-Sys-Training (分布式训练框架, DeepSpeed, Megatron, FSDP, Horovod) (23)](#ai-sys-training)
- [AI-Sys-FineTuning (轻量微调, LoRA, PEFT, QLoRA, Unsloth, Adapter) (10)](#ai-sys-finetuning)
- [AI-Sys-RLHF (后训练对齐, RLHF, PPO, DPO, GRPO, TRL, OpenRLHF) (14)](#ai-sys-rlhf)
- [AI-Sys-Cluster (集群调度与编排, Kubernetes, Ray, Slurm, Skypilot) (1)](#ai-sys-cluster)
- [AI-Data-Dataset (开源数据集, HuggingFace-Datasets, FineWeb, CommonCrawl) (1)](#ai-data-dataset)
- [AI-Data-Crawl (网页抓取与爬虫, Crawlee, Scrapy, Firecrawl) (5)](#ai-data-crawl)
- [AI-Sys-Inference (推理引擎与后端, vLLM, TGI, TensorRT-LLM, llama.cpp, SGLang) (13)](#ai-sys-inference)
- [AI-Algo-LLM (语言模型架构, Llama, Qwen, Mistral, DeepSeek, GLM) (6)](#ai-algo-llm)
- [AI-Algo-Multi (多模态与新架构, CLIP, Mamba, MoE, LLaVA, VLM) (7)](#ai-algo-multi)
- [AI-Algo-Vision (计算机视觉与生成, Stable Diffusion, YOLO, SAM, OpenCV) (5)](#ai-algo-vision)
- [AI-Algo-Audio (语音识别与合成, Whisper, TTS, ASR, Bark) (1)](#ai-algo-audio)
- [AI-Algo-Game (游戏AI与仿真, Unity ML-Agents, Gymnasium, PettingZoo) (1)](#ai-algo-game)
- [AI-App-Framework (应用编排框架, Dify, Flowise, Langflow, LangGraph) (4)](#ai-app-framework)
- [AI-App-RAG (检索增强生成, LangChain, LlamaIndex, Haystack) (6)](#ai-app-rag)
- [AI-App-Agent (智能体, 规划与记忆, AutoGPT, MetaGPT, CrewAI) (18)](#ai-app-agent)
- [AI-App-MCP (Model Context Protocol, MCP Server) (8)](#ai-app-mcp)
- [AI-Algo-Theory (纯理论代码, 论文复现, 数学库, NumPy, SciPy) (1)](#ai-algo-theory)
- [Research-Paper (论文代码复现, Arxiv, PapersWithCode) (4)](#research-paper)
- [Dev-Web-FullStack (现代Web开发, Next.js, React, Vue, FastAPI, Django) (9)](#dev-web-fullstack)
- [Dev-Infra-Cloud (云原生与容器, Docker, Kubernetes, Terraform, Pulumi) (1)](#dev-infra-cloud)
- [Dev-Lang-Core (编程语言核心资源, Rust, Python, Go, C++) (1)](#dev-lang-core)
- [AI-App-Coding (AI编程助手, Cursor, Copilot, Aider, Continue) (3)](#ai-app-coding)
- [Tools-Efficiency (生产力与终端工具, Oh-My-Zsh, Raycast, Obsidian, Neovim) (13)](#tools-efficiency)
- [Tools-Media (图像视频处理工具, FFmpeg, ImageMagick, yt-dlp) (3)](#tools-media)
- [CS-Education (教程与面试, 系统设计, LeetCode, 学习路线图) (9)](#cs-education)
- [Uncategorized (无法分类) (2)](#uncategorized)

---
## <span id='ai-sys-framework'>AI-Sys-Framework (深度学习框架底座, PyTorch, TensorFlow, JAX, MXNet)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [huggingface/transformers](https://github.com/huggingface/transformers) | 🤗 Transformers: the model-definition framework for state-of-the-art machine learning models in text, | 158765 | Python |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | Tensors and Dynamic neural networks in Python with strong GPU acceleration | 98795 | Python |
| [tigert1998/mytorch](https://github.com/tigert1998/mytorch) | A toy Python DL training library with PyTorch like API | 38 | Python |

## <span id='ai-sys-kernel'>AI-Sys-Kernel (高性能算子与底层优化, FlashAttention, CUTLASS, Triton)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Fast and memory-efficient exact attention | 23119 | Python |
| [triton-lang/triton](https://github.com/triton-lang/triton) | Development repository for the Triton language and compiler | 18840 | MLIR |
| [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | FlashMLA: Efficient Multi-head Latent Attention Kernels | 12551 | C++ |
| [facebookresearch/xformers](https://github.com/facebookresearch/xformers) | Hackable and optimized Transformers building blocks, supporting a composable construction. | 10403 | Python |
| [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | Efficient Triton Kernels for LLM Training | 6258 | Python |
| [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | 🚀 Efficient implementations of state-of-the-art linear attention models | 4803 | Python |
| [stepfun-ai/Step-3.5-Flash](https://github.com/stepfun-ai/Step-3.5-Flash) | Fast, Sharp & Reliable Agentic Intelligence | 1975 | C++ |
| [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | FP16xINT4 LLM inference kernel that can achieve near-ideal ~4x speedups up to medium batchsizes of 1 | 1048 | Python |
| [RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel) | Autoresearch for GPU kernels. Give it any PyTorch model, go to sleep, wake up to optimized Triton ke | 921 | Python |
| [hustvl/MoDA](https://github.com/hustvl/MoDA) | An hardware-aware Efficient Implementation for "Mixture-of-Depths Attention". | 152 | Python |
| [DefTruth/CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes) | 📚200+ Tensor/CUDA Cores Kernels, ⚡️flash-attn-mma, ⚡️hgemm with WMMA, MMA and CuTe (98%~100% TFLOPS  | 74 | Cuda |
| [jt-zhang/Sparse_Attention_API](https://github.com/jt-zhang/Sparse_Attention_API) |  | 66 | Python |
| [liangyuwang/MetaProfiler](https://github.com/liangyuwang/MetaProfiler) | MetaProfiler is a lightweight, structure-agnostic operator-level profiler for PyTorch models that le | 2 | Python |

## <span id='ai-sys-training'>AI-Sys-Training (分布式训练框架, DeepSpeed, Megatron, FSDP, Horovod)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [jingyaogong/minimind](https://github.com/jingyaogong/minimind) | 🚀🚀 「大模型」2小时完全从0训练26M的小参数GPT！🌏 Train a 26M-parameter GPT from scratch in just 2h! | 45541 | Python |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Ongoing research training transformer models at scale | 15907 | Python |
| [deepspeedai/DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples) | Example models using DeepSpeed | 6813 | Python |
| [pytorch/torchtitan](https://github.com/pytorch/torchtitan) | A PyTorch native platform for training generative AI models | 5207 | Python |
| [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | NanoGPT (124M) in 3 minutes | 5051 | Python |
| [PKU-Alignment/align-anything](https://github.com/PKU-Alignment/align-anything) | Align Anything: Training All-modality Model with Feedback | 4639 | Python |
| [huggingface/nanotron](https://github.com/huggingface/nanotron) | Minimalistic large language model 3D-parallelism training | 2632 | Python |
| [qibin0506/Cortex](https://github.com/qibin0506/Cortex) | 个人构建MoE大模型：从预训练到DPO的完整实践 | 2567 | Python |
| [Open-Reasoner-Zero/Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) | Official Repo for Open-Reasoner-Zero | 2087 | Python |
| [stepfun-ai/SteptronOss](https://github.com/stepfun-ai/SteptronOss) | A lightweight, AI-native training framework for large language models. Designed for fast iteration,  | 526 | Python |
| [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) | Pytorch Distributed native training library for LLMs/VLMs with OOTB Hugging Face support | 406 | Python |
| [Victorwz/Open-Qwen2VL](https://github.com/Victorwz/Open-Qwen2VL) | [COLM 2025] Open-Qwen2VL: Compute-Efficient Pre-Training of Fully-Open Multimodal LLMs on Academic R | 312 | Python |
| [liangyuwang/zo2](https://github.com/liangyuwang/zo2) | ZO2 (Zeroth-Order Offloading): Full Parameter Fine-Tuning 175B LLMs with 18GB GPU Memory [COLM2025] | 203 | Python |
| [OpenDCAI/DataFlex](https://github.com/OpenDCAI/DataFlex) | DataFlex is a data-centric training framework that enhances model performance by either selecting th | 144 | Python |
| [MiroMindAI/MiroTrain](https://github.com/MiroMindAI/MiroTrain) | MiroTrain is an efficient and algorithm-first framework for post-training large agentic models.  | 140 | Python |
| [nex-agi/NexRL](https://github.com/nex-agi/NexRL) | NexRL is an ultra-loosely-coupled LLM post-training framework. | 104 | Python |
| [liangyuwang/Tiny-FSDP](https://github.com/liangyuwang/Tiny-FSDP) | Tiny-FSDP, a minimalistic re-implementation of the PyTorch FSDP | 101 | Python |
| [CoinCheung/gdGPT](https://github.com/CoinCheung/gdGPT) | Train llm (bloom, llama, baichuan2-7b, chatglm3-6b) with deepspeed pipeline mode. Faster than zero/z | 97 | Python |
| [wxhcore/bumblecore](https://github.com/wxhcore/bumblecore) | An LLM training framework built from the ground up, featuring a custom BumbleBee architecture and en | 64 | Python |
| [liangyuwang/Tiny-DeepSpeed](https://github.com/liangyuwang/Tiny-DeepSpeed) | Tiny-DeepSpeed, a minimalistic re-implementation of the DeepSpeed library | 50 | Python |
| [XU-YIJIE/hobo-llm-from-scratch](https://github.com/XU-YIJIE/hobo-llm-from-scratch) | From Llama to Deepseek, grpo/mtp implemented. With pt/sft/lora/qlora included | 30 | Python |
| [liangyuwang/Tiny-Megatron](https://github.com/liangyuwang/Tiny-Megatron) | Tiny-Megatron, a minimalistic re-implementation of the Megatron library | 23 | Python |
| [liangyuwang/Streaming-Dataloader](https://github.com/liangyuwang/Streaming-Dataloader) |  A memory-efficient streaming data loader designed for LLM pretraining under limited CPU and GPU mem | 3 | Python |

## <span id='ai-sys-finetuning'>AI-Sys-FineTuning (轻量微调, LoRA, PEFT, QLoRA, Unsloth, Adapter)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) | Unified Efficient Fine-Tuning of 100+ LLMs & VLMs (ACL 2024) | 69478 | Python |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | Fine-tuning & Reinforcement Learning for LLMs. 🦥 Train OpenAI gpt-oss, DeepSeek, Qwen, Llama, Gemma, | 59298 | Python |
| [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | Use PEFT or Full-parameter to CPT/SFT/DPO/GRPO 600+ LLMs (Qwen3.5, DeepSeek-R1, GLM-5, InternLM3, Ll | 13520 | Python |
| [imoneoi/openchat](https://github.com/imoneoi/openchat) | OpenChat: Advancing Open-source Language Models with Imperfect Data | 5477 | Python |
| [adonis-dym/memory_reduced_optimizer](https://github.com/adonis-dym/memory_reduced_optimizer) |  | 530 | Python |
| [AI-Study-Han/Zero-Qwen-VL](https://github.com/AI-Study-Han/Zero-Qwen-VL) | 训练一个对中文支持更好的LLaVA模型，并开源训练代码和数据。 | 82 | Python |
| [qibin0506/llm_trainer](https://github.com/qibin0506/llm_trainer) |  | 62 | Python |
| [0x0C001/OpenSFT](https://github.com/0x0C001/OpenSFT) |  | 46 | Python |
| [yafo-ai/y-trainer](https://github.com/yafo-ai/y-trainer) | y-trainerY-Trainer 是一个LLM模型微调训练框架。  📊 核心优势： 📉 精准对抗过拟合： 专门优化，有效解决SFT中的过拟合难题。  🧩 突破遗忘瓶颈： 无需依赖通用语料，即可卓越 | 43 | Python |
| [liangyuwang/Tiny-transformers](https://github.com/liangyuwang/Tiny-transformers) |  | 3 | Python |

## <span id='ai-sys-rlhf'>AI-Sys-RLHF (后训练对齐, RLHF, PPO, DPO, GRPO, TRL, OpenRLHF)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [volcengine/verl](https://github.com/volcengine/verl) | verl: Volcano Engine Reinforcement Learning for LLMs | 20425 | Python |
| [huggingface/trl](https://github.com/huggingface/trl) | Train transformer language models with reinforcement learning. | 17908 | Python |
| [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray (PPO & GRPO & REINFORCE++  | 9298 | Python |
| [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) | Solve Visual Understanding with Reinforced VLMs | 5926 | Python |
| [THUDM/slime](https://github.com/THUDM/slime) | slime is an LLM post-training framework for RL Scaling. | 5110 | Python |
| [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) | Lightning-Fast RL for LLM Reasoning and Agents. Made Simple & Flexible. | 4978 | Python |
| [hiyouga/EasyR1](https://github.com/hiyouga/EasyR1) | EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework based on veRL | 4802 | Python |
| [alibaba/ROLL](https://github.com/alibaba/ROLL) | An Efficient and User-Friendly Scaling Library for Reinforcement Learning with Large Language Models | 3046 | Python |
| [ChenmienTan/RL2](https://github.com/ChenmienTan/RL2) |  | 1252 | Python |
| [radixark/miles](https://github.com/radixark/miles) | Miles is an enterprise-facing reinforcement learning framework for LLM and VLM post-training, forked | 1043 | Python |
| [MiroMindAI/MiroRL](https://github.com/MiroMindAI/MiroRL) | MiroRL is  an MCP-first reinforcement learning framework for deep research agent. | 244 | Python |
| [0x0C001/OpenDPO](https://github.com/0x0C001/OpenDPO) |  | 33 | Python |
| [DeepLink-org/LightRFT](https://github.com/DeepLink-org/LightRFT) | LightRFT (Light Reinforcement Fine-Tuning) is an advanced reinforcement learning fine-tuning framewo | 16 | Python |
| [nideyongbao/LightRFT](https://github.com/nideyongbao/LightRFT) | LightRFT (Light Reinforcement Fine-Tuning) is an advanced reinforcement learning fine-tuning framewo | 1 | N/A |

## <span id='ai-sys-cluster'>AI-Sys-Cluster (集群调度与编排, Kubernetes, Ray, Slurm, Skypilot)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [ray-project/ray](https://github.com/ray-project/ray) | Ray is an AI compute engine. Ray consists of a core distributed runtime and a set of AI Libraries fo | 41943 | Python |

## <span id='ai-data-dataset'>AI-Data-Dataset (开源数据集, HuggingFace-Datasets, FineWeb, CommonCrawl)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [awesomedata/awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets) | A topic-centric list of HQ open datasets. | 73808 | N/A |

## <span id='ai-data-crawl'>AI-Data-Crawl (网页抓取与爬虫, Crawlee, Scrapy, Firecrawl)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [NanmiCoder/MediaCrawler](https://github.com/NanmiCoder/MediaCrawler) | 小红书笔记 \| 评论爬虫、抖音视频 \| 评论爬虫、快手视频 \| 评论爬虫、B 站视频 ｜ 评论爬虫、微博帖子 ｜ 评论爬虫、百度贴吧帖子 ｜ 百度贴吧评论回复爬虫  \| 知乎问答文章｜评论爬虫 | 47221 | Python |
| [wechat-article/wechat-article-exporter](https://github.com/wechat-article/wechat-article-exporter) | 一款在线的 微信公众号文章批量下载 工具，支持导出阅读量与评论数据，无需搭建任何环境，可通过 在线网站 使用，支持 docker 私有化部署和 Cloudflare 部署。  支持下载各种文件格式，其 | 8283 | TypeScript |
| [cv-cat/Spider_XHS](https://github.com/cv-cat/Spider_XHS) | 小红书爬虫数据采集，小红书全域运营解决方案 | 4879 | JavaScript |
| [cwjcw/xhs_douyin_content](https://github.com/cwjcw/xhs_douyin_content) | 自动抓取抖音和小红书创作者中心里的每条笔记/视频的播放，完播，点击，播放时长，点赞，分享，评论，收藏，主页访问，粉丝增量等互动数据 | 268 | Python |
| [yhslgg-arch/url-reader](https://github.com/yhslgg-arch/url-reader) | 智能网页内容读取器 - Claude Code Skill，支持微信公众号、小红书、今日头条等中国主流平台 | 159 | Python |

## <span id='ai-sys-inference'>AI-Sys-Inference (推理引擎与后端, vLLM, TGI, TensorRT-LLM, llama.cpp, SGLang)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [ollama/ollama](https://github.com/ollama/ollama) | Get up and running with OpenAI gpt-oss, DeepSeek-R1, Gemma 3 and other models. | 167032 | Go |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | LLM inference in C/C++ | 101171 | C++ |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | A high-throughput and memory-efficient inference and serving engine for LLMs | 75184 | Python |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | SGLang is a high-performance serving framework for large language models and multimodal models. | 25398 | Python |
| [liguodongiot/llm-action](https://github.com/liguodongiot/llm-action) | 本项目旨在分享大模型相关技术原理以及实战经验（大模型工程化、大模型应用落地） | 23860 | HTML |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | Nano vLLM | 12674 | Python |
| [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) | A compact implementation of SGLang, designed to demystify the complexities of modern LLM serving sys | 3912 | Python |
| [CalvinXKY/InfraTech](https://github.com/CalvinXKY/InfraTech) | 分享AI Infra知识&代码练习：PyTorch/vLLM/SGLang框架入门⚡️、性能加速🚀、大模型基础🧠、AI软硬件🔧等 | 1514 | Jupyter Notebook |
| [naklecha/simple-llm](https://github.com/naklecha/simple-llm) | ~950 line, minimal, extensible LLM inference engine built from scratch. | 467 | Python |
| [slwang-ustc/nano-vllm-v1](https://github.com/slwang-ustc/nano-vllm-v1) | Nano vLLM with vLLM v1's request scheduling strategy and chunked prefill | 74 | Python |
| [difey/nano-vllm-v1](https://github.com/difey/nano-vllm-v1) | Nano vLLM v1 engine | 15 | N/A |
| [cosmoliu2002/nano-vllm-triton](https://github.com/cosmoliu2002/nano-vllm-triton) | Nano vLLM Triton | 12 | Python |
| [RealJosephus/radix-turn-aware-nano-vllm](https://github.com/RealJosephus/radix-turn-aware-nano-vllm) | Radix Tree KV Cache with Turn-Aware Growth | 10 | Python |

## <span id='ai-algo-llm'>AI-Algo-LLM (语言模型架构, Llama, Qwen, Mistral, DeepSeek, GLM)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) | Kronos: A Foundation Model for the Language of Financial Markets | 11460 | Python |
| [wgwang/awesome-LLMs-In-China](https://github.com/wgwang/awesome-LLMs-In-China) | 中国大模型 | 6430 | N/A |
| [Duxiaoman-DI/XuanYuan](https://github.com/Duxiaoman-DI/XuanYuan) | 轩辕：度小满中文金融对话大模型 | 1307 | Python |
| [wdndev/llama3-from-scratch-zh](https://github.com/wdndev/llama3-from-scratch-zh) | 从零实现一个 llama3 中文版 | 1032 | Jupyter Notebook |
| [wdndev/tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh) | 从零实现一个小参数量中文大语言模型。 | 992 | Python |
| [Emericen/tiny-qwen](https://github.com/Emericen/tiny-qwen) | A minimal PyTorch re-implementation of Qwen 3.5  | 400 | Python |

## <span id='ai-algo-multi'>AI-Algo-Multi (多模态与新架构, CLIP, Mamba, MoE, LLaVA, VLM)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | Qwen3-VL is the multimodal large language model series developed by Qwen team, Alibaba Cloud. | 18868 | Jupyter Notebook |
| [MoonshotAI/Kimi-VL](https://github.com/MoonshotAI/Kimi-VL) | Kimi-VL: Mixture-of-Experts Vision-Language Model for Multimodal Reasoning, Long-Context Understandi | 1174 | N/A |
| [TinyLLaVA/TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) | A Framework of Small-scale Large Multimodal Models | 973 | Python |
| [hkproj/pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma) | Coding a Multimodal (Vision) Language Model from scratch in PyTorch with full explanation: https://w | 601 | Python |
| [bytedance/tarsier](https://github.com/bytedance/tarsier) | Tarsier -- a family of large-scale video-language models, which is designed to generate high-quality | 537 | Python |
| [TinyLoopX/RLLaVA](https://github.com/TinyLoopX/RLLaVA) | RLLaVA is a user-friendly framework for multi-modal RL research and optimized for resource-constrain | 58 | Python |
| [Layjins/Spider](https://github.com/Layjins/Spider) | Code for paper "Spider: Any-to-Many Multimodal LLM" | 15 | Python |

## <span id='ai-algo-vision'>AI-Algo-Vision (计算机视觉与生成, Stable Diffusion, YOLO, SAM, OpenCV)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) | The most powerful and modular diffusion model GUI, api and backend with a graph/nodes interface. | 107712 | Python |
| [NVlabs/Sana](https://github.com/NVlabs/Sana) | SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer | 5044 | Python |
| [AIDC-AI/Pixelle-Video](https://github.com/AIDC-AI/Pixelle-Video) | 🚀 AI 全自动短视频引擎 \| AI Fully Automated Short Video Engine | 3401 | Python |
| [RanFeng/clipsketch-ai](https://github.com/RanFeng/clipsketch-ai) | 将视频瞬间转化为手绘故事 Turn Video Moments into Hand-Drawn Stories | 1692 | TypeScript |
| [forXuyx/Cinego](https://github.com/forXuyx/Cinego) | 🚀 轻量视频🎥 大模型🤖 | 22 | Python |

## <span id='ai-algo-audio'>AI-Algo-Audio (语音识别与合成, Whisper, TTS, ASR, Bark)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [QuentinFuxa/WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | Simultaneous speech-to-text model | 10035 | Python |

## <span id='ai-algo-game'>AI-Algo-Game (游戏AI与仿真, Unity ML-Agents, Gymnasium, PettingZoo)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | PyTorch version of Stable Baselines, reliable implementations of reinforcement learning algorithms.  | 13023 | Python |

## <span id='ai-app-framework'>AI-App-Framework (应用编排框架, Dify, Flowise, Langflow, LangGraph)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | The best ChatGPT that $100 can buy. | 50971 | Python |
| [danielmiessler/Fabric](https://github.com/danielmiessler/Fabric) | Fabric is an open-source framework for augmenting humans using AI. It provides a modular system for  | 40383 | Go |
| [microsoft/qlib](https://github.com/microsoft/qlib) | Qlib is an AI-oriented Quant investment platform that aims to use AI tech to empower Quant Research, | 40132 | Python |
| [dataelement/bisheng](https://github.com/dataelement/bisheng) | BISHENG is an open LLM devops platform for next generation Enterprise AI applications. Powerful and  | 11285 | TypeScript |

## <span id='ai-app-rag'>AI-App-RAG (检索增强生成, LangChain, LlamaIndex, Haystack)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 🦜🔗 The platform for reliable agents. | 132264 | Python |
| [open-webui/open-webui](https://github.com/open-webui/open-webui) | User-friendly AI Interface (Supports Ollama, OpenAI API, ...) | 129913 | Python |
| [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) | Collection of awesome LLM apps with AI Agents and RAG using OpenAI, Anthropic, Gemini and opensource | 104421 | Python |
| [chatchat-space/Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) | Langchain-Chatchat（原Langchain-ChatGLM）基于 Langchain 与 ChatGLM, Qwen 与 Llama 等语言模型的 RAG 与 Agent 应用 \|  | 37737 | Python |
| [lfnovo/open-notebook](https://github.com/lfnovo/open-notebook) | An Open Source implementation of Notebook LM with more flexibility and features | 21719 | TypeScript |
| [mangiucugna/json_repair](https://github.com/mangiucugna/json_repair) | A python module to repair invalid JSON from LLMs | 4633 | Python |

## <span id='ai-app-agent'>AI-App-Agent (智能体, 规划与记忆, AutoGPT, MetaGPT, CrewAI)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [x1xhlol/system-prompts-and-models-of-ai-tools](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools) | FULL Augment Code, Claude Code, Cluely, CodeBuddy, Comet, Cursor, Devin AI, Junie, Kiro, Leap.new, L | 134257 | N/A |
| [lobehub/lobe-chat](https://github.com/lobehub/lobe-chat) | 🤯 LobeHub - an open-source, modern design AI Agent Workspace. Supports multiple AI providers, Knowle | 74704 | TypeScript |
| [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) | 🌟 The Multi-Agent Framework: First AI Software Company, Towards Natural Language Programming | 66613 | Python |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | AI agents running research on single-GPU nanochat training automatically | 65072 | Python |
| [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) | How can we build a true AI agent? Like Claude Code. | 47888 | TypeScript |
| [666ghj/BettaFish](https://github.com/666ghj/BettaFish) | 微舆：人人可用的多Agent舆情分析助手，打破信息茧房，还原舆情原貌，预测未来走向，辅助决策！从0实现，不依赖任何框架。 | 40142 | Python |
| [datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents) | 📚 《从零开始构建智能体》——从零开始的智能体原理与实践教程 | 33479 | Python |
| [continuedev/continue](https://github.com/continuedev/continue) | ⏩ Ship faster with Continuous AI. Open-source CLI that can be used in TUI mode as a coding agent or  | 32272 | TypeScript |
| [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) | Tongyi Deep Research, the Leading Open-source Deep Research Agent | 18594 | Python |
| [tukuaiai/vibe-coding-cn](https://github.com/tukuaiai/vibe-coding-cn) | 我的开发经验+提示词库=vibecoding工作站；My development experience + prompt dictionary = Vibecoding workstation；ניס | 11182 | Python |
| [shareAI-lab/Kode-cli](https://github.com/shareAI-lab/Kode-cli) | Kode CLI — Design for post-human workflows. One unit agent for every human & computer task. | 4878 | TypeScript |
| [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch) | Claude Autoresearch Skill — Autonomous goal-directed iteration for Claude Code. Inspired by Karpathy | 3144 | Shell |
| [Anning01/AIMedia](https://github.com/Anning01/AIMedia) | AIMedia 是一款自动抓取热点，AI创作文章，自动发布的集成软件。支持头条，小红书，公众号等 | 2001 | Python |
| [MeetKai/functionary](https://github.com/MeetKai/functionary) | Chat language model that can use tools and interpret the results | 1595 | Python |
| [leo-lilinxiao/codex-autoresearch](https://github.com/leo-lilinxiao/codex-autoresearch) | Codex Autoresearch Skill — A self-directed iterative system for Codex that continuously cycles throu | 1124 | Python |
| [study8677/antigravity-workspace-template](https://github.com/study8677/antigravity-workspace-template) | 🪐 The ultimate starter kit for Google Antigravity IDE. Optimized for Gemini 3 Agentic Workflows, "De | 1088 | Python |
| [vibesurf-ai/VibeSurf](https://github.com/vibesurf-ai/VibeSurf) | A powerful browser assistant for vibe surfing 一个开源的AI浏览器智能助手 | 479 | Python |
| [chmod777john/github-hunter](https://github.com/chmod777john/github-hunter) | AI 发掘潜在的爆火项目 | 76 | Jupyter Notebook |

## <span id='ai-app-mcp'>AI-App-MCP (Model Context Protocol, MCP Server)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [PDFMathTranslate/PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate) | [EMNLP 2025 Demo] PDF scientific paper translation with preserved formats - 基于 AI 完整保留排版的 PDF 文档全文双语 | 32681 | Python |
| [xpzouying/xiaohongshu-mcp](https://github.com/xpzouying/xiaohongshu-mcp) | MCP for xiaohongshu.com | 12563 | Go |
| [idosal/git-mcp](https://github.com/idosal/git-mcp) | Put an end to code hallucinations! GitMCP is a free, open-source, remote MCP server for any GitHub p | 7871 | TypeScript |
| [agent-infra/sandbox](https://github.com/agent-infra/sandbox) | All-in-One Sandbox for AI Agents that combines Browser, Shell, File, MCP and VSCode Server in a sing | 4041 | Python |
| [iFurySt/RedNote-MCP](https://github.com/iFurySt/RedNote-MCP) | 🚀MCP server for accessing RedNote(XiaoHongShu, xhs). | 1032 | TypeScript |
| [instavm/open-skills](https://github.com/instavm/open-skills) | OpenSkills: Run Claude Skills Locally using any LLM | 391 | Python |
| [AI-QL/chat-ui](https://github.com/AI-QL/chat-ui) | Single-File AI Chatbot UI with Multimodal & MCP Support: An All-in-One HTML File for a Streamlined C | 90 | HTML |
| [jswortz/antigravity-claude-skills](https://github.com/jswortz/antigravity-claude-skills) |  | 8 | Python |

## <span id='ai-algo-theory'>AI-Algo-Theory (纯理论代码, 论文复现, 数学库, NumPy, SciPy)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [CoinCheung/pytorch-loss](https://github.com/CoinCheung/pytorch-loss) | label-smooth, amsoftmax, partial-fc, focal-loss, triplet-loss, lovasz-softmax. Maybe useful  | 2262 | Python |

## <span id='research-paper'>Research-Paper (论文代码复现, Arxiv, PapersWithCode)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [wyf3/llm_related](https://github.com/wyf3/llm_related) | 复现大模型相关算法及一些学习记录 | 3212 | Python |
| [firechecking/CleanTransformer](https://github.com/firechecking/CleanTransformer) | an implementation of transformer, bert, gpt, and diffusion models for learning purposes | 160 | Python |
| [jqlong17/attnres-toy-jupyter](https://github.com/jqlong17/attnres-toy-jupyter) | A beginner-friendly Jupyter toy reproduction of MoonshotAI Attention Residuals with Chinese explanat | 4 | Jupyter Notebook |
| [hanfang/chatgpt-usage-taxonomies](https://github.com/hanfang/chatgpt-usage-taxonomies) | Taxonomies and classification prompts from the 'How People Use ChatGPT' research paper (NBER Working | 3 | N/A |

## <span id='dev-web-fullstack'>Dev-Web-FullStack (现代Web开发, Next.js, React, Vue, FastAPI, Django)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [slidevjs/slidev](https://github.com/slidevjs/slidev) | Presentation Slides for Developers | 45419 | TypeScript |
| [vnpy/vnpy](https://github.com/vnpy/vnpy) | 基于Python的开源量化交易平台开发框架 | 38805 | Python |
| [DayuanJiang/next-ai-draw-io](https://github.com/DayuanJiang/next-ai-draw-io) | A next.js web application that integrates AI capabilities with draw.io diagrams. This app allows you | 25902 | TypeScript |
| [alshedivat/al-folio](https://github.com/alshedivat/al-folio) | A beautiful, simple, clean, and responsive Jekyll theme for academics | 15419 | HTML |
| [rainxchzed/Github-Store](https://github.com/rainxchzed/Github-Store) | A free, open-source app store for GitHub releases — browse, discover, and install apps with one clic | 10160 | Kotlin |
| [jnsahaj/tweakcn](https://github.com/jnsahaj/tweakcn) | A visual no-code theme editor for shadcn/ui components | 9634 | TypeScript |
| [gamosoft/NoteDiscovery](https://github.com/gamosoft/NoteDiscovery) | Your Self-Hosted Knowledge Base | 2406 | JavaScript |
| [hezhizheng/go-wxpush](https://github.com/hezhizheng/go-wxpush) | 极简且免费的微信消息推送服务 (基于golang) | 1619 | Go |
| [dqbd/tiktokenizer](https://github.com/dqbd/tiktokenizer) | Online playground for OpenAPI tokenizers | 1550 | TypeScript |

## <span id='dev-infra-cloud'>Dev-Infra-Cloud (云原生与容器, Docker, Kubernetes, Terraform, Pulumi)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [DigitalPlatDev/FreeDomain](https://github.com/DigitalPlatDev/FreeDomain) | DigitalPlat FreeDomain: Free Domain For Everyone | 155594 | HTML |

## <span id='dev-lang-core'>Dev-Lang-Core (编程语言核心资源, Rust, Python, Go, C++)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [Lulzx/tinypdf](https://github.com/Lulzx/tinypdf) | Minimal PDF creation library. <400 LOC, zero dependencies, makes real PDFs. | 1662 | TypeScript |

## <span id='ai-app-coding'>AI-App-Coding (AI编程助手, Cursor, Copilot, Aider, Continue)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [1rgs/nanocode](https://github.com/1rgs/nanocode) | Minimal Claude Code alternative. Single Python file, zero dependencies, ~250 lines. | 2225 | Python |
| [CloudAI-X/claude-workflow-v2](https://github.com/CloudAI-X/claude-workflow-v2) | Universal Claude Code workflow plugin with agents, skills, hooks, and commands | 1311 | Python |
| [jokemon/antiPM-Workflow](https://github.com/jokemon/antiPM-Workflow) | A collection of Antigravity workflows for Product Managers. (产品经理专属的 Antigravity 工作流合集) | 26 | N/A |

## <span id='tools-efficiency'>Tools-Efficiency (生产力与终端工具, Oh-My-Zsh, Raycast, Obsidian, Neovim)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [tw93/Mole](https://github.com/tw93/Mole) | 🐹 Deep clean and optimize your Mac. | 45305 | Shell |
| [lbjlaq/Antigravity-Manager](https://github.com/lbjlaq/Antigravity-Manager) | Professional Antigravity Account Manager & Switcher. One-click seamless account switching for Antigr | 27715 | Rust |
| [ourongxing/newsnow](https://github.com/ourongxing/newsnow) | Elegant reading of real-time and hottest news | 19246 | TypeScript |
| [githubnext/monaspace](https://github.com/githubnext/monaspace) | An innovative superfamily of fonts for code | 18798 | Shell |
| [rendercv/rendercv](https://github.com/rendercv/rendercv) | CV/resume generator for academics and engineers, YAML to PDF | 16187 | Python |
| [iamgio/quarkdown](https://github.com/iamgio/quarkdown) | 🪐 Markdown with superpowers: from ideas to papers, presentations, websites, books, and knowledge bas | 10351 | Kotlin |
| [funstory-ai/BabelDOC](https://github.com/funstory-ai/BabelDOC) | Yet Another Document Translator | 8054 | Python |
| [Diorser/LiteMonitor](https://github.com/Diorser/LiteMonitor) | 一款轻量、可定制的开源桌面硬件监控软件 — 实时监测 CPU、GPU、内存、磁盘、网络等系统性能。支持横竖屏显示、多语言、主题切换、透明度显示、三色报警，界面简洁且高度可配置。A lightweigh | 4584 | C# |
| [axtonliu/axton-obsidian-visual-skills](https://github.com/axtonliu/axton-obsidian-visual-skills) | Visual Skills Pack for Obsidian: generate Canvas, Excalidraw, and Mermaid diagrams from text with Cl | 2114 | N/A |
| [OpenGithubs/github-daily-rank](https://github.com/OpenGithubs/github-daily-rank) | Github开源项目:每天📈飙升榜 top10,每天早上8:30更新 | 848 | N/A |
| [PKM-er/awesome-obsidian-zh](https://github.com/PKM-er/awesome-obsidian-zh) | Obsidian 优秀中文插件、主题与资源 | 525 | N/A |
| [zimya/zhihu_obsidian](https://github.com/zimya/zhihu_obsidian) | Zhihu on Obsidian \| 知乎 Obsidian 插件 | 213 | TypeScript |
| [simwy/Side-Markdown](https://github.com/simwy/Side-Markdown) | A sleek edge-mounted Markdown editor—accessible yet non-intrusive. Full support for headings, lists  | 4 | TypeScript |

## <span id='tools-media'>Tools-Media (图像视频处理工具, FFmpeg, ImageMagick, yt-dlp)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [VERT-sh/VERT](https://github.com/VERT-sh/VERT) | The next-generation file converter. Open source, fully local* and free forever. | 14509 | Svelte |
| [dreammis/social-auto-upload](https://github.com/dreammis/social-auto-upload) | 自动化上传视频到社交媒体：抖音、小红书、视频号、tiktok、youtube、bilibili | 9679 | Python |
| [op7418/Youtube-clipper-skill](https://github.com/op7418/Youtube-clipper-skill) |  | 1646 | Python |

## <span id='cs-education'>CS-Education (教程与面试, 系统设计, LeetCode, 学习路线图)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [harvard-edge/cs249r_book](https://github.com/harvard-edge/cs249r_book) | Introduction to Machine Learning Systems | 23333 | JavaScript |
| [WangRongsheng/awesome-LLM-resources](https://github.com/WangRongsheng/awesome-LLM-resources) | 🧑‍🚀 全世界最好的LLM资料总结（多模态生成、Agent、辅助编程、AI审稿、数据处理、模型训练、模型推理、o1 模型、MCP、小语言模型、视觉语言模型） \| Summary of the wor | 7999 | N/A |
| [itcharge/AlgoNote](https://github.com/itcharge/AlgoNote) | ⛽️「算法通关手册」：从零开始的「算法与数据结构」学习教程，200 道「算法面试热门题目」，1000+ 道「LeetCode 题目解析」，持续更新中！ | 7670 | Python |
| [zhaochenyang20/Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial) | My learning notes for ML SYS. | 5853 | Python |
| [xlite-dev/Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) | 📚A curated list of Awesome LLM/VLM Inference Papers with Codes: Flash-Attention, Paged-Attention, WI | 5121 | Python |
| [changyeyu/LLM-RL-Visualized](https://github.com/changyeyu/LLM-RL-Visualized) | 🌟100+ 原创 LLM / RL 原理图📚，《大模型算法》作者巨献！💥（100+  LLM/RL Algorithm Maps ） | 3950 | Python |
| [ginobefun/BestBlogs](https://github.com/ginobefun/BestBlogs) | bestblogs.dev - 汇集顶级编程、人工智能、产品、科技文章，大语言模型摘要评分辅助阅读，探索编程和技术未来 | 3330 | N/A |
| [ChinmayK0607/heiretsu](https://github.com/ChinmayK0607/heiretsu) | Educational WIP  | 71 | Python |
| [KylinC/Awesome-Awesome-LLM](https://github.com/KylinC/Awesome-Awesome-LLM) | awesome LLM papers！ 🚀 🚀 🚀 | 37 | N/A |

## <span id='uncategorized'>Uncategorized (无法分类)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [Tencent/Wechat-YATT](https://github.com/Tencent/Wechat-YATT) |  | 63 | Python |
| [codfish-zz/cn-trader](https://github.com/codfish-zz/cn-trader) | Python back testing system for trading strategies, based on backtrader and AkShare, customized for C | 29 | Python |

