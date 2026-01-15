# 🌟 My Awesome AI Stars

> 🤖 自动生成于 GitHub Actions, Powered by LLM.

## 目录
- [AI-Sys-Framework (深度学习框架底座, PyTorch, TensorFlow, JAX, MXNet) (3)](#ai-sys-framework)
- [AI-Sys-Kernel (高性能算子与底层优化, FlashAttention, CUTLASS, Triton) (7)](#ai-sys-kernel)
- [AI-Sys-Training (分布式训练框架, DeepSpeed, Megatron, FSDP, Horovod) (17)](#ai-sys-training)
- [AI-Sys-FineTuning (轻量微调, LoRA, PEFT, QLoRA, Unsloth, Adapter) (7)](#ai-sys-finetuning)
- [AI-Sys-RLHF (后训练对齐, RLHF, PPO, DPO, GRPO, TRL, OpenRLHF) (12)](#ai-sys-rlhf)
- [AI-Sys-Cluster (集群调度与编排, Kubernetes, Ray, Slurm, Skypilot) (1)](#ai-sys-cluster)
- [AI-Data-Dataset (开源数据集, HuggingFace-Datasets, FineWeb, CommonCrawl) (1)](#ai-data-dataset)
- [AI-Data-Crawl (网页抓取与爬虫, Crawlee, Scrapy, Firecrawl) (3)](#ai-data-crawl)
- [AI-Sys-Inference (推理引擎与后端, vLLM, TGI, TensorRT-LLM, llama.cpp, SGLang) (10)](#ai-sys-inference)
- [AI-Algo-LLM (语言模型架构, Llama, Qwen, Mistral, DeepSeek, GLM) (5)](#ai-algo-llm)
- [AI-Algo-Multi (多模态与新架构, CLIP, Mamba, MoE, LLaVA, VLM) (4)](#ai-algo-multi)
- [AI-Algo-Vision (计算机视觉与生成, Stable Diffusion, YOLO, SAM, OpenCV) (5)](#ai-algo-vision)
- [AI-Algo-Audio (语音识别与合成, Whisper, TTS, ASR, Bark) (1)](#ai-algo-audio)
- [AI-App-Framework (应用编排框架, Dify, Flowise, Langflow, LangGraph) (3)](#ai-app-framework)
- [AI-App-RAG (检索增强生成, LangChain, LlamaIndex, Haystack) (6)](#ai-app-rag)
- [AI-App-Agent (智能体, 规划与记忆, AutoGPT, MetaGPT, CrewAI) (13)](#ai-app-agent)
- [AI-App-MCP (Model Context Protocol, MCP Server) (7)](#ai-app-mcp)
- [AI-Algo-Theory (纯理论代码, 论文复现, 数学库, NumPy, SciPy) (1)](#ai-algo-theory)
- [Research-Paper (论文代码复现, Arxiv, PapersWithCode) (2)](#research-paper)
- [Dev-Web-FullStack (现代Web开发, Next.js, React, Vue, FastAPI, Django) (6)](#dev-web-fullstack)
- [Dev-Infra-Cloud (云原生与容器, Docker, Kubernetes, Terraform, Pulumi) (1)](#dev-infra-cloud)
- [Dev-Lang-Core (编程语言核心资源, Rust, Python, Go, C++) (1)](#dev-lang-core)
- [AI-App-Coding (AI编程助手, Cursor, Copilot, Aider, Continue) (2)](#ai-app-coding)
- [Tools-Efficiency (生产力与终端工具, Oh-My-Zsh, Raycast, Obsidian, Neovim) (9)](#tools-efficiency)
- [Tools-Media (图像视频处理工具, FFmpeg, ImageMagick, yt-dlp) (2)](#tools-media)
- [CS-Education (教程与面试, 系统设计, LeetCode, 学习路线图) (6)](#cs-education)

---
## <span id='ai-sys-framework'>AI-Sys-Framework (深度学习框架底座, PyTorch, TensorFlow, JAX, MXNet)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [huggingface/transformers](https://github.com/huggingface/transformers) | 🤗 Transformers: the model-definition framework for state-of-the-art machine learning models in text, | 155072 | Python |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | Tensors and Dynamic neural networks in Python with strong GPU acceleration | 96618 | Python |
| [tigert1998/mytorch](https://github.com/tigert1998/mytorch) | A toy Python DL training library with PyTorch like API | 39 | Python |

## <span id='ai-sys-kernel'>AI-Sys-Kernel (高性能算子与底层优化, FlashAttention, CUTLASS, Triton)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | FlashMLA: Efficient Multi-head Latent Attention Kernels | 11967 | C++ |
| [facebookresearch/xformers](https://github.com/facebookresearch/xformers) | Hackable and optimized Transformers building blocks, supporting a composable construction. | 10262 | Python |
| [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | Efficient Triton Kernels for LLM Training | 6036 | Python |
| [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | FP16xINT4 LLM inference kernel that can achieve near-ideal ~4x speedups up to medium batchsizes of 1 | 982 | Python |
| [jt-zhang/Sparse_Attention_API](https://github.com/jt-zhang/Sparse_Attention_API) |  | 66 | Python |
| [DefTruth/CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes) | 📚200+ Tensor/CUDA Cores Kernels, ⚡️flash-attn-mma, ⚡️hgemm with WMMA, MMA and CuTe (98%~100% TFLOPS  | 60 | Cuda |
| [liangyuwang/MetaProfiler](https://github.com/liangyuwang/MetaProfiler) | MetaProfiler is a lightweight, structure-agnostic operator-level profiler for PyTorch models that le | 2 | Python |

## <span id='ai-sys-training'>AI-Sys-Training (分布式训练框架, DeepSpeed, Megatron, FSDP, Horovod)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [jingyaogong/minimind](https://github.com/jingyaogong/minimind) | 🚀🚀 「大模型」2小时完全从0训练26M的小参数GPT！🌏 Train a 26M-parameter GPT from scratch in just 2h! | 37286 | Python |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Ongoing research training transformer models at scale | 14908 | Python |
| [pytorch/torchtitan](https://github.com/pytorch/torchtitan) | A PyTorch native platform for training generative AI models | 4961 | Python |
| [PKU-Alignment/align-anything](https://github.com/PKU-Alignment/align-anything) | Align Anything: Training All-modality Model with Feedback | 4619 | Python |
| [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | NanoGPT (124M) in 3 minutes | 4134 | Python |
| [qibin0506/Cortex](https://github.com/qibin0506/Cortex) | 个人构建MoE大模型：从预训练到DPO的完整实践 | 2239 | Python |
| [Open-Reasoner-Zero/Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) | Official Repo for Open-Reasoner-Zero | 2084 | Python |
| [Victorwz/Open-Qwen2VL](https://github.com/Victorwz/Open-Qwen2VL) | [COLM 2025] Open-Qwen2VL: Compute-Efficient Pre-Training of Fully-Open Multimodal LLMs on Academic R | 300 | Python |
| [liangyuwang/zo2](https://github.com/liangyuwang/zo2) | ZO2 (Zeroth-Order Offloading): Full Parameter Fine-Tuning 175B LLMs with 18GB GPU Memory [COLM2025] | 198 | Python |
| [MiroMindAI/MiroTrain](https://github.com/MiroMindAI/MiroTrain) | MiroTrain is an efficient and algorithm-first framework for post-training large agentic models.  | 124 | Python |
| [CoinCheung/gdGPT](https://github.com/CoinCheung/gdGPT) | Train llm (bloom, llama, baichuan2-7b, chatglm3-6b) with deepspeed pipeline mode. Faster than zero/z | 98 | Python |
| [liangyuwang/Tiny-FSDP](https://github.com/liangyuwang/Tiny-FSDP) | Tiny-FSDP, a minimalistic re-implementation of the PyTorch FSDP | 93 | Python |
| [nex-agi/NexRL](https://github.com/nex-agi/NexRL) | NexRL is an ultra-loosely-coupled LLM post-training framework. | 68 | Python |
| [liangyuwang/Tiny-DeepSpeed](https://github.com/liangyuwang/Tiny-DeepSpeed) | Tiny-DeepSpeed, a minimalistic re-implementation of the DeepSpeed library | 49 | Python |
| [XU-YIJIE/hobo-llm-from-scratch](https://github.com/XU-YIJIE/hobo-llm-from-scratch) | From Llama to Deepseek, grpo/mtp implemented. With pt/sft/lora/qlora included | 31 | Python |
| [liangyuwang/Tiny-Megatron](https://github.com/liangyuwang/Tiny-Megatron) | Tiny-Megatron, a minimalistic re-implementation of the Megatron library | 21 | Python |
| [liangyuwang/Streaming-Dataloader](https://github.com/liangyuwang/Streaming-Dataloader) |  A memory-efficient streaming data loader designed for LLM pretraining under limited CPU and GPU mem | 3 | Python |

## <span id='ai-sys-finetuning'>AI-Sys-FineTuning (轻量微调, LoRA, PEFT, QLoRA, Unsloth, Adapter)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | Fine-tuning & Reinforcement Learning for LLMs. 🦥 Train OpenAI gpt-oss, DeepSeek, Qwen, Llama, Gemma, | 50701 | Python |
| [imoneoi/openchat](https://github.com/imoneoi/openchat) | OpenChat: Advancing Open-source Language Models with Imperfect Data | 5465 | Python |
| [adonis-dym/memory_reduced_optimizer](https://github.com/adonis-dym/memory_reduced_optimizer) |  | 530 | Python |
| [AI-Study-Han/Zero-Qwen-VL](https://github.com/AI-Study-Han/Zero-Qwen-VL) | 训练一个对中文支持更好的LLaVA模型，并开源训练代码和数据。 | 78 | Python |
| [0x0C001/OpenSFT](https://github.com/0x0C001/OpenSFT) |  | 46 | Python |
| [qibin0506/llm_trainer](https://github.com/qibin0506/llm_trainer) |  | 44 | Python |
| [liangyuwang/Tiny-transformers](https://github.com/liangyuwang/Tiny-transformers) |  | 3 | Python |

## <span id='ai-sys-rlhf'>AI-Sys-RLHF (后训练对齐, RLHF, PPO, DPO, GRPO, TRL, OpenRLHF)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [volcengine/verl](https://github.com/volcengine/verl) | verl: Volcano Engine Reinforcement Learning for LLMs | 18339 | Python |
| [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray (PPO & GRPO & REINFORCE++  | 8781 | Python |
| [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) | Solve Visual Understanding with Reinforced VLMs | 5806 | Python |
| [hiyouga/EasyR1](https://github.com/hiyouga/EasyR1) | EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework based on veRL | 4428 | Python |
| [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) | Lightning-Fast RL for LLM Reasoning and Agents. Made Simple & Flexible. | 3394 | Python |
| [THUDM/slime](https://github.com/THUDM/slime) | slime is an LLM post-training framework for RL Scaling. | 3329 | Python |
| [alibaba/ROLL](https://github.com/alibaba/ROLL) | An Efficient and User-Friendly Scaling Library for Reinforcement Learning with Large Language Models | 2644 | Python |
| [ChenmienTan/RL2](https://github.com/ChenmienTan/RL2) |  | 1040 | Python |
| [MiroMindAI/MiroRL](https://github.com/MiroMindAI/MiroRL) | MiroRL is  an MCP-first reinforcement learning framework for deep research agent. | 217 | Python |
| [0x0C001/OpenDPO](https://github.com/0x0C001/OpenDPO) |  | 33 | Python |
| [DeepLink-org/LightRFT](https://github.com/DeepLink-org/LightRFT) | LightRFT (Light Reinforcement Fine-Tuning) is an advanced reinforcement learning fine-tuning framewo | 10 | Python |
| [nideyongbao/LightRFT](https://github.com/nideyongbao/LightRFT) | LightRFT (Light Reinforcement Fine-Tuning) is an advanced reinforcement learning fine-tuning framewo | 1 | N/A |

## <span id='ai-sys-cluster'>AI-Sys-Cluster (集群调度与编排, Kubernetes, Ray, Slurm, Skypilot)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [ray-project/ray](https://github.com/ray-project/ray) | Ray is an AI compute engine. Ray consists of a core distributed runtime and a set of AI Libraries fo | 40773 | Python |

## <span id='ai-data-dataset'>AI-Data-Dataset (开源数据集, HuggingFace-Datasets, FineWeb, CommonCrawl)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [awesomedata/awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets) | A topic-centric list of HQ open datasets. | 72076 | N/A |

## <span id='ai-data-crawl'>AI-Data-Crawl (网页抓取与爬虫, Crawlee, Scrapy, Firecrawl)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [NanmiCoder/MediaCrawler](https://github.com/NanmiCoder/MediaCrawler) | 小红书笔记 \| 评论爬虫、抖音视频 \| 评论爬虫、快手视频 \| 评论爬虫、B 站视频 ｜ 评论爬虫、微博帖子 ｜ 评论爬虫、百度贴吧帖子 ｜ 百度贴吧评论回复爬虫  \| 知乎问答文章｜评论爬虫 | 42581 | Python |
| [cv-cat/Spider_XHS](https://github.com/cv-cat/Spider_XHS) | 小红书爬虫数据采集，小红书全域运营解决方案 | 4048 | JavaScript |
| [cwjcw/xhs_douyin_content](https://github.com/cwjcw/xhs_douyin_content) | 自动抓取抖音和小红书创作者中心里的每条笔记/视频的播放，完播，点击，播放时长，点赞，分享，评论，收藏，主页访问，粉丝增量等互动数据 | 251 | Python |

## <span id='ai-sys-inference'>AI-Sys-Inference (推理引擎与后端, vLLM, TGI, TensorRT-LLM, llama.cpp, SGLang)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [ollama/ollama](https://github.com/ollama/ollama) | Get up and running with OpenAI gpt-oss, DeepSeek-R1, Gemma 3 and other models. | 159446 | Go |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | LLM inference in C/C++ | 92980 | C++ |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | A high-throughput and memory-efficient inference and serving engine for LLMs | 67525 | Python |
| [liguodongiot/llm-action](https://github.com/liguodongiot/llm-action) | 本项目旨在分享大模型相关技术原理以及实战经验（大模型工程化、大模型应用落地） | 22787 | HTML |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | SGLang is a high-performance serving framework for large language models and multimodal models. | 22444 | Python |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | Nano vLLM | 10755 | Python |
| [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) | A compact implementation of SGLang, designed to demystify the complexities of modern LLM serving sys | 2936 | Python |
| [difey/nano-vllm-v1](https://github.com/difey/nano-vllm-v1) | Nano vLLM v1 engine | 13 | N/A |
| [cosmoliu2002/nano-vllm-triton](https://github.com/cosmoliu2002/nano-vllm-triton) | Nano vLLM Triton | 13 | Python |
| [RealJosephus/radix-turn-aware-nano-vllm](https://github.com/RealJosephus/radix-turn-aware-nano-vllm) | Radix Tree KV Cache with Turn-Aware Growth | 10 | Python |

## <span id='ai-algo-llm'>AI-Algo-LLM (语言模型架构, Llama, Qwen, Mistral, DeepSeek, GLM)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) | Kronos: A Foundation Model for the Language of Financial Markets | 10029 | Python |
| [wgwang/awesome-LLMs-In-China](https://github.com/wgwang/awesome-LLMs-In-China) | 中国大模型 | 6361 | N/A |
| [Duxiaoman-DI/XuanYuan](https://github.com/Duxiaoman-DI/XuanYuan) | 轩辕：度小满中文金融对话大模型 | 1291 | Python |
| [wdndev/llama3-from-scratch-zh](https://github.com/wdndev/llama3-from-scratch-zh) | 从零实现一个 llama3 中文版 | 1004 | Jupyter Notebook |
| [wdndev/tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh) | 从零实现一个小参数量中文大语言模型。 | 925 | Python |

## <span id='ai-algo-multi'>AI-Algo-Multi (多模态与新架构, CLIP, Mamba, MoE, LLaVA, VLM)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | Qwen3-VL is the multimodal large language model series developed by Qwen team, Alibaba Cloud. | 17764 | Jupyter Notebook |
| [MoonshotAI/Kimi-VL](https://github.com/MoonshotAI/Kimi-VL) | Kimi-VL: Mixture-of-Experts Vision-Language Model for Multimodal Reasoning, Long-Context Understandi | 1136 | N/A |
| [hkproj/pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma) | Coding a Multimodal (Vision) Language Model from scratch in PyTorch with full explanation: https://w | 587 | Python |
| [Layjins/Spider](https://github.com/Layjins/Spider) | Code for paper "Spider: Any-to-Many Multimodal LLM" | 14 | Python |

## <span id='ai-algo-vision'>AI-Algo-Vision (计算机视觉与生成, Stable Diffusion, YOLO, SAM, OpenCV)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) | The most powerful and modular diffusion model GUI, api and backend with a graph/nodes interface. | 100219 | Python |
| [NVlabs/Sana](https://github.com/NVlabs/Sana) | SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer | 4893 | Python |
| [AIDC-AI/Pixelle-Video](https://github.com/AIDC-AI/Pixelle-Video) | 🚀 AI 全自动短视频引擎 \| AI Fully Automated Short Video Engine | 1542 | Python |
| [RanFeng/clipsketch-ai](https://github.com/RanFeng/clipsketch-ai) | 将视频瞬间转化为手绘故事 Turn Video Moments into Hand-Drawn Stories | 1399 | TypeScript |
| [forXuyx/Cinego](https://github.com/forXuyx/Cinego) | 🚀 轻量视频🎥 大模型🤖 | 20 | Python |

## <span id='ai-algo-audio'>AI-Algo-Audio (语音识别与合成, Whisper, TTS, ASR, Bark)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [QuentinFuxa/WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | Simultaneous speech-to-text model | 9490 | Python |

## <span id='ai-app-framework'>AI-App-Framework (应用编排框架, Dify, Flowise, Langflow, LangGraph)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | The best ChatGPT that $100 can buy. | 40294 | Python |
| [danielmiessler/Fabric](https://github.com/danielmiessler/Fabric) | Fabric is an open-source framework for augmenting humans using AI. It provides a modular system for  | 38160 | Go |
| [dataelement/bisheng](https://github.com/dataelement/bisheng) | BISHENG is an open LLM devops platform for next generation Enterprise AI applications. Powerful and  | 10932 | TypeScript |

## <span id='ai-app-rag'>AI-App-RAG (检索增强生成, LangChain, LlamaIndex, Haystack)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 🦜🔗 The platform for reliable agents. | 124197 | Python |
| [open-webui/open-webui](https://github.com/open-webui/open-webui) | User-friendly AI Interface (Supports Ollama, OpenAI API, ...) | 120673 | Python |
| [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) | Collection of awesome LLM apps with AI Agents and RAG using OpenAI, Anthropic, Gemini and opensource | 87862 | Python |
| [chatchat-space/Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) | Langchain-Chatchat（原Langchain-ChatGLM）基于 Langchain 与 ChatGLM, Qwen 与 Llama 等语言模型的 RAG 与 Agent 应用 \|  | 37083 | Python |
| [lfnovo/open-notebook](https://github.com/lfnovo/open-notebook) | An Open Source implementation of Notebook LM with more flexibility and features | 17918 | TypeScript |
| [mangiucugna/json_repair](https://github.com/mangiucugna/json_repair) | A python module to repair invalid JSON from LLMs | 4321 | Python |

## <span id='ai-app-agent'>AI-App-Agent (智能体, 规划与记忆, AutoGPT, MetaGPT, CrewAI)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [x1xhlol/system-prompts-and-models-of-ai-tools](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools) | FULL Augment Code, Claude Code, Cluely, CodeBuddy, Comet, Cursor, Devin AI, Junie, Kiro, Leap.new, L | 108201 | N/A |
| [lobehub/lobe-chat](https://github.com/lobehub/lobe-chat) | 🤯 LobeHub - an open-source, modern design AI Agent Workspace. Supports multiple AI providers, Knowle | 70121 | TypeScript |
| [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) | 🌟 The Multi-Agent Framework: First AI Software Company, Towards Natural Language Programming | 62963 | Python |
| [666ghj/BettaFish](https://github.com/666ghj/BettaFish) | 微舆：人人可用的多Agent舆情分析助手，打破信息茧房，还原舆情原貌，预测未来走向，辅助决策！从0实现，不依赖任何框架。 | 34487 | Python |
| [continuedev/continue](https://github.com/continuedev/continue) | ⏩ Ship faster with Continuous AI. Open-source CLI that can be used in TUI mode as a coding agent or  | 30890 | TypeScript |
| [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) | Tongyi Deep Research, the Leading Open-source Deep Research Agent | 17933 | Python |
| [datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents) | 📚 《从零开始构建智能体》——从零开始的智能体原理与实践教程 | 16260 | Python |
| [tukuaiai/vibe-coding-cn](https://github.com/tukuaiai/vibe-coding-cn) | 我的开发经验+提示词库=vibecoding工作站；My development experience + prompt dictionary = Vibecoding workstation；ניס | 6710 | Python |
| [Anning01/AIMedia](https://github.com/Anning01/AIMedia) | AIMedia 是一款自动抓取热点，AI创作文章，自动发布的集成软件。支持头条，小红书，公众号等 | 1600 | Python |
| [MeetKai/functionary](https://github.com/MeetKai/functionary) | Chat language model that can use tools and interpret the results | 1592 | Python |
| [study8677/antigravity-workspace-template](https://github.com/study8677/antigravity-workspace-template) | 🪐 The ultimate starter kit for Google Antigravity IDE. Optimized for Gemini 3 Agentic Workflows, "De | 552 | Python |
| [vibesurf-ai/VibeSurf](https://github.com/vibesurf-ai/VibeSurf) | A powerful browser assistant for vibe surfing 一个开源的AI浏览器智能助手 | 393 | Python |
| [chmod777john/github-hunter](https://github.com/chmod777john/github-hunter) | AI 发掘潜在的爆火项目 | 63 | Jupyter Notebook |

## <span id='ai-app-mcp'>AI-App-MCP (Model Context Protocol, MCP Server)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [xpzouying/xiaohongshu-mcp](https://github.com/xpzouying/xiaohongshu-mcp) | MCP for xiaohongshu.com | 8040 | Go |
| [idosal/git-mcp](https://github.com/idosal/git-mcp) | Put an end to code hallucinations! GitMCP is a free, open-source, remote MCP server for any GitHub p | 7374 | TypeScript |
| [agent-infra/sandbox](https://github.com/agent-infra/sandbox) | All-in-One Sandbox for AI Agents that combines Browser, Shell, File, MCP and VSCode Server in a sing | 2031 | Python |
| [iFurySt/RedNote-MCP](https://github.com/iFurySt/RedNote-MCP) | 🚀MCP server for accessing RedNote(XiaoHongShu, xhs). | 941 | TypeScript |
| [instavm/open-skills](https://github.com/instavm/open-skills) | OpenSkills: Run Claude Skills Locally using any LLM | 297 | Python |
| [AI-QL/chat-ui](https://github.com/AI-QL/chat-ui) | Single-File AI Chatbot UI with Multimodal & MCP Support: An All-in-One HTML File for a Streamlined C | 86 | HTML |
| [jswortz/antigravity-claude-skills](https://github.com/jswortz/antigravity-claude-skills) |  | 5 | Python |

## <span id='ai-algo-theory'>AI-Algo-Theory (纯理论代码, 论文复现, 数学库, NumPy, SciPy)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [CoinCheung/pytorch-loss](https://github.com/CoinCheung/pytorch-loss) | label-smooth, amsoftmax, partial-fc, focal-loss, triplet-loss, lovasz-softmax. Maybe useful  | 2258 | Python |

## <span id='research-paper'>Research-Paper (论文代码复现, Arxiv, PapersWithCode)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [wyf3/llm_related](https://github.com/wyf3/llm_related) | 复现大模型相关算法及一些学习记录 | 2842 | Python |
| [hanfang/chatgpt-usage-taxonomies](https://github.com/hanfang/chatgpt-usage-taxonomies) | Taxonomies and classification prompts from the 'How People Use ChatGPT' research paper (NBER Working | 2 | N/A |

## <span id='dev-web-fullstack'>Dev-Web-FullStack (现代Web开发, Next.js, React, Vue, FastAPI, Django)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [vnpy/vnpy](https://github.com/vnpy/vnpy) | 基于Python的开源量化交易平台开发框架 | 35433 | Python |
| [DayuanJiang/next-ai-draw-io](https://github.com/DayuanJiang/next-ai-draw-io) | A next.js web application that integrates AI capabilities with draw.io diagrams. This app allows you | 18419 | TypeScript |
| [jnsahaj/tweakcn](https://github.com/jnsahaj/tweakcn) | A visual no-code theme editor for shadcn/ui components | 9066 | TypeScript |
| [rainxchzed/Github-Store](https://github.com/rainxchzed/Github-Store) | A free, open-source app store for GitHub releases — browse, discover, and install apps with one clic | 3720 | Kotlin |
| [gamosoft/NoteDiscovery](https://github.com/gamosoft/NoteDiscovery) | Your Self-Hosted Knowledge Base | 2070 | JavaScript |
| [hezhizheng/go-wxpush](https://github.com/hezhizheng/go-wxpush) | 极简且免费的微信消息推送服务 (基于golang) | 1206 | Go |

## <span id='dev-infra-cloud'>Dev-Infra-Cloud (云原生与容器, Docker, Kubernetes, Terraform, Pulumi)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [DigitalPlatDev/FreeDomain](https://github.com/DigitalPlatDev/FreeDomain) | DigitalPlat FreeDomain: Free Domain For Everyone | 140466 | HTML |

## <span id='dev-lang-core'>Dev-Lang-Core (编程语言核心资源, Rust, Python, Go, C++)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [Lulzx/tinypdf](https://github.com/Lulzx/tinypdf) | Minimal PDF creation library. <400 LOC, zero dependencies, makes real PDFs. | 1334 | TypeScript |

## <span id='ai-app-coding'>AI-App-Coding (AI编程助手, Cursor, Copilot, Aider, Continue)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [CloudAI-X/claude-workflow-v2](https://github.com/CloudAI-X/claude-workflow-v2) | Universal Claude Code workflow plugin with agents, skills, hooks, and commands | 1136 | Python |
| [jokemon/antiPM-Workflow](https://github.com/jokemon/antiPM-Workflow) | A collection of Antigravity workflows for Product Managers. (产品经理专属的 Antigravity 工作流合集) | 16 | N/A |

## <span id='tools-efficiency'>Tools-Efficiency (生产力与终端工具, Oh-My-Zsh, Raycast, Obsidian, Neovim)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [tw93/Mole](https://github.com/tw93/Mole) | 🐹 Deep clean and optimize your Mac. | 29194 | Shell |
| [githubnext/monaspace](https://github.com/githubnext/monaspace) | An innovative superfamily of fonts for code | 17869 | Shell |
| [ourongxing/newsnow](https://github.com/ourongxing/newsnow) | Elegant reading of real-time and hottest news | 17268 | TypeScript |
| [rendercv/rendercv](https://github.com/rendercv/rendercv) | CV/resume generator for academics and engineers, YAML to PDF | 14717 | Python |
| [lbjlaq/Antigravity-Manager](https://github.com/lbjlaq/Antigravity-Manager) | Professional Antigravity Account Manager & Switcher. One-click seamless account switching for Antigr | 13288 | Rust |
| [funstory-ai/BabelDOC](https://github.com/funstory-ai/BabelDOC) | Yet Another Document Translator | 6879 | Python |
| [Diorser/LiteMonitor](https://github.com/Diorser/LiteMonitor) | 一款轻量、可定制的开源桌面硬件监控软件 — 实时监测 CPU、GPU、内存、磁盘、网络等系统性能。支持横竖屏显示、多语言、主题切换、透明度显示、三色报警，界面简洁且高度可配置。A lightweigh | 3245 | C# |
| [OpenGithubs/github-daily-rank](https://github.com/OpenGithubs/github-daily-rank) | Github开源项目:每天📈飙升榜 top10,每天早上8:30更新 | 760 | N/A |
| [simwy/Side-Markdown](https://github.com/simwy/Side-Markdown) | A sleek edge-mounted Markdown editor—accessible yet non-intrusive. Full support for headings, lists  | 4 | TypeScript |

## <span id='tools-media'>Tools-Media (图像视频处理工具, FFmpeg, ImageMagick, yt-dlp)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [VERT-sh/VERT](https://github.com/VERT-sh/VERT) | The next-generation file converter. Open source, fully local* and free forever. | 13269 | Svelte |
| [dreammis/social-auto-upload](https://github.com/dreammis/social-auto-upload) | 自动化上传视频到社交媒体：抖音、小红书、视频号、tiktok、youtube、bilibili | 8031 | Python |

## <span id='cs-education'>CS-Education (教程与面试, 系统设计, LeetCode, 学习路线图)</span>

| Project | Description | Stars | Language |
|---|---|---|---|
| [harvard-edge/cs249r_book](https://github.com/harvard-edge/cs249r_book) | Introduction to Machine Learning Systems | 16953 | JavaScript |
| [itcharge/AlgoNote](https://github.com/itcharge/AlgoNote) | ⛽️「算法通关手册」：从零开始的「算法与数据结构」学习教程，200 道「算法面试热门题目」，1000+ 道「LeetCode 题目解析」，持续更新中！ | 7524 | Python |
| [WangRongsheng/awesome-LLM-resources](https://github.com/WangRongsheng/awesome-LLM-resources) | 🧑‍🚀 全世界最好的LLM资料总结（多模态生成、Agent、辅助编程、AI审稿、数据处理、模型训练、模型推理、o1 模型、MCP、小语言模型、视觉语言模型） \| Summary of the wor | 7281 | N/A |
| [zhaochenyang20/Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial) | My learning notes for ML SYS. | 5042 | Python |
| [changyeyu/LLM-RL-Visualized](https://github.com/changyeyu/LLM-RL-Visualized) | 🌟100+ 原创 LLM / RL 原理图📚，《大模型算法》作者巨献！💥（100+  LLM/RL Algorithm Maps ） | 2947 | Python |
| [ginobefun/BestBlogs](https://github.com/ginobefun/BestBlogs) | bestblogs.dev - 汇集顶级编程、人工智能、产品、科技文章，大语言模型摘要评分辅助阅读，探索编程和技术未来 | 2801 | N/A |

