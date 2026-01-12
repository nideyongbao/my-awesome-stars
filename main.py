import os
import json
import time
import github
from github import Github
from openai import OpenAI
from tqdm import tqdm

# --- 配置部分 ---
GITHUB_TOKEN = os.getenv("GH_TOKEN")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 配置文件路径
CACHE_FILE = "stars_cache.json"
CATEGORY_FILE = "categories.json"

# --- Prompt 模板 (JSON 输出 + 思维链) ---
# 基于 note.md 中的 "6+3" 个人研究体系框架
PROMPT_TEMPLATE = """
你是一个资深的 AI 系统工程师和开源社区专家。你的任务是将 GitHub 仓库精准分类到**个人研究体系框架**中最匹配的类别。

### 输入信息
- 仓库名: {repo_name}
- 描述: {description}
- Topics: {topics}

### 预设分类体系 (6 大核心研究域 + 3 大支撑基石)
{categories_json}

### 核心决策逻辑 (Priority Logic)

#### 第一步：判断项目类型
1. **AI 相关项目**：描述或 Topics 中包含 LLM、ML、AI、模型、训练、推理、Transformer、attention、embedding 等关键词 → 进入 AI 分类决策
2. **通用开发项目**：不包含上述关键词 → 使用 `Dev-`、`Tools-`、`CS-Education` 等通用分类

#### 第二步：AI 项目的 6 大研究域分类

**Domain 1: Compute & Acceleration (计算加速)**
- **AI-Sys-Hardware**: CUDA/ROCm 驱动、硬件抽象层、Ascend/Metal 后端
- **AI-Sys-Kernel**: FlashAttention、CUTLASS、Triton 算子、手写 CUDA kernel
- **AI-Sys-Compiler**: TVM、MLIR、XLA、TorchCompile、计算图优化
- **AI-Sys-Framework**: PyTorch/TensorFlow/JAX 深度学习框架底座
- **AI-Sys-MLOps**: MLflow/WandB 实验管理与监控

**Domain 2: Training Systems (训练系统)**
- **AI-Sys-Training**: 分布式训练框架如 DeepSpeed ZeRO、Megatron、FSDP、NanoGPT 等从零训练大模型
- **AI-Sys-FineTuning**: LoRA/QLoRA/PEFT/Adapter 轻量微调、Unsloth 等
- **AI-Sys-RLHF**: PPO/DPO/GRPO 后训练对齐、TRL、OpenRLHF、verl 等 RL 微调框架
- **AI-Sys-Cluster**: Ray/Slurm/Skypilot 集群调度

**Domain 3: Data Systems (数据系统)**
- **AI-Data-Pipeline**: Datatrove/Data-Juicer 数据清洗管线
- **AI-Data-Synthetic**: Self-Instruct/Distilabel 合成数据生成
- **AI-Data-Vector**: Milvus/Faiss/Chroma 向量数据库（注意：纯向量库选此项，RAG 框架选 AI-App-RAG）
- **AI-Data-Dataset**: 公开数据集仓库
- **AI-Data-Crawl**: 爬虫、数据抓取工具

**Domain 4: Inference Systems (推理系统)**
- **AI-Sys-Inference**: vLLM/SGLang/TGI/llama.cpp/ollama 推理引擎
- **AI-Sys-Quantization**: GPTQ/AWQ/GGUF 量化压缩

**Domain 5: Model Behavior & Control (模型行为与控制)**
- **AI-Algo-LLM**: Llama/Qwen/DeepSeek/GLM 等语言模型架构
- **AI-Algo-Multi**: CLIP/LLaVA/VLM 多模态模型、Mamba/MoE 新架构
- **AI-Algo-Vision**: Stable Diffusion/YOLO/SAM 视觉生成与检测
- **AI-Algo-Audio**: Whisper/TTS 语音识别与合成
- **AI-Algo-Robotics**: LeRobot/RT-X/VLA 机器人与具身智能

**Domain 6: AI Applications (AI 系统化应用)**
- **AI-App-Framework**: Dify/Flowise/LangGraph 应用编排框架
- **AI-App-RAG**: LangChain/LlamaIndex RAG 检索增强
- **AI-App-Agent**: AutoGPT/MetaGPT/CrewAI 智能体框架
- **AI-App-MCP**: 任何提及 "Model Context Protocol" 或 "MCP Server" 的项目 → 必须归入此类
- **AI-App-Coding**: Cursor/Copilot/Aider AI 编程助手、Claude/Antigravity workflow

#### 第三步：支撑基石分类

**Pillar A: Theoretical Grounding (理论基石)**
- **AI-Algo-Theory**: 损失函数实现、数学库、算法可视化
- **Research-Paper**: 论文复现代码、arXiv 实现

**Pillar B: Engineering & Delivery (工程基石)**
- **Dev-Web-FullStack**: Next.js/React/Vue/FastAPI 全栈开发
- **Dev-Infra-Cloud**: Docker/K8s/Terraform 云原生
- **Dev-DB-Storage**: PostgreSQL/Redis/MongoDB 数据库
- **Dev-Lang-Core**: 编程语言核心资源
- **Dev-Sec**: 安全、渗透、逆向

**Pillar C: AI-Native Workflow (AI工作流基石)**
- **Tools-Efficiency**: 终端工具、效率工具、Neovim/Obsidian
- **Tools-Media**: FFmpeg/yt-dlp 音视频处理

#### 其他通用分类
- **CS-Education**: 教程、面试、学习路线


#### 特殊判断规则
1. **MCP 最高优先级**: 凡是提及 "MCP"、"Model Context Protocol" 或 "mcp-server" → 直接归入 `AI-App-MCP`
2. **Awesome-list 集合**: 如果是资源汇总/awesome-list → 归入 `CS-Education`
3. **Tutorial/教程**: 优先 `CS-Education`
4. **兜底规则**: 无法确定 → `Uncategorized (无法分类)`

### 输出格式 (JSON)
{{
    "category": "选择最匹配的完整分类名，如 'AI-Sys-Inference (推理引擎与后端, vLLM, TGI, TensorRT-LLM, llama.cpp, SGLang)'",
    "reasoning": "简短理由（一句话说明为何选择此分类）"
}}
"""


def load_categories():
    """从 categories.json 加载分类体系"""
    if not os.path.exists(CATEGORY_FILE):
        raise FileNotFoundError(f"分类配置文件 {CATEGORY_FILE} 不存在，请先创建！")
    with open(CATEGORY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_categories(categories):
    """保存分类体系（包含动态扩展的新分类）"""
    with open(CATEGORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)


def get_llm_classification(repo_name, description, topics, current_categories):
    """调用 LLM 进行分类，返回 JSON 结果"""
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    default_result = {"category": "Uncategorized", "confidence": "low", "reasoning": "API Error or Invalid Response"}

    prompt = PROMPT_TEMPLATE.format(
        repo_name=repo_name,
        description=description,
        topics=", ".join(topics) if topics else "N/A",
        categories_json=json.dumps(current_categories, ensure_ascii=False, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},  # 强制 JSON 模式
            temperature=0.1
        )
        
        # 调试日志：打印完整响应结构
        print(f"📡 API Response for {repo_name}:")
        print(f"   - Model: {LLM_MODEL}")
        print(f"   - Choices count: {len(response.choices) if response.choices else 0}")
        if response.choices:
            print(f"   - Finish reason: {response.choices[0].finish_reason}")
        
        # 校验 response 结构
        if not response.choices:
            print(f"LLM Error: Empty choices for {repo_name}")
            return default_result
        
        content = response.choices[0].message.content
        print(f"📝 LLM Raw Response for {repo_name}: {content[:200] if content else 'None'}...")  # 调试日志
        
        if not content:
            print(f"LLM Error: Empty content for {repo_name}")
            return default_result
        
        result = json.loads(content)
        print(f"✅ Parsed Result: category={result.get('category', 'N/A')}, reasoning={result.get('reasoning', 'N/A')[:50]}...")
        
        # 校验 result 是否为有效 dict 且包含 category 字段
        if not isinstance(result, dict):
            print(f"LLM Error: Result is not a dict for {repo_name}, got: {type(result)}")
            return default_result
        
        if "category" not in result:
            print(f"LLM Warning: Missing 'category' key for {repo_name}, using Uncategorized")
            result["category"] = "Uncategorized"
        
        return result
    except Exception as e:
        if "429" in str(e):
            print(f"⚠️ LLM Rate Limit hit for {repo_name}. Sleeping for 60s...")
            time.sleep(60)
            return get_llm_classification(repo_name, description, topics, current_categories)  # 重试
        else:
            print(f"LLM Error: {e}")
        return default_result


def update_readme(data, categories):
    """生成 README.md"""
    # 动态收集所有分类
    all_categories = set()
    for repo in data.values():
        all_categories.add(repo['category'])

    # 排序：优先 categories 列表顺序，新分类按字母序，Uncategorized 最后
    sorted_cats = []
    seen = set()

    for cat in categories:
        if cat in all_categories:
            sorted_cats.append(cat)
            seen.add(cat)

    remaining = [c for c in all_categories if c not in seen and c != "Uncategorized"]
    remaining.sort()
    sorted_cats.extend(remaining)

    if "Uncategorized" in all_categories:
        sorted_cats.append("Uncategorized")

    # 分组
    grouped = {cat: [] for cat in sorted_cats}
    for repo in data.values():
        cat = repo['category']
        if cat in grouped:
            grouped[cat].append(repo)
        else:
            if "Uncategorized" not in grouped:
                grouped["Uncategorized"] = []
            grouped["Uncategorized"].append(repo)

    # 生成内容
    md = "# 🌟 My Awesome AI Stars\n\n> 🤖 自动生成于 GitHub Actions, Powered by LLM.\n\n"
    md += "## 目录\n"
    for cat in sorted_cats:
        cat_key = cat.split(" ")[0]
        if " " not in cat:
            cat_key = cat
        count = len(grouped[cat])
        md += f"- [{cat} ({count})](#{cat_key.lower()})\n"

    md += "\n---\n"

    for cat in sorted_cats:
        repos = grouped[cat]
        if not repos:
            continue

        cat_key = cat.split(" ")[0]
        if " " not in cat:
            cat_key = cat

        md += f"## <span id='{cat_key.lower()}'>{cat}</span>\n\n"
        md += "| Project | Description | Stars | Language |\n"
        md += "|---|---|---|---|\n"
        repos.sort(key=lambda x: x['stars'], reverse=True)
        for r in repos:
            desc = (r.get('description') or "").replace("|", r"\|").replace("\n", " ")
            lang = r.get('language') or "N/A"
            md += f"| [{r['name']}]({r['url']}) | {desc[:100]} | {r['stars']} | {lang} |\n"
        md += "\n"

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(md)


def main():
    # 1. 加载分类体系
    categories = load_categories()

    # 2. 读取缓存
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}

    # 3. 获取 GitHub Stars (使用新版 Auth)
    auth = github.Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)
    user = g.get_user()
    print(f"Fetching stars for user: {user.login}...")

    starred_repos = user.get_starred()

    # 4. 增量更新逻辑
    new_cache = {}

    for repo in tqdm(starred_repos, total=starred_repos.totalCount):
        repo_id = str(repo.id)

        # 如果缓存里有，且不是 Uncategorized，直接复用
        if repo_id in cache and cache[repo_id].get('category') != 'Uncategorized':
            cache[repo_id]['stars'] = repo.stargazers_count
            cache[repo_id]['language'] = repo.language
            new_cache[repo_id] = cache[repo_id]
        else:
            # 新仓库或需要重新分类
            print(f"🤖 Classifying: {repo.full_name}")

            # 获取 Topics
            try:
                topics = repo.get_topics()
            except Exception:
                topics = []

            result = get_llm_classification(
                repo.name,
                repo.description or "",
                topics,
                categories
            )

            category_name = result.get("category", "Uncategorized")

            # --- 动态扩展逻辑 ---
            if category_name not in categories and "(" in category_name:
                print(f"✨ 发现新领域，自动扩展分类体系: {category_name}")
                categories.append(category_name)
                save_categories(categories)

            entry = {
                "name": repo.full_name,
                "url": repo.html_url,
                "description": repo.description,
                "stars": repo.stargazers_count,
                "category": category_name,
                "language": repo.language,
                "topics": topics,
                "confidence": result.get("confidence", "unknown"),
                "reasoning": result.get("reasoning", ""),
                "crawled_at": time.time()
            }
            new_cache[repo_id] = entry
            time.sleep(1)  # 避免 LLM Rate Limit

    # 5. 保存缓存
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_cache, f, ensure_ascii=False, indent=2)

    # 6. 保存分类体系
    save_categories(categories)

    # 7. 生成 README
    update_readme(new_cache, categories)
    print("Done!")


if __name__ == "__main__":
    main()
