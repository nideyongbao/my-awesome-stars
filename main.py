import os
import json
import time
from github import Github
from openai import OpenAI
from tqdm import tqdm

# --- 配置部分 ---
GITHUB_TOKEN = os.getenv("GH_TOKEN")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com") # 默认使用DeepSeek，可改
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat") 

# 你的分类体系
CATEGORIES = [
    "AI-Sys-Train (训练框架, DeepSpeed, Megatron)",
    "AI-Sys-Inference (推理与部署, vLLM, TGI)",
    "AI-Sys-Perf (性能优化, CUDA, Kernel)",
    "AI-Sys-Core (DL框架底座, PyTorch, JAX)",
    "AI-Algo-Model (模型架构, Llama, Qwen)",
    "AI-App-Agent (Agent, CoT, Planner)",
    "AI-App-Utils (LangChain, RAG, PDF解析)",
    "AI-Data (数据集, 数据处理)",
    "Dev-Web (前后端开发)",
    "Tools-CLI (命令行工具, 效率脚本)",
    "Proj-RedLoop (RedLoop相关项目)",
    "Research-Other (其他研究, 量化等)",
    "Uncategorized (无法分类)"
]

CACHE_FILE = "stars_cache.json"

def get_llm_category(repo_name, description):
    """调用 LLM 进行分类"""
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    
    prompt = f"""
    你是一个专业的技术仓库分类器。请根据以下 GitHub 仓库信息，从给定的分类列表中选择最匹配的一个。
    
    仓库名: {repo_name}
    描述: {description}
    
    可选分类列表:
    {json.dumps(CATEGORIES, ensure_ascii=False)}
    
    规则：
    1. 只能返回列表中的某一个字符串，不要解释。
    2. 如果是分布式训练相关，优先选 AI-Sys-Train。
    3. 如果是 Agent 或 MCP 相关，优先选 AI-App-Agent。
    
    输出分类名称：
    """
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 如果是 Rate Limit (429)，打印更明显的警告
        if "429" in str(e):
             print(f"⚠️ LLM Rate Limit hit for {repo_name}. Sleeping for 60s...")
             time.sleep(60)
        else:
             print(f"LLM Error: {e}")
        return "Uncategorized"

def update_readme(data):
    """生成 Markdown"""
    # 按分类分组
    grouped = {cat.split(" ")[0]: [] for cat in CATEGORIES}
    grouped["Uncategorized"] = []
    
    for repo in data.values():
        cat_key = repo['category'].split(" ")[0] # 提取 "AI-Sys-Train" 这种短名
        if cat_key not in grouped:
            cat_key = "Uncategorized"
        grouped[cat_key].append(repo)
    
    # 生成内容
    md = "# 🌟 My Awesome AI Stars\n\n> 🤖 自动生成于 GitHub Actions, Powered by LLM.\n\n"
    md += "## 目录\n"
    for cat in grouped.keys():
        if grouped[cat]:
            md += f"- [{cat}](#{cat.lower()})\n"
    
    md += "\n---\n"
    
    for cat, repos in grouped.items():
        if not repos: continue
        md += f"## {cat}\n\n"
        md += "| Project | Description | Stars | Category |\n"
        md += "|---|---|---|---|\n"
        # 按 Star 数倒序排列
        repos.sort(key=lambda x: x['stars'], reverse=True)
        for r in repos:
            desc = (r['description'] or "").replace("|", "\|") # 转义表格符
            md += f"| [{r['name']}]({r['url']}) | {desc} | {r['stars']} | {r['category']} |\n"
        md += "\n"
        
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(md)

def main():
    # 1. 读取缓存
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}

    # 2. 获取 GitHub Stars
    g = Github(GITHUB_TOKEN)
    user = g.get_user()
    print(f"Fetching stars for user: {user.login}...")
    
    starred_repos = user.get_starred()
    
    # 3. 增量更新逻辑
    new_cache = {}
    is_updated = False
    
    # 注意：这里为了演示只取前 500 个，全量同步可去掉切片，但要注意 API 速率
    for repo in tqdm(starred_repos, total=starred_repos.totalCount):
        repo_id = str(repo.id)
        
        # 如果缓存里有，且不需要强制刷新，直接复用
        # CHANGE: 如果之前是 Uncategorized，则重新尝试分类
        if repo_id in cache and cache[repo_id].get('category') != 'Uncategorized':
            # 更新 star 数
            cache[repo_id]['stars'] = repo.stargazers_count
            new_cache[repo_id] = cache[repo_id]
        else:
            # 新发现的仓库，调用 LLM
            print(f"🤖 Classifying new repo: {repo.full_name}")
            category = get_llm_category(repo.name, repo.description or "")
            
            entry = {
                "name": repo.full_name,
                "url": repo.html_url,
                "description": repo.description,
                "stars": repo.stargazers_count,
                "category": category,
                "crawled_at": time.time()
            }
            new_cache[repo_id] = entry
            is_updated = True
            time.sleep(1) # 避免 LLM Rate Limit
            
    # 4. 保存缓存
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_cache, f, ensure_ascii=False, indent=2)
        
    # 5. 生成 Readme
    update_readme(new_cache)
    print("Done!")

if __name__ == "__main__":
    main()
