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
    
    可选分类列表 (仅供参考，如果没有合适的，你可以新建一个符合格式的分类):
    {json.dumps(CATEGORIES, ensure_ascii=False)}
    
    规则：
    1. 只能返回分类名称字符串，不要解释。
    2. 如果现有分类不合适，请生成一个新的分类，格式必须为 "Category-Name (Description)"，例如 "AI-Audio (语音合成与识别)"。
    3. 如果是分布式训练相关，优先选 AI-Sys-Train。
    
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
    # 动态收集所有分类
    all_categories = set()
    for repo in data.values():
        all_categories.add(repo['category'])
    
    # 将标准分类和新发现的分类合并并排序
    # 优先展示配置好的 CATEGORIES 顺序，新分类按字母序排在后面
    sorted_cats = []
    seen = set()
    
    # 1. 先加预定义的
    for cat in CATEGORIES:
        if cat in all_categories:
            sorted_cats.append(cat)
            seen.add(cat)
            
    # 2. 再加新生成的 (排除 Uncategorized)
    remaining = [c for c in all_categories if c not in seen and c != "Uncategorized"]
    remaining.sort()
    sorted_cats.extend(remaining)
    
    # 3. 最后加 Uncategorized
    if "Uncategorized" in all_categories:
        sorted_cats.append("Uncategorized")

    # 分组
    grouped = {cat: [] for cat in sorted_cats}
    for repo in data.values():
        cat = repo['category']
        if cat in grouped:
            grouped[cat].append(repo)
        else:
            # Fallback 如果有些奇奇怪怪的分类没被捕获
            if "Uncategorized" not in grouped:
                grouped["Uncategorized"] = []
            grouped["Uncategorized"].append(repo)
    
    # 生成内容
    md = "# 🌟 My Awesome AI Stars\n\n> 🤖 自动生成于 GitHub Actions, Powered by LLM.\n\n"
    md += "## 目录\n"
    for cat in sorted_cats:
        cat_key = cat.split(" ")[0] # 提取 "AI-Sys-Train" 用于锚点
        # 兼容一下，如果生成的分类没有空格，直接用全文
        if " " not in cat: 
             cat_key = cat
        
        count = len(grouped[cat])
        md += f"- [{cat} ({count})](#{cat_key.lower()})\n"
    
    md += "\n---\n"
    
    for cat in sorted_cats:
        repos = grouped[cat]
        if not repos: continue
        
        cat_key = cat.split(" ")[0]
        if " " not in cat: cat_key = cat
        
        md += f"## <span id='{cat_key.lower()}'>{cat}</span>\n\n"
        md += "| Project | Description | Stars | Language |\n"
        md += "|---|---|---|---|\n"
        # 按 Star 数倒序排列
        repos.sort(key=lambda x: x['stars'], reverse=True)
        for r in repos:
            desc = (r['description'] or "").replace("|", "\|") # 转义表格符
            lang = r.get('language') or "N/A"
            md += f"| [{r['name']}]({r['url']}) | {desc} | {r['stars']} | {lang} |\n"
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
    # DeprecationWarning: Argument login_or_token is deprecated, please use auth=github.Auth.Token(...) instead
    from github import Auth
    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)
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
            # 更新动态数据: stars, language
            cache[repo_id]['stars'] = repo.stargazers_count
            cache[repo_id]['language'] = repo.language
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
                "language": repo.language,
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
