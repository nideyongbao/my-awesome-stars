#!/usr/bin/env python3
"""Build an offline, dependency-free GitHub Pages site from the star snapshot."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import main as stars


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "site"
OUTPUT = ROOT / "_site"


def public_data(cache: dict, taxonomy: dict) -> dict:
    """Export display fields only; derive links from validated repository names."""
    stars.validate_cache(cache, taxonomy)
    vaults = {v["name"]: v["id"] for v in taxonomy["vault_categories"]}
    repositories = []
    for repo_id, item in cache.items():
        name = item["name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name):
            raise ValueError(f"Invalid GitHub repository name: {name!r}")
        repositories.append({
            "id": str(repo_id),
            "name": name,
            "url": f"https://github.com/{name}",
            "description": item.get("description") or "这个项目暂未提供简介，前往 GitHub 了解更多。",
            "stars": max(0, int(item.get("stars") or 0)),
            "language": item.get("language") or "未标注",
            "topics": item.get("topics") or [],
            "category": item["category_id"],
            "vault": vaults[item["vault_category"]],
            "first_seen_at": item.get("first_seen_at") or "",
            "pushed_at": item.get("pushed_at") or "",
            "archived": bool(item.get("archived")),
        })
    repositories.sort(key=lambda r: (-r["stars"], r["name"].casefold()))
    return {
        "repositories": repositories,
        "vaults": [{"id": v["id"], "name": v["name"], "description": v["description"]}
                   for v in taxonomy["vault_categories"]],
        "categories": [{"id": t["id"], "name": t["name"], "description": t["description"],
                        "vault": vaults[t["vault_category"]]}
                       for t in taxonomy["topic_categories"]],
    }


def json_for_html(value: object) -> str:
    # Prevent descriptions such as </script> from ending the JSON data element.
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e"))


def short_label(name: str) -> str:
    return name.removeprefix("Domain-").removeprefix("Pillar-")


def render_card(repo: dict, vault_names: dict, category_names: dict) -> str:
    owner, project = repo["name"].split("/", 1)
    tags = "".join(f'<span class="tag">{escape(str(tag))}</span>' for tag in repo["topics"][:3])
    archived = '<span class="archived">已归档</span>' if repo["archived"] else ""
    return f'''<article class="repo-card" data-repo-id="{escape(repo['id'], quote=True)}">
      <div class="card-top"><span class="category-tag">{escape(vault_names[repo['vault']])}</span>{archived}</div>
      <h3><a href="{escape(repo['url'], quote=True)}" target="_blank" rel="noopener noreferrer"><span class="owner">{escape(owner)} /</span> {escape(project)}<span class="outbound" aria-hidden="true">↗</span></a></h3>
      <p class="repo-description">{escape(repo['description'])}</p>
      <div class="tags">{tags}</div>
      <div class="card-bottom"><span class="repo-stars" aria-label="{repo['stars']} stars">☆ {repo['stars']:,}</span><span class="language"><span class="language-dot" aria-hidden="true"></span>{escape(repo['language'])}</span><span class="topic-label" title="{escape(category_names[repo['category']], quote=True)}">{escape(category_names[repo['category']])}</span></div>
    </article>'''


def render_page(data: dict, template: str, built_at: str) -> str:
    repos = data["repositories"]
    counts = Counter(r["vault"] for r in repos)
    language_counts = Counter(r["language"] for r in repos)
    vault_names = {v["id"]: short_label(v["name"]) for v in data["vaults"]}
    category_names = {t["id"]: t["name"] for t in data["categories"]}
    nav = [f'<button class="category-button" data-vault="" aria-pressed="true"><span>全部项目</span><span class="count">{len(repos)}</span></button>']
    previous_group = None
    for vault in data["vaults"]:
        group = "研究领域 / DOMAIN" if vault["name"].startswith("Domain-") else "知识支柱 / PILLAR" if vault["name"].startswith("Pillar-") else "更多 / MORE"
        if group != previous_group:
            nav.append(f'<p class="nav-group">{group}</p>')
            previous_group = group
        nav.append(f'<button class="category-button" data-vault="{escape(vault["id"], quote=True)}" aria-pressed="false"><span>{escape(vault_names[vault["id"]])}</span><span class="count">{counts[vault["id"]]}</span></button>')
    replacements = {
        "NAV": "\n".join(nav),
        "CARDS": "\n".join(render_card(r, vault_names, category_names) for r in repos),
        "DATA": json_for_html(data),
        "TOTAL": str(len(repos)),
        "VAULT_COUNT": str(sum(1 for count in counts.values() if count)),
        "LANGUAGE_COUNT": str(len(set(language_counts) - {"未标注"})),
        "LANGUAGES": "".join(f'<option value="{escape(lang, quote=True)}">{escape(lang)} ({count})</option>' for lang, count in sorted(language_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
        "BUILT_AT": escape(built_at),
    }
    # One-pass substitution: untrusted text containing a template token stays literal.
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: replacements[m[1]], template)


def build(output: Path = OUTPUT) -> dict:
    taxonomy = stars.load_taxonomy()
    cache = json.loads(stars.CACHE_FILE.read_text(encoding="utf-8"))
    data = public_data(cache, taxonomy)
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = render_page(data, (SOURCE / "index.html").read_text(encoding="utf-8"), built_at)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(page, encoding="utf-8")
    for filename in ("style.css", "app.js", "favicon.svg"):
        shutil.copyfile(SOURCE / filename, output / filename)
    (output / ".nojekyll").touch()
    print(f"Built {len(data['repositories'])} repositories -> {output}")
    return data


if __name__ == "__main__":
    build()
