#!/usr/bin/env python3
"""Synchronize and classify a GitHub user's starred repositories."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from openai import OpenAI


ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / "stars_cache.json"
CATEGORY_FILE = ROOT / "categories.json"
REMOVED_FILE = ROOT / "removed_stars.json"
README_FILE = ROOT / "README.md"

GITHUB_TOKEN = os.getenv("GH_TOKEN")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

DEFAULT_MAX_REMOVAL_RATIO = 0.20
UNCATEGORIZED_ID = "other"
PERSONAL_OWNER = "nideyongbao"
PERSONAL_CATEGORY_ID = "personal-repositories"


PROMPT_TEMPLATE = """
你是 GitHub 仓库分类器。请将给定仓库归入下方 taxonomy 中唯一一个细分类。

仓库名: {repo_name}
描述: {description}
Topics: {topics}

分类体系:
{taxonomy_json}

规则:
1. 只能选择 taxonomy 中已经存在的 category_id，不得创建新分类。
2. vault_category 由程序根据 category_id 推导，不需要自行生成。
3. MCP 项目优先选择 app-tools-mcp。
4. Awesome List、课程和教程优先选择 theory-reproduction。
5. 无法可靠判断时选择 other。

只返回 JSON:
{{
  "category_id": "taxonomy 中的 id",
  "confidence": "high|medium|low",
  "reasoning": "一句简短理由"
}}
"""


def configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp_to_iso(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return (
            datetime.fromtimestamp(value, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    return str(value)


def datetime_to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, datetime):
        return str(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = handle.name
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def save_json(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, content)


def topic_label(topic: dict[str, Any]) -> str:
    return f"{topic['name']} ({topic['description']})"


def is_personal_repository(repo_name: str) -> bool:
    owner, separator, _ = str(repo_name or "").partition("/")
    return bool(separator) and owner.casefold() == PERSONAL_OWNER.casefold()


def get_repository_override(
    repo_id: Union[int, str],
    taxonomy: dict[str, Any],
) -> Optional[dict[str, Any]]:
    override = taxonomy.get("repository_overrides", {}).get(str(repo_id))
    return override if isinstance(override, dict) else None


def taxonomy_indexes(
    taxonomy: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    topics_by_id = {
        topic["id"]: topic for topic in taxonomy["topic_categories"]
    }
    topics_by_name = {}
    for topic in taxonomy["topic_categories"]:
        topics_by_name[topic["name"]] = topic
        for legacy_name in topic.get("legacy_names", []):
            topics_by_name[legacy_name] = topic
    topics_by_label = {
        topic_label(topic): topic for topic in taxonomy["topic_categories"]
    }
    return topics_by_id, topics_by_name, topics_by_label


def load_taxonomy(path: Path = CATEGORY_FILE) -> dict[str, Any]:
    taxonomy = load_json(path, {})
    if not isinstance(taxonomy, dict) or taxonomy.get("schema_version") != 2:
        raise ValueError("categories.json 必须使用 schema_version 2。")

    vault_categories = taxonomy.get("vault_categories")
    topic_categories = taxonomy.get("topic_categories")
    if not isinstance(vault_categories, list) or not vault_categories:
        raise ValueError("taxonomy 缺少 vault_categories。")
    if not isinstance(topic_categories, list) or not topic_categories:
        raise ValueError("taxonomy 缺少 topic_categories。")

    vault_ids = [item.get("id") for item in vault_categories]
    vault_names = [item.get("name") for item in vault_categories]
    topic_ids = [item.get("id") for item in topic_categories]
    topic_names = [item.get("name") for item in topic_categories]

    if None in vault_ids or len(vault_ids) != len(set(vault_ids)):
        raise ValueError("vault category id 必须存在且唯一。")
    if None in vault_names or len(vault_names) != len(set(vault_names)):
        raise ValueError("vault category name 必须存在且唯一。")
    if None in topic_ids or len(topic_ids) != len(set(topic_ids)):
        raise ValueError("topic category id 必须存在且唯一。")
    if None in topic_names or len(topic_names) != len(set(topic_names)):
        raise ValueError("topic category name 必须存在且唯一。")
    if UNCATEGORIZED_ID not in set(topic_ids):
        raise ValueError("taxonomy 必须定义 other。")
    if PERSONAL_CATEGORY_ID not in set(topic_ids):
        raise ValueError("taxonomy 必须定义 personal-repositories。")

    repository_overrides = taxonomy.get("repository_overrides", {})
    if not isinstance(repository_overrides, dict):
        raise ValueError("repository_overrides 必须是对象。")
    for repo_id, override in repository_overrides.items():
        if not str(repo_id).isdigit() or not isinstance(override, dict):
            raise ValueError("repository_overrides 必须按 GitHub 数字 ID 定义。")
        if override.get("category_id") not in set(topic_ids):
            raise ValueError(f"仓库覆盖规则 {repo_id} 使用了无效 category_id。")
        if override.get("category_id") == PERSONAL_CATEGORY_ID:
            raise ValueError("个人仓库应使用 owner 规则，不应写入覆盖规则。")
        if not override.get("name") or not override.get("reasoning"):
            raise ValueError(f"仓库覆盖规则 {repo_id} 缺少 name 或 reasoning。")
        if not override.get("classified_at"):
            raise ValueError(f"仓库覆盖规则 {repo_id} 缺少 classified_at。")

    valid_vault_names = set(vault_names)
    for topic in topic_categories:
        if topic.get("vault_category") not in valid_vault_names:
            raise ValueError(
                f"{topic.get('id')} 引用了不存在的 vault_category。"
            )
        if not topic.get("description"):
            raise ValueError(f"{topic.get('id')} 缺少 description。")

    return taxonomy


def infer_category_id(
    entry: dict[str, Any],
    taxonomy: dict[str, Any],
) -> str:
    topics_by_id, topics_by_name, topics_by_label = taxonomy_indexes(taxonomy)

    category_id = str(entry.get("category_id") or "").strip().lower()
    if category_id in topics_by_id:
        return category_id
    for topic in taxonomy["topic_categories"]:
        legacy_ids = {
            str(name).strip().lower()
            for name in topic.get("legacy_names", [])
        }
        if category_id in legacy_ids:
            return topic["id"]

    for field in ("category", "topic_category"):
        value = str(entry.get(field) or "").strip()
        if value in topics_by_label:
            return topics_by_label[value]["id"]
        name = value.split(" (", 1)[0]
        if name in topics_by_name:
            return topics_by_name[name]["id"]

    return UNCATEGORIZED_ID


def apply_repository_override(
    repo_id: Union[int, str],
    entry: dict[str, Any],
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    override = get_repository_override(repo_id, taxonomy)
    if override is None:
        return entry

    topics_by_id, _, _ = taxonomy_indexes(taxonomy)
    topic = topics_by_id[override["category_id"]]
    entry["category_id"] = topic["id"]
    entry["category"] = topic_label(topic)
    entry["vault_category"] = topic["vault_category"]
    entry["confidence"] = "high"
    entry["reasoning"] = override["reasoning"]
    entry["classified_at"] = override["classified_at"]
    entry["classification_source"] = "manual_override"
    return entry


def apply_personal_category(
    entry: dict[str, Any],
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    if not is_personal_repository(str(entry.get("name") or "")):
        return entry

    topics_by_id, _, _ = taxonomy_indexes(taxonomy)
    category_id = infer_category_id(entry, taxonomy)
    if category_id != PERSONAL_CATEGORY_ID:
        subject = topics_by_id[category_id]
        entry["subject_category_id"] = category_id
        entry["subject_category"] = topic_label(subject)
        entry["subject_vault_category"] = subject["vault_category"]
        if entry.get("reasoning"):
            entry["subject_reasoning"] = entry["reasoning"]

    personal_topic = topics_by_id[PERSONAL_CATEGORY_ID]
    entry["category_id"] = PERSONAL_CATEGORY_ID
    entry["category"] = topic_label(personal_topic)
    entry["vault_category"] = personal_topic["vault_category"]
    entry["confidence"] = "high"
    entry["reasoning"] = (
        f"仓库所有者为 {PERSONAL_OWNER}，按 owner 规则归入个人仓库。"
    )
    entry["classification_source"] = "owner_rule"
    return entry


def migrate_entry(
    repo_id: str,
    entry: dict[str, Any],
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    topics_by_id, _, _ = taxonomy_indexes(taxonomy)
    migrated = dict(entry)
    category_id = infer_category_id(migrated, taxonomy)
    topic = topics_by_id[category_id]

    try:
        github_id: Union[int, str] = int(repo_id)
    except ValueError:
        github_id = repo_id

    legacy_timestamp = timestamp_to_iso(migrated.pop("crawled_at", None))
    migrated["github_id"] = github_id
    migrated["category_id"] = category_id
    migrated["category"] = topic_label(topic)
    migrated["vault_category"] = topic["vault_category"]
    migrated["first_seen_at"] = (
        migrated.get("first_seen_at") or legacy_timestamp
    )
    migrated["classified_at"] = (
        migrated.get("classified_at") or legacy_timestamp
    )
    if not migrated.get("classification_source"):
        historical_reasoning = str(migrated.get("reasoning") or "")
        if historical_reasoning:
            migrated["legacy_reasoning"] = historical_reasoning
        migrated["reasoning"] = (
            "由历史分类按 2026-07-30 Vault taxonomy 审计迁移；"
            "原始理由保存在 legacy_reasoning。"
        )
        migrated["classification_source"] = "legacy_migration"
    migrated = apply_repository_override(repo_id, migrated, taxonomy)
    return apply_personal_category(migrated, taxonomy)


def migrate_cache(
    cache: dict[str, Any],
    taxonomy: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not isinstance(cache, dict):
        raise ValueError("stars_cache.json 顶层必须是对象。")
    migrated: dict[str, dict[str, Any]] = {}
    for repo_id, entry in cache.items():
        if not isinstance(entry, dict):
            raise ValueError(f"仓库 {repo_id} 的缓存条目不是对象。")
        migrated[str(repo_id)] = migrate_entry(str(repo_id), entry, taxonomy)
    return migrated


def validate_cache(
    cache: dict[str, dict[str, Any]],
    taxonomy: dict[str, Any],
) -> None:
    topics_by_id, _, _ = taxonomy_indexes(taxonomy)
    for repo_id, entry in cache.items():
        category_id = entry.get("category_id")
        if category_id not in topics_by_id:
            raise ValueError(f"仓库 {repo_id} 使用了无效 category_id。")
        expected_vault_category = topics_by_id[category_id]["vault_category"]
        if entry.get("vault_category") != expected_vault_category:
            raise ValueError(f"仓库 {repo_id} 的 vault_category 不一致。")
        if not entry.get("name"):
            raise ValueError(f"仓库 {repo_id} 缺少 name。")
        is_personal = is_personal_repository(str(entry["name"]))
        if is_personal and category_id != PERSONAL_CATEGORY_ID:
            raise ValueError(f"个人仓库 {repo_id} 未归入 personal-repositories。")
        if category_id == PERSONAL_CATEGORY_ID and not is_personal:
            raise ValueError(
                f"非 {PERSONAL_OWNER} 仓库 {repo_id} 不应归入 personal-repositories。"
            )
        override = get_repository_override(repo_id, taxonomy)
        if override and (
            category_id != override["category_id"]
            or entry.get("classification_source") != "manual_override"
        ):
            raise ValueError(f"仓库 {repo_id} 未应用人工覆盖规则。")


def get_repo_topics(repo: Any, previous: dict[str, Any]) -> list[str]:
    try:
        return list(repo.get_topics())
    except Exception as exc:
        print(f"⚠️ 无法刷新 {repo.full_name} 的 topics，保留旧值: {exc}")
        return list(previous.get("topics") or [])


def fresh_repo_metadata(repo: Any, topics: list[str]) -> dict[str, Any]:
    return {
        "github_id": repo.id,
        "name": repo.full_name,
        "url": repo.html_url,
        "clone_url": getattr(repo, "clone_url", None),
        "description": repo.description,
        "stars": repo.stargazers_count,
        "language": repo.language,
        "topics": topics,
        "default_branch": getattr(repo, "default_branch", None),
        "archived": bool(getattr(repo, "archived", False)),
        "fork": bool(getattr(repo, "fork", False)),
        "size_kb": getattr(repo, "size", None),
        "pushed_at": datetime_to_iso(getattr(repo, "pushed_at", None)),
        "updated_at": datetime_to_iso(getattr(repo, "updated_at", None)),
    }


def get_llm_classification(
    client: "OpenAI",
    repo_name: str,
    description: str,
    topics: list[str],
    taxonomy: dict[str, Any],
    max_attempts: int = 3,
) -> dict[str, str]:
    topics_by_id, _, _ = taxonomy_indexes(taxonomy)
    llm_category_ids = set(topics_by_id) - {PERSONAL_CATEGORY_ID}
    prompt_topics = [
        {
            "id": item["id"],
            "name": item["name"],
            "description": item["description"],
            "vault_category": item["vault_category"],
        }
        for item in taxonomy["topic_categories"]
        if item["id"] in llm_category_ids
    ]
    prompt = PROMPT_TEMPLATE.format(
        repo_name=repo_name,
        description=description,
        topics=", ".join(topics) if topics else "N/A",
        taxonomy_json=json.dumps(prompt_topics, ensure_ascii=False, indent=2),
    )
    fallback = {
        "category_id": UNCATEGORIZED_ID,
        "confidence": "low",
        "reasoning": "LLM API 失败或返回了无效分类",
    }

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("LLM 返回空响应。")

            result = json.loads(response.choices[0].message.content)
            if not isinstance(result, dict):
                raise ValueError("LLM 响应不是 JSON 对象。")

            category_id = str(result.get("category_id") or "").strip().lower()
            if category_id not in llm_category_ids:
                raise ValueError(f"LLM 返回未知 category_id: {category_id}")

            confidence = str(result.get("confidence") or "unknown").lower()
            if confidence not in {"high", "medium", "low", "unknown"}:
                confidence = "unknown"
            return {
                "category_id": category_id,
                "confidence": confidence,
                "reasoning": str(result.get("reasoning") or ""),
            }
        except Exception as exc:
            is_rate_limit = "429" in str(exc)
            if is_rate_limit and attempt < max_attempts:
                delay = 60 * attempt
                print(
                    f"⚠️ LLM Rate Limit: {repo_name}; "
                    f"{delay} 秒后进行第 {attempt + 1} 次尝试。"
                )
                time.sleep(delay)
                continue
            print(f"⚠️ LLM 分类失败 {repo_name}: {exc}")
            return fallback

    return fallback


def build_entry(
    repo: Any,
    previous: Optional[dict[str, Any]],
    taxonomy: dict[str, Any],
    llm_client: Optional["OpenAI"],
    reclassify_other: bool,
) -> tuple[dict[str, Any], bool]:
    topics_by_id, _, _ = taxonomy_indexes(taxonomy)
    previous = (
        migrate_entry(str(repo.id), previous, taxonomy) if previous else {}
    )
    topics = get_repo_topics(repo, previous)
    personal_repository = is_personal_repository(repo.full_name)
    repository_override = get_repository_override(repo.id, taxonomy)
    category_id = (
        PERSONAL_CATEGORY_ID
        if personal_repository
        else (
            repository_override["category_id"]
            if repository_override
            else infer_category_id(previous, taxonomy)
        )
    )
    needs_classification = (
        not personal_repository
        and repository_override is None
        and (
            not previous
            or (
                reclassify_other
                and category_id == UNCATEGORIZED_ID
            )
        )
    )

    if needs_classification:
        if llm_client is None:
            raise RuntimeError(
                f"{repo.full_name} 需要分类，但 LLM_API_KEY 未配置。"
            )
        classification = get_llm_classification(
            llm_client,
            repo.full_name,
            repo.description or "",
            topics,
            taxonomy,
        )
        category_id = classification["category_id"]
        confidence = classification["confidence"]
        reasoning = classification["reasoning"]
        classified_at = utc_now_iso()
        classification_source = "llm"
    else:
        confidence = str(previous.get("confidence") or "unknown")
        reasoning = str(previous.get("reasoning") or "")
        classified_at = previous.get("classified_at")
        classification_source = str(
            previous.get("classification_source") or "legacy_migration"
        )

    topic = topics_by_id[category_id]
    entry = fresh_repo_metadata(repo, topics)
    entry.update(
        {
            "category_id": category_id,
            "category": topic_label(topic),
            "vault_category": topic["vault_category"],
            "confidence": confidence,
            "reasoning": reasoning,
            "classification_source": classification_source,
            "first_seen_at": previous.get("first_seen_at") or utc_now_iso(),
            "classified_at": classified_at or utc_now_iso(),
        }
    )
    if previous.get("legacy_reasoning"):
        entry["legacy_reasoning"] = previous["legacy_reasoning"]
    for field in (
        "subject_category_id",
        "subject_category",
        "subject_vault_category",
        "subject_reasoning",
    ):
        if previous.get(field):
            entry[field] = previous[field]
    entry = apply_repository_override(repo.id, entry, taxonomy)
    return apply_personal_category(entry, taxonomy), needs_classification


def ensure_safe_removal(
    old_cache: dict[str, Any],
    new_cache: dict[str, Any],
    max_removal_ratio: float,
    allow_large_removal: bool,
) -> set[str]:
    removed_ids = set(old_cache) - set(new_cache)
    if not old_cache or not removed_ids:
        return removed_ids

    removal_ratio = len(removed_ids) / len(old_cache)
    if removal_ratio > max_removal_ratio and not allow_large_removal:
        raise RuntimeError(
            f"本次有 {len(removed_ids)}/{len(old_cache)} "
            f"({removal_ratio:.1%}) 个仓库消失，超过安全阈值 "
            f"{max_removal_ratio:.1%}。如已确认 Token 权限和 Star 列表正常，"
            "使用 --allow-large-removal。"
        )
    return removed_ids


def update_removed_records(
    removed_records: dict[str, Any],
    old_cache: dict[str, dict[str, Any]],
    new_cache: dict[str, dict[str, Any]],
    removed_ids: set[str],
) -> dict[str, Any]:
    updated = dict(removed_records)

    for repo_id in new_cache:
        updated.pop(repo_id, None)

    removed_at = utc_now_iso()
    for repo_id in sorted(removed_ids):
        previous = old_cache[repo_id]
        updated[repo_id] = {
            "github_id": previous.get("github_id", repo_id),
            "name": previous.get("name"),
            "url": previous.get("url"),
            "category_id": previous.get("category_id"),
            "vault_category": previous.get("vault_category"),
            "removed_at": removed_at,
            "reason": "missing_from_visible_stars",
            "last_entry": previous,
        }
    return updated


def markdown_escape(value: Any) -> str:
    return (
        str(value or "")
        .replace("|", r"\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def render_readme(
    data: dict[str, dict[str, Any]],
    taxonomy: dict[str, Any],
) -> str:
    topics_by_id, _, _ = taxonomy_indexes(taxonomy)
    vault_order = taxonomy["vault_categories"]
    topic_order = taxonomy["topic_categories"]

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        vault["name"]: {} for vault in vault_order
    }
    for repo in data.values():
        category_id = repo.get("category_id", UNCATEGORIZED_ID)
        topic = topics_by_id.get(category_id, topics_by_id[UNCATEGORIZED_ID])
        grouped.setdefault(topic["vault_category"], {}).setdefault(
            topic["id"], []
        ).append(repo)

    lines = [
        "# 🌟 My Awesome AI Stars",
        "",
        "> 🤖 由 GitHub Actions 自动更新；主分类与本地 Vault 6+3 体系一致。",
        "",
        f"当前收录 **{len(data)}** 个仓库。",
        "",
        "分类设计与旧 39 类迁移记录见 "
        "[分类体系审计](docs/category_audit.md)。",
        "",
        "## 目录",
        "",
    ]

    for vault in vault_order:
        count = sum(
            len(repos) for repos in grouped.get(vault["name"], {}).values()
        )
        if count:
            lines.append(
                f"- [{vault['name']} ({count})](#vault-{vault['id']})"
            )

    lines.extend(["", "---", ""])

    for vault in vault_order:
        topic_groups = grouped.get(vault["name"], {})
        count = sum(len(repos) for repos in topic_groups.values())
        if not count:
            continue

        lines.extend(
            [
                (
                    f"## <span id=\"vault-{vault['id']}\">"
                    f"{vault['name']} ({count})</span>"
                ),
                "",
                vault["description"],
                "",
            ]
        )

        for topic in topic_order:
            repos = topic_groups.get(topic["id"], [])
            if not repos:
                continue
            repos.sort(
                key=lambda item: int(item.get("stars") or 0),
                reverse=True,
            )
            lines.extend(
                [
                    (
                        f"### <span id=\"topic-{topic['id']}\">"
                        f"{topic['name']} ({len(repos)})</span>"
                    ),
                    "",
                    topic["description"],
                    "",
                    "| Project | Description | Stars | Language |",
                    "|---|---|---:|---|",
                ]
            )
            for repo in repos:
                lines.append(
                    "| "
                    f"[{markdown_escape(repo.get('name'))}]"
                    f"({repo.get('url')}) | "
                    f"{markdown_escape(repo.get('description'))[:160]} | "
                    f"{int(repo.get('stars') or 0)} | "
                    f"{markdown_escape(repo.get('language') or 'N/A')} |"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def persist_outputs(
    cache: dict[str, dict[str, Any]],
    removed_records: dict[str, Any],
    taxonomy: dict[str, Any],
) -> None:
    validate_cache(cache, taxonomy)
    readme = render_readme(cache, taxonomy)
    save_json(CACHE_FILE, cache)
    save_json(REMOVED_FILE, removed_records)
    atomic_write_text(README_FILE, readme)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="同步 GitHub Stars，并按 Vault 6+3 + 细分类生成快照。"
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="只迁移现有缓存并重建 README，不访问 GitHub 或 LLM。",
    )
    parser.add_argument(
        "--reclassify-other",
        action="store_true",
        help="重新调用 LLM 分类 other 条目。",
    )
    parser.add_argument(
        "--max-removal-ratio",
        type=float,
        default=DEFAULT_MAX_REMOVAL_RATIO,
        help="单次允许从可见 Star 集合消失的最大比例，默认 0.20。",
    )
    parser.add_argument(
        "--allow-large-removal",
        action="store_true",
        help="确认 Token 和 Star 列表正常后，允许超过安全阈值的移除。",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    configure_utf8()
    args = build_parser().parse_args(argv)
    if not 0 <= args.max_removal_ratio <= 1:
        print("❌ --max-removal-ratio 必须位于 0 到 1 之间。")
        return 2

    try:
        taxonomy = load_taxonomy()
        raw_cache = load_json(CACHE_FILE, {})
        old_cache = migrate_cache(raw_cache, taxonomy)
        removed_records = load_json(REMOVED_FILE, {})
        if not isinstance(removed_records, dict):
            raise ValueError("removed_stars.json 顶层必须是对象。")

        if args.render_only:
            persist_outputs(old_cache, removed_records, taxonomy)
            print(
                f"✅ 已迁移 {len(old_cache)} 个缓存条目并按 6+3 重建 README。"
            )
            return 0

        if not GITHUB_TOKEN:
            raise RuntimeError("缺少 GH_TOKEN，无法读取 GitHub Stars。")

        try:
            import github
            from github import Github
            from openai import OpenAI
            from tqdm import tqdm
        except ImportError as exc:
            raise RuntimeError(
                "缺少运行依赖，请先执行 pip install -r requirements.txt。"
            ) from exc

        auth = github.Auth.Token(GITHUB_TOKEN)
        github_client = Github(auth=auth)
        user = github_client.get_user()
        starred_repos = user.get_starred()
        expected_total = starred_repos.totalCount
        print(f"Fetching {expected_total} stars for user: {user.login}...")

        new_cache: dict[str, dict[str, Any]] = {}
        llm_client: Optional["OpenAI"] = None
        classified_count = 0

        for repo in tqdm(starred_repos, total=expected_total):
            repo_id = str(repo.id)
            previous = old_cache.get(repo_id)
            needs_classification = (
                not is_personal_repository(repo.full_name)
                and get_repository_override(repo_id, taxonomy) is None
                and (
                    not previous
                    or (
                        args.reclassify_other
                        and infer_category_id(previous, taxonomy)
                        == UNCATEGORIZED_ID
                    )
                )
            )
            if needs_classification and llm_client is None:
                if not LLM_API_KEY:
                    raise RuntimeError(
                        f"{repo.full_name} 需要分类，但缺少 LLM_API_KEY。"
                    )
                llm_client = OpenAI(
                    api_key=LLM_API_KEY,
                    base_url=LLM_BASE_URL,
                )

            entry, classified = build_entry(
                repo,
                previous,
                taxonomy,
                llm_client,
                args.reclassify_other,
            )
            new_cache[repo_id] = entry
            if classified:
                classified_count += 1
                time.sleep(1)

        if len(new_cache) != expected_total:
            raise RuntimeError(
                f"GitHub 返回 totalCount={expected_total}，"
                f"实际只处理 {len(new_cache)} 个仓库；拒绝覆盖旧快照。"
            )

        removed_ids = ensure_safe_removal(
            old_cache,
            new_cache,
            args.max_removal_ratio,
            args.allow_large_removal,
        )
        updated_removed = update_removed_records(
            removed_records,
            old_cache,
            new_cache,
            removed_ids,
        )
        persist_outputs(new_cache, updated_removed, taxonomy)
        print(
            f"✅ 同步完成：活跃 {len(new_cache)}，"
            f"新分类 {classified_count}，移除 {len(removed_ids)}。"
        )
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"❌ {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
