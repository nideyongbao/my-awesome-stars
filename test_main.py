"""Offline tests for taxonomy migration and snapshot safety."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import main as app


LEGACY_EXPECTATIONS = {
    "AI-Sys-Framework": "compute-system-framework",
    "AI-Sys-Hardware": "compute-system-framework",
    "AI-Sys-Kernel": "compute-kernel-operator",
    "AI-Sys-Compiler": "compute-compiler-runtime",
    "AI-Sys-MLOps": "engineering-practice",
    "AI-Sys-Training": "training-distributed-pretrain",
    "AI-Sys-FineTuning": "training-sft-peft",
    "AI-Sys-RLHF": "training-rl-posttraining",
    "AI-Sys-Cluster": "training-distributed-pretrain",
    "AI-Data-Pipeline": "data-systems",
    "AI-Data-Synthetic": "data-systems",
    "AI-Data-Vector": "data-systems",
    "AI-Data-Dataset": "data-systems",
    "AI-Data-Crawl": "data-systems",
    "AI-Data-Labeling": "data-systems",
    "AI-Sys-Inference": "inference-engines-serving",
    "AI-Sys-Quantization": "inference-optimization",
    "AI-Algo-LLM": "model-architecture-components",
    "AI-Algo-Multi": "model-multimodal-vlm",
    "AI-Algo-Vision": "model-multimodal-vlm",
    "AI-Algo-Audio": "model-multimodal-vlm",
    "AI-Algo-Robotics": "model-multimodal-vlm",
    "AI-Algo-Game": "training-rl-posttraining",
    "AI-App-Framework": "app-agent-rag",
    "AI-App-RAG": "app-agent-rag",
    "AI-App-Agent": "app-agent-rag",
    "AI-App-MCP": "app-tools-mcp",
    "AI-Algo-Theory": "theory-reproduction",
    "Research-Paper": "theory-reproduction",
    "Dev-Web-FullStack": "engineering-practice",
    "Dev-Infra-Cloud": "engineering-practice",
    "Dev-DB-Storage": "engineering-practice",
    "Dev-Lang-Core": "workflow-knowledge-tools",
    "Dev-Sec": "engineering-practice",
    "AI-App-Coding": "workflow-ai-coding",
    "Tools-Efficiency": "workflow-knowledge-tools",
    "Tools-Media": "workflow-knowledge-tools",
    "CS-Education": "theory-reproduction",
    "Uncategorized": "other",
}


class FakeRepo:
    id = 42
    full_name = "new-owner/renamed-repo"
    html_url = "https://github.com/new-owner/renamed-repo"
    clone_url = "https://github.com/new-owner/renamed-repo.git"
    description = "A renamed repository"
    stargazers_count = 123
    language = "Python"
    default_branch = "trunk"
    archived = False
    fork = True
    size = 2048
    pushed_at = datetime(2026, 7, 30, tzinfo=timezone.utc)
    updated_at = datetime(2026, 7, 29, tzinfo=timezone.utc)

    def get_topics(self):
        return ["llm", "training"]


class PersonalRepo(FakeRepo):
    id = 43
    full_name = "nideyongbao/new-experiment"
    html_url = "https://github.com/nideyongbao/new-experiment"
    clone_url = "https://github.com/nideyongbao/new-experiment.git"


class OverriddenRepo(FakeRepo):
    id = 1277572326
    full_name = "NVIDIA-NeMo/labs-molt"
    html_url = "https://github.com/NVIDIA-NeMo/labs-molt"
    clone_url = "https://github.com/NVIDIA-NeMo/labs-molt.git"


class TaxonomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = app.load_taxonomy()

    def test_taxonomy_matches_vault_shape(self):
        self.assertEqual(len(self.taxonomy["vault_categories"]), 10)
        self.assertEqual(len(self.taxonomy["topic_categories"]), 21)
        topic_ids = {
            item["id"] for item in self.taxonomy["topic_categories"]
        }
        self.assertNotIn("ai-algo-audio", topic_ids)
        self.assertNotIn("ai-algo-game", topic_ids)
        self.assertIn("model-multimodal-vlm", topic_ids)
        self.assertIn("training-rl-posttraining", topic_ids)
        self.assertIn("personal-repositories", topic_ids)

    def test_all_legacy_categories_have_an_explicit_migration(self):
        for legacy_name, expected in LEGACY_EXPECTATIONS.items():
            with self.subTest(legacy_name=legacy_name):
                actual = app.infer_category_id(
                    {"category": f"{legacy_name} (legacy description)"},
                    self.taxonomy,
                )
                self.assertEqual(actual, expected)

    def test_audio_and_game_are_merged(self):
        audio = app.migrate_entry(
            "1",
            {"name": "owner/audio", "category": "AI-Algo-Audio (old)"},
            self.taxonomy,
        )
        game = app.migrate_entry(
            "2",
            {"name": "owner/cleanrl", "category": "AI-Algo-Game (old)"},
            self.taxonomy,
        )
        self.assertEqual(audio["category_id"], "model-multimodal-vlm")
        self.assertEqual(game["category_id"], "training-rl-posttraining")

    def test_cached_classification_survives_metadata_refresh(self):
        previous = {
            "name": "old-owner/old-name",
            "category": "AI-Sys-Training (old)",
            "reasoning": "legacy classification",
            "confidence": "high",
            "crawled_at": 1_700_000_000,
        }
        entry, classified = app.build_entry(
            FakeRepo(),
            previous,
            self.taxonomy,
            None,
            False,
        )
        self.assertFalse(classified)
        self.assertEqual(entry["name"], "new-owner/renamed-repo")
        self.assertEqual(entry["default_branch"], "trunk")
        self.assertEqual(entry["clone_url"], FakeRepo.clone_url)
        self.assertEqual(
            entry["category_id"],
            "training-distributed-pretrain",
        )
        self.assertEqual(entry["classification_source"], "legacy_migration")
        self.assertEqual(
            entry["legacy_reasoning"],
            "legacy classification",
        )

    def test_personal_owner_rule_overrides_and_preserves_subject(self):
        entry = app.migrate_entry(
            "43",
            {
                "name": "nideyongbao/new-experiment",
                "category_id": "training-rl-posttraining",
                "reasoning": "RL training experiment",
                "classification_source": "llm",
            },
            self.taxonomy,
        )
        self.assertEqual(entry["category_id"], "personal-repositories")
        self.assertEqual(entry["vault_category"], "Pillar-工作流与工具链")
        self.assertEqual(entry["classification_source"], "owner_rule")
        self.assertEqual(
            entry["subject_category_id"],
            "training-rl-posttraining",
        )
        self.assertEqual(entry["subject_reasoning"], "RL training experiment")

    def test_new_personal_repo_does_not_require_llm(self):
        entry, classified = app.build_entry(
            PersonalRepo(),
            None,
            self.taxonomy,
            None,
            True,
        )
        self.assertFalse(classified)
        self.assertEqual(entry["category_id"], "personal-repositories")
        self.assertEqual(entry["classification_source"], "owner_rule")

    def test_repository_override_does_not_require_llm(self):
        entry, classified = app.build_entry(
            OverriddenRepo(),
            None,
            self.taxonomy,
            None,
            True,
        )
        self.assertFalse(classified)
        self.assertEqual(
            entry["category_id"],
            "training-rl-posttraining",
        )
        self.assertEqual(entry["classification_source"], "manual_override")
        self.assertEqual(entry["confidence"], "high")

    def test_requested_repository_overrides_are_complete(self):
        expected = {
            "1277572326": "training-rl-posttraining",
            "1023367592": "training-rl-posttraining",
            "1244657633": "workflow-ai-coding",
        }
        for repo_id, category_id in expected.items():
            with self.subTest(repo_id=repo_id):
                override = app.get_repository_override(
                    repo_id,
                    self.taxonomy,
                )
                self.assertIsNotNone(override)
                self.assertEqual(override["category_id"], category_id)

    def test_readme_uses_vault_then_topic_hierarchy(self):
        cache = {
            "42": {
                "name": "new-owner/renamed-repo",
                "url": FakeRepo.html_url,
                "description": "A renamed repository",
                "stars": 123,
                "language": "Python",
                "category_id": "training-distributed-pretrain",
                "vault_category": "Domain-训练框架",
            }
        }
        readme = app.render_readme(cache, self.taxonomy)
        self.assertIn("Domain-训练框架 (1)", readme)
        self.assertIn("Training-Distributed-Pretrain (1)", readme)
        self.assertNotIn("AI-Sys-Training", readme)


class RemovalSafetyTests(unittest.TestCase):
    def test_large_removal_is_blocked(self):
        old_cache = {str(index): {} for index in range(10)}
        new_cache = {str(index): {} for index in range(5)}
        with self.assertRaises(RuntimeError):
            app.ensure_safe_removal(old_cache, new_cache, 0.20, False)

    def test_confirmed_large_removal_is_allowed(self):
        old_cache = {str(index): {} for index in range(10)}
        new_cache = {str(index): {} for index in range(5)}
        removed = app.ensure_safe_removal(
            old_cache,
            new_cache,
            0.20,
            True,
        )
        self.assertEqual(len(removed), 5)

    def test_restar_clears_removed_record(self):
        active = {
            "1": {
                "github_id": 1,
                "name": "owner/repo",
                "category_id": "other",
                "vault_category": "其他",
            }
        }
        updated = app.update_removed_records(
            {"1": {"name": "owner/repo"}},
            active,
            active,
            set(),
        )
        self.assertNotIn("1", updated)


if __name__ == "__main__":
    unittest.main()
