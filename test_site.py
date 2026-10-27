"""Offline checks for snapshot fidelity, safe rendering, and Pages assets."""

import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import build_site as site


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cards = []
        self.assets = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.cards.append(attrs["data-repo-id"])
        if tag == "script":
            self.scripts.append(attrs)
            if "src" in attrs:
                self.assets.append(attrs["src"])
        if tag == "link":
            self.assets.append(attrs["href"])


class SiteTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = site.stars.load_taxonomy()
        self.entry = {
            "name": "example/project", "category_id": "other", "vault_category": "其他",
            "url": "javascript:alert(1)", "stars": 42, "language": "Python",
            "description": '</script><script>alert("x")</script><img src=x onerror=alert(1)> {{TOTAL}}',
            "topics": ['<unsafe>'], "reasoning": "not-for-the-site", "classification_source": "llm",
        }

    def test_external_text_cannot_inject_markup_or_script(self):
        data = site.public_data({"example-id": self.entry}, self.taxonomy)
        self.assertEqual(data["repositories"][0]["url"], "https://github.com/example/project")
        self.assertNotIn("reasoning", data["repositories"][0])
        encoded = site.json_for_html(data)
        self.assertNotIn("</script>", encoded)
        self.assertEqual(json.loads(encoded), data)
        html = site.render_page(data, (site.SOURCE / "index.html").read_text(encoding="utf-8"), "test")
        parser = PageParser()
        parser.feed(html)
        self.assertEqual(len(parser.scripts), 2)
        self.assertNotIn('<img src=x', html)
        self.assertIn("&lt;unsafe&gt;", html)
        self.assertIn("{{TOTAL}}", html)  # Repository content stays literal.

    def test_invalid_repository_name_is_rejected(self):
        self.entry["name"] = 'example/repo" onclick="alert(1)'
        with self.assertRaises(ValueError):
            site.public_data({"example-id": self.entry}, self.taxonomy)

    def test_empty_snapshot_is_renderable(self):
        data = site.public_data({}, self.taxonomy)
        html = site.render_page(data, (site.SOURCE / "index.html").read_text(encoding="utf-8"), "test")
        self.assertIn("0 个项目", html)
        self.assertNotIn('class="repo-card"', html)

    def test_real_snapshot_and_project_relative_assets(self):
        cache = json.loads(site.stars.CACHE_FILE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            data = site.build(output)
            html = (output / "index.html").read_text(encoding="utf-8")
            parser = PageParser()
            parser.feed(html)
            self.assertEqual(set(parser.cards), set(cache))
            self.assertEqual(len(parser.cards), len(cache))
            self.assertEqual(sum(r["stars"] for r in data["repositories"]), sum(int(r.get("stars") or 0) for r in cache.values()))
            self.assertNotIn("{{", html)
            for asset in parser.assets:
                self.assertTrue(asset.startswith("./"), asset)
                self.assertTrue((output / asset).is_file(), asset)
            self.assertEqual({p.name for p in output.iterdir()}, {"index.html", "style.css", "app.js", "favicon.svg", ".nojekyll"})


if __name__ == "__main__":
    unittest.main()
