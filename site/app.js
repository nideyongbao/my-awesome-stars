"use strict";

(() => {
  const data = JSON.parse(document.querySelector("#site-data").textContent);
  const $ = (selector) => document.querySelector(selector);
  const search = $("#search"), language = $("#language"), topic = $("#topic"), sort = $("#sort");
  const grid = $("#repo-grid");
  const cards = new Map([...grid.children].map((card) => [card.dataset.repoId, card]));
  const vaults = new Map(data.vaults.map((vault) => [vault.id, vault]));
  const categories = new Map(data.categories.map((category) => [category.id, category]));
  const searchable = new Map(data.repositories.map((repo) => [repo.id,
    [repo.name, repo.description, ...repo.topics].join(" ").toLocaleLowerCase()
  ]));
  const buttons = [...document.querySelectorAll("[data-vault]")];
  let selectedVault = "";

  function populateTopics(selected = "") {
    topic.replaceChildren(new Option("全部细分类", ""));
    const available = new Set(data.repositories.filter((repo) => !selectedVault || repo.vault === selectedVault).map((repo) => repo.category));
    for (const category of data.categories) {
      if (available.has(category.id)) topic.add(new Option(category.name, category.id));
    }
    topic.value = [...topic.options].some((option) => option.value === selected) ? selected : "";
  }

  function syncUrl() {
    const url = new URL(window.location.href);
    for (const [key, value] of Object.entries({ q: search.value.trim(), category: selectedVault, topic: topic.value, language: language.value, sort: sort.value === "stars" ? "" : sort.value })) {
      if (value) url.searchParams.set(key, value);
      else url.searchParams.delete(key);
    }
    // Local file previews may not permit history changes; filtering still works.
    try { window.history.replaceState(null, "", url); } catch { /* file:// preview */ }
  }

  function render(updateUrl = true) {
    const terms = search.value.trim().toLocaleLowerCase().split(/\s+/u).filter(Boolean);
    const visible = data.repositories.filter((repo) =>
      (!selectedVault || repo.vault === selectedVault) &&
      (!topic.value || repo.category === topic.value) &&
      (!language.value || repo.language === language.value) &&
      terms.every((term) => searchable.get(repo.id).includes(term))
    );
    const timestamp = (value) => Date.parse(value) || 0;
    const compareName = (a, b) => a.name.localeCompare(b.name, "en", { sensitivity: "base" });
    visible.sort((a, b) => {
      if (sort.value === "name") return compareName(a, b);
      if (sort.value === "recent") return timestamp(b.first_seen_at) - timestamp(a.first_seen_at) || compareName(a, b);
      if (sort.value === "updated") return timestamp(b.pushed_at) - timestamp(a.pushed_at) || compareName(a, b);
      return b.stars - a.stars || compareName(a, b);
    });
    grid.replaceChildren(...visible.map((repo) => cards.get(repo.id)));
    $("#empty-state").hidden = visible.length !== 0;
    $("#result-count").textContent = `${visible.length} / ${data.repositories.length} 个项目`;
    const vault = vaults.get(selectedVault);
    $("#results-title").textContent = vault ? vault.name.replace(/^(Domain|Pillar)-/, "") : "全部项目";
    $("#category-description").textContent = categories.get(topic.value)?.description || vault?.description || "按主题整理的开源收藏，找到值得深入的下一个项目。";
    for (const button of buttons) button.setAttribute("aria-pressed", String(button.dataset.vault === selectedVault));
    if (updateUrl) syncUrl();
  }

  function restoreUrl() {
    const params = new URLSearchParams(window.location.search);
    search.value = params.get("q") || "";
    selectedVault = vaults.has(params.get("category")) ? params.get("category") : "";
    populateTopics(params.get("topic") || "");
    language.value = [...language.options].some((option) => option.value === params.get("language")) ? params.get("language") : "";
    sort.value = ["stars", "recent", "updated", "name"].includes(params.get("sort")) ? params.get("sort") : "stars";
    render(false);
  }

  function reset() {
    selectedVault = "";
    search.value = language.value = "";
    sort.value = "stars";
    populateTopics();
    render();
  }

  for (const button of buttons) button.addEventListener("click", () => {
    selectedVault = button.dataset.vault;
    populateTopics();
    render();
  });
  search.addEventListener("input", () => render());
  for (const select of [language, topic, sort]) select.addEventListener("change", () => render());
  $("#reset").addEventListener("click", reset);
  $("#empty-reset").addEventListener("click", reset);
  window.addEventListener("popstate", restoreUrl);
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey &&
      !event.target.closest("input, textarea, select, [contenteditable]")) {
      event.preventDefault();
      search.focus();
    }
  });
  restoreUrl();
})();
