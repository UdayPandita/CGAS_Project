"use strict";

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pretty = (s) => String(s ?? "").replace(/_/g, " ");
const fmt = (v, d = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));

const GOALS = [
  ["energy", "energy"], ["fat", "fat"], ["sugar", "sugar"], ["sodium", "sodium"],
  ["protein", "protein"], ["fiber", "fibre"], ["carbs", "carbs"], ["cholesterol", "cholesterol"],
];
const NUTRIENTS = [
  ["energy_kcal", "Energy", "kcal", 0], ["protein_g", "Protein", "g", 1], ["fat_g", "Fat", "g", 1],
  ["carbohydrate_g", "Carbohydrate", "g", 1], ["sugars_g", "Sugars", "g", 1], ["fiber_g", "Fibre", "g", 1],
  ["sodium_mg", "Sodium", "mg", 0], ["cholesterol_mg", "Cholesterol", "mg", 0],
];

const state = { q: "butter", mode: "flavor", goals: {}, card: null, results: [] };

async function api(path) {
  const r = await fetch(path);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(body.error || `Request failed (${r.status})`), { status: r.status });
  return body;
}

/* ------------------------------------------------------------------ controls */
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === mode)));
  $("#opts-flavor").hidden = mode !== "flavor";
  $("#opts-nutrient").hidden = mode !== "nutrient";
}

function buildGoalChips() {
  const box = $("#goal-chips");
  box.innerHTML = GOALS.map(([key, label]) =>
    `<button type="button" data-goal="${key}" data-state="any" aria-label="${label}: any">${label} <span>·</span></button>`).join("");
  box.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-goal]");
    if (!b) return;
    const next = { any: "lower", lower: "higher", higher: "any" }[b.dataset.state];
    b.dataset.state = next;
    state.goals[b.dataset.goal] = next;
    b.querySelector("span").textContent = next === "lower" ? "↓" : next === "higher" ? "↑" : "·";
    b.setAttribute("aria-label", `${b.textContent.trim()}: ${next}`);
    run();
  });
}

function params() {
  const p = new URLSearchParams({ ingredient: state.q, mode: state.mode });
  if (state.mode === "flavor") {
    p.set("weighting", $("#weighting").value);
    p.set("k", $("#k-f").value);
    if ($("#minconf-f").value) p.set("min_confidence", $("#minconf-f").value);
    if ($("#samecat-f").checked) p.set("same_category", "1");
  } else {
    p.set("catalog", $("#catalog").value);
    p.set("features", $("#features").value);
    p.set("k", $("#k-n").value);
    if ($("#samecat-n").checked) p.set("same_category", "1");
    const lower = Object.keys(state.goals).filter((g) => state.goals[g] === "lower");
    const higher = Object.keys(state.goals).filter((g) => state.goals[g] === "higher");
    if (lower.length) p.set("lower", lower.join(","));
    if (higher.length) p.set("higher", higher.join(","));
  }
  return p;
}

function syncUrl() {
  const u = new URL(location.href);
  u.searchParams.set("q", state.q);
  u.searchParams.set("mode", state.mode);
  history.replaceState(null, "", u.pathname + u.search + location.hash);
}

/* ------------------------------------------------------------------ autocomplete */
function setupSearch() {
  const input = $("#q"), list = $("#suggest");
  let items = [], active = -1, timer;
  const close = () => { list.hidden = true; input.setAttribute("aria-expanded", "false"); active = -1; };
  const choose = (name) => { input.value = pretty(name); state.q = name; close(); run(); };
  const paint = () => {
    list.innerHTML = items.map((it, i) => {
      const modes = [it.flavor ? "flavor" : null, it.nutrient ? "nutrient" : null].filter(Boolean).join(" · ");
      return `<li role="option" id="opt-${i}" aria-selected="${i === active}" data-name="${esc(it.name)}">
        <span>${esc(it.label)}</span><span class="meta">${esc(modes)}</span></li>`;
    }).join("");
    list.hidden = items.length === 0;
    input.setAttribute("aria-expanded", String(!list.hidden));
    if (active >= 0) input.setAttribute("aria-activedescendant", `opt-${active}`);
  };
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (!q) { items = []; paint(); return; }
    timer = setTimeout(async () => {
      try { items = await api(`/api/search?q=${encodeURIComponent(q)}`); active = -1; paint(); } catch { /* keep old list */ }
    }, 120);
  });
  input.addEventListener("keydown", (e) => {
    if (list.hidden) return;
    if (e.key === "ArrowDown") { active = Math.min(items.length - 1, active + 1); paint(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { active = Math.max(0, active - 1); paint(); e.preventDefault(); }
    else if (e.key === "Enter" && active >= 0) { choose(items[active].name); e.preventDefault(); }
    else if (e.key === "Escape") close();
  });
  list.addEventListener("mousedown", (e) => { const li = e.target.closest("li"); if (li) { e.preventDefault(); choose(li.dataset.name); } });
  document.addEventListener("click", (e) => { if (!e.target.closest(".combo")) close(); });
  $("#controls").addEventListener("submit", (e) => {
    e.preventDefault();
    close();
    state.q = input.value.trim() || state.q;
    run();
  });
}

/* ------------------------------------------------------------------ rendering: ingredient card */
function renderCard(card) {
  const el = $("#ingredient");
  if (!card) { el.innerHTML = `<p class="skeleton">Ingredient not found.</p>`; return; }
  const tags = [
    card.group ? `<span class="tag">${esc(card.group)}</span>` : "",
    card.flavor ? `<span class="tag f">${card.compounds} flavor compounds</span>` : `<span class="tag">no flavor data</span>`,
    card.nutrient ? `<span class="tag n">nutrients per 100 g</span>` : `<span class="tag">no nutrient data</span>`,
  ].join("");
  let nut = "";
  if (card.nutrients) {
    const show = [["energy_kcal", "Energy", "kcal", 0], ["protein_g", "Protein", "g", 1], ["fat_g", "Fat", "g", 1],
      ["carbohydrate_g", "Carbs", "g", 1], ["sugars_g", "Sugars", "g", 1], ["sodium_mg", "Sodium", "mg", 0]];
    nut = `<dl class="nut-grid">${show.map(([k, l, u, d]) =>
      `<div><dt>${l}</dt><dd>${fmt(card.nutrients[k], d)} <small>${u}</small></dd></div>`).join("")}</dl>
      <p class="source">Source: ${esc(card.nutrient_source.description)}${card.nutrient_source.origin === "foodb_auto" ? " (automatic FooDB profile)" : ""}</p>`;
  }
  const cooks = card.real_substitutes?.length
    ? `<div class="cooks"><h4>Real cooks substituted it with</h4><ul>${card.real_substitutes.map((s) =>
        `<li><button type="button" data-goto="${esc(s.name)}">${esc(pretty(s.name))}<span>×${s.count}</span></button></li>`).join("")}</ul></div>`
    : `<p class="source">No reported substitutions for this ingredient in Recipe1MSubs.</p>`;
  el.innerHTML = `<h3>${esc(card.label)}</h3><div class="tags">${tags}</div>${nut}${cooks}`;
}

/* ------------------------------------------------------------------ rendering: results */
function renderResults(data) {
  const head = $("#results-head"), list = $("#results"), legend = $("#legend");
  const title = state.mode === "flavor" ? `Tastes like ${pretty(data.query)}` : `Nutritionally like ${pretty(data.query)}`;
  const goals = Object.entries(state.goals).filter(([, v]) => v !== "any").map(([g, v]) => `${v} ${g}`).join(", ");
  const setting = state.mode === "flavor"
    ? ($("#weighting").value === "idf" ? "IDF-weighted Jaccard" : "Jaccard")
    : `${$("#features").value === "macro" ? "macros" : "14 nutrients"}${goals ? " · " + goals : ""}`;
  head.innerHTML = `<h3>${esc(title)}</h3><span class="mono">${esc(setting)}</span>`;
  $("#note").hidden = !data.note;
  $("#note").textContent = data.note || "";

  if (!data.results.length) {
    list.innerHTML = `<li class="skeleton">No ingredient meets these settings. Loosen a goal or a filter.</li>`;
    legend.textContent = "";
    return;
  }
  list.innerHTML = data.results.map((r, i) => state.mode === "flavor" ? flavorRow(r, i) : nutrientRow(r, i)).join("");
  legend.innerHTML = state.mode === "flavor"
    ? "Bar: Jaccard similarity, the share of all their compounds the two ingredients have in common. Whiskers: 95% bootstrap interval. Confidence: High when both have at least 20 known compounds, Low when either has fewer than 5. Select a row to see the shared compounds."
    : "Bar: nutrient similarity, 1 / (1 + distance) between standardised profiles. Values per 100 g. Flavor score is the Jaccard overlap, shown as a taste check. Select a row to compare all nutrients.";
}

function badges(r) {
  return `<div class="badges"><span class="pill ${esc(r.confidence)}">${esc(r.confidence)}</span>${
    r.cooks_used ? `<span class="pill cooks" title="Real cooks reported this substitution ${r.cooks_used} times">cooks ×${r.cooks_used}</span>` : ""}</div>`;
}

function flavorRow(r, i) {
  const w = Math.max(0, Math.min(1, r.similarity)) * 100;
  const lo = (r.ci_low ?? r.similarity) * 100, hi = (r.ci_high ?? r.similarity) * 100;
  return `<li class="res" data-i="${i}">
    <button type="button" class="res-main" aria-expanded="false">
      <span class="rank">${r.rank}</span>
      <span class="who"><b>${esc(r.label)}</b><small>${esc(pretty(r.category))}</small></span>
      <span class="meter">
        <span class="bar"><span class="fill" style="width:${w}%;background:var(--flavor)"></span>
          <span class="ci" style="left:${lo}%;width:${Math.max(0.5, hi - lo)}%"></span></span>
        <span class="meter-txt"><b>J ${fmt(r.similarity)}</b><span>CI ${fmt(r.ci_low)}–${fmt(r.ci_high)}</span>
          <span>${r.shared_compounds} shared</span><span>top-5 in ${Math.round((r.top5_stability ?? 0) * 100)}% of resamples</span></span>
      </span>
      ${badges(r)}
    </button>
    <div class="detail" hidden></div>
  </li>`;
}

function nutrientRow(r, i) {
  const w = Math.max(0, Math.min(1, r.nutrient_similarity)) * 100;
  const fs = r.flavor_similarity === null || r.flavor_similarity === undefined ? "no flavor data" : `flavor ${fmt(r.flavor_similarity)}`;
  return `<li class="res" data-i="${i}">
    <button type="button" class="res-main" aria-expanded="false">
      <span class="rank">${r.rank}</span>
      <span class="who"><b>${esc(r.label)}</b><small>${esc(pretty(r.category))}${r.origin === "foodb_auto" ? " · FooDB" : ""}</small></span>
      <span class="meter">
        <span class="bar"><span class="fill" style="width:${w}%;background:var(--nutrient)"></span></span>
        <span class="meter-txt"><b>${fmt(r.nutrient_similarity)}</b><span>${fmt(r.energy_kcal, 0)} kcal</span>
          <span>P ${fmt(r.protein_g, 1)}</span><span>F ${fmt(r.fat_g, 1)}</span><span>C ${fmt(r.carbohydrate_g, 1)} g</span><span>${esc(fs)}</span></span>
      </span>
      ${badges(r)}
    </button>
    <div class="detail" hidden></div>
  </li>`;
}

async function toggleDetail(li) {
  const btn = $(".res-main", li), det = $(".detail", li);
  const open = btn.getAttribute("aria-expanded") === "true";
  btn.setAttribute("aria-expanded", String(!open));
  det.hidden = open;
  if (open || det.dataset.loaded) return;
  const r = state.results[Number(li.dataset.i)];
  if (state.mode === "flavor") {
    det.innerHTML = `<p class="skeleton">Loading shared compounds…</p>`;
    try {
      const s = await api(`/api/shared?a=${encodeURIComponent(state.q)}&b=${encodeURIComponent(r.substitute)}`);
      det.innerHTML = `<p>${esc(pretty(s.a))} has ${s.n_a} known compounds and ${esc(pretty(s.b))} has ${s.n_b}; they share ${s.n_shared}.
        ${s.n_shared ? "Rarest first, with how many of 1,525 ingredients contain each:" : ""}</p>
        <div class="chips">${s.compounds.map((c) => `<span class="chip">${esc(c.name)} <i>${c.found_in}</i></span>`).join("")}</div>`;
    } catch (e) { det.innerHTML = `<p class="skeleton">${esc(e.message)}</p>`; }
  } else {
    const q = state.card?.nutrients;
    const rows = NUTRIENTS.map(([k, l, u, d]) => {
      const a = q ? q[k] : null, b = r[k];
      const delta = a === null || a === undefined || b === null ? "" : b - a;
      const cls = delta === "" || Math.abs(delta) < 1e-9 ? "" : delta > 0 ? "up" : "down";
      const sign = delta === "" ? "–" : `${delta > 0 ? "+" : ""}${Number(delta).toFixed(d)}`;
      return `<tr><td>${l} (${u})</td><td>${fmt(a, d)}</td><td>${fmt(b, d)}</td><td class="${cls}">${sign}</td></tr>`;
    }).join("");
    det.innerHTML = `<div class="cmp"><table><thead><tr><th>per 100 g</th><th>${esc(pretty(state.q))}</th><th>${esc(r.label)}</th><th>change</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="source">${r.match_type === "proxy" ? "Nutrients come from a close stand-in food. " : ""}${r.identical_record ? "Same food record as the query, so nutrition cannot tell them apart." : ""}</p>`;
  }
  det.dataset.loaded = "1";
}

/* ------------------------------------------------------------------ run */
let runId = 0;
async function run() {
  const id = ++runId;
  syncUrl();
  $("#error").hidden = true;
  $("#results").innerHTML = `<li class="skeleton">Finding substitutes…</li>`;
  const [card, rec] = await Promise.allSettled([
    api(`/api/ingredient/${encodeURIComponent(state.q)}`), api(`/api/recommend?${params()}`),
  ]);
  if (id !== runId) return;
  state.card = card.status === "fulfilled" ? card.value : null;
  renderCard(state.card);
  if (state.card) $("#q").value = pretty(state.card.label === state.card.name ? state.card.name : state.card.label);
  if (rec.status === "fulfilled") {
    state.results = rec.value.results;
    renderResults(rec.value);
  } else {
    state.results = [];
    $("#results").innerHTML = "";
    $("#results-head").innerHTML = "";
    $("#legend").textContent = "";
    $("#note").hidden = true;
    const err = $("#error");
    err.hidden = false;
    const other = state.mode === "flavor" ? "nutrient" : "flavor";
    const canSwitch = state.card && state.card[other];
    err.innerHTML = `${esc(rec.reason.message)}${canSwitch ? ` <button type="button" id="switch">Switch to ${other} mode</button>` : ""}`;
    if (canSwitch) $("#switch").onclick = () => { setMode(other); run(); };
  }
}

/* ------------------------------------------------------------------ boot */
function boot() {
  const u = new URL(location.href);
  if (u.searchParams.get("q")) state.q = u.searchParams.get("q");
  setMode(u.searchParams.get("mode") === "nutrient" ? "nutrient" : "flavor");
  $("#q").value = pretty(state.q);
  buildGoalChips();
  setupSearch();
  document.querySelectorAll(".mode button").forEach((b) => b.addEventListener("click", () => { setMode(b.dataset.mode); run(); }));
  ["#weighting", "#minconf-f", "#samecat-f", "#k-f", "#catalog", "#features", "#samecat-n", "#k-n"]
    .forEach((s) => $(s).addEventListener("change", run));
  $("#results").addEventListener("click", (e) => { const li = e.target.closest(".res"); if (li && e.target.closest(".res-main")) toggleDetail(li); });
  $("#ingredient").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-goto]");
    if (b) { state.q = b.dataset.goto; $("#q").value = pretty(state.q); run(); }
  });
  run();
}

boot();
