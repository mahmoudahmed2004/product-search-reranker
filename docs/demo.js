const get = (id) => document.getElementById(id);
const labelNames = { E: "Exact", S: "Substitute", C: "Complement", I: "Irrelevant" };
const formats = {
  score: (value) => value === null ? "—" : value.toFixed(4),
  delta: (value) => value === null ? "—" : (value > 0 ? "+" : "") + value.toFixed(4),
  count: new Intl.NumberFormat("en-US"),
};
const slices = [
  ["all", "All queries"],
  ["short", "Short queries (1–2 tokens)"],
  ["exact_title", "Exact title among candidates"],
];

function element(tag, className, value) {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = value;
  return node;
}

function tone(delta) {
  return delta > 0 ? "positive" : delta < 0 ? "negative" : "";
}

function productRow(product, field, showMovement) {
  const row = document.createElement("li");
  row.className = "result";
  row.dataset.productId = product.product_id;
  const rank = product[field];
  row.value = rank;
  row.append(element("span", "rank", String(rank)));
  row.firstChild.setAttribute("aria-hidden", "true");

  const copy = document.createElement("div");
  const title = element("button", "product", product.title || "Untitled product");
  title.type = "button";
  title.setAttribute("aria-pressed", "false");
  copy.append(title);
  const detail = document.createElement("div");
  detail.className = "detail";
  if (Object.hasOwn(labelNames, product.label)) {
    const relevance = element("span", "relevance", labelNames[product.label]);
    relevance.dataset.label = product.label;
    detail.append(relevance);
  }
  if (showMovement) {
    const change = product.bm25_rank - rank;
    const movement = change > 0 ? "Up " + change : change < 0 ? "Down " + -change : "Same rank";
    detail.append(element("span", tone(change), movement + " · was #" + product.bm25_rank));
  }
  if (detail.childElementCount) copy.append(detail);
  row.append(copy);
  return row;
}

function renderRanking(list, products, field, showMovement = false) {
  const rows = products
    .filter((product) => product[field] <= 10)
    .sort((left, right) => left[field] - right[field])
    .map((product) => productRow(product, field, showMovement));
  list.replaceChildren(...rows);
}

function bindGridControls(grid) {
  grid.addEventListener("click", (event) => {
    const method = event.target.closest(".rank-switch button");
    if (method) {
      grid.dataset.view = method.dataset.view;
      for (const button of grid.querySelectorAll(".rank-switch button")) {
        button.setAttribute("aria-pressed", String(button === method));
      }
      return;
    }

    const clicked = event.target.closest("button.product");
    if (!clicked) return;
    const selected = clicked.getAttribute("aria-pressed") === "true"
      ? null : clicked.closest(".result").dataset.productId;
    for (const row of grid.querySelectorAll(".result")) {
      const active = selected === row.dataset.productId;
      row.classList.toggle("linked", active);
      row.querySelector("button.product").setAttribute("aria-pressed", String(active));
    }
  });
}

function renderCase(entry) {
  renderRanking(get("bm25"), entry.products, "bm25_rank");
  renderRanking(get("model"), entry.products, "lambdarank_rank", true);
  const delta = entry.lambdarank - entry.bm25;
  const stats = [
    ["BM25 nDCG@10", formats.score(entry.bm25), ""],
    ["LambdaRank nDCG@10", formats.score(entry.lambdarank), ""],
    ["Change", formats.delta(delta), tone(delta)],
  ];
  get("case-details").replaceChildren(...stats.map(([label, value, className]) => {
    const score = document.createElement("div");
    score.className = "case-score";
    score.append(element("span", "", label), element("strong", className, value));
    return score;
  }));
  get("status").textContent = "Showing " + Math.min(10, entry.candidate_count) + " of " +
    entry.candidate_count + " judged candidates. Select a title to highlight matching rows.";
}

function renderBenchmark(report) {
  get("benchmark-note").textContent = report.dataset.name +
    (report.dataset.mode === "smoke" ? " · smoke run" : "");
  get("metrics").replaceChildren(...slices.map(([key, label]) => {
    const slice = report.slices[key];
    const row = document.createElement("tr");
    const heading = element("th", "", label);
    heading.scope = "row";
    row.append(
      heading,
      element("td", "", formats.count.format(slice.queries)),
      element("td", "", formats.score(slice.bm25)),
      element("td", "", formats.score(slice.lambdarank)),
      element("td", tone(slice.delta), formats.delta(slice.delta)),
    );
    return row;
  }));
  const { lower, upper } = report.bootstrap;
  get("interval").textContent = "Paired 95% interval for the overall change: " +
    formats.delta(lower) + " to " + formats.delta(upper) + ".";
  get("benchmark").hidden = false;
}

async function loadBenchmark() {
  const picker = get("query");
  try {
    const response = await fetch("benchmark.json");
    if (!response.ok) throw new Error("Missing benchmark");
    const report = await response.json();
    if (!report.cases.length) throw new Error("Missing sample queries");
    renderBenchmark(report);
    picker.replaceChildren(...report.cases.map((entry, index) => new Option(entry.query, index)));
    picker.addEventListener("change", () => renderCase(report.cases[Number(picker.value)]));
    renderCase(report.cases[0]);
    picker.disabled = false;
    get("rankings").hidden = false;
  } catch {
    get("status").textContent = "The benchmark examples could not be loaded. Reload the page to try again.";
  }
}

async function setupLiveRanking() {
  const status = get("live-status");
  const button = get("rank-button");
  const form = get("live-form");
  try {
    if (location.hostname.endsWith(".github.io")) throw new Error("Static hosting");
    const response = await fetch("api/health", { signal: AbortSignal.timeout(4000) });
    if (!response.ok || !(await response.json()).ready) throw new Error("Model unavailable");
  } catch {
    status.textContent = "Live ranking is available when this demo runs with the Python server and model files.";
    form.hidden = true;
    return;
  }

  status.textContent = "The model runs on the titles you enter. Human relevance grades are shown for benchmark queries.";
  button.disabled = false;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    get("live-results").hidden = true;
    const query = get("live-query").value.trim();
    const titles = get("candidates").value.split(/\r?\n/u).map((line) => line.trim()).filter(Boolean);
    if (!query || titles.length < 2 || titles.length > 40 ||
        titles.some((title) => title.length > 240) || new Set(titles).size !== titles.length) {
      status.textContent = "Enter a query and 2–40 distinct product titles, each up to 240 characters.";
      return;
    }
    button.disabled = true;
    form.setAttribute("aria-busy", "true");
    status.textContent = "Ranking products…";
    try {
      const response = await fetch("api/rank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, candidates: titles.map((title) => ({ title })) }),
      });
      if (response.status === 422) {
        status.textContent = "Use letters or numbers in the query and enter distinct product titles.";
        return;
      }
      if (!response.ok) throw new Error("Ranking request failed");
      const result = await response.json();
      renderRanking(get("live-bm25"), result.products, "bm25_rank");
      renderRanking(get("live-model"), result.products, "lambdarank_rank", true);
      status.textContent = "Showing " + Math.min(10, result.candidate_count) + " of " +
        result.candidate_count + " products for “" + result.query + "”.";
      get("live-results").hidden = false;
    } catch {
      status.textContent = "The model could not rank these products. Try again in a moment.";
    } finally {
      form.removeAttribute("aria-busy");
      button.disabled = false;
    }
  });
}

bindGridControls(get("rankings"));
bindGridControls(get("live-results"));
loadBenchmark();
setupLiveRanking();
