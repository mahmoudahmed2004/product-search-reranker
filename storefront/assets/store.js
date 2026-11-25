import { clearContext, latestContext, recordSearch, recordView } from "./home-state.js";

const select = (id) => document.getElementById(id);
const main = select("main");
const dialog = select("bag");
const storageKey = "assembly-bag-v1";
const dollars = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const labels = {
  computers: "Computers", phones: "Phones", tablets: "Tablets",
  audio: "Audio", accessories: "Accessories", workspace: "Workspace",
  kitchen: "Kitchen & dining", sports: "Sports & outdoors", home: "Home & living",
  beauty: "Beauty & care", style: "Style", pantry: "Pantry", pets: "Pets",
};

function element(tag, classes = "", text) {
  const node = document.createElement(tag);
  node.className = classes;
  if (text !== undefined) node.textContent = text;
  return node;
}

function link(text, href, classes = "") {
  const node = element("a", classes, text);
  node.href = href;
  return node;
}

function button(text, classes = "") {
  const node = element("button", classes, text);
  node.type = "button";
  return node;
}

function photo(url, title, lazy = true) {
  const frame = element("div", "photo");
  const img = element("img");
  img.alt = title;
  img.loading = lazy ? "lazy" : "eager";
  img.src = url;
  img.addEventListener("error", () => {
    frame.replaceChildren(element("span", "photo-fallback", title || "Photo unavailable"));
  });
  frame.append(img);
  return frame;
}

const productUrl = (product) => "/store/p/" + encodeURIComponent(product.slug);
const price = (product) => dollars.format(product.price_cents / 100);

function card(product, reason) {
  const article = element("article", "product-card");
  const anchor = link(undefined, productUrl(product), "card-link");
  const media = photo(product.image, "");
  media.classList.add("card-visual");
  const info = element("div", "card-info");
  if (product.brand) info.append(element("p", "card-brand", product.brand));
  info.append(element("h3", "card-title", product.title), element("p", "card-price", price(product)));
  if (reason) info.append(element("p", "rec-reason", reason));
  anchor.append(media, info);
  article.append(anchor);
  return article;
}

async function api(path) {
  const response = await fetch("/api/store" + path);
  if (!response.ok) {
    const error = new Error("Catalog request failed.");
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function hero(products) {
  const area = element("section", "hero");
  const copy = element("div", "hero-copy");
  copy.append(
    element("h1", "", "Good finds for every day."),
    element("p", "", "Find kitchen tools, sports gear, and essentials for home and work."),
    link("Shop all departments", "#catalog", "primary"),
  );
  const displays = element("div", "aisle-display");
  displays.append(...[52, 152, 119].map((id) => products.find((product) => product.id === id))
    .filter(Boolean).map((product) => {
      const tile = link(undefined, "/store/?collection=" + product.collection, "aisle");
      tile.append(photo(product.image, "", false),
        element("span", "aisle-label", labels[product.collection]));
      return tile;
    }));
  area.append(copy, displays);
  return area;
}

async function readHomeContext() {
  const context = latestContext();
  if (!context) return null;
  try {
    const result = await api("/home-context?" + context);
    return result.source === "none" ? null : result;
  } catch {
    return null;
  }
}

function contextPanel(context, products) {
  const panel = element("section", "context-panel");
  const heading = element("div", "context-header");
  const copy = element("div");
  const source = context.source === "search" ? "From your latest search" : "Recently viewed";
  const topic = labels[context.collection];
  copy.append(element("h1", "", context.label), element("p", "context-source",
    topic ? source + " · " + topic : source));
  heading.append(copy);
  const clear = button("Clear activity", "quiet");
  clear.addEventListener("click", () => {
    clearContext();
    const replacement = hero(products);
    replacement.id = "home-discovery";
    panel.replaceWith(replacement);
    select("announcer").textContent = "Shopping activity cleared.";
  });
  heading.append(clear);
  panel.append(heading);
  if (context.products.length) {
    const grid = element("div", "context-grid");
    grid.append(...context.products.map((product) => card(product)));
    panel.append(grid);
  } else {
    panel.append(element("p", "context-empty", "No close recommendations. Browse the departments below."));
  }
  return panel;
}

async function browse() {
  const params = new URLSearchParams(location.search);
  const query = params.get("q")?.trim() || "";
  const collection = params.get("collection") || "";
  const rank = params.get("rank") === "reranked" ? "reranked" : "hybrid";
  select("catalog-search").value = query;
  for (const anchor of document.querySelectorAll("[data-collection]")) {
    if (anchor.dataset.collection === collection) anchor.setAttribute("aria-current", "page");
  }
  const filters = new URLSearchParams({ q: query, collection, rank });
  const home = !query && !collection;
  const [{ products, total }, context] = await Promise.all([
    api("/catalog?" + filters),
    home ? readHomeContext() : null,
  ]);
  if (query) recordSearch(query, collection);
  const results = element("section", "catalog");
  results.id = "catalog";
  const heading = element("div", "section-head");
  const intro = element("div");
  const title = query ? "Results for “" + query + "”" : labels[collection] || "Shop all";
  intro.append(element(query || collection ? "h1" : "h2", "", title),
    element("p", "muted", total + (total === 1 ? " product" : " products")));
  heading.append(intro);
  results.append(heading);
  if (products.length) {
    const sort = element("select");
    sort.add(new Option(query ? "Relevance" : "Featured", "default"));
    sort.add(new Option("Price: low to high", "low"));
    sort.add(new Option("Price: high to low", "high"));
    const label = element("label", "sort", "Sort by");
    label.append(sort);
    const controls = element("div", "result-controls");
    if (query) {
      const mode = element("select");
      mode.setAttribute("aria-label", "Ranking");
      mode.add(new Option("Hybrid", "hybrid"));
      mode.add(new Option("Hybrid + reranker", "reranked"));
      mode.value = rank;
      const field = element("label", "sort", "Ranking");
      field.append(mode);
      mode.addEventListener("change", () => {
        filters.set("rank", mode.value);
        location.search = filters.toString();
      });
      controls.append(field);
    }
    controls.append(label);
    heading.append(controls);
    const grid = element("div", "product-grid");
    const more = button("Show more products", "quiet catalog-more");
    let visible = 16;
    const order = () => sort.value === "default" ? products : [...products].sort(
      (a, b) => (a.price_cents - b.price_cents) * (sort.value === "low" ? 1 : -1),
    );
    const render = () => {
      grid.replaceChildren(...order().slice(0, visible).map((product) => card(product)));
      more.hidden = visible >= products.length;
    };
    sort.addEventListener("change", () => {
      visible = 16;
      render();
    });
    more.addEventListener("click", () => {
      const batch = order().slice(visible, visible + 16);
      grid.append(...batch.map((product) => card(product)));
      visible += batch.length;
      more.hidden = visible >= products.length;
      select("announcer").textContent = "Showing " + visible + " products.";
    });
    render();
    results.append(grid, more);
  } else {
    const empty = element("div", "empty");
    empty.append(element("h2", "", "No products found"),
      element("p", "", "Try a different search or browse all products."),
      link("View all products", "/store/", "quiet"));
    results.append(empty);
  }
  if (home) {
    const display = context ? contextPanel(context, products) : hero(products);
    display.id = "home-discovery";
    main.replaceChildren(display, results);
    window.addEventListener("pageshow", async (event) => {
      if (!event.persisted) return;
      const fresh = await readHomeContext();
      const next = fresh ? contextPanel(fresh, products) : hero(products);
      next.id = "home-discovery";
      select("home-discovery")?.replaceWith(next);
    });
  } else {
    main.replaceChildren(results);
  }
}

function gallery(product) {
  const section = element("div", "gallery");
  const urls = product.images.length ? product.images : [product.image];
  const makePhoto = (url) => {
    const display = photo(url, product.title, false);
    display.classList.add("main-photo");
    return display;
  };
  let display = makePhoto(urls[0]);
  section.append(display);
  if (urls.length <= 1) return section;

  const selectors = element("div", "thumbnails");
  selectors.setAttribute("aria-label", "Product images");
  const pickers = urls.map((url, index) => {
    const picker = button(undefined, "thumbnail");
    picker.setAttribute("aria-label", "View photo " + (index + 1));
    picker.setAttribute("aria-pressed", String(index === 0));
    picker.append(photo(url, ""));
    picker.addEventListener("click", () => {
      const next = makePhoto(url);
      display.replaceWith(next);
      display = next;
      for (const peer of pickers) peer.setAttribute("aria-pressed", String(peer === picker));
    });
    return picker;
  });
  selectors.append(...pickers);
  section.append(selectors);
  return section;
}

function recommendationRail(title, subtitle, rows, empty) {
  const section = element("section", "rec-section");
  const head = element("div", "rec-head");
  head.append(element("h2", "", title), element("p", "", subtitle));
  section.append(head);
  if (rows.length) {
    const grid = element("div", "rec-grid");
    grid.append(...rows.map(({ product, reason }) => card(product, reason)));
    section.append(grid);
  } else {
    section.append(element("p", "rec-empty", empty));
  }
  return section;
}

async function showProduct(slug) {
  const { product, similar, complements } = await api("/products/" + encodeURIComponent(slug));
  recordView(product.slug);
  document.title = product.title + " · Assembly";
  const trail = element("nav", "crumbs");
  trail.setAttribute("aria-label", "Breadcrumb");
  trail.append(link("Shop", "/store/"), element("span", "", "/"),
    link(labels[product.collection], "/store/?collection=" + product.collection));
  const item = element("div", "product-detail");
  const description = element("div", "detail-copy");
  if (product.brand) description.append(element("p", "card-brand", product.brand));
  description.append(
    element("h1", "", product.title),
    element("p", "detail-price", price(product)),
    element("p", "description", product.description),
    element("p", product.stock ? "stock" : "stock unavailable", product.stock ? "In stock" : "Out of stock"),
  );
  const add = button("Add to bag", "primary");
  add.disabled = !product.stock;
  add.addEventListener("click", () => {
    changeQuantity(product, 1);
    showBag();
  });
  description.append(add);
  item.append(gallery(product), description);
  main.replaceChildren(
    trail,
    item,
    recommendationRail("Similar products", "Explore alternatives in the same product type.", similar,
      "There are no other in-stock products of this type."),
    recommendationRail("Complete your setup", "Find products that go with this item.", complements,
      "No matching add-ons are available for this product."),
  );
}

function readBag() {
  try {
    const rows = JSON.parse(localStorage.getItem(storageKey) || "[]");
    return Array.isArray(rows) ? rows.filter(({ product, quantity }) =>
      product?.id && Number.isInteger(quantity) && quantity > 0) : [];
  } catch {
    return [];
  }
}

let bag = readBag();

function updateBag() {
  try {
    localStorage.setItem(storageKey, JSON.stringify(bag));
  } catch {
    // Keep this session usable when browser storage is disabled.
  }
  select("bag-count").textContent = bag.reduce((sum, item) => sum + item.quantity, 0);
  if (dialog.open) renderBag();
}

function changeQuantity(product, difference) {
  const index = bag.findIndex((row) => row.product.id === product.id);
  const before = index >= 0 ? bag[index].quantity : 0;
  const quantity = Math.min(product.stock, before + difference);
  if (quantity > 0 && index >= 0) bag[index] = { product, quantity };
  else if (quantity > 0) bag.push({ product, quantity });
  else if (index >= 0) bag.splice(index, 1);
  select("announcer").textContent = quantity === before
    ? "You have the maximum available quantity." : product.title + " quantity " + quantity;
  updateBag();
}

function renderBag() {
  const rows = bag.map(({ product, quantity }) => {
    const row = element("div", "bag-row");
    const info = element("div");
    info.append(link(product.title, productUrl(product)), element("p", "muted", price(product)));
    const controls = element("div", "quantity");
    const fewer = button("−", "quiet");
    fewer.setAttribute("aria-label", "Remove one " + product.title);
    fewer.addEventListener("click", () => changeQuantity(product, -1));
    const more = button("+", "quiet");
    more.setAttribute("aria-label", "Add one " + product.title);
    more.disabled = quantity >= product.stock;
    more.addEventListener("click", () => changeQuantity(product, 1));
    controls.append(fewer, element("span", "", String(quantity)), more);
    row.append(info, controls);
    return row;
  });
  select("bag-items").replaceChildren(
    ...(rows.length ? rows : [element("p", "page-status", "Your bag is empty.")]),
  );
  const subtotal = bag.reduce((sum, row) => sum + row.product.price_cents * row.quantity, 0);
  select("bag-total").textContent = dollars.format(subtotal / 100);
}

function showBag() {
  renderBag();
  if (!dialog.open) dialog.showModal();
}

select("open-bag").addEventListener("click", showBag);
updateBag();
const slug = location.pathname.split("/")[3];
(slug ? showProduct(slug) : browse()).catch((error) => {
  const section = element("section", "empty");
  section.append(element("h1", "", error.status === 404 ? "Product not found" : "We couldn’t load the store"),
    element("p", "", "Return to the catalog and try again."),
    link("Back to the catalog", "/store/", "primary"));
  main.replaceChildren(section);
});
