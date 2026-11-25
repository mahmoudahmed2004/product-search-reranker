const key = "assembly-context-v1";
const ttl = 24 * 60 * 60 * 1000;

function load() {
  try {
    return JSON.parse(localStorage.getItem(key) || "{}") || {};
  } catch {
    return {};
  }
}

function update(fields) {
  try {
    localStorage.setItem(key, JSON.stringify({ ...load(), ...fields }));
  } catch {
    // Browsing still works when local storage is unavailable.
  }
}

export function recordSearch(query, collection) {
  update({ search: { query, collection, timestamp: Date.now() } });
}

export function recordView(slug) {
  update({ view: { slug, timestamp: Date.now() } });
}

export function latestContext() {
  const { search, view } = load();
  const now = Date.now();
  const fresh = (event) => Number.isFinite(event?.timestamp)
    && event.timestamp <= now && now - event.timestamp <= ttl;
  if (fresh(search) && typeof search.query === "string" && search.query.trim()) {
    return new URLSearchParams({ q: search.query, collection: search.collection || "" });
  }
  if (fresh(view) && typeof view.slug === "string" && view.slug) {
    return new URLSearchParams({ viewed: view.slug });
  }
  return null;
}

export function clearContext() {
  try {
    localStorage.removeItem(key);
  } catch {
    // The current homepage can still be cleared in this session.
  }
}
