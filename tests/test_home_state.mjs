import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";
import { clearContext, latestContext, recordSearch, recordView } from "../storefront/assets/home-state.js";

beforeEach(() => {
  const store = new Map();
  globalThis.localStorage = {
    getItem: (key) => store.get(key) ?? null,
    setItem: (key, value) => store.set(key, value),
    removeItem: (key) => store.delete(key),
  };
});

test("the newest search takes priority over subsequent product views", () => {
  recordSearch("steel cookware", "kitchen");
  recordView("iphone-13-pro");
  assert.equal(latestContext().get("q"), "steel cookware");
  assert.equal(latestContext().get("collection"), "kitchen");
  recordSearch("tennis racket", "sports");
  assert.equal(latestContext().get("q"), "tennis racket");
});

test("expired search yields to a new view and clear removes both signals", (t) => {
  const clock = t.mock.method(Date, "now", () => 100);
  recordSearch("cat food", "pets");
  clock.mock.mockImplementation(() => 24 * 60 * 60 * 1000 + 101);
  assert.equal(latestContext(), null);
  recordView("tennis-racket");
  assert.equal(latestContext().get("viewed"), "tennis-racket");
  clearContext();
  assert.equal(latestContext(), null);
});
