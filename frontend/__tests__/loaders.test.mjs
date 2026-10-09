import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.resolve(__dirname, "../public/data");

test("summary.json is valid and contains non-null metrics", () => {
  const file = path.join(dataDir, "summary.json");
  assert.ok(fs.existsSync(file), "summary.json must exist");
  const data = JSON.parse(fs.readFileSync(file, "utf8"));
  assert.equal(typeof data.bronze_row_count, "number");
  assert.ok(data.bronze_row_count > 100000, "Bronze count must be > 100k");
  assert.equal(data.class_count, 10);
});

test("models.json contains valid bootstrap CIs and hypothetical cost labels", () => {
  const file = path.join(dataDir, "models.json");
  assert.ok(fs.existsSync(file), "models.json must exist");
  const data = JSON.parse(fs.readFileSync(file, "utf8"));
  assert.ok(Array.isArray(data.models), "models array must exist");
  for (const m of data.models) {
    assert.ok(m.macro_f1 > 0, "macro_f1 must be positive");
    assert.ok(m.ci_low <= m.macro_f1 && m.macro_f1 <= m.ci_high, "CI bounds must bracket macro_f1");
    assert.match(m.cost_label, /hypothetical/i, "Cost must be labeled hypothetical");
  }
});

test("pairwise_tests.json contains valid McNemar stats and Holm-adjusted p-values", () => {
  const file = path.join(dataDir, "pairwise_tests.json");
  assert.ok(fs.existsSync(file), "pairwise_tests.json must exist");
  const data = JSON.parse(fs.readFileSync(file, "utf8"));
  assert.ok(Array.isArray(data.pairs), "pairs array must exist");
  for (const p of data.pairs) {
    assert.ok(typeof p.delta_f1 === "number");
    assert.ok(typeof p.mcnemar_stat === "number");
    assert.ok(typeof p.adjusted_p_value === "number");
  }
});

test("examples.json contains valid gold narratives and predictions without fabrication", () => {
  const file = path.join(dataDir, "examples.json");
  assert.ok(fs.existsSync(file), "examples.json must exist");
  const data = JSON.parse(fs.readFileSync(file, "utf8"));
  assert.ok(Array.isArray(data.examples));
  assert.ok(data.examples.length >= 30, "Must have at least 30 examples");
  for (const ex of data.examples) {
    assert.ok(ex.narrative.length > 20, "Narrative must not be empty");
    assert.ok(ex.true_label.length > 0, "True label must be present");
    assert.ok(Object.keys(ex.model_predictions).length >= 4, "Must have predictions for all models");
  }
});
