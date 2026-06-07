#!/usr/bin/env node
//
// embed_code.cjs — embed excerpts of the Slang shaders into the reference docs.
//
// The reference docs (docs/ReSTIR.md, docs/NeuralGuiding.md) show, beneath each
// equation, the Slang that computes it. Those excerpts are NOT hand-typed: they
// are sliced out of the live `.slang` sources by this generator, so they cannot
// silently drift. A renamed/removed shader function becomes a build failure
// (run with `--check`) instead of a stale paragraph.
//
//   Slang side   : // DOC:<key>#<slice> start  …  // DOC:<key>#<slice> end
//                  comment pairs wrap the regions to excerpt. Markers are
//                  comments only — no shader semantics, no `.spv` recompile.
//   Markdown side: <!-- CODE:<key> slice1,slice2 -->
//                  ```slang  (generated body)  ```
//                  <!-- /CODE:<key> -->
//                  The generator fills the fenced block with `// from <file>.slang`
//                  provenance + the named slices in order, joining non-adjacent
//                  source slices with an elision line (`    // …`).
//
// Usage (run from the repo root; needs only Node, no packages):
//   node docs/diagrams/embed_code.cjs          # rewrite the docs in place
//   node docs/diagrams/embed_code.cjs --check   # verify, exit non-zero on drift
//
// Adding an excerpt: wrap the source lines in a `// DOC:key#slice start/end`
// pair, drop a `<!-- CODE:key slice -->`…`<!-- /CODE:key -->` placeholder under
// the equation, and re-run. Multi-slice excerpts (signature + key line) list the
// slices comma-separated: `<!-- CODE:rqs sig,core -->`.

const fs = require("fs");
const path = require("path");

const REPO = path.resolve(__dirname, "..", "..");
const SHADER_ROOT = path.join(REPO, "src", "skinny", "shaders");
const DOCS = [
  path.join(REPO, "docs", "ReSTIR.md"),
  path.join(REPO, "docs", "NeuralGuiding.md"),
];

const CHECK = process.argv.includes("--check");

// ── collect every .slang under the shader root ─────────────────────────────
function walk(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.name.endsWith(".slang")) out.push(p);
  }
  return out;
}

// ── parse `// DOC:key#slice start/end` regions → slices[`key#slice`] ────────
// Each slice records its source file basename and the exact code lines between
// (exclusive of) the marker lines, plus the source line numbers so the doc side
// can decide whether two slices were adjacent.
const slices = {};        // "key#slice" -> { file, firstLine, lastLine, lines: [] }
const errors = [];

const START_RE = /^\s*\/\/\s*DOC:([A-Za-z0-9_-]+)#([A-Za-z0-9_-]+)\s+start\s*$/;
const END_RE = /^\s*\/\/\s*DOC:([A-Za-z0-9_-]+)#([A-Za-z0-9_-]+)\s+end\s*$/;

for (const file of walk(SHADER_ROOT)) {
  const base = path.basename(file);
  const lines = fs.readFileSync(file, "utf8").split("\n");
  let open = null; // { id, firstLine, lines: [] }
  lines.forEach((line, i) => {
    const s = START_RE.exec(line);
    const e = END_RE.exec(line);
    if (s) {
      const id = `${s[1]}#${s[2]}`;
      if (open) errors.push(`${base}:${i + 1}: DOC marker '${id}' opened while '${open.id}' still open`);
      open = { id, firstLine: i + 2, lines: [] };
    } else if (e) {
      const id = `${e[1]}#${e[2]}`;
      if (!open) errors.push(`${base}:${i + 1}: DOC end '${id}' with no matching start`);
      else {
        if (open.id !== id) errors.push(`${base}:${i + 1}: DOC end '${id}' closes a different start '${open.id}'`);
        if (slices[open.id]) errors.push(`${base}:${i + 1}: duplicate DOC marker '${open.id}'`);
        slices[open.id] = { file: base, firstLine: open.firstLine, lastLine: i, lines: open.lines };
        open = null;
      }
    } else if (open) {
      open.lines.push(line);
    }
  });
  if (open) errors.push(`${base}: DOC marker '${open.id}' opened but never closed`);
}

// ── render one CODE placeholder's body from its ordered slice list ──────────
const referenced = new Set();

function renderBlock(key, sliceNames) {
  if (sliceNames.length === 0) {
    errors.push(`CODE:${key}: empty slice list`);
    return null;
  }
  const picked = [];
  let file = null;
  for (const name of sliceNames) {
    const id = `${key}#${name}`;
    const sl = slices[id];
    if (!sl) {
      errors.push(`CODE:${key}: no Slang DOC marker for slice '${id}'`);
      return null;
    }
    referenced.add(id);
    file = file || sl.file;
    picked.push(sl);
  }
  const body = [`// from ${file}`];
  picked.forEach((sl, i) => {
    if (i > 0) {
      const prev = picked[i - 1];
      // adjacent in source (and same file) ⇒ no elision; otherwise elide.
      const adjacent = sl.file === prev.file && sl.firstLine <= prev.lastLine + 1;
      if (!adjacent) body.push("    // …");
    }
    body.push(...sl.lines);
  });
  return ["```slang", ...body, "```"].join("\n");
}

// ── rewrite each doc's CODE regions ────────────────────────────────────────
const OPEN_RE = /^<!--\s*CODE:([A-Za-z0-9_-]+)\s+([A-Za-z0-9_,-]+)\s*-->\s*$/;
const CLOSE_RE = /^<!--\s*\/CODE:([A-Za-z0-9_-]+)\s*-->\s*$/;

let changed = false;
for (const doc of DOCS) {
  if (!fs.existsSync(doc)) {
    errors.push(`${path.basename(doc)}: doc not found`);
    continue;
  }
  const src = fs.readFileSync(doc, "utf8");
  const lines = src.split("\n");
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const m = OPEN_RE.exec(lines[i]);
    if (!m) {
      out.push(lines[i]);
      continue;
    }
    const key = m[1];
    const sliceNames = m[2].split(",").map((s) => s.trim()).filter(Boolean);
    out.push(lines[i]); // keep the opening placeholder
    // find the matching close
    let j = i + 1;
    while (j < lines.length && !CLOSE_RE.test(lines[j])) j++;
    if (j >= lines.length) {
      errors.push(`${path.basename(doc)}: CODE:${key} has no closing <!-- /CODE:${key} -->`);
      continue;
    }
    const block = renderBlock(key, sliceNames);
    if (block !== null) out.push(block);
    out.push(lines[j]); // closing placeholder
    i = j;
  }
  const result = out.join("\n");
  if (result !== src) {
    changed = true;
    if (!CHECK) fs.writeFileSync(doc, result);
    console.log(`${CHECK ? "would update" : "updated"} ${path.relative(REPO, doc)}`);
  }
}

// ── warn on DOC markers no doc references (not an error) ────────────────────
for (const id of Object.keys(slices)) {
  if (!referenced.has(id)) console.warn(`warning: DOC marker '${id}' (${slices[id].file}) is not referenced by any doc`);
}

if (errors.length) {
  console.error("\nembed_code: errors:");
  for (const e of errors) console.error("  ✗ " + e);
  process.exit(1);
}
if (CHECK && changed) {
  console.error("\nembed_code --check: docs are stale; run `node docs/diagrams/embed_code.cjs` and commit.");
  process.exit(1);
}
console.log(CHECK ? "embed_code --check: docs in sync." : "embed_code: done.");
