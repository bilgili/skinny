# Design — docs-equation-code-mapping

## Context

Both reference docs share one shape and one gap:

```
per equation today:                  per equation after:
  ![eq](diagrams/.../x.svg)            ![eq](diagrams/.../x.svg)
  prose                                prose
  > Implements: fn (file)              <!-- CODE:key sig,core -->
                                       ```slang  (generated)   ```
                                       <!-- /CODE:key -->
                                       | symbol | code | meaning |   (hand-written)
                                       > Implements: fn (file)
```

The `Implements:` pointers already exist and all the named functions exist and
are clean (verified: `reservoirUpdate/Merge/Finalize`, `nf_rqs_fwd/inv`,
`restirEvalRef`, `_mixPdf`, …). The work is surfacing the arithmetic and the
symbol binding next to the equation without letting the copy rot.

## Decisions

### D1 — Granularity: signature + key line (not whole function, not bare line)

A snippet shows the function signature plus the line(s) that perform the
equation's arithmetic. Tiny functions (`reservoirFinalize`, `reservoirMerge`)
are ≈5 lines, so signature + key line is effectively the whole body. Large
functions (`nf_rqs_fwd` ≈26 lines) contribute only their signature and the 2–3
core lines. Rejected: whole-function (adds hundreds of lines across ~21 blocks,
rots fastest) and bare-line (loses the call context the reader needs).

### D2 — Drift control: generator + markers, zero hand-sync

Snippets are sliced out of the live Slang at doc-build time, never typed into the
Markdown. This is the only option that makes a renamed/moved function a build
failure instead of a silent stale paragraph. Precedent: the docs already carry
Node `*.cjs` generators under `docs/diagrams/` for the equation SVGs, so a Node
`embed_code.cjs` is in-idiom and needs no new toolchain.

### D3 — Marker grammar: multi-slice snippets joined by elision (the crux)

"Signature + key line" + "slice contiguous marked regions" only compose if one
snippet can be assembled from several non-adjacent slices, because in a large
function the signature and the key line are far apart.

Slang side — comment-pair markers, `key#slice`:

```slang
// DOC:rqs#sig start
float nf_rqs_fwd(float x, float widths[NF_BINS], float heights[NF_BINS],
                 float derivs[NF_BINS+1], out float logdet)
// DOC:rqs#sig end
        ...
// DOC:rqs#core start
    float num = hgt * (delta*theta*theta + d0*t1);
    float den = delta + (d0 + d1 - 2.0*delta) * t1;
    float y   = y0 + num / den;
// DOC:rqs#core end
```

Markdown side — placeholder names the ordered slice list:

```markdown
<!-- CODE:rqs sig,core -->
```slang
(generated body)
```
<!-- /CODE:rqs -->
```

Generator output: concatenate the named slices in listed order; between two
slices that were **not** adjacent in the source, emit one elision line
(`    // …`) at the leading indent; prepend a single `// from neural_flow.slang`
provenance line. Tiny single-slice snippets (`<!-- CODE:ris-W core -->`) just
emit that slice verbatim.

### D4 — Symbol map: per-equation table (uniform), inline comments allowed too

Each equation gets a small `| symbol | code | meaning |` table. Uniformity is
worth the minor repetition. The table earns most of its keep in Neural, where
names diverge from symbols (`θ→theta`, `s→delta`, `h_k→hgt`, `w_k→w`,
`d_k→d0`, `d_{k+1}→d1`, `y_k→y0`); in ReSTIR the names already ≈ the symbols
(`wSum`, `W`, `pHat`, `M`) so those tables are short, near-confirmation. The
tables are hand-written prose (not generated) — they map intent, which the code
cannot supply.

### D5 — Markers are comments → `.spv` stays valid

`// DOC:` lines change no Slang semantics, so the checked-in `main_pass.spv` /
`skin.spv` need no recompile for this change. Avoids coupling a docs change to a
shader-build step.

### D6 — Generator is also a linter (orphan check both directions)

`embed_code.cjs --check` (CI / pre-archive) fails if: a `CODE:` placeholder names
a `key#slice` with no matching Slang markers; a slice list is empty; or markers
are unbalanced (`start` without `end`). Reverse check (a `DOC:` marker no doc
references) is a warning, not an error — keeps deliberately-reserved markers
legal. This is what turns D2's promise ("can't drift") into something enforced.

### D7 — Scope: both docs in one change, one shared generator

~7 ReSTIR blocks + ~11 Neural blocks = ~18–21 generated snippets, ~8 slang files
marked, one generator. Splitting per-doc would duplicate the generator design
review; the marker grammar must be settled once for both.

## Risks

- **Indentation / brace balance in slices** — a slice may cut mid-block. Mitigated
  by D1 (snippets are signature + a few self-contained arithmetic lines, chosen to
  read standalone) and the fenced ```slang block not needing to compile.
- **Generator becomes load-bearing for the docs** — if `embed_code.cjs` is not run,
  docs go stale silently. Mitigated by D6's `--check` in the verify step and a note
  in `CLAUDE.md` upkeep.
- **Symbol-table drift** (hand-written) — names in the table can lag a rename. The
  generated snippet sitting directly above it makes the mismatch visible in review.
