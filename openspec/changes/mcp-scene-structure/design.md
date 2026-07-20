# Design: mcp-scene-structure

## Context

The in-process MCP server (`mcp_server.py`) serves three path-addressed tools
over the render-thread command queue; every renderer touch happens inside a
posted closure awaited with a timeout (spec `mcp-scene-control`). The renderer
already implements structural editing against a non-destructive USD edit
sublayer: `add_model` (AddReference under a new Xform, `renderer.py:6034`),
`add_light` (UsdLux schema table, `:6085`), `remove_node` (`SetActive(False)`,
`:6173`), `save_edits` (`:6219`), each with rollback on failure and a full
`_resync_geometry_from_stage()` re-read. `render_session.py` exposes them as
awaited proxy verbs; both GUI docks call them. The MCP client is an AI agent on
the same machine (loopback + bearer token), with its own file tools — it can
author arbitrary `.usda` content to disk and reference it.

Two gaps prevent agent-driven scene composition: the MCP surface lacks the
structural verbs, and a bare gprim defined in the edit layer would bind to the
protected fallback material (slot 0, `apply_material_override` refuses edits at
`renderer.py:7003`) and could never be re-colored.

## Goals / Non-Goals

**Goals:**

- An MCP agent can compose a scene: add referenced USD files, primitives with
  editable materials, lights; remove nodes; persist the edit layer.
- Filesystem arguments are confined to configurable allowlisted roots, enforced
  deeply enough to catch USD composition pulling files from elsewhere.
- A structural call whose resync outlasts the request window degrades to a
  pollable job instead of a cancelled-but-completed mystery.

**Non-Goals:**

- No `usd_text` upload parameter — the agent writes its own files (C1). The
  server never becomes a file-content sink.
- No rendered-image tool (accumulation reset makes an immediate readback
  near-noise; unchanged from `mcp-scene-control`).
- No OS-level sandboxing; the allowlist is a guardrail against agent mistakes
  within one trust domain, not a privilege boundary.
- No new GPU/loader work: `tessellate_gprim` (`usd_loader.py:1979`) already
  meshes Sphere/Cube/Cylinder/Cone/Capsule/Plane; resync/upload paths reused.

## Decisions

### D1 — Agent writes files; tools take paths (C1 over wire-content C2)

The client is local and already has file tools; the server is loopback+token.
Accepting USD text over the socket would duplicate a client capability and turn
the renderer into a file writer. Rejected alternative: `usd_text` param for
sandboxed remote clients — no such client exists in this deployment.

### D2 — Path allowlist enforced at three levels (P3 + D2 + T2)

Roots resolution: `--mcp-roots` CLI flag > `SKINNY_MCP_ROOTS` env > default
`[realpath(tempfile.gettempdir()), realpath("/tmp"), realpath(os.getcwd())]`.
Both `/tmp` spellings are required: macOS `gettempdir()` is per-user
`/var/folders/…` while an agent's `/tmp/foo.usda` realpaths to `/private/tmp` —
omitting either produces mystery rejections. Precedence mirrors
`backend_select.py`.

Enforcement (all realpath-then-prefix checks):

1. **Argument** — the `usd_path` / `scene_save` path itself.
2. **Composed layer stack** — after `AddReference` recomposes, walk
   `stage.GetUsedLayers()`; anonymous layers (edit sublayer, session) exempt.
   Compare against the pre-add layer set so pre-existing out-of-root layers of
   the operator's own scene never fail an agent add.
3. **Asset attributes** — traverse the added subtree; every asset-valued
   attribute's **resolved** path (`attr.Get().resolvedPath` /
   `Sdf.AssetPath.resolvedPath`, not the authored string, which is usually
   relative) resolves under the roots. The traversal uses
   `Usd.PrimRange(prim, Usd.TraverseInstanceProxies())` and loads payloads on
   the subtree first (F3, below). Empty resolution (asset-not-found, or a udim
   `<UDIM>` template that never resolves to one path) ⇒ **reject** with a
   message naming the unresolved asset — fail-closed, never fail-open. The
   matching guard on asset-typed `scene_set` writes keeps later edits coherent
   (D-set below).

**F3 — layer walk and asset walk each have an escape; both closed here:**

- **Deferred payloads:** a referenced file whose out-of-root content sits behind
  a `payload` does not compose at `AddReference` time, so its layer never enters
  `GetUsedLayers()`. Before walking, `added_prim.Load(Usd.LoadWithDescendants)`
  so payloads compose and their layers/assets become visible.
- **Instanced prototypes:** a plain `UsdPrimRange(prim)` stops at
  `instanceable=true` boundaries, so a texture bound only inside an instanced
  prim escapes both walks. `TraverseInstanceProxies()` descends into prototypes.

Any residual gap after these (e.g. resolver plugins, dynamic payloads) is owned
explicitly in the "guardrail, not a sandbox" caveat rather than implied closed.

Violation ⇒ prim rollback (the fixed `add_model` rollback from D3, which now
also removes auto-created parents) and a tool error naming the offending path
and the roots. Rejected: argument-only check (nested references escape
trivially); resolver-hook enforcement in `usd_loader` (would police the human
operator's normal loads — wrong scope).

### D3 — Validator seam on `add_model` (add_model only)

Enforcement levels 2–3 must run post-reference/pre-resync on the render thread,
against the prim that `add_model` authored (its path is chosen internally via
`_unique_prim_path`, so the caller cannot precompute it). `add_model` grows an
optional `validate=` callable **invoked as `validate(stage, added_prim)`** after
the reference recomposes but before `_resync_geometry_from_stage()`; a raised
exception triggers rollback and propagates to the tool error. The MCP closure
passes `lambda stage, prim: validate_added_subtree(stage, prim, pre_layers,
roots)`, capturing `pre_layers = stage.GetUsedLayers()` **inside the same posted
closure** (render thread) immediately before calling `add_model`. Docks pass
nothing → byte-identical.

Scoped to `add_model` only (F12/YAGNI): `add_light` and `add_primitive` author
no external files, so nothing to veto. Rejected: MCP-layer reimplementation of
authoring (two paths drift); post-resync check (pays a full resync before
rejecting).

**`add_model` rollback must be fixed first (F2).** `add_model`'s current
`except` runs only `RemovePrim(prim_path)` (renderer.py:6079-6082) — unlike
`add_light` (:6125-6133, :6162-6164) it does not track and remove parent Xforms
it auto-created. A validator veto on an add whose parent chain was synthesized
would otherwise leave stray Xforms in the edit layer, persisted on the next
`scene_save`, breaking the "nothing authored"/"rolled back" spec scenarios. Port
`add_light`'s `missing_parent_paths` tracking + reversed removal into
`add_model` before wiring the seam.

### D4 — `add_primitive` authors gprim + `UsdPreviewSurface` (M2)

New renderer verb mirroring `add_light`'s schema-table shape: define the gprim,
define a sibling `UsdShade.Material` + `UsdPreviewSurface` shader
(`diffuseColor`, `roughness`, `metallic` from optional tool params, sane
defaults), bind it, resync. The material becomes a real `scene.materials` entry
with its own scene-graph node — every later adjustment flows through the
existing `scene_set` dispatch. Rejected: bare gprim (permanently gray — slot 0
is edit-protected); fat material schema on the tool (duplicates `scene_set`).

### D5 — Tool surface: six flat tools (S2), no polymorphic union

`scene_add_model`, `scene_add_primitive`, `scene_add_light`, `scene_remove`,
`scene_save`, `scene_job_status`. Flat per-tool schemas route better in
tool-calling LLMs than one branchy `scene_add(kind=…)`. `scene_remove` keeps
the docks' `is_deletable` guard (root and `/Skinny/*` refused) and documents
`SetActive(False)` semantics. Transform edits need no new tool — `scene_set`
vec3f dispatch already recomposes TRS. Lights take optional `intensity` /
`color` (universal across the 5 UsdLux types); dome texture stays a post-add
`scene_set` (applies to 1 of 5 types).

**`add_light` must gain the params (F5).** It currently hardcodes
`CreateIntensityAttr().Set(1.0)` / color `(1,1,1)` (renderer.py:6140-6141), so
the tool cannot satisfy its own "light carries those values" scenario — and a
post-add `scene_set` would not persist on save. Extend `add_light(intensity=None,
color=None)` to `Set` at define time when given (one line beside the existing
`Create*Attr().Set`).

**Parent default (F14):** the docks derive the add-parent from the operator's
*selection* (`add_parent_for_node`). MCP has no selection, so each add tool's
concrete default parent is `/World` (auto-created if absent), not "the editor's
selection."

### D6 — Transforms: TRS or raw matrix, mutually exclusive (X3)

`translate` / `rotate_euler_deg` (XYZ) / `scale` (scalar or vec3) composed via
`compose_trs_matrix` (`scene_graph.py:1111` — same convention the dock round-
trips, so `scene_get` readback matches what was written), or `matrix` (16
floats, row-major). Both given ⇒ error. The add tools author a single
`xformOp:transform` matrix (`_author_local_transform`); the TRS round-trip
scenario holds only because the scene-graph builder decomposes that composed
local matrix back to TRS via `decompose_trs_matrix` (the exact inverse of
`compose_trs_matrix`) — an implementation test must confirm the builder path,
not just the two helpers in isolation, and confirm the documented Euler order
(intrinsic XYZ, `Rz·Ry·Rx`, row-vector) is what clients are told. Documented
caveat: sheared matrices do not round-trip through the TRS props `scene_get`
shows.

### D7 — Hybrid async jobs (W3 + J2)

Structural tools (`scene_add_*`, `scene_remove`, `scene_save`) post the closure
and wait an inline grace (~2 s). Completed ⇒ `{status: "done", …result}`.
Not completed ⇒ the future is parked in a job store and `{status: "pending",
job_id}` returns; the render queue serializes execution either way, so
semantics are identical — the split is purely ergonomic. `scene_job_status`
returns pending / done+result / failed+error; unknown id is a tool error.
Results carry the final prim path. Retention: last 50 jobs or 10 minutes,
whichever prunes first. Read tools keep the flat 10 s timeout.

**Concurrency reality (F6).** FastMCP runs sync tool bodies directly on the
uvicorn event loop (no `to_thread`), so every blocking `future.result()` freezes
the whole loop — the 2 s grace is a deliberate loop-wide stall, and *no other
MCP request (including a `scene_job_status` poll) is serviced during it*. This
is acceptable for one sequential local agent (it mirrors the existing 10 s read
stall) but the design does **not** claim the server keeps serving during the
grace. Two consequences bind the implementation: keep the grace short (≤2 s),
and `scene_job_status` must be strictly non-blocking — `future.done()` then an
immediate `result()`, never `result(timeout=…)` — so a poll can never extend the
freeze. The job store is written by the render thread (settling futures) and
read by the loop thread; guard it with a lock.

Rationale over flat long timeout (W2): a cancelled-after-start add still
completes on the render thread, so the client is told "unknown outcome" while
the scene silently changes — a retrying agent then duplicates the prim. The
job pattern makes the outcome observable instead. Rejected pure-ticket (J1):
poll ceremony on a 50 ms light-add doubles latency and tokens of the common
case forever.

### D-set — Generic asset check on `scene_set` (F8)

The asset-typed `scene_set` root check gates on the property being asset-valued
generically (the same asset branch `_coerce` already recognizes —
`string`/`token` are *not* asset paths, but `texture_file`/`lens_file`/`asset`
and any future asset type are), not a hardcoded name whitelist, so a new
asset-valued property cannot silently bypass the allowlist.

### D8 — `scene_save` with the partial-save caveat (V2)

Wraps `renderer.save_edits`; target path checked against the roots. **The path
is required and explicit (F9):** `save_edits(path=None)` defaults to
`<scene>.edits.usda` beside the loaded scene — typically *outside* the roots and
unchecked — so the tool refuses an omitted path rather than writing out-of-root
silently. Loudly documented in the tool description: property edits made via
`scene_set` mutate in-memory render state without authoring to USD, so a save
captures structure (adds / removes / transforms) but omits material and light
parameter tweaks — the same partial-save behavior the dock's save button ships
today. `save_edits` with no edit layer raises
`RuntimeError("save_edits: no edit layer attached")`, which `_wrap` would surface
verbatim; map it (like all structural tools) to the friendly "no editable USD
stage is loaded" message. The `mcp-scene-control` exclusion requirement narrows
accordingly (see the spec delta): rendered-image return stays excluded; the old
combined requirement is **REMOVED and re-ADDED** as an image-only exclusion so
its title stops contradicting its body (F7).

### D9 — Errors are retry-grade

Every rejection names the actual problem and the relevant facts (offending
path + configured roots; expected type; bounds), because the client retries
based on message text. `has_editable_stage` false ⇒ "no editable USD stage
loaded" rather than an attribute error. New prim names uniquified via
`_unique_prim_path`; the chosen path is returned in the result (and in the
pending-job result once done).

**`scene_remove` distinguishes not-found from not-deletable (F10).** The tool
resolves a `SceneGraphNode`; a path that is a valid stage prim but absent from
the *derived* scene graph yields `find_node_by_path → None → is_deletable(None)
→ False`, which reads as a permission failure. Return "no such node: <path>" for
the None case and "not deletable: <path>" only for a resolved-but-guarded node.

**Inactive-prim non-resurrection is load-bearing (F11).** `remove_node` does
`SetActive(False)` (the prim stays on the stage), and `_unique_prim_path` relies
on `GetPrimAtPath` reporting the inactive prim as valid so it skips the burned
name — never `Define`-ing onto the dead spec and resurrecting its old children.
Correct in USD today; an implementation test (remove `Sphere`, re-add
`name="Sphere"` → distinct path, old subtree not resurrected) pins it.

## Risks / Trade-offs

- **[Allowlist is not containment]** Deferred payloads and instanced-prototype
  textures are now closed (payload load + `TraverseInstanceProxies`, D2/F3), but
  the check is still add-time only and cannot see resolver plugins, dynamic
  payloads, or edits made outside MCP. → Accepted: same trust domain; documented
  as a guardrail, not a sandbox, with the residual gaps named explicitly rather
  than implied closed.
- **[Layer walk false positives]** Operator loads a scene referencing assets
  outside the roots, agent adds an unrelated model, naive walk flags the old
  layers. → Diff against the pre-add `GetUsedLayers()` snapshot; only newly
  introduced layers are checked.
- **[Render-thread stall during big adds]** A 28 M-tri reference stalls frames
  for the whole resync, job pattern or not (dock behaves identically today).
  → Out of scope; job pattern at least reports honestly.
- **[Job store lifetime]** Unpolled completed jobs pin their results.
  → last-50 / 10-min pruning; store lives in `SceneTools`, per-process.
- **[Spec title collision]** The old exclusion requirement is REMOVED and an
  image-only exclusion ADDED (F7), rather than MODIFIED under a title that would
  then contradict its body. Single in-flight change touches this spec; archive
  alone.
- **[`scene_set` asset guard tightens existing behavior]** Previously any
  string passed; now out-of-root texture paths are refused. Operator dock is
  unaffected (different code path). → Called out in proposal as spec-level
  change.

## Open Questions

None. The interview resolved all forks (C1, P3, D2, T2, R2, S2, M2, X3, W3/J2,
L2, V2); the xhigh design review's 14 findings (F1 validator signature, F2
add_model parent rollback, F3 payload/instance escapes, F4 asset resolution, F5
add_light params, F6 event-loop framing, F7–F14) are folded into the decisions,
risks, and tasks above.
