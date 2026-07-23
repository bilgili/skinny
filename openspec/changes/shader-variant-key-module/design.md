# Design: shader-variant-key-module

## Context

Every compiled kernel's identity is a point in a variant matrix, but the
matrix has no owning module. The true axes, as found in code (survey
2026-07-23, worktree `arch-proposals`):

1. **Target** — Vulkan SPIR-V (`slangc`, `-fvk-use-scalar-layout`, flat `-D`
   token tuples) vs native Metal (in-process SlangPy session, defines dict,
   `SKINNY_METAL=1`, `column_major` matrix layout, no scalar-layout flag).
2. **Pipeline family** — megakernel (`SKINNY_COMPUTE_PIPELINE`, no
   `SKINNY_WAVEFRONT`); wavefront full-tree (`SKINNY_COMPUTE_PIPELINE` +
   `SKINNY_WAVEFRONT`); wavefront foundation (`SKINNY_WAVEFRONT` only —
   `vk_wavefront._slang_flags`, **Vulkan-only**: Metal compiles every
   wavefront kernel through `_metal_slang_session`'s full CP+METAL+WAVEFRONT
   set); preview (`SKINNY_COMPUTE_PIPELINE` only — Vulkan `PreviewPipeline`
   and Metal `PreviewPipelineMetal._build` at `metal_compute.py:995`);
   debug-raster (bare `SKINNY_METAL`, `DebugRasterMetal` at
   `metal_compute.py:1139`, **Metal-only**: the Vulkan debug viewport is a
   graphics rasteriser, not this compute path).
3. **Spectral** — `SKINNY_SPECTRAL=1` + `_spectral` filename suffix.
4. **MLT** — `SKINNY_MLT=1` + `_mlt` tag (wavefront MLT kernels only; the
   megakernel is never compiled with it — that is the recorded byte-identity
   guarantee).
5. **Neural build config** — `NeuralBuildConfig` (layers/bins/hidden ×
   precision fp32|fp16-storage|fp16-compute|fp8 × encoding E0/E1/E3(+`P<n>`)
   × coupling × chart) → `slang_defines()` (`NF_*` tokens; default = empty)
   and `cache_tag` (`L6B24H96…_fp16-compute` slugs; 35 tagged `.spv` files on
   disk carry `L6B24H96`).
6. **Metal-only gates** — `SKINNY_METAL_NEURAL`, `SKINNY_METAL_RECORDS`
   (argument-table trimming, change metal-record-drain). Deliberately have no
   Vulkan counterpart; today that asymmetry is implicit in call-site dicts.

Emission sites: `vk_compute.py` (2 compile methods + 2 duplicated
`_cache_key` copies), `vk_wavefront.py` (3), `metal_compute.py` (3),
`metal_wavefront.py` (6). Cache encodings are split across two mechanisms:
opaque blake2b keys in `build/spv_cache/` (flags hashed in) and human-readable
filename tags (`f"{out_name}{tag}{spectral_suffix}.spv"`) assembled by string
concatenation with the neural part sourced three files away.

## Goals / Non-Goals

**Goals**

- One module owns key → defines → cache token. Both backends, both pipeline
  families, and the `wavefront_layout` sizers consume it.
- Cross-backend define agreement becomes a hostless test, not a convention.
- Metal-only defines become an explicit, named set instead of implicit
  call-site knowledge.
- Every recorded byte-identity guarantee survives, verified per migration
  step.

**Non-Goals**

- No new variants, axes, or dispatch changes.
- No change to `NeuralBuildConfig` internals or the NFW1 format.
- No unification of the two cache mechanisms (blake2b `spv_cache` vs filename
  tags) — they serve different needs (content invalidation vs anti-clobber)
  and both keep their exact current outputs.
- No shader edits; no `.spv` recompiles required to land the change.

## Decisions

### D1 — Key shape: frozen dataclass composing NeuralBuildConfig

`src/skinny/shader_variants.py`:

```python
@dataclass(frozen=True)
class ShaderVariantKey:
    target: Target              # VULKAN | METAL
    family: Family              # MEGAKERNEL | WAVEFRONT | WAVEFRONT_FOUNDATION | PREVIEW
    spectral: bool = False
    mlt: bool = False
    metal_neural: bool = False    # SKINNY_METAL_NEURAL (METAL only)
    metal_records: bool = False   # SKINNY_METAL_RECORDS (METAL only)
    neural: NeuralBuildConfig | None = None
```

Derivations: `slangc_defines()` (Vulkan form, see D4 for its segmented
shape), `session_defines() -> dict[str, str]` (SlangPy form),
`cache_token() -> str` (the filename tag). One shared internal
`_defines() -> dict[str, str]` feeds both public forms so they cannot diverge.
`__post_init__` enforces a **(target, family) validity table** plus the axis
rules — the table makes illegal states unrepresentable instead of silently
dropped:

| Family | VULKAN | METAL |
|--------|--------|-------|
| MEGAKERNEL | ✅ | ✅ |
| WAVEFRONT | ✅ | ✅ |
| WAVEFRONT_FOUNDATION | ✅ | ❌ (no Metal foundation compile) |
| PREVIEW | ✅ | ✅ |
| DEBUG_RASTER | ❌ (Vulkan uses a graphics rasteriser) | ✅ |

Axis rules: `metal_neural`/`metal_records` only on METAL keys; `mlt` only in
the WAVEFRONT family.

Rationale: cross-backend statements ("for every key pair differing only in
target") are only meaningful over families valid on both targets; without the
table the agreement test would construct phantom keys.

Rationale: the codebase already has the pattern — `NeuralBuildConfig` is a
working variant-key module for its own axes (`slang_defines()` +
`cache_tag`). The new module extends the same idea one level up and composes
it rather than re-deriving `NF_*` logic.

### D2 — Module home: new hostless `src/skinny/shader_variants.py`

Not inside `vk_compute` or `metal_compute` (each is one adapter; the point is
a neutral owner both import) and not inside `sampling/neural_weights.py`
(that module is about the neural net, not the renderer-wide matrix). No GPU
imports — importable in hostless tests like `wavefront_layout`.

### D3 — Cache tokens stay byte-identical (no cache flush)

`cache_token()` reproduces today's derivation exactly:
`f"{'_' + neural.cache_tag if neural-defines-nonempty else ''}{'_mlt' if mlt else ''}{'_spectral' if spectral else ''}"`
with the default key yielding `""`. Consequences: tagged `.spv` filenames are
unchanged, and `build/spv_cache` blake2b keys are unchanged because the
hashed flag tuples are unchanged (D4). No one-time flush; a flush is accepted
only if a future change deliberately alters a flag list, at which point the
blake2b scheme already invalidates correctly.

### D4 — Segmented define emission; per-site splice keeps hashes byte-equal

A single contiguous define block cannot reproduce today's flag tuples,
because the Vulkan sites interleave defines with `-fvk-use-scalar-layout`
in **three different orders** — all hashed positionally by `_cache_key`:

- `vk_compute.ComputePipeline._compile_slang` (:324–340):
  `-D SKINNY_COMPUTE_PIPELINE=1`, `-fvk-use-scalar-layout`, then
  `-D SKINNY_SPECTRAL=1` appended **after** the scalar-layout flag.
- `vk_wavefront._slang_flags` (:33–37): `-fvk-use-scalar-layout` **before**
  `-D SKINNY_WAVEFRONT=1`.
- `vk_wavefront._compile_full_spv` (:472–478): all defines (CP, WAVEFRONT,
  spectral, neural, MLT) **before** `-fvk-use-scalar-layout`.

Decision: the module emits defines as **ordered segments**, not one blob —
`slangc_defines()` returns named groups (`base`, `spectral`, `neural`,
`mlt`), each an ordered `-D` token tuple; every Vulkan site keeps its own
flag scaffolding (`-target`/`-entry`/`-I`/`-fvk-use-scalar-layout`) and
splices each group at its **recorded historical position** (the megakernel
splices `spectral` after the scalar-layout flag; `_compile_full_spv` splices
everything before it). The segment contents and order-within-group come from
the module; the splice positions stay site-owned and are locked by the
task-1.1 golden flag tuples. Result: flag tuples byte-identical → blake2b
`_cache_key` unchanged → no cache migration, including for the spectral
megakernel. The rejected alternative — canonical contiguous order plus a
one-time accepted `spv_cache` invalidation for the spectral megakernel — was
declined: the cache-hit guarantee is a stated spec requirement and the
segmented form is barely more code. `session_defines()` is unaffected (dicts
are order-insensitive for the Metal session).

### D5 — Metal-only defines are an explicit named set

`METAL_ONLY_DEFINES = frozenset({"SKINNY_METAL", "SKINNY_METAL_NEURAL",
"SKINNY_METAL_RECORDS"})` lives in the module beside the table. The
cross-backend agreement test is defined as: **for every family valid on both
targets** (per the D1 table — foundation and debug-raster are excluded as
single-target), for each key pair differing only in target,
`session_defines().keys() - METAL_ONLY_DEFINES ==
parse(slangc_defines()).keys()` and values match — modulo the module's
explicit **recorded-asymmetry table**. The survey found one real entry: the
Metal SPPM pass compiles with the active `NeuralBuildConfig`'s `NF_*` defines
(`metal_wavefront.py:786`, `_defines_dict(neural_config.slang_defines())`)
while the Vulkan SPPM compile passes none (`vk_wavefront.py:927`,
`_compile_full_spv(..., spectral=...)` with default `defines=()`), so a
non-default neural config makes the two SPPM define sets diverge today.
The module records this as an explicit table entry (asymmetric today,
harmless at the default config which emits zero `NF_*` flags) rather than
silently normalizing it — whether to align Vulkan SPPM (or stop passing
neural defines to Metal SPPM) is a separate follow-up change, out of this
pure refactor's scope. Both deliberate-asymmetry sets are reviewable
constants instead of folklore. (MLT was checked and is symmetric.)

### D6 — wavefront_layout consumes the key's axes, not the key

Sizers keep their `spectral=`/`msl=` keyword signatures (hostless tests and
many call sites use them) but the renderer-side call path reads those
booleans off the active `ShaderVariantKey` (`key.spectral`,
`key.target is METAL`). Smallest diff; the agreement is enforced where the
key is created, not by rewriting every sizer signature.

### D7 — Alternatives rejected

- **String-keyed registry / config table**: more machinery than seven fields;
  the dataclass with `__post_init__` validation is smaller and typo-proof.
- **Fold everything into `NeuralBuildConfig`**: wrong owner; neural is one
  axis of six.
- **Unify the two cache mechanisms**: behavior change risk for zero user
  benefit; out of scope (Non-Goals).
- **Also migrate `vk_skinning.py` / one-off tool compiles**: they take no
  variant axes today (fixed flag lists); migrating them adds churn without
  removing a drift risk. Revisit only if they grow an axis.

## Risks / Trade-offs

- [Risk] A migrated site emits flags in a different order → different blake2b
  key → silent full `spv_cache` miss (slow, not wrong) or, worse, a changed
  define set → different SPIR-V. → Mitigation: D4's segmented per-site
  splice + the per-step byte-identity check in the Migration Plan (hash the emitted flag
  tuple and the produced `.spv` before/after each consumer migration).
- [Risk] The SlangPy `opts.defines` copy-on-read gotcha (documented in
  `metal_compute.py:673`) — assigning via the module must still build the
  full dict and assign once. → Mitigation: `session_defines()` returns a
  fresh complete dict; consumers assign it in one statement; the existing
  spectral-megakernel regression stays green.
- [Risk] Key validation is stricter than today's permissive call sites and
  refuses a combination some in-flight branch relies on. → Mitigation:
  validation rules are transcribed from the Compatibility matrix (CLAUDE.md)
  only; anything the matrix calls valid constructs.
- [Risk] Hostless agreement test overfits to today's define spelling and
  blocks a legitimate future change. → Mitigation: the test compares the
  module against itself across targets (semantic agreement) plus a small
  golden list for the recorded guarantees; only the golden list needs
  touching when a variant is deliberately added.

## Migration Plan

Each step lands with byte-identity verified before the next starts.
Verification harness (step 0) is a small hostless script/test that records,
for a fixed sweep of keys: the Vulkan flag tuple, the Metal defines dict, and
the cache token, plus the blake2b `_cache_key` over a pinned source tree.

1. **Module + tests first** (no consumers): `shader_variants.py` +
   `tests/test_shader_variants.py` asserting (a) cross-backend agreement per
   D5, (b) `cache_token()` golden values incl. empty default, (c) recorded
   guarantees: default neural key emits zero `NF_*` flags; no VULKAN key ever
   contains a `METAL_ONLY_DEFINES` member; `mlt=False` keys contain no
   `SKINNY_MLT`; `spectral=False` keys contain no `SKINNY_SPECTRAL`.
2. **vk_compute**: `ComputePipeline._compile_slang` + `PreviewPipeline`
   consume `slangc_defines()`. Verify: flag tuples byte-equal → `_cache_key`
   unchanged → cached `.spv` reused (assert cache hit, no recompile).
3. **vk_wavefront**: `_slang_flags`, `_compile_full_spv` (defines + tag), MLT
   site. Verify: all 28 RGB wavefront kernel flag tuples and out-filenames
   byte-equal; `_mlt`/`_spectral`/neural-tagged names unchanged.
4. **metal_compute**: three `opts.defines` sites consume
   `session_defines()`. Verify: dicts equal to the previous literals
   (hostless — compare dicts, no GPU needed).
5. **metal_wavefront**: `_metal_slang_session` + five per-pass sites; delete
   the local `_defines_dict`. Verify: per-pass dicts equal previous
   assemblies for representative (neural, records, spectral, mlt) states.
6. **wavefront_layout / renderer seam** (D6): renderer builds one key per
   compile request and reads sizer booleans off it. Verify: hostless sizer
   outputs unchanged for both RGB and spectral, scalar and MSL.
7. **GPU smoke** (one guarded Metal process, dispatch-hygiene rules): one
   megakernel frame + one wavefront frame per backend, RGB and spectral —
   confirms no runtime regression; plus `tests/test_metal_cleanup.py`
   hostless subset since compile plumbing was touched.
8. **Docs**: `docs/Architecture.md` section naming the module as owner;
   CLAUDE.md/README untouched (no user-facing change).

## Open Questions

- (Resolved during review) Family modeling is pinned: `PREVIEW` is valid on
  both targets (`vk_compute.PreviewPipeline` / `PreviewPipelineMetal._build`
  at `metal_compute.py:995`, both `SKINNY_COMPUTE_PIPELINE`); the bare
  `SKINNY_METAL` site at `metal_compute.py:1139` is `DebugRasterMetal`, a
  Metal-only `DEBUG_RASTER` family; `WAVEFRONT_FOUNDATION` is Vulkan-only.
  See the D1 validity table.
- (Resolved during review) MLT define symmetry was checked: the Metal MLT
  session matches the Vulkan MLT flag set modulo `METAL_ONLY_DEFINES` — no
  asymmetry entry needed.
- The Metal-SPPM neural-defines asymmetry (D5) is recorded, not resolved:
  should Vulkan SPPM also receive the active neural `NF_*` defines, or
  should Metal SPPM stop receiving them? Follow-up change; this refactor
  keeps the observed behavior and its explicit table entry either way.
