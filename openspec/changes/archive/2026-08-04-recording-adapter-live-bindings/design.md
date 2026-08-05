## Context

`recording_compute.RecordingContext` duck-types a GPU context closely enough for
the resource constructors, and carries a `Recorder` that logs every allocation,
binding, and dispatch. Two of its queries are the point:

- `dispatch_entries()` — the pass sequence, in dispatch order.
- `missing_bindings()` — `(entry, global_name)` for every shader global a
  recorded dispatch left unbound.

`missing_bindings()` needs to know which globals a pass *declares*. Today that
arrives through `ComputePipeline.reflect_globals(set)`, called by the test with a
literal set. The bind side arrives through `dispatch(binds={...})`, also a
literal. So the current tests assert that two literals the same author wrote
disagree in the expected places.

That is a fine unit test **of the recorder**. It is not a test of any pass. The
distinction matters because the thing being prevented — a Metal global that is
declared but never bound, which reads as zero rather than raising — can only be
caught by comparing what a shader *actually declares* against what the host
*actually binds*.

Two facts shape the design. First, the adapters are already surface-checked by
AST rather than by import (`gpu_backend.adapter_surface`), because the
conformance test must run on a host with neither backend — so source-level
derivation is an established idiom here, not a new one. Second, a pass's bind
map is built by host code that does not need a device: the renderer assembles a
name→resource dict and hands it to `dispatch()`.

## Goals / Non-Goals

**Goals:**

- `missing_bindings()` over a real pass compares source-declared globals against
  the host's real bind map.
- Registering a pass is one entry; coverage follows automatically.
- The gate can fail — proven by a negative control in the suite.
- Everything stays hostless: no GPU, neither backend package importable.

**Non-Goals:**

- Driving a full `Renderer` frame against a `RecordingContext`. Much larger, and
  unnecessary for binding coverage.
- Replacing the recorder's own unit tests. They keep the hand-driven form, which
  is correct *for testing the recorder*.
- Radiometry. The recorder records; it does not simulate. Image correctness stays
  the parity matrix's job.
- A Slang front-end. The declared globals come from the compiler's reflection
  (D1); no Slang is parsed in-tree.

- **The Vulkan bind path.** Coverage drives the **Metal** name table only, which
  is the backend that binds by name: an unbound Metal global reads as zero, so
  the gap is silent. A Vulkan dispatch binds through a descriptor set, where a
  missing write is a validation error the driver already reports. Deliberate
  scope for this change (confirmed with the user), not an oversight — a Vulkan
  provider is a follow-up.

- **The wavefront passes.** Their bind maps come from the staged pass object
  (`vk_wavefront` / `metal_wavefront`), which owns the per-stream buffers in
  descriptor set 1; `SceneResourceSet` supplies only set 0. All 45 are recorded
  exclusions with that reason, and are the largest remaining gap.

- **Non-default build variants of a registered pass.** Each pass is reflected
  under ONE `ShaderVariantKey` — today the default RGB, no-graph key. The
  spectral megakernel's extra globals (bindings 45–51) and a graph-active build's
  `graphParamsCombined` are therefore not covered: they belong to variants that
  are not registered, exactly like the wavefront and Vulkan gaps above. Broader
  variant coverage means registering `(pass, key)` pairs and reflecting each —
  a follow-up, not a silent hole (codex review): the golden and its bind map are
  both scoped to the registered key, so neither side claims a variant it did not
  check.

## Decisions

### D1 — A pass's declared globals come from the compiler's reflection, checked in as a golden

`tests/fixtures/gen_recording_pass_globals.py` compiles each registered pass with
`slangc … -reflection-json` under that pass's `ShaderVariantKey` defines and takes
the top-level `parameters` — the globals the compiled kernel actually declares,
uniform block included. The result is checked in as
`tests/fixtures/recording_pass_globals.json`; the hostless gate reads it and never
runs the compiler.

The failure mode that matters is **under-reporting**: a declared set that silently
misses a global makes the gate pass while the binding is missing, worse than no
gate. The compiler cannot miss a declaration it compiled, so the golden is
authoritative by construction — the whole class of parser under-reports is gone.
The golden is a checked-in generated artifact, trusted the way the parity
harness trusts its checked-in reference EXRs: a `gpu`-marked freshness test
re-runs the compiler and diffs, so a stale golden is caught, while the hostless
gate stays device-free.

*History (revised after review).* This decision first shipped a hand-written Slang
parser (`declared_globals`) that derived the set at test time and refused on
anything it could not classify. Codex pre-merge review repeatedly found valid
declaration spellings it under-reported (qualified globals, split-across-lines,
two-per-line, block-comment-prefixed): a line/regex parser cannot separate a
file-scope resource global from a function parameter of resource type without full
scope tracking, and every partial fix exposed another spelling. Reflecting through
the real compiler — offline, into a checked-in golden, not at test time — removes
the heuristic entirely while keeping the gate hostless. That is why the earlier
"reflect through a real Slang session" alternative, rejected for needing a device
at test time, is exactly this decision **moved offline**.

### D2 — Coverage compares the golden against the host's real bind map

`reflect_globals()` stays, because the recorder's own unit tests legitimately
want to inject a set. But a **registered pass** takes its globals from the D1
golden and its binds from the host, and the gate asserts against those. The
registry entry names how to obtain the bind map without a device.

This is what turns the assertion from "two literals agree" into "the compiler and
the host agree".

### D3 — A registry, and a meta-test that fails when a pass is missing from it

`RECORDABLE_PASSES` lists each pass: entry module, entry point, and its bind-map
provider. The coverage gate iterates it.

A meta-test asserts that every compute entry point the shader tree declares is
either registered or listed in a recorded exclusion set with a reason. This is
the same shape as the parity matrix's integrator-coverage check, which fails the
build when the app exposes an integrator with no validity entry — the mechanism
that makes coverage enforced rather than aspirational.

*Alternative rejected:* register passes ad hoc as tests are written. That is
today's situation, and it is why the recorder has covered nothing for as long as
it has existed.

### D4 — The gate must be provably able to fail

A coverage test that cannot fail is indistinguishable from one that passes, and
this repo has shipped both (a source-order test that passed textually while the
code inside it violated the rule; a gate matching a string the removed tables
never held). So the suite carries a **negative control**: a fixture pass whose
bind map deliberately omits one declared global, asserted to be reported by
`missing_bindings()`.

The negative control lives beside the gate and uses the same code path — not a
parallel hand-built one, which would prove only that the parallel path works.

### D5 — The golden is validated against an independently maintained table

The golden is authoritative but could still be **stale** — regenerated against
the wrong defines, or not regenerated after a shader edit. It is cross-checked
against a table maintained for a completely different reason: the binding numbers
and Metal names `gpu_resources.DECLARATIONS` records. The megakernel golden must
name exactly the default-layout inventory resources it reaches (all but the three
neural buffers it strips at build), so a drift on either side fails. Combined with
the `gpu`-marked freshness diff, a golden that disagrees with reality is caught two
ways.

## Risks / Trade-offs

- **The golden under-reports and the gate passes vacuously** → the compiler
  produced it, so it cannot miss a compiled declaration (D1); the freshness test
  re-runs the compiler and diffs; D4 proves the gate can fail; D5 cross-checks
  against an independently maintained table. The original hand-parser needed
  three legs because it was a heuristic; the reflection golden needs them only
  against staleness, not misreading.

- **A stale golden** (shader changed, golden not regenerated) → the `gpu`-marked
  freshness test fails, and D5 fails if the drift touches a default-layout
  resource. Regeneration is one command, documented in `docs/Contributing.md`.

- **`#if`-gated declarations** — the generator compiles under each pass's
  `ShaderVariantKey.session_defines()`, the same defines the renderer builds with,
  so `#if`-gated globals resolve exactly as the shipped kernel sees them. No gate
  vocabulary is maintained by hand.

## Migration Plan

None. The change is additive: existing recorder tests keep working unchanged,
relabelled to say what they actually test.

## Open Questions

Resolved. Task 1.2's gate-vocabulary question and task 2.1's bind-map-shape
question were answered during implementation; the compiler now owns the gate
resolution, and every registered pass's bind map comes from
`SceneResourceSet.metal_binds`.
