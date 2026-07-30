## MODIFIED Requirements

### Requirement: One shared, canonical bring-up sequence for every front-end

The renderer bring-up orchestration SHALL be owned by a single shared bring-up
module — startup-integrator resolution, execution-mode resolution, render-flag
validation, the refusal guards (`sppm`-without-wavefront, MLT envelope,
spectral envelope, MCP support, denoiser flags), GPU backend selection, and the
backend-dependent denoiser guard —
and SHALL run in one canonical order for every front-end (`skinny`,
`skinny-gui`, `skinny-render`, `skinny-web`). Front-ends MUST NOT re-implement
or reorder the sequence; they invoke the shared module and keep only
surface-specific wiring (GLFW window, Qt threading, web server/session
lifecycle, MCP flag plumbing). Adding a new refusal guard to the shared
sequence SHALL make it effective on all front-ends without per-front-end edits.

GPU backend selection SHALL stay the last step that can fail before the plan is
returned, except for guards that **need** the resolved backend. A guard that
needs the resolved backend SHALL run immediately after backend selection and
before the plan is returned, so it still refuses before the renderer's GPU
context is constructed. Every guard that does not need the resolved backend
SHALL run before backend selection.

The plan SHALL carry the resolved denoiser name and denoise scale, because both
are guard-vetted and identical on every front-end.

#### Scenario: All four front-ends run the same guard sequence

- **WHEN** any of `skinny`, `skinny-gui`, `skinny-render`, or `skinny-web` is
  launched with a render-flag combination that today is refused at startup
  (for example `--integrator sppm --execution-mode megakernel`, an
  out-of-envelope `--spectral` combination, an unsupported `--mcp`, or a
  denoiser the resolved backend cannot run)
- **THEN** the launch is refused by the shared bring-up sequence with the same
  refusal semantics and byte-identical message text as before this change for
  every pre-existing refusal

Message prefixes are asymmetric and SHALL stay so: the shared refusal guards
print their own fixed `skinny:` prefix on every front-end (the MCP guard prints
none), while a backend-selection failure is prefixed with the invoking
front-end's program name (`skinny:`, `skinny-gui:`, `skinny-render:`,
`skinny-web:`). That is the pre-change behavior on all four; repointing the
guard prefixes at the invoking program would be a user-visible output change
and is out of scope here. The denoiser guards SHALL print the fixed `skinny:`
prefix, matching the other shared guards.

#### Scenario: A guard added to the shared sequence covers every front-end

- **WHEN** a new refusal guard is added to the shared bring-up sequence
- **THEN** the guard is enforced on all four front-ends without any
  front-end-specific bring-up code being modified

#### Scenario: A backend-dependent guard still precedes context construction

- **WHEN** a denoiser is requested that the resolved backend cannot run
- **THEN** the launch is refused after backend selection and before the plan is
  returned, so the renderer's GPU context is never constructed
