## Context

The RGB path integrator samples every non-delta bounce through
`sampling.proposal::sampleBounceDirection`. That helper:

1. selects BSDF, environment, or neural according to `fc.proposalMask` and
   `fc.proposalAlpha`;
2. returns the selected tangent-space direction and the full one-sample-MIS
   mixture pdf; and
3. couples NEE through `mixtureProposalPdf`.

The spectral NEE helper already calls `mixtureProposalPdf`, and the wavefront
flat shade stage already obtains its `BSDFSample` from the shared proposal seam.
The missing pieces are:

- `SpectralPathTracer` still calls `mat.sample()` directly; and
- `Renderer._active_proposals()` clamps every spectral selection to BSDF.

The direction and every proposal pdf are scalar, wavelength-independent
geometry. Only the throughput numerator must be spectral. For the BSDF-only fast
path it is `flatResponseS(mat, cols, wo, wi)`; a mixed continuous proposal must
use `flatResponseNEE`, whose opacity factor matches the unconditional BSDF
density returned by `mat.evaluate()`.

## Goals / Non-Goals

**Goals**

- Make `{bsdf,env}` and `{env}` effective in spectral path tracing under the
  megakernel and wavefront modes.
- Keep the NEE, environment-miss, emissive-hit, and sphere-hit MIS companions
  coupled to the exact mixture pdf that generated the bounce.
- Preserve BSDF-only behavior and RGB compiled behavior.
- Report the actual resolved analytic proposal selection.

**Non-goals**

- Spectral neural inference or online training.
- Spectral ReSTIR reuse.
- Adding the proposal seam to BDPT or SPPM.
- Changing wavelength sampling, spectral upsampling, or environment CDFs.

## Decisions

### D1: Reuse `sampleBounceDirection`; recolor only its numerator

The megakernel spectral path will build a `ProposalContext` and call
`sampleBounceDirection`. For a non-delta sample, `bs.pdf` is the full mixture
pdf. The spectral throughput remains:

`flatResponseS(mat, cols, wo, bs.wi) / bs.pdf` for the BSDF-only fast path, and
`flatResponseNEE(mat, cols, wo, bs.wi) / bs.pdf` when a continuous proposal
mixture is active.

The helper's RGB `weight` is deliberately ignored. Delta samples continue to
pass through unmixed and use the existing spectral transmission treatment. The
mixed-path opacity factor is required because `FlatMaterial.sample()` chooses
the continuous surface branch with probability `opacity`, while
`FlatMaterial.evaluate().pdf` includes that probability in the mixture density;
the BSDF-only estimator instead receives the conditional sample pdf and gets
the same factor through stochastic branch selection.

### D2: Treat the returned pdf as the spawning proposal pdf everywhere

The value carried as `misBsdfPdf` / `prevBsdfPdf` is semantically the bounce
mixture pdf when an environment proposal is active. It is used unchanged for:

- environment-miss versus environment NEE MIS;
- emissive-triangle hit versus light NEE MIS; and
- sphere-light hit versus light NEE MIS.

This matches the scene-sampling contract: downstream MIS uses the density that
actually generated the ray.

### D3: Resolve spectral path presets to their analytic subset

For the spectral path integrator the renderer retains `bsdf` and `env`, removes
`neural`, and falls back to `bsdf` if removing unsupported proposals leaves an
empty set. Thus:

| Requested | Resolved |
| --- | --- |
| `bsdf` | `bsdf` |
| `bsdf,env` | `bsdf,env` |
| `env` | `env` |
| `bsdf,neural` | `bsdf` (reported pin) |
| `neural` | `bsdf` (reported pin) |

The CLI accepts only tokens in `{bsdf,env}` for explicit spectral requests and
continues to reject neural before GPU initialization. An explicit environment
proposal with spectral BDPT/SPPM is refused because those integrators do not
consume the seam. If an interactive session switches from path to BDPT/SPPM,
runtime resolution pins the active proposal set to BSDF and reports the pin.

### D4: Wavefront uses the same opacity-aware numerator

`wavefront/flat_bounce.slang` already calls `sampleBounceDirection`, and
`wf_shade_common.slang` already consumes the direction and pdf from that seam.
Removing the host clamp activates it, while the spectral finisher must select
the same opacity-aware numerator as the megakernel for mixed proposals. The
implementation locks both routes with source and rendered convergence tests.

## Risks / Trade-offs

- The environment proposal can choose a below-horizon direction. The shared
  proposal helper returns a zero-contribution invalid sample, which is the
  existing unbiased one-sample estimator behavior.
- Spectral and RGB paths consume different RNG streams because wavelength
  sampling is spectral-only; this change does not promise sample-by-sample RGB
  parity.
- BDPT and SPPM still use their native sampling strategies. Supporting the
  directional-proposal seam there would require separate OpenSpec changes.

## Validation

- Hostless tests prove explicit spectral `{bsdf,env}` / `{env}` acceptance,
  neural refusal, and analytic-subset runtime/config resolution.
- Source-contract tests prove both spectral path implementations route bounce
  sampling through the shared seam and choose the correct opacity-aware response
  divided by its pdf.
- The parity matrix enumerates the environment proposal as a path-only axis and
  GPU convergence tests compare `{bsdf}` with `{bsdf,env}` under megakernel and
  wavefront using `compute_metrics`.
- `slangc` compiles the spectral megakernel and spectral wavefront flat shade
  entry; focused Python tests, Ruff, and the required Metal cleanup/kill harness
  pass.
