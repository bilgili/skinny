## 1. Host envelope and runtime resolution

- [x] 1.1 Update `reject_spectral_unsupported` to accept only the analytic
      spectral proposal tokens `bsdf` and `env`, while continuing to reject
      `neural` and ReSTIR reuse. Add focused CLI tests and run them.
- [x] 1.2 Resolve spectral runtime proposal presets to their supported analytic
      subset (`bsdf` / `env`) and update the configuration matrix to report an
      actual environment selection or a neural pin. Add observability/runtime
      tests and run them.

## 2. Spectral path shader proposal sampling

- [x] 2.1 Route megakernel `SpectralPathTracer` through
      `sampleBounceDirection`, using the returned mixture pdf with
      `flatResponseS` and every downstream MIS companion. Add a source-contract
      regression and compile the spectral megakernel.
- [x] 2.2 Verify the existing wavefront spectral path already routes through the
      shared proposal sampler and spectral response/pdf weighting; update stale
      comments, add a source-contract regression, and compile the spectral
      wavefront flat-shade entry.

## 3. Documentation and final validation

- [x] 3.1 Update `docs/Spectral.md`, `docs/Wavefront.md`, `README.md`, and
      `CHANGELOG.md` for the widened analytic spectral-proposal envelope. Review
      all other Markdown documents for affected claims.
- [x] 3.2 Run focused tests, Ruff on touched Python, OpenSpec strict validation,
      and review the final diff.
- [x] 3.3 Mark the change implementation-complete after every task and
      validation passes.

## 4. Review remediation

- [x] 4.1 Correct partial-opacity throughput under mixed spectral proposals in
      both megakernel and wavefront paths, preserving the BSDF-only estimator.
      Update source contracts and compile both spectral entries.
- [x] 4.2 Add the environment proposal to the renderer parity matrix as a
      spectral path-only axis, with explicit validity and coverage tests.
- [x] 4.3 Add and run rendered spectral `{bsdf}` versus `{bsdf,env}`
      convergence/self-consistency gates under megakernel and wavefront using
      `compute_metrics` and a recorded stochastic tolerance.
- [x] 4.4 Run the mandatory Metal cleanup tests: the hostless suite and the
      guarded GPU kill harness.
- [x] 4.5 Re-run focused tests, Ruff, shader compilation/byte-identity,
      OpenSpec strict validation, review the diff, and restore the
      implementation-complete marker.
