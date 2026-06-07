## ADDED Requirements

### Requirement: Mixture unbiasedness under a time-varying proposal density
The scene-sampling mixture-MIS contract SHALL hold when a proposal's density changes
between frames, by keying each sample's density to the network version stamped on it
rather than the proposal's current state, so that asynchronous weight updates change
variance but not the expected value of the estimator.

#### Scenario: Density keyed to the sample's version
- **WHEN** the mixture computes the combined `Σ α_k·pdf_k` for a sample whose neural
  proposal has since been updated
- **THEN** the neural term uses the density of the version that drew the sample, keeping the
  one-sample-MIS weighting consistent

#### Scenario: NEE coupling stays consistent across a swap
- **WHEN** a light-sampled direction is evaluated against the neural proposal density after
  a weight swap
- **THEN** the neural density used for the NEE/MIS coupling is that of the active render-side
  version for that frame, preserving the unbiased coupling
