## ADDED Requirements

### Requirement: Backend-aware scene-readiness for enabling the loop

The front-ends SHALL defer enabling the online-training loop until the scene is
built, detected via a backend-aware readiness signal rather than the Vulkan-only
descriptor sets. The native Metal backend binds resources by name and never
allocates Vulkan descriptor sets, so the readiness check SHALL NOT depend on
them; it SHALL be satisfied on both backends once the scene bindings the record
drain reads are present. Enabling and the per-frame drain SHALL use the same
readiness signal so they agree.

#### Scenario: Enabling works on the native Metal backend

- **WHEN** an interactive front-end runs with `--backend metal --execution-mode
  wavefront --online-training`, a neural proposal is active, and the scene is
  built
- **THEN** the loop is enabled (the `NeuralTrainer` is constructed and training
  runs), even though no Vulkan descriptor sets are ever allocated

#### Scenario: Vulkan readiness is unchanged

- **WHEN** an interactive front-end runs on Vulkan with the same flags
- **THEN** enabling still waits for the per-frame scene descriptor sets to exist,
  exactly as before
