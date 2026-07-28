# metropolis-light-transport (delta)

## ADDED Requirements

### Requirement: MLT chain-buffer binding identity has a single host declaration checked against the shader

Each MLT chain-state buffer SHALL be declared once on the host, and that single
declaration SHALL carry its size key, its Vulkan descriptor binding number, and
its Metal shader-global name together. The Vulkan MLT pass, the Metal MLT pass,
and the scene resource set's creation-time dummy writes SHALL all derive their
binding information from that one declaration; none of them may carry its own
binding table. The declaration SHALL agree with the shader's own
`[[vk::binding(N)]] … <name>` declarations, entry for entry, and a disagreement
MUST fail a hostless test.

Behaviour MUST be unchanged: the same six buffers, at the same sizes, at the
same binding numbers, under the same Metal global names, written in the same
order.

#### Scenario: One declaration feeds both backends and the dummy writes

- **WHEN** the Vulkan MLT pass, the Metal MLT pass, and the scene resource set
  each resolve the chain buffers' binding information
- **THEN** all three derive it from the same declaration, and no module states
  a binding number or a Metal global name for these buffers independently

#### Scenario: A transposed pairing fails the build

- **WHEN** a declaration's Vulkan binding number or Metal global name is paired
  with a different chain buffer than the shader pairs it with — while every
  binding number, name and size remains individually present and valid
- **THEN** a hostless test fails, naming the buffer whose pairing disagrees

#### Scenario: The shader-agreement check cannot pass vacuously

- **WHEN** the shader-agreement test runs
- **THEN** it first asserts that it parsed exactly as many MLT binding
  declarations out of the shader sources as the host table declares, so a parse
  that matches nothing fails instead of silently reporting agreement

#### Scenario: A chain buffer added to the shader is not silently ignored

- **WHEN** a new MLT chain buffer is declared in the shader without a
  corresponding host declaration
- **THEN** the shader-agreement test fails on the count mismatch rather than
  comparing only the entries the host happens to declare

#### Scenario: Backends stay bit-identical across the move

- **WHEN** an MLT render is run on Vulkan and on native Metal at equal budget,
  in RGB and in spectral, after the declaration is adopted
- **THEN** the images match to the same tolerance recorded before the change,
  and no per-combo baseline or self-consistency tolerance is loosened
