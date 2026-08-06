## ADDED Requirements

### Requirement: A pass's declared globals come from the compiler's reflection

The set of shader globals a pass declares SHALL be the top-level parameters the
Slang compiler reflects when the pass's entry module is compiled under that pass's
build-variant defines. It SHALL NOT be supplied by the caller for any registered
pass, and it SHALL NOT be derived by a hand-written source parser — a line parser
cannot separate a file-scope resource global from a function parameter of resource
type without full scope tracking, and a missed global makes the coverage gate pass
while the binding is absent, which is worse than having no gate.

The reflection SHALL be generated **offline** into a checked-in golden and the
hostless gate SHALL read that golden without running the compiler, exactly as the
checked-in compiled shader binary is trusted. A separate device-permitted test
SHALL re-run the compiler and compare, so a stale golden is caught.

The hand-supplied form MAY remain for unit tests of the recorder itself, and
those tests SHALL be labelled as testing the recorder rather than any pass.

#### Scenario: Globals come from the compiler, not the test

- **WHEN** the declared globals of a registered pass are requested
- **THEN** they come from the checked-in reflection golden for that pass, and no
  literal set appears at the call site

#### Scenario: The golden is regenerated from the compiler

- **WHEN** the golden is regenerated
- **THEN** each pass's globals are the compiler's reflected top-level parameters
  for that pass's entry module under its build-variant defines

#### Scenario: A stale golden is caught

- **WHEN** the checked-in golden differs from a fresh compiler reflection
- **THEN** the device-permitted freshness test fails, naming that the golden must
  be regenerated

#### Scenario: The golden agrees with an independently maintained table

- **WHEN** the megakernel golden is compared with the binding identities the GPU
  resource inventory declares for it
- **THEN** the two agree, so a stale or mis-generated golden is caught by a table
  maintained for a different reason

### Requirement: Binding coverage compares the shader against the host's real bind map

For a registered pass, the recorded coverage SHALL compare the globals derived
from its shader against the bind map the **host** builds for that pass. Neither
side SHALL be written by the test.

`missing_bindings()` SHALL report every global the shader declares that the
host's bind map does not supply, as `(entry, global_name)`. A key present with no
resource SHALL count as unbound, not as bound.

#### Scenario: A forgotten binding is reported

- **WHEN** a registered pass's host bind map omits a global its shader declares
- **THEN** the gate fails, naming the entry point and the global

#### Scenario: A fully bound pass reports nothing

- **WHEN** a registered pass binds every global its shader declares
- **THEN** `missing_bindings()` is empty for it

#### Scenario: Coverage runs with no GPU

- **WHEN** the coverage gate runs in a process with neither backend package
  importable
- **THEN** it completes and reports its result

### Requirement: Recordable passes are registered, and omission fails a meta-test

Each recordable pass SHALL be declared once in a registry naming its entry
module, its entry point, and how its bind map is obtained. The coverage gate
SHALL iterate the registry.

A meta-test SHALL assert that every compute entry point in the shader tree is
either registered or listed in a recorded exclusion set with a stated reason.
Adding a pass without doing one of the two SHALL fail the build.

#### Scenario: A new pass without registration fails the build

- **WHEN** a compute entry point is added to the shader tree and registered
  nowhere
- **THEN** the meta-test fails, naming the entry point

#### Scenario: An exclusion states its reason

- **WHEN** the exclusion set is enumerated
- **THEN** every entry carries a reason, and no entry names a pass that no longer
  exists

### Requirement: The coverage gate is proven able to fail

The suite SHALL carry a negative control: a fixture pass whose bind map
deliberately omits one declared global, asserted to be reported.

The negative control SHALL exercise the **same** code path as the real gate. A
parallel hand-built path would prove only that the parallel path works, which is
the defect this capability exists to remove.

#### Scenario: The negative control is reported

- **WHEN** the fixture pass with one deliberately omitted global is run through
  the gate
- **THEN** exactly that global is reported, through the same call the real gate
  uses

#### Scenario: Disabling the check fails the negative control

- **WHEN** the coverage comparison is disabled or weakened
- **THEN** the negative control fails, so the gate cannot silently become inert
