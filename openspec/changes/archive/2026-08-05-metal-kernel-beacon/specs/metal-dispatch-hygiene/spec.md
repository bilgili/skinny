# Metal dispatch hygiene

## ADDED Requirements

### Requirement: The timeout path reports the hung kernel identity

The parent SHALL read the beacon cell and report the hung kernel's identity by
name when its wall-clock timeout expires and it SIGTERMs the child. The report
SHALL name the kernel whose id the child stamped last, resolved
through the kernel-identity table. The SIGTERM path SHALL keep its current
contract: the chained handler runs `MetalContext.destroy()`, the parent waits the
grace period, and it escalates to SIGKILL only after it confirms the child holds
no in-flight dispatch. The beacon report SHALL be additive; it SHALL NOT change
the SIGTERM-first, never-SIGKILL-first order.

#### Scenario: a wedged dispatch is reported by kernel name, not a guess
- **WHEN** a child dispatch hangs and the parent times out
- **THEN** the parent SIGTERMs the child (running `destroy()`), then reports the
  hung kernel by its entry-point name instead of "a Metal dispatch did not
  return"

#### Scenario: the beacon report does not weaken the kill order
- **WHEN** the parent adds the beacon report to the timeout path
- **THEN** the parent still sends SIGTERM first, still waits the grace period, and
  still escalates to SIGKILL only after it confirms no in-flight dispatch
