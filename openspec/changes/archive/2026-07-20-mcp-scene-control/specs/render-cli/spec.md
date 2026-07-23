## ADDED Requirements

### Requirement: MCP server flags on the interactive front-ends

The interactive front-ends SHALL accept a flag that enables the in-process MCP
server and a flag that overrides its port, each with an environment-variable
equivalent, following the established precedence of flag over environment over
default. The server SHALL be disabled by default.

The port option SHALL accept a port number only and SHALL NOT accept a host or
bind-address component, so that the loopback-only bind cannot be widened through
configuration.

These flags SHALL NOT be offered by the non-interactive front-ends, which have no
running renderer for a client to attach to.

#### Scenario: Default is disabled

- **WHEN** an interactive front-end is started with neither the flag nor its
  environment variable set
- **THEN** the MCP server is not started and no port is bound

#### Scenario: Flag overrides environment

- **WHEN** both the port flag and its environment variable are set to different
  values
- **THEN** the flag value is used

#### Scenario: Port option rejects a host component

- **WHEN** a value containing a host or bind address is supplied to the port option
- **THEN** startup fails with an error, and no server is started

#### Scenario: Not offered by non-interactive front-ends

- **WHEN** the headless render front-end is invoked with the MCP flag
- **THEN** the flag is not recognized and startup fails with an unrecognized-argument
  error

### Requirement: MCP flags are rejected against unsupported configurations at startup

Enabling the MCP server without its optional dependency installed SHALL fail at
startup with an explicit message naming the extra to install, consistent with the
existing startup rejection paths, rather than failing later at first client
connection.

#### Scenario: Enabled without the optional dependency

- **WHEN** an interactive front-end is started with MCP enabled and the MCP server
  dependency is not importable
- **THEN** startup fails with a message naming the extra to install, and the
  renderer does not start

#### Scenario: Enabled with the dependency present

- **WHEN** an interactive front-end is started with MCP enabled and the dependency
  is importable
- **THEN** startup proceeds and the registration command is printed
