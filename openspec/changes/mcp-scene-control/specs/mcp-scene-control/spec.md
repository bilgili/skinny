## ADDED Requirements

### Requirement: Opt-in in-process MCP server attached to a running renderer

The interactive front-ends SHALL be able to host an MCP server inside the running
renderer process, serving over streamable HTTP on a loopback socket. The server
SHALL be disabled unless explicitly enabled, SHALL attach to the renderer that is
already running, and SHALL NOT construct a `Renderer` or a GPU context of its own.
The server thread SHALL be a daemon thread so it cannot delay process exit.

#### Scenario: Server absent by default

- **WHEN** a front-end starts without the MCP flag or its environment equivalent
- **THEN** no listening socket is opened, no MCP dependency is imported, and
  startup behavior is byte-for-byte unchanged from before this capability existed

#### Scenario: Server attaches to the live renderer

- **WHEN** a front-end starts with MCP enabled and a scene is loaded
- **THEN** an MCP client can connect and enumerate the scene currently displayed
  in that process's window, and no second renderer or GPU context is created

#### Scenario: Optional dependency absent

- **WHEN** MCP is enabled but the MCP server dependency is not installed
- **THEN** startup fails with an explicit message naming the extra to install, in
  the same style as the other startup rejection paths, rather than degrading
  silently or failing later at first connection

### Requirement: The MCP server thread never touches the renderer directly

Every MCP tool handler SHALL perform all renderer reads and writes inside a
callback submitted to the render-thread command queue and awaited for its reply.
The server thread SHALL NOT access `Renderer` attributes, the scene graph, or any
GPU resource directly, because `Renderer` has no internal lock and its scene graph
may be swapped by the streaming load thread.

Every wait on a queued reply SHALL carry a timeout and SHALL return an MCP error
on expiry rather than blocking indefinitely.

#### Scenario: Reads are marshalled, not just writes

- **WHEN** an MCP tool inspects the scene graph
- **THEN** the inspection executes on the render thread via the command queue and
  returns a detached copy, never a live reference read from the server thread

#### Scenario: Render thread stalled

- **WHEN** an MCP tool posts work and the render thread does not drain within the
  timeout
- **THEN** the tool returns an MCP error describing the timeout, and the server
  remains able to serve subsequent requests

#### Scenario: Renderer exception becomes a tool error

- **WHEN** a renderer call raised inside a queued callback propagates back to the
  server thread
- **THEN** it is reported as an MCP tool error naming the failure, not as an
  unhandled transport-level error

### Requirement: The server does not alter process signal handling

Starting the MCP server SHALL NOT install, replace, or remove process signal
handlers. The GPU backend registers interrupt and termination handlers to guarantee
context teardown, and overwriting them would leave abandoned GPU work unrecoverable
on the host.

The listening socket SHALL be created by the front-end rather than by the server
runtime, so that the loopback assertion and the bind-collision path are both
handled before the server runtime starts.

#### Scenario: Signal handlers survive server startup

- **WHEN** the MCP server has started
- **THEN** the process's interrupt and termination handlers are the same objects
  they were before it started

#### Scenario: Bind failure is detected before startup is reported

- **WHEN** the configured port is already bound
- **THEN** the failure is detected at socket creation, before any registration line
  is printed

### Requirement: Path-addressed scene tools

The server SHALL expose scene inspection and mutation as a small set of tools
addressed by USD prim path plus property name — enumeration, node read, and
property write — rather than one tool per renderer method and rather than an
arbitrary code-execution tool.

Property writes SHALL be routed by a single dispatch function shared with the
graphical scene-graph editor, so that an MCP edit and the equivalent editor edit
execute the same code rather than two tables that can drift.

This SHALL cover scalar, boolean, colour, vector, and file-path property writes.
The graphical editor's *file-chooser flows* are exempt: they own a dialog and
report failure asynchronously, and against the editor's proxy the underlying
call returns a future rather than a boolean, so they cannot share the
synchronous routing path. The routing decision those flows encode SHALL still
live in the shared function, so the client reaches the same verb.

Dispatch SHALL be determined from the resolved property and its node — the
property's type and metadata, the node's type and reference, an ancestor walk where
the node itself carries no renderer reference, and sibling property values where a
write recomposes a compound value. Dispatch SHALL NOT be determined from the
addressed path and property name alone.

#### Scenario: Material parameter on a shader prim

- **WHEN** a client writes a material parameter addressed at a shader prim, which
  carries no renderer reference of its own
- **THEN** the write resolves through an ancestor walk to the enclosing material and
  applies, rather than failing to route

#### Scenario: Compound transform write

- **WHEN** a client writes one component of a node's transform
- **THEN** the remaining components are read from the node and the full transform is
  recomposed, so the unwritten components are preserved

#### Scenario: Editor and client take the same path

- **WHEN** a client writes a property, and the operator makes the equivalent edit in
  the graphical editor
- **THEN** both execute the same dispatch function and produce the same visible
  result

#### Scenario: Unsupported property is an explicit error

- **WHEN** a client writes a property that has no dispatch route for that node kind
- **THEN** the tool returns an error naming the path and the property, and SHALL
  NOT report success for an edit that did not take effect

#### Scenario: Unknown path is an explicit error

- **WHEN** a client addresses a prim path that does not resolve in the current scene
- **THEN** the tool returns an error naming the unresolved path

### Requirement: Tool results are self-describing

Node reads SHALL include, for each property, whether it is editable and any range
metadata the scene model carries for it, so that a client can determine what it may
set and to what bounds from the tool result alone.

Writes SHALL be validated against the published range metadata and a violation
SHALL be rejected with an error quoting the bounds. Writes SHALL NOT be silently
clamped, because the published ranges are editor affordances rather than legal
bounds — the graphical editor itself permits exceeding them for properties marked
growable, and clamping would both make a client less capable than the operator and
silently alter the render.

Properties marked growable SHALL be exempt from the bounds check.

Writes SHALL also be validated against the property's declared type before
reaching renderer code, and a mismatch SHALL be rejected rather than coerced — a
client can send any value the wire format permits, so a string must not become a
boolean, a non-finite number must not reach a material override, and a compound
value must carry exactly its declared component count.

#### Scenario: Type mismatch is rejected, not coerced

- **WHEN** a client writes a string to a boolean property, or a non-finite number to
  a numeric one
- **THEN** the write is rejected with an error naming the expected type, and nothing
  is applied

#### Scenario: Compound value of the wrong length is rejected

- **WHEN** a client writes a two-component value to a three-component property
- **THEN** the write is rejected rather than reaching transform recomposition

#### Scenario: Client reads editability and bounds

- **WHEN** a client reads a node carrying properties with range metadata
- **THEN** the result reports each property's editable flag and its bounds without
  requiring documentation outside the tool result

#### Scenario: Out-of-range write is rejected

- **WHEN** a client writes a value outside a non-growable property's published bounds
- **THEN** the write is rejected with an error quoting the bounds, and the value is
  neither applied nor silently clamped

#### Scenario: Growable property accepts a value beyond its published maximum

- **WHEN** a client writes a value above the published maximum of a property marked
  growable
- **THEN** the write is applied, matching what the graphical editor permits

### Requirement: Enumeration returns structure, not properties

Scene enumeration SHALL return tree structure only — path, name, type, and child
count — and SHALL NOT include node properties. Property detail SHALL be obtained by
reading an individual node.

Enumeration SHALL accept a starting path, a traversal depth bound with a small
default, and an optional filter by node kind, so that a client can locate nodes in
a large scene without retrieving the whole tree.

#### Scenario: Large scene enumerates without property payload

- **WHEN** a client enumerates a scene containing many prims
- **THEN** the result contains structural entries only, bounded by the requested
  depth, and contains no property values

#### Scenario: Filter by node kind

- **WHEN** a client enumerates with a kind filter such as lights
- **THEN** only nodes of that kind are returned, resolved from the reference kind
  already carried by the scene model

### Requirement: Concurrent operator edits use last-write-wins

The server SHALL NOT lock the scene, SHALL NOT disable the graphical controls while
a client is connected, and SHALL NOT reject a write because the scene changed since
the client last read it.

Every tool result SHALL report both the structural scene-graph version and the
material version, because property edits advance only the latter — the renderer
deliberately does not bump the structural version on a property edit, since the
editor's widgets are bound to the live property objects. Reporting only the
structural version would leave a client unable to observe any property edit.

Client writes SHALL be awaited rather than coalesced: resolution, validation, and
routing all execute on the renderer's owning thread, and the client SHALL be told
whether its write applied or why it did not. A fire-and-forget write would report
success for an edit a validation check then discarded. The awaited round-trip
itself paces a client, so a single client cannot grow the command queue without
bound.

Where the graphical editor does coalesce its own high-rate edits, compound values
such as transforms SHALL be coalesced per node rather than per component, because
a component write recomposes from its siblings and a per-component key would lose
updates.

#### Scenario: Operator edits during a client session

- **WHEN** the operator edits a control in the graphical front-end while a client
  holds an earlier read of the scene
- **THEN** the operator's edit applies normally, the graphical controls remain
  enabled, and a subsequent client write is applied rather than rejected

#### Scenario: Property edit is observable

- **WHEN** either party edits a material or light property between two tool calls
- **THEN** the material version reported by the second result differs from the first

#### Scenario: Structural change is observable

- **WHEN** the scene structure changes between two tool calls
- **THEN** the structural scene-graph version reported by the second result differs
  from the first

### Requirement: Persistence, node authoring, and rendered output are excluded

The server SHALL NOT expose a tool that exports or saves scene edits, SHALL NOT
expose tools that add or remove scene nodes, and SHALL NOT expose a tool that
returns rendered images.

Saving is excluded because material, light, and instance edits mutate in-memory
render state without authoring to the USD edit layer, so an export would silently
omit most client edits.

Node authoring is excluded because adds and removes author solely into the USD edit
layer, which this capability provides no way to export — so their entire output
would be discarded at process exit — and because they are inoperative on scenes
with no USD stage. They belong with the export capability.

Image return is excluded because any edit resets progressive accumulation, so an
immediate readback would return a near-noise frame.

#### Scenario: No save tool is advertised

- **WHEN** a client lists available tools
- **THEN** no tool for saving, exporting, or persisting scene edits is present

#### Scenario: No node authoring tools are advertised

- **WHEN** a client lists available tools
- **THEN** no tool for adding or removing scene nodes is present

#### Scenario: No image tool is advertised

- **WHEN** a client lists available tools
- **THEN** no tool returning rendered image data is present

### Requirement: Loopback-only bind

The server SHALL bind only the loopback interface. The bind address SHALL NOT be
configurable, and the configured port option SHALL NOT accept a host component. The
loopback address SHALL be asserted at socket creation rather than merely defaulted.

#### Scenario: Bind is loopback

- **WHEN** the server starts
- **THEN** it listens on the loopback interface only and is not reachable from
  another host

#### Scenario: Non-loopback bind is refused

- **WHEN** code or configuration attempts to bind a non-loopback address
- **THEN** the server refuses to start rather than listening on that address

### Requirement: Requests are authenticated and browser-origin requests refused

The server SHALL require a bearer token on every request, compared using a
constant-time comparison. The token SHALL be generated once with a
cryptographically secure random source, stored under the user settings directory
with owner-only permissions applied at creation, reused across restarts, and
overridable by environment variable.

The server SHALL validate the `Host` header against the loopback host and the bound
port, so that a rebound-DNS request — which need carry no `Origin` at all — is
refused.

The server SHALL refuse any request carrying an `Origin` header. This is a
deliberate trade: it blocks browser-hosted MCP clients and browser-based debugging
tools, which is an accepted cost of closing the drive-by path.

#### Scenario: Missing or wrong token

- **WHEN** a request arrives without a bearer token, or with an incorrect one
- **THEN** the request is refused as unauthorized and no renderer work is enqueued

#### Scenario: Token persists across restarts

- **WHEN** the front-end is restarted with MCP enabled
- **THEN** the previously generated token remains valid, so an existing client
  configuration continues to work without reconfiguration

#### Scenario: Token file permissions

- **WHEN** the token file is created
- **THEN** it is owner-readable only, with the restrictive mode applied as it is
  created rather than after the fact, and published atomically so an interrupted
  write cannot leave a truncated token that locks out later launches

#### Scenario: Token file is not a symlink or another user's file

- **WHEN** the token path resolves to a symlink, a non-regular file, or a file owned
  by another user, on a platform providing the necessary primitives
- **THEN** the server refuses to start rather than reading it

#### Scenario: Platform without the necessary primitives

- **WHEN** the host does not provide no-follow opens or POSIX ownership
- **THEN** the server still starts, relying on the platform's own access control
  for the settings directory, and the reduced guarantee is documented rather than
  implied to be equivalent

#### Scenario: Token cannot be published atomically

- **WHEN** the filesystem cannot create the token exclusively
- **THEN** the server refuses to start and directs the operator to supply the token
  by environment variable, rather than falling back to a non-exclusive write that
  could destroy a token another instance is already serving

#### Scenario: Registration names the token source actually in force

- **WHEN** the token comes from the environment override rather than the file
- **THEN** the printed registration command references the environment variable,
  not the file, which may be absent or hold a different value

#### Scenario: Browser-originated request refused

- **WHEN** a request arrives carrying an `Origin` header
- **THEN** the request is refused regardless of whether it carries a valid token

#### Scenario: Mismatched Host refused

- **WHEN** a request arrives whose `Host` header does not name the loopback host and
  bound port
- **THEN** the request is refused, even when it carries no `Origin` header

#### Scenario: Absent or portless Host refused

- **WHEN** a request arrives with no `Host` header, or with a bare hostname carrying
  no port
- **THEN** the request is refused — both are shapes a rebound request can take, and
  neither is what a client talking to this server sends

### Requirement: Client configuration is printed at startup

When MCP is enabled, the front-end SHALL print at startup the complete client
registration command, including the transport, the loopback URL with the actual
port, and the authorization header, so that setup is copy-paste.

The printed command SHALL reference the token file rather than embedding the token
value, so the secret does not enter terminal scrollback or captured logs.

#### Scenario: Startup prints the registration line

- **WHEN** a front-end starts with MCP enabled and binds successfully
- **THEN** it prints a registration command containing the bound port and the
  authorization header

#### Scenario: Token value is not printed

- **WHEN** the registration line is printed
- **THEN** it contains a reference to the token file rather than the token's value

### Requirement: Port collision leaves the renderer running

The server SHALL use a fixed default port, overridable by option. If the port is
already bound — for example by a second front-end instance — the front-end SHALL
log a warning and continue running with MCP disabled. It SHALL NOT exit, and SHALL
NOT silently select a different port.

#### Scenario: Second instance collides

- **WHEN** a second front-end starts with MCP enabled while the port is already bound
- **THEN** it warns that MCP is unavailable, continues to start and render normally,
  and does not listen on any other port
