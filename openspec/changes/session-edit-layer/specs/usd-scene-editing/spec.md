# Delta: usd-scene-editing

## MODIFIED Requirements

### Requirement: Stage is the authoritative model with a non-destructive edit layer

On loading a USD scene the renderer SHALL retain the live `Usd.Stage` as the
authoritative scene model and author non-destructive edits into the stage's
**session layer** (`Usd.Stage.GetSessionLayer()`), set as the stage edit target.
The session layer is stronger than the entire root layer stack, so an edit
overrides any opinion authored in the root/file layer. The original root layer
SHALL NOT be written to disk by any editing operation. The flat scene and GPU
buffers SHALL be a derived cache that the renderer refreshes from the stage on
each edit.

A transform edit (`set_transform` / the shared local-transform author) SHALL be
able to override a prim's `xformOp:transform` even when that op is authored in
the loaded file, without raising a duplicate-op error. The author path SHALL
reuse an existing single `xformOp:transform` op by setting its value in the edit
target, rather than clearing and re-adding the op.

#### Scenario: Edit target is the session layer

- **WHEN** a USD scene is loaded for editing
- **THEN** the stage edit target is the session layer, and no editing sublayer is
  inserted into the root layer's sublayer stack

#### Scenario: Original file untouched by edits

- **WHEN** any editing operation authors to the stage
- **THEN** the opinion is written to the session (edit) layer and the original
  file on disk is unchanged

#### Scenario: File-authored transform can be overridden

- **WHEN** `set_transform` is called on a prim whose `xformOp:transform` is
  authored in the loaded root/file layer
- **THEN** no duplicate-op error is raised and the prim's composed local
  transform equals the newly authored matrix
