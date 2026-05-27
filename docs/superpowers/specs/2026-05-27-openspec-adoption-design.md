# OpenSpec Adoption for skinny — Design

**Date:** 2026-05-27
**Status:** Approved (brainstorm)
**Topic:** Adopt OpenSpec as the spec-driven development standard for skinny

## Context

skinny is a mature Python/Slang physically-based skin renderer with rich
existing documentation (`Architecture.md`, `SkinRendering.md`, `CHANGELOG.md`,
`README.md`, `CLAUDE.md`, `AGENTS.md`) and an informal spec→plan habit captured
under `docs/superpowers/{specs,plans}`. The OpenSpec CLI (v1.3.1) is already
installed at `/opt/homebrew/bin/openspec`; Node v25.9 is available.

Goal: adopt OpenSpec as the standard for documenting and developing further
features, without discarding the existing curated docs.

## Decisions

1. **Spec scope: minimal scaffold, grow per-change.** Initialize OpenSpec and
   leave `openspec/specs/` near-empty. Specs accrete as change proposals are
   archived. No upfront backfill of existing subsystems. (OpenSpec's recommended
   pattern; avoids stale retro-specs on a fast-moving codebase.)
2. **Workflow: brainstorm first, then OpenSpec.** Keep the superpowers
   brainstorm dialogue for aligning on intent/approach, but the resulting
   artifacts live as an OpenSpec change proposal — not in `docs/superpowers/`.

## Design

### 1. Scaffold

Run `openspec init --tools claude`. It creates:

- `openspec/AGENTS.md` — OpenSpec conventions for AI assistants. Tool-managed;
  never hand-edit. Re-synced via `openspec update`.
- `openspec/project.md` — project context, filled in by us: stack, key
  directories, build/test/lint commands, naming conventions, and links to
  `Architecture.md`, `SkinRendering.md`, and `CLAUDE.md`.
- `openspec/specs/` — empty now; grows per change.
- `openspec/changes/` — empty now; holds active change proposals.
- `.claude/commands/openspec/*` — Claude Code slash commands (proposal / apply /
  archive).
- A marker-delimited managed block appended to root `CLAUDE.md` pointing at
  `openspec/AGENTS.md`. Reversible (removable via `openspec update`). The diff is
  shown before acceptance; curated `CLAUDE.md`/`AGENTS.md` prose stays intact.

### 2. Existing docs relationship

- `Architecture.md`, `SkinRendering.md`, `CHANGELOG.md`, `README.md` remain
  as-is (reference / architecture docs).
- `openspec/project.md` links to them so proposals carry context.
- `docs/superpowers/{specs,plans}` (prior camera work) left untouched; not
  migrated.
- No spec backfill (per Decision 1).

### 3. Workflow going forward (brainstorm → OpenSpec)

For each non-trivial feature:

1. Brainstorm dialogue (superpowers) to align on intent + approach.
2. Capture as `openspec/changes/<slug>/`: `proposal.md`, `tasks.md`, and spec
   deltas under `changes/<slug>/specs/<capability>/spec.md`.
3. `openspec validate <slug> --strict`.
4. User approves → implement the tasks.
5. `openspec archive <slug>` folds the deltas into `openspec/specs/` (the source
   of truth) and moves the change to the archive.

Trivial fixes (typos, one-liners) skip the proposal. OpenSpec becomes the
artifact home for feature work; the brainstorm dialogue remains the thinking
step that precedes a proposal.

### 4. Bootstrap note

OpenSpec is not initialized yet, so this adoption is performed directly rather
than as its own change proposal. The change-proposal flow begins with the next
feature.

## Out of scope (YAGNI)

- No retro-spec of existing subsystems.
- No migration of existing docs into `openspec/`.
- No CI hook for `openspec validate` (can be added later if wanted).
