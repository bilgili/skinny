# OpenSpec Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap OpenSpec (v1.3.1) in the skinny repo as the spec-driven development standard, with a project-context file and the brainstorm→OpenSpec workflow ready for future features.

**Architecture:** Run `openspec init --tools claude` to scaffold `openspec/` + Claude Code slash commands, fill `openspec/project.md` with skinny's stack/commands/conventions, verify with the OpenSpec CLI, then commit (force-adding past the `*` .gitignore so specs are version-controlled).

**Tech Stack:** OpenSpec CLI 1.3.1 (`/opt/homebrew/bin/openspec`), Node v25.9, git. No application code changes.

**Note on verification:** This is a tooling bootstrap — there is no application code and therefore no unit tests. "Verify" steps run the OpenSpec CLI and inspect files/output instead of pytest.

**Tracking decision (from `git ls-files`):** `.gitignore` is `*` (whole tree ignored; tracked files were force-added deliberately). Repo convention: `docs/superpowers/{specs,plans}/*.md` ARE tracked; `.claude/` tracks only `settings.local.json`. So this plan commits `openspec/config.yaml` plus this session's design + plan docs (force-add), and leaves the regenerable `.claude/` OpenSpec files (slash commands + skills) untracked — they are recreated by `openspec init` / `openspec update`.

**Divergence from the design's init assumptions (OpenSpec 1.3.1 actual behavior):**
- Init does NOT modify `CLAUDE.md` or `AGENTS.md` — both diffs were empty. Curated files untouched.
- Init does NOT create `openspec/project.md` or `openspec/AGENTS.md`. `project.md` is *legacy* in 1.3.1 (the CLI's migration hint says move its content to `openspec/config.yaml`'s `context` field).
- Integration lives in `.claude/skills/openspec-{propose,explore,apply-change,archive-change}/SKILL.md` + `.claude/commands/opsx/{propose,explore,apply,archive}.md` (slash command `/opsx:*`).
- `openspec init --tools claude` ran non-interactively and reported "Config: skipped", so no `config.yaml` was created by init — Task 2 creates it.
- Default workflow schema is `spec-driven`.

---

### Task 1: Initialize the OpenSpec scaffold

**Files (1.3.1 actual):**
- Create: `openspec/specs/`, `openspec/changes/` (+ `changes/archive/`). No `openspec/project.md` or `openspec/AGENTS.md` (legacy in 1.3.1).
- Create: `.claude/commands/opsx/{propose,explore,apply,archive}.md`, `.claude/skills/openspec-{propose,explore,apply-change,archive-change}/SKILL.md`
- Does NOT modify `CLAUDE.md` or `AGENTS.md`.

- [ ] **Step 1: Run init non-interactively**

Run:
```bash
openspec init --tools claude
```
Expected: success output listing created files; `openspec/` and `.claude/commands/openspec/` now exist. If it prompts about legacy cleanup, re-run with `openspec init --tools claude --force`.

- [ ] **Step 2: Verify the scaffold exists**

Run:
```bash
ls -la openspec && echo "---commands---" && ls .claude/commands/openspec
```
Expected: `AGENTS.md`, `project.md`, `specs/`, `changes/` under `openspec/`; one or more `*.md` slash-command files under `.claude/commands/openspec/`.

- [ ] **Step 3: Inspect what init changed in tracked root files**

Run:
```bash
git status --short && echo "---CLAUDE.md diff---" && git diff -- CLAUDE.md
```
Expected: `CLAUDE.md` shows an added block delimited by `<!-- OPENSPEC:START -->` / `<!-- OPENSPEC:END -->` pointing at `openspec/AGENTS.md`. New `openspec/` and `.claude/commands/openspec/` files appear as ignored/untracked (because `.gitignore` is `*`). Confirm the curated CLAUDE.md prose above/below the block is untouched. Do NOT commit yet (Task 3 commits).

- [ ] **Step 4: Confirm the CLI sees the project**

Run:
```bash
openspec list ; openspec list --specs
```
Expected: both run without error and report no changes / no specs yet (empty is correct — specs grow per change).

---

### Task 2: Write `openspec/config.yaml` (project context)

**Files:**
- Create: `openspec/config.yaml` (init skipped config; `project.md` is legacy in 1.3.1)

- [ ] **Step 1: Write config.yaml with `schema` + `context`**

Create `openspec/config.yaml` with `schema: spec-driven` and a `context:` YAML
block scalar. The `context` is injected as `<project_context>` into every
artifact-generation prompt. The substance below (originally drafted as a
`project.md`) was adapted into that `context:` block — see the committed
`openspec/config.yaml` for the exact YAML. Source substance:

```markdown
# Project: skinny

## Purpose
Physically-based skin renderer. Layered skin optics (BSSRDF, GGX specular),
path + bidirectional path tracing, MaterialX materials, USD scene loading.
Interactive GLFW app with progressive accumulation; Vulkan and Metal backends.

## Tech Stack
- Python 3.11+ (`src/` layout, package `skinny`)
- Slang shaders compiled to SPIR-V via `slangc`; `main_pass.slang` is the compute entry
- Vulkan backend: `vk_context.py`, `vk_compute.py` (instance/device/swapchain, SPIR-V reflection)
- Metal backend: `metal_backend.py` (SlangPy/RHI, `DeviceType.metal`)
- MaterialX >=1.39 built from source with `PyMaterialXGenSlang` (the PyPI wheel lacks it)
- USD loading (`usd_loader.py`), GLFW window, ImGui/Qt UI (`ui/`)

## Key Directories
- `src/skinny/` — Python package; `app.py` entry point, `renderer.py` orchestration
- `src/skinny/shaders/` — Slang sources (materials/, integrators/, lights/, samplers/, cameras/)
- `tests/` — pytest, organised by subsystem; `harnesses/` Slang test shaders, `kernels/` reference kernels
- `hdrs/` — Radiance HDR environments; `heads/` — OBJ head models + texture maps; `assets/`, `tattoos/`

## Commands
- Setup: `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`
- Run: `.venv/bin/skinny` (or `python -m skinny.app`); force backend with `--backend metal|vulkan`
- Lint: `.venv/bin/ruff check src/`
- Test: `.venv/bin/pytest`
- Compile shader: `slangc src/skinny/shaders/main_pass.slang -target spirv -entry mainImage -stage compute -o src/skinny/shaders/main_pass.spv -I src/skinny/shaders`
- Headless render: use the repo-root Python 3.13 venv (`./bin/python3.13`) with `VULKAN_SDK` + `DYLD_LIBRARY_PATH` set (see CLAUDE.md and tests/test_headless.py)

## Conventions
- Ruff, 100-char lines. `snake_case` functions/vars/modules, `PascalCase` classes, UPPERCASE for true constants.
- `SkinParameters.pack()` std140 bytes must match the `SkinParams` Slang struct exactly.
- Any shader change requires recompiling `main_pass.spv` with `slangc`.
- Keep Slang modular; share utilities via `common.slang`.
- Default backend: Metal on macOS (auto-falls back to Vulkan); `SKINNY_BACKEND` env var overrides.

## Reference Docs
- `Architecture.md` — module map + authoritative descriptor binding map
- `SkinRendering.md` — skin estimator chain (sections 1–6)
- `CLAUDE.md` / `AGENTS.md` — contributor and agent guidance
- `CHANGELOG.md`, `README.md`
```

- [ ] **Step 2: Verify the context is read by OpenSpec**

Run:
```bash
openspec new change verify-context >/dev/null 2>&1
openspec instructions proposal --change verify-context 2>&1 | sed -n '/<project_context>/,/<\/project_context>/p' | head -5
rm -rf openspec/changes/verify-context && openspec list
```
Expected: the `<project_context>` block contains skinny's background (mentions "skinny ... renderer"); cleanup leaves "No active changes found".

---

### Task 3: Verify end-to-end and commit

**Files:**
- Commit (force-add): `openspec/config.yaml`, `docs/superpowers/specs/2026-05-27-openspec-adoption-design.md`, `docs/superpowers/plans/2026-05-27-openspec-adoption.md`
- Leave untracked: `.claude/commands/opsx/*`, `.claude/skills/openspec-*` (regenerable)

- [ ] **Step 1: Sanity-check the whole OpenSpec install**

Run:
```bash
openspec list && openspec list --specs && openspec update --help >/dev/null && echo OK
```
Expected: no errors, prints `OK`. (No changes/specs yet is correct.)

- [ ] **Step 2: Confirm no curated tracked files changed**

Run:
```bash
git diff -- CLAUDE.md AGENTS.md
```
Expected: empty. 1.3.1 init does not modify these.

- [ ] **Step 3: Stage the tracked artifacts (force past `.gitignore`)**

Run:
```bash
git add -f openspec/config.yaml \
  docs/superpowers/specs/2026-05-27-openspec-adoption-design.md \
  docs/superpowers/plans/2026-05-27-openspec-adoption.md
git status --short
```
Expected: the three files staged; no `CLAUDE.md`/`AGENTS.md` changes; `.claude/` OpenSpec files remain untracked.

- [ ] **Step 4: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore(openspec): adopt OpenSpec for spec-driven development

Scaffold openspec/ (project.md + empty specs/changes) and Claude Code
slash commands. Future features: brainstorm to align, then capture as an
openspec change proposal; archive folds spec deltas into openspec/specs.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```
Expected: commit succeeds.

- [ ] **Step 5: Verify the commit**

Run:
```bash
git status && git log --oneline -1
```
Expected: clean tree (modulo pre-existing `.claude/settings.local.json`); top commit is the OpenSpec adoption commit.

---

## Self-Review

**Spec coverage:**
- Design §1 (Scaffold) → Task 1 (init, verify structure, confirm curated files untouched) + Task 3 (commit config.yaml). Mechanism differs from design (config.yaml not project.md; no CLAUDE.md block) — see Divergence note.
- Design §2 (Existing docs) → Task 2 config.yaml `context` links to Architecture.md/SkinRendering.md/CLAUDE.md; no migration; no backfill (nothing in plan creates specs). Covered.
- Design §3 (Workflow going forward) → documented in the commit message + `openspec/AGENTS.md` (tool-managed); no code needed now. Covered.
- Design §4 (Bootstrap note) → whole plan is the direct bootstrap; first change proposal is the next feature, not part of this plan. Covered.
- Design "Out of scope" → plan creates no retro-specs, migrates no docs, adds no CI hook. Consistent.

**Placeholder scan:** No TBD/TODO; project.md content is given in full; all commands are concrete with expected output.

**Type consistency:** N/A (no code/types). File paths consistent across tasks (`openspec/project.md`, `.claude/commands/openspec`).
