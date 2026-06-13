# Tasks — combine-graph-param-buffers

## 1. Codegen: single buffer + offset table

- [x] 1.1 `megakernel_sources.py`: add a host-side layout helper that, for the
      ordered `graph_fragments`, computes each graph's stride (= host param-pack
      stride, the value already used to size `_graph_param_buffers`) and a
      16-B-aligned cumulative byte base; returns `[(target, base, stride, n)]` +
      total combined size.
- [x] 1.2 `emit_megakernel_aggregator`: replace the per-graph `[[vk::binding(25+
      idx)]] StructuredBuffer<GraphParams_X>` decls with one
      `[[vk::binding(GRAPH_BINDING_BASE,0)]] ByteAddressBuffer graphParamsCombined;`
      plus `static const uint GRAPH_BASE_<name>` / `GRAPH_STRIDE_<name>`
      constants; rewrite each `_graphParams_X(matId)` to
      `graphParamsCombined.Load<GraphParams_X>(GRAPH_BASE_X + matId*GRAPH_STRIDE_X)`.
      Zero-graph path unchanged (no decl referenced).
- [x] 1.3 `emit_wavefront_shade_module` (in `vk_compute.py` /
      `megakernel_sources.py`, wherever the wavefront half lives): mirror 1.2 so
      each per-material shade module declares the same single combined buffer +
      its own graph's `GRAPH_BASE`/`GRAPH_STRIDE` and `Load<T>` accessor.
- [x] 1.4 `emit_generated_materials` return value: replace the per-graph
      `{target: 25+idx}` binding map with the single-slot binding + the offset
      table (consumed by the host packer + binder).
- [x] 1.5 Codegen unit test (`tests/` non-GPU): 3-fragment fixture → one
      `ByteAddressBuffer` at slot 25, monotonic 16-B-aligned bases, strides match
      the fixture struct sizes, accessors reference the constants.

## 2. Host: pack + bind one combined buffer

- [x] 2.1 `renderer.py` `_upload_graph_param_buffers`: build one
      `self._graph_params_combined` StorageBuffer of the codegen total size;
      copy each target's packed blob to its `GRAPH_BASE` offset (zero-fill pad).
      Drop the per-target `_graph_param_buffers` dict (or keep as staging only).
- [x] 2.2 `renderer.py` `_scene_graph_bindings` + every bind site
      (`_upload_graph_param_buffers` specs, `build_wavefront_shade_passes`,
      `_ensure_wavefront_path_pass`, megakernel aggregator bind): emit one
      `graphParamsCombined` bind at slot 25 instead of the per-target loop.
- [x] 2.3 `vk_compute.py`: collapse the `for idx … binding GRAPH_BINDING_BASE+idx`
      descriptor spec to a single storage-buffer descriptor at
      `GRAPH_BINDING_BASE`.
- [x] 2.4 `metal_compute.py` + `metal_wavefront.py`: replace the per-target
      `graph_param_layouts` reflection + per-name bind entries with one
      `graphParamsCombined` reflection + bind; assert the reflected
      `GraphParams_X` strides equal the emitted `GRAPH_STRIDE_<name>` (loud fail
      on drift).

## 3. Env-CDF combine (the completing −1 slot)

Graph-param combine alone leaves the neural+online-training wavefront build 1
slot over at baseline (independent of graph count). Reclaim it by folding the
two env importance-sampling CDF buffers into one.

- [x] 3.1 `environment.slang`: replace `envMarginalCdf`(31)+`envCondCdf`(32) with
      one `[[vk::binding(31)]] ByteAddressBuffer`-free `StructuredBuffer<float>
      envDistCdf` = `[marginal | conditional]`; conditional reads add
      `ENV_COND_CDF_BASE = ENV_DIST_H+1`. Update `envImagePdf` + `sampleEnvDir`.
- [x] 3.2 Host (`renderer.py`): `env_dist_buffer` replaces the two buffers
      (alloc sized `(H+1)+H*(W+1)`; upload `marg+cond`; bind-by-name `envDistCdf`;
      Vulkan descriptor write at 31 only; destroy one). `vk_compute.py`: drop the
      binding-32 layout entry. Fix the descriptor-pool count (env 2→1 buffer;
      graph N→1).
- [x] 3.3 Guarded GPU build (one Metal compile process): the reported command
      builds the wavefront path pass (`wfPathShadeFlat` ≤ 31 buffers) and runs —
      neural ACTIVE, online-training APPROVED, **no SLANG_FAIL**. RED before
      §1–3, GREEN after. ✅ verified.

## 4. Equivalence + regression

- [x] 4.1 Correctness render: `three_materials_demo.usda` on Vulkan megakernel
      (uses `graphParamsCombined` + `envDistCdf`) — all 3 graph materials read
      distinct, non-black radiance via the structural AOV; scene env-lit. ✅
      (Full Metal↔Vulkan megakernel A/B skipped: the Metal megakernel
      cold-compile is the RAM-spike path the box-safety guard warns against; the
      Metal **wavefront** build success + Vulkan render together cover it, since
      `Load<T>` scalar layout is backend-identical.)
- [x] 4.2 Zero-graph path preserved (emitter test: no `graphParamsCombined` decl
      when no graphs); single-combined-buffer bind covers single- and multi-graph.
- [x] 4.3 `ruff check` clean (no new findings — worktree renderer drops 1
      pre-existing finding); `pytest -m 'not gpu'` green (99 passed; the only
      failures are pre-existing / worktree-missing-`.mtlx`-asset, not this change).
      `main_pass.spv` is an untracked runtime artifact (regenerated in-process on
      every compile via the content-hash cache); the Vulkan render recompiled it.

## 5. Docs

- [ ] 5.1 `docs/Architecture.md`: descriptor binding map — slot 25 is now the
      single combined graph-param buffer (was 25..25+N−1).
- [ ] 5.2 `docs/Wavefront.md` + `docs/Megakernel.md`: graph-param layout (one
      byte-addressed buffer, per-graph compile-time offsets).
- [ ] 5.3 `CHANGELOG.md` entry; CLAUDE.md compatibility note only if the
      supported-combo wording changes (the reported combo now runs).
- [ ] 5.4 `openspec validate combine-graph-param-buffers --strict`; archive after
      merge.
