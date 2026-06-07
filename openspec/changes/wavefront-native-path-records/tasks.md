> Status: PROPOSAL (deferred). Not implemented. Motivated by the NVIDIA/Windows
> megakernel TDR found while landing `neural-online-training` (task 1.2). All
> boxes intentionally unchecked.

## 1. Wavefront record emission

- [ ] 1.1 Add a per-lane record stack (`RecVertex[REC_MAX_BOUNCES]`: pos/normal/wo/wiLocal/L_k/beta_in_k/depth) to the wavefront path-state buffer; bump the stride in `vk_wavefront.py`
- [ ] 1.2 Emit/push on a guideable bounce in the wavefront shade/scatter kernel — same guard (flat/graph, reflective, `wiLocal.y > 1e-4`, `pdf > 0`) and pre-update `beta_in_k` snapshot as `estimateRadianceRecord`
- [ ] 1.3 Terminate-time backward attribution: on lane termination splat `contrib_k = max((L_final − L_k)/beta_in_k, 0)` via the existing bounds-safe `emitRecord` (binding 36 append + binding 37 counter); drop non-finite
- [ ] 1.4 Share the attribution math with `integrators/path_record.slang` (one source of truth for `contrib_k`)

## 2. Record-mode gate

- [ ] 2.1 Frame-constants flag enabling record emission only while online training is active; default-off keeps the wavefront render byte-identical (no stack writes, no emit)
- [ ] 2.2 Thread the flag from `Renderer._online_training` through `_pack_uniforms` / the wavefront passes

## 3. Renderer rewire

- [ ] 3.1 `drain_path_records_to_replay` reads the wavefront-produced records (no `mainImageRecord` dispatch) when record-mode is on
- [ ] 3.2 Keep `mainImageRecord` + `dump_path_records` for the offline `.nrec` dump only (off the per-frame path)

## 4. Verification

- [ ] 4.1 GPU end to end on NVIDIA/Windows: `{bsdf,neural}` online loop drains real wavefront records → train → swap → unbiased (the test the megakernel drain could not run)
- [ ] 4.2 Stream parity: wavefront record stream ≡ megakernel `.nrec` dump (same records modulo order) on a box where the megakernel runs
- [ ] 4.3 Default-render invariance: record-mode off ⇒ wavefront image unchanged vs the current backend
- [ ] 4.4 `py_compile` + `ruff check src/` green; `openspec validate wavefront-native-path-records --strict` passes

## 5. Docs

- [ ] 5.1 `docs/Wavefront.md`: record-emission stage + per-lane vertex stack + terminate-time splat
- [ ] 5.2 `docs/Architecture.md`: bindings 36/37 are wavefront-fed on the online path (megakernel only for the offline dump)
