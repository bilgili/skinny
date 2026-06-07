> Status: IMPLEMENTED on the motivating NVIDIA/Windows box (RTX 4090, driver
> 596.21). Storage revised from the original proposal: the per-lane record stack
> lives in SEPARATE gated set-1 buffers, NOT inside `WavefrontPathState` (which is
> copied by value in every kernel — inlining it would spill to scratch / ~8× the
> path-state bandwidth in every kernel even when not training). The drain is made
> source-selectable so megakernel training stays available. Task 4.2 (megakernel
> parity) is gated to a non-TDR box.

## 1. Wavefront record emission

- [x] 1.1 Add a per-lane record stack (`RecVertex`: pos/normal/wo/wiLocal/L_k/beta_in/depth, 76 B) in SEPARATE set-1 buffers — `wfRecStack` (binding 9) + per-lane `wfRecCount` (binding 10) in `wavefront/wf_records.slang`, sized full only while recording (else 1-element dummy) in `vk_wavefront.py` (`record_capacity`); host mirror `REC_VERTEX_STRIDE`/`REC_MAX_BOUNCES` in `wavefront_layout.py`. (Revised from "inside WavefrontPathState" — see banner.)
- [x] 1.2 Emit/push on a guideable bounce in `wfFinishShade` — same guard (flat/python, reflective, `wiLocal.y > 1e-4`, `pdf > 0`) and pre-update `beta_in` snapshot as `estimateRadianceRecord` (`wfPushRecord`)
- [x] 1.3 Terminate-time backward attribution: on lane termination splat `contrib_k = recordContrib(L_final, L_k, beta_in_k)` via the shared bounds-safe `emitRecord` (binding 36 append + binding 37 counter); drop non-finite. Termination points: `wfTerminate` (miss/RR/no-bsdf/sphere-light, both copies) + `wfPathResolve` (max-depth survivors)
- [x] 1.4 Share the attribution math with the megakernel record entry — `recordContrib` + `PathRecord` + `emitRecord` + `REC_MAX_BOUNCES` live in one module `integrators/path_record_common.slang`, imported by both `path_record.slang` and `wf_records.slang`

## 2. Record-mode gate

- [x] 2.1 `FrameConstants.recordMode` (`common.slang`) enabling record emission only while the wavefront drain is active; default-off keeps the wavefront render byte-identical (no stack writes, no emit) — verified diff = 0
- [x] 2.2 Thread the flag from `Renderer._wf_record_active` through `_pack_uniforms`; the path pass rebuild key + `record_capacity` follow it

## 3. Renderer rewire (dual-source drain)

- [x] 3.1 `drain_path_records_to_replay` reads the wavefront-produced records (no `mainImageRecord` dispatch) when the resolved source is `wavefront` (`_drain_wavefront_records` + `_ensure_wf_record_drain`); `_resolve_record_source` (`auto`/`megakernel`/`wavefront`)
- [x] 3.2 Keep `mainImageRecord` + `dump_path_records` for the offline `.nrec` dump AND as a selectable (`megakernel`) live-drain source on non-TDR boxes (off the per-frame wavefront path)

## 4. Verification

- [x] 4.1 GPU end to end on NVIDIA/Windows (RTX 4090): wavefront render drains 21k+ real records → replay, no megakernel dispatch, no device-lost — the test the megakernel drain could not run (`tests/test_wavefront_records.py::test_wavefront_drain_end_to_end`, `test_record_stream_format_roundtrip`)
- [ ] 4.2 Stream parity: wavefront record stream ≈ megakernel `.nrec` dump (statistically — the wavefront path uses RR, so equal in expectation, not byte-exact). Gated behind `SKINNY_RUN_MEGAKERNEL_PARITY=1` (the megakernel record pipeline ~400 s-compiles / device-losts under the Windows TDR here) — `test_wavefront_megakernel_parity`
- [x] 4.3 Default-render invariance: record-mode off ⇒ wavefront image bit-identical (diff = 0) — `test_record_mode_invariance`; record-source resolution — `test_record_source_resolution`
- [x] 4.4 `py_compile` + `ruff check` green on changed files (5 pre-existing repo-wide ruff errors unrelated to this change); `tests/test_wavefront_state.py` + `tests/test_headless.py::*pack_uniforms_size` green; `openspec validate wavefront-native-path-records --strict` passes

## 5. Docs

- [x] 5.1 `docs/Wavefront.md`: wavefront-native record emission stage + per-lane vertex stack + terminate-time splat + dual-source drain
- [x] 5.2 `docs/Architecture.md`: bindings 36/37 are wavefront-fed when `recordMode` is set (megakernel only for the offline dump); `docs/NeuralGuiding.md`: shared `recordContrib` (regenerated embed)
