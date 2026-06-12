# Tasks — metal-record-drain

## 1. Shader build flavor + slot budget (D1, D2)

- [x] 1.1 `wavefront/wf_records.slang`: change the stub gate from
      `#if defined(SKINNY_METAL)` to
      `#if defined(SKINNY_METAL) && !defined(SKINNY_METAL_RECORDS)` so the
      Metal records build compiles the real emitters
      (`wfResetRecords`/`wfPushRecord`/`wfEmitRecords`).
- [x] 1.2 `integrators/path_record_common.slang` + `bindings.slang`: extend the
      compile-out conditions so the neural+records build keeps
      `recordBuf`/`recordCounter` (36/37) declared while `toolBuffer` stays
      compiled out (`SKINNY_METAL && SKINNY_METAL_NEURAL &&
      !SKINNY_METAL_RECORDS` for 36/37; toolBuffer condition unchanged).
- [x] 1.3 `metal_wavefront.py`: `records_active` ctor flag on the path pass →
      `defines["SKINNY_METAL_RECORDS"] = "1"`; fold the flag into the pipeline
      cache key so records-on/off builds never clobber each other.
- [x] 1.4 Post-build reflected slot check: after session link, count each entry
      point's buffer bindings; raise `RuntimeError` naming the kernel and count
      when >31. If the neural+records build trips it, compile out
      `gizmoSegments`/`lightSplatBuffer` (wavefront-dead) behind
      `SKINNY_METAL_RECORDS` and re-verify.
- [x] 1.5 Guarded GPU smoke: neural+records pass builds on this host; log the
      heaviest kernel's slot count in the test output (thermal rule:
      scripts/guarded_metal.sh, one process).

## 2. Renderer drain + gating (D3, D4)

- [x] 2.1 `renderer.py`: replace the `descriptor_sets is not None` gate in
      `enable_online_training` with a backend-neutral check; on Metal request
      the path-pass rebuild with `records_active=True` (and rebuild without it
      on `disable_online_training`), resetting accumulation via the existing
      `_last_state_hash = None` path.
- [x] 2.2 `_ensure_wf_record_drain`: Metal branch — allocate the drain target +
      counter via `self._gpu.StorageBuffer`, seed `[0, capacity]` by
      `upload_sync`, and route the buffers into `_build_metal_binds()` under
      the existing `recordBuf`/`recordCounter` names.
- [x] 2.3 `_drain_wavefront_records`: backend split — Vulkan path unchanged;
      Metal path reads the counter and record bytes via `download_sync`,
      resets the counter, and feeds `records_from_buffer` → `ReplayBuffer`
      exactly like Vulkan.
- [x] 2.4 `recordMode` in the MSL uniform pack: confirm `_pack_uniforms_msl`
      carries `fc.recordMode` at the MSL-correct offset; add it if absent and
      extend `tests/test_metal_msl_uniform_offsets.py` either way.
- [x] 2.5 Unit tests (no GPU): record-source resolution arms `_wf_record_active`
      on a fake Metal ctx; drain math (counter clamp, capacity, reset) on fake
      buffers; records-off leaves `recordMode` 0 and requests no records build.

## 3. GPU verification (D5)

- [x] 3.1 Guarded GPU test: records-enabled Metal wavefront render on the flat
      Cornell scene drains records — count > 0, finite contributions, stack
      bounded (≤ REC_MAX_BOUNCES per lane), bytes parse via
      `records_from_buffer`.
- [x] 3.2 Guarded A/B: same scene/config/sample budget recorded on Metal and
      Vulkan → equivalent record sets (same vertices + contributions,
      order-independent; the wavefront-vs-megakernel equivalence criterion).
- [x] 3.3 Guarded bit-identity: Metal wavefront render with records off is
      byte-identical to the pre-change render (records build never constructed).
- [x] 3.4 Guarded end-to-end: fully-on-Metal online loop — wavefront path +
      neural proposal + `--neural-handoff interop` + numpy trainer; records
      drain each frame, weights publish GPU-side, `networkVersion` advances,
      converged `{bsdf,neural}` energy matches `{bsdf}` (unbiasedness gate).
      No Vulkan device, no NFW1 file.
- [x] 3.5 Measure path-pass frame time records-on vs records-off on the test
      scene; note the overhead number in the change.

## 4. Docs + housekeeping

- [x] 4.1 `docs/NeuralGuiding.md`: drop the "record drain remains Vulkan-only"
      caveat; document the Metal records build flavor + fully-on-Metal recipe.
- [x] 4.2 `docs/Wavefront.md` Metal section + `docs/Architecture.md` slot-cap
      and drain notes (36/37 on Metal, SKINNY_METAL_RECORDS).
- [x] 4.3 `README.md` online-training platform notes + `CHANGELOG.md` entry.
- [x] 4.4 `ruff check src/`, `node docs/diagrams/embed_code.cjs --check`,
      `pytest -m 'not gpu'` sweep, guarded GPU tests, `openspec validate
      metal-record-drain`.
