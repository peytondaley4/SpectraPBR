# Device-Side Path Guiding Rework — Implementation Plan

> **For agentic workers:** Executed inline (executing-plans) in the authoring session. TDD steps are replaced by a
> consistency-check protocol because this machine (macOS) cannot compile CUDA/OptiX; the build/run checkpoints below
> are executed by the user on the RTX box. No commits are made by the agent; tier boundaries are the suggested commit
> points.

**Goal:** Replace the CPU round-trip cell-discovery pipeline with device-resident allocation (Tier 1), cut per-vertex
guide overhead ~8× via single-cell conditioning (Tier 2), and lift the guide's variance-reduction ceiling with a
4-lobe vMF mixture (Tier 3).

**Architecture:** One persistent device hash table (atomicCAS insert-on-first-touch, bump allocator, stable cell
indices for the lifetime of the grid). Cells materialize the moment a path vertex needs them and are usable at the
next refit (≤4 frames) instead of after a ~10-frame staging→readback→CPU-merge→swap round trip. Refinement becomes a
periodic device kernel that inserts children of well-fed cells (sample-count criterion, PPG-style); coarsening is
dropped (indices are stable, memory is bounded by the fixed capacity). The CPU keeps only debug/UI mirrors.

**Tech stack:** CUDA 12.x, OptiX 7+, existing build (PTX via nvcc custom command; `path_guide_kernels.cu` compiled
into the main target).

---

## Locked design decisions

### Device hash table (Tier 1)
- Key: `(uint64)level << 48 | morton` (matches existing packing). `PG_KEY_EMPTY = 0xFFFF'FFFF'FFFF'FFFF`.
- Values: cell index, or `PG_VALUE_PENDING = 0xFFFFFFFF` (slot claimed, cell not yet published) /
  `PG_VALUE_FULL = 0xFFFFFFFE` (claimed but allocator exhausted). Lookup treats both sentinels as miss.
- Insert: probe (Fibonacci hash, linear probe, cap 32); on EMPTY: pre-check `*counter < capacity` (plain read), then
  `atomicCAS` the key. Winner: `idx = atomicAdd(counter, 1)`; on overflow write `FULL`; else initialize the cell
  payload, `__threadfence()`, then publish `values[slot] = idx`. Racing readers that see `PENDING` treat the cell as
  missing for this launch — never spin (warp-safe under OptiX).
- `cell_keys[idx] = key` written by the winner — gives the host an O(cells) mirror readback for debug viz/inspector
  and gives kernels a way to decode a cell's level/coords from its index.
- Table sizing: `hash_table_size = next_pow2(2 × cell_capacity)` (≤50% load). Default `cell_capacity = 1 << 19`.
- The data array is pre-zeroed once; each index is allocated at most once, so winners need not clear their payload
  (Tier 3 adds an explicit lobe init by the winner).
- Lives in `optix_programs/path_guide_hash_device.h` with **explicit-argument functions only** (no `params`), shared
  by `raygen.cu` (OptiX) and `path_guide_kernels.cu` (plain CUDA). Morton encode/spread move here.

### Lookup (Tier 1)
Top-down replaces fine-to-coarse: probe `start_level`; while a child cell containing the position exists, descend
(child morton = `parent << 3 | octant`). Unsubdivided steady state = 1 hit + 1 miss ≈ 2 probes (was ~5–6). If the
`start_level` probe misses, raygen **inserts** the base cell (bounds-checked — fixes the out-of-bounds seeding drip)
and uses it immediately as the training target (kappa = 0 ⇒ never sampled before its first refit, so PDFs stay
consistent within the launch). `min_level` is retired from the hot path.

### Subdivision (Tier 1, replaces runRefinementPass)
Kernel over allocated cells, every `refine_interval_frames` (30) while training: if `level < max_level` and
`cumCount ≥ subdivide_count_threshold` (default 8192, config knob), insert the 8 children; children warm-start with
the parent's lobe(s) and 1/8 of the parent's cumulative stats (keeps guiding + Tier 2 confidence alive through the
split). Count criterion replaces the `kappa < 2` / log-likelihood rules — wide incident radiance is a property of the
lighting, not insufficient spatial resolution, and the old rule exploded cell counts in diffuse regions.
The LL helper (`vmf_fitting.{h,cpp}`) is deleted.

### Cell layout
Tier 1–2 keeps stride 16 (`PG_RESERVED` becomes `PG_EXP_NEG2K`). Tier 3 stride 64:

```
[ 0..23]  4 lobes × {mu.xyz, kappa, expNeg2K, weight}      (sampling hot data, 96B)
[24..39]  per-lobe interval sums {sx, sy, sz, sw}           (atomicAdd by shaders)
[40..55]  per-lobe cumulative EMA sums                      (owned by refit)
[56] lastHitFrame  [57] interval count  [58] cum count (EMA)  [59..63] reserved
```

### Tier 2 — single-cell conditioning (Müller 2017 jittered lookup)
Per vertex: top-down lookup at `pos` → level L; jitter `pos` by ±0.5 cell at L; second top-down lookup at the
jittered position picks **the** context cell (fall back to the unjittered cell on miss). That one cell is used for
sampling, the combined PDF, the NEE MIS pdf, and training. Unbiased: the pdf matches the realized technique
conditioned on the jitter, which is drawn independently of the direction. The 8-cell trilinear machinery
(`TrilinearInfo`, `computeTrilinearNeighbors`, `filterTrilinearByValidLobes`, `stochasticSelectCell`,
`trilinearGuidePdf`) is deleted.

Eligibility/alpha: `alpha = mis_weight × (1 − pSpec) × confidence`, `confidence = min(1, cumCount / 32)`, gated on
`kappa ≥ 2` (Tier 3: on the eligible-lobe subset). Wide lobes are strictly worse than cosine sampling and waste up to
half their samples below the horizon — they fall back to pure BSDF. Deterministic from read-only cell data ⇒
consistent across all three PDF uses.

Deposits: per-path gate `rand < 0.25`, weights ×4 (cuts the contended atomic traffic 4×; training estimates are
unchanged in expectation). Refit caches `exp(−2κ)` per lobe so `vmfPdf`/`vmfSample` cost 1 `expf` instead of 2.

### Tier 3 — K = 4 vMF mixture, hard-assignment stepwise EM
- Insert-time init: tetrahedral directions, κ = 0 (ineligible until trained), π = 0.25 each.
- Deposit: assign to `argmax_k π_k · vmf(max(κ_k, 0.5), μ_k·d)` (κ floor partitions the sphere while lobes are
  untrained); 4 atomicAdds into that lobe's interval sums + count + lastHit ⇒ same atomic budget as the single lobe.
- Refit M-step per lobe: EMA-fold interval→cumulative, `π_k = W_k/ΣW`, μ/κ via Banerjee from per-lobe resultant,
  cache `exp(−2κ)`. Dead lobes (π < 0.02 with ΣW ≥ 32) re-seed deterministically near the strongest lobe with κ = 1.
- Sampling: CDF over **eligible** lobes (π ≥ 0.05, κ ≥ 2), renormalized; pdf is the same renormalized subset mixture
  ⇒ consistent. Cell eligible iff Σπ_eligible ≥ 0.3 and the confidence ramp passes.
- `clear()` re-runs the lobe-init kernel (zeroed lobes would collapse the hard assignment).
- Inspector/hemisphere UI shows the top-2 lobes by weight (the 2-lobe UI fields already exist).

### Deletions (Tier 1)
`PathGuideStagingDescriptor`/staging buffers, `pathGuideStagingAppend`, the async readback state machine
(`beginAsyncReadback`/`pollAsyncReadback`/`finishBuildFromReadback`/`swapGrids`), double-buffered `GridBuffers`,
`buildStructureArrays`, `uploadStructure`, `buildAndUploadHashTable` (CPU), `gatherCellsKernel`,
`runRefinementPass` (CPU), `hierarchicalCellLookup`, the binary-search fallback, morton/level-offset arrays in
LaunchParams, app build-thread machinery (`m_buildFuture`, `m_buildThreadActive`, `m_pathGuideBuildInFlight`,
`m_buildThisFrame`, `m_pathGuideAutoBuildInterval`), `vmf_fitting.{h,cpp}`.

### LaunchParams delta (gpu_types.h and shared_types.h stay byte-identical)
- Remove: `path_guide_morton_codes`, `path_guide_level_offsets`, `path_guide_staging_{buffer,count,capacity}`.
- Add: `path_guide_cell_keys` (u64*), `path_guide_cell_counter` (u32*), `path_guide_cell_capacity` (u32).
- Keep: data, hash table fields, bounds, level resolutions, levels, enabled, mis_weight, debug stats.

### Host class (PathGuideGrid) after rewrite
- `init/shutdown/clear` (clear = zero data + re-init lobes; structure persists).
- `refitLobes(frame, stream)` / `runSubdivisionPass(frame, stream)` — capacity-sized launches, device-side counter
  bound (no host count needed).
- `requestCellCountAsync(stream)` + `lastCellCount()` — 4-byte pinned poll for UI (every 60 frames).
- `refreshHostMirror()` — synchronous counter + `cell_keys` readback (debug only: G-toggle, inspector click,
  after subdivision passes while visualizing). Feeds `generateEdgeVertices*` and `inspectCellAtPosition`.
- StepOnce mode = one refit + one subdivision pass, then Paused. `updatePathGuideAutomationStatus`'s "builds" slot
  reports subdivision passes.

### CMake
- Add `optix_programs/path_guide_hash_device.h` to `OPTIX_HEADER_DEPS`.
- Add `${CMAKE_CURRENT_SOURCE_DIR}/optix_programs` to `target_include_directories(SpectraPBR …)` so
  `path_guide_kernels.cu` can include the shared header.
- Remove `src/rendering/vmf_fitting.cpp`.

---

## Task sequence

- [x] **T1.1** `optix_programs/path_guide_hash_device.h` — table struct, sentinels, morton, lookup, insert.
- [x] **T1.2** `optix_programs/path_guide_grid_device.h` — descriptor rework, top-down lookup, insert-on-miss; delete
      staging/binary search/hierarchical lookup (trilinear survives Tier 1).
- [x] **T1.3** `path_guide_kernels.{h,cu}` — counter-bounded refit, `subdivideCellsKernel`, delete gather.
- [x] **T1.4** `path_guide_grid.{h,cpp}` — class rewrite per above.
- [x] **T1.5** LaunchParams (both headers), `optix_engine.{h,cpp}` setter, `raygen.cu` grid init + seeding block,
      `application.{h,cpp}` cadence + inspector, CMakeLists.
- [x] **T1.6** Consistency pass: grep deleted symbols to zero; LaunchParams headers diff-identical in the guide block;
      every new param set.
- [ ] **CHECKPOINT (RTX box):** build, run, enable guiding. Expect: no staging-overflow warnings (gone), cells appear
      within ~4 frames, wireframe viz works, inspector works, image matches pre-change unguided render when guiding off.
- [x] **T2.1** Single-cell conditioning + kappa gate + confidence ramp in `raygen.cu`; delete trilinear machinery.
- [x] **T2.2** `exp(−2κ)` cache (refit + vmf_device.h cached variants) + per-path deposit subsampling.
- [ ] **CHECKPOINT (RTX box):** A/B guiding on/off frame time; expect most of the guiding-on cost gone; check
      below-horizon stat ratio drops.
- [x] **T3.1** Mixture layout + init/refit/subdivide kernels + raygen deposit/sampling/pdf.
- [x] **T3.2** Inspector/hemisphere top-2 lobe mapping.
- [ ] **CHECKPOINT (RTX box):** multi-light/interior scene; expect visibly faster convergence at equal time vs Tier 2.
- [x] Final fresh-eyes review of the full diff (subagent); 4 bugs + 4 nits found and fixed (insert publish-before-init, subdivision same-pass cascade, clamp-after-scale, StepOnce gate; stride/scale/capacity guards, count-unit docs).
