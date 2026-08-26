# Performance Diagnosis and Measurement Experiment

## Purpose

This document records a static performance review of SpectraPBR and defines an experiment for measuring the renderer correctly. It is intended as a handoff for an implementation agent running on the target NVIDIA RTX system.

The central question is why SpectraPBR can perform reasonably well while recent rendering papers often show a scene becoming legible in only a few 15–30 ms frames.

The short answer is: both hardware and implementation specialization matter, but the larger difference is usually the estimator. SpectraPBR currently generates essentially fresh Monte Carlo evidence every frame. Many real-time research systems generate one new path per pixel and then make that path count many times through spatial reuse, temporal reuse, radiance caches, reconstruction, or temporal denoising. Their effective sample count is therefore much larger than their literal samples per pixel.

This review identifies likely latency sources, but it does not assign costs without measurements from the target GPU. Hypotheses below must be confirmed or rejected using GPU timestamps and profiler captures.

## Separate the Three Performance Questions

Do not reduce performance to a single FPS number. Measure these independently:

1. **GPU frame time:** time spent executing the renderer, guide maintenance, denoiser, and presentation work.
2. **Time to legibility:** elapsed time or number of frames required to reach a defined image-quality threshold after an accumulation reset.
3. **Interaction-to-display latency:** time from a camera or scene change until the corresponding image becomes visible.

A technique can improve one while worsening another. For example, triple buffering improves throughput but adds presentation latency. Path guiding may improve long-run convergence while making each frame slower and providing little benefit during its initial training period.

## Current Architecture: Positive Findings

The application plumbing is generally sensible:

- Release PTX is compiled with `-O3 --use_fast_math -lineinfo` in [CMakeLists.txt](../CMakeLists.txt).
- VSync is disabled by default in [src/graphics/gl_context.cpp](../src/graphics/gl_context.cpp).
- Scene display uses three CUDA/OpenGL PBOs so CPU submission and GPU work can overlap.
- Launch parameters use pinned staging buffers rather than pageable asynchronous copies.
- Path guiding and denoising are disabled by default and opt in independently.
- Acceleration structures use `OPTIX_BUILD_FLAG_PREFER_FAST_TRACE`.

The likely performance gap is therefore not broad editor overhead or an obviously unoptimized build. The strongest candidates are concentrated in the path kernel, the guide's cost-to-benefit timing, ray count, and the absence of screen-space sample reuse.

## Finding 1: The Existing CPU Diagnostic Does Not Measure GPU Render Time

`Application::renderFrame()` labels a CPU interval as `render`, but `OptixEngine::render()` submits an asynchronous parameter copy and `optixLaunch()`. The optional denoiser is also queued asynchronously. The CPU interval therefore measures submission overhead, not completion of tracing and denoising.

Most GPU execution time is observed later in `syncRdr`, when `waitForRender()` synchronizes the event associated with an older PBO. This can make the render bucket appear cheap while synchronization appears expensive, even though the synchronization is merely where earlier GPU work becomes visible to the CPU.

Relevant code:

- [src/core/application.cpp](../src/core/application.cpp), `Application::renderFrame()`
- [src/optix/optix_engine.cpp](../src/optix/optix_engine.cpp), `OptixEngine::render()`
- [src/cuda/cuda_interop.cpp](../src/cuda/cuda_interop.cpp), `CudaInterop::waitForRender()`

### Required correction

Use CUDA events on the actual render stream. At minimum, record non-blocking event pairs around:

- path-guide refit and subdivision kernels;
- launch-parameter upload plus OptiX launch;
- OptiX launch alone, if practical;
- denoiser average-color calculation, invocation, and output copy;
- output PBO map through render-complete event;
- total queued render-stream work for one frame.

Read completed timing events several frames later. Do not synchronize the current frame merely to print a timing value, because that would change the scheduling behavior being measured.

CPU timing should remain available for:

- event polling and scene bookkeeping;
- UI collection/render submission;
- OpenGL display submission;
- `glfwSwapBuffers()`;
- time blocked waiting for an old render buffer.

Report CPU and GPU timings separately rather than summing asynchronous intervals.

## Finding 2: Triple Buffering Adds Deliberate Presentation Latency

The display path selects a buffer roughly two frames older than the buffer currently being rendered. This is a valid throughput optimization: it lets rendering continue without waiting for the immediately preceding frame. It also means a new camera view or scene state may not be visible for approximately two render intervals.

At a 25 ms render interval, this queue alone can contribute roughly 50 ms of interaction-to-display latency. This does not mean the GPU takes 50 ms to render a frame. Research papers generally report algorithmic GPU time rather than editor click-to-photon latency, so these values are not directly comparable.

Measure and report:

- render throughput with the current three-buffer path;
- age, in submitted frames, of the displayed PBO;
- interaction-to-display latency after a camera movement or explicit accumulation reset.

Do not remove buffering as part of the initial experiment. First quantify its throughput and latency effects.

## Finding 3: The Megakernel Carries Heavy Per-Path State

The strongest kernel-level suspicion is the amount of live state in [optix_programs/raygen.cu](../optix_programs/raygen.cu).

`tracePath()` includes material and medium state, MIS state, texture-footprint state, guide lookup state, and an eight-entry training array:

```cpp
TrainRecord train[MAX_TRAIN_VERTICES]; // MAX_TRAIN_VERTICES = 8
```

A `TrainRecord` is logically about 64 bytes, making this array approximately 512 bytes per thread before the rest of the function's state is considered. Whether the compiler scalarizes, spills, or otherwise transforms it must be measured, but the likely risks are:

- high register count per thread;
- reduced active warps and occupancy;
- local-memory loads and stores;
- large live ranges across `optixTrace()` calls;
- greater penalties when paths in the same warp take different material branches or terminate at different depths.

Because guiding is a runtime mode in the same compiled program, the training machinery remains part of the megakernel even when guiding is disabled. Runtime-disabled accesses may disappear, but the compiled function and its resource requirements still need inspection.

This is also a plausible reason for the optional wavefront implementation to become valuable even if it is not faster yet. Wavefront scheduling adds queues and kernel launches, but it can shrink each kernel's live state and compact surviving paths after divergence. The checked-out `main` revision used for this review, `e80dc36`, contains no wavefront files or symbols, so the optional path was not evaluated here.

### Required profiler evidence

Capture the OptiX launch with Nsight Compute or the closest supported profiler and record:

- registers per thread;
- local-memory bytes and local load/store transactions;
- achieved occupancy and active warps;
- branch efficiency or divergent-branch stalls;
- long-scoreboard and memory-dependency stalls;
- L1/L2 hit rates;
- atomic serialization;
- ray-traversal utilization, where available.

If OptiX JIT reporting exposes stack or register information, retain that output with the experiment results.

## Finding 4: Guiding Performs Significant Work Before It Can Help

When guiding is active, each eligible nonterminal vertex can perform:

- top-down sparse-grid hash lookup;
- first-touch insertion using `atomicCAS`;
- stochastic cell jitter and another lookup;
- repeated parent lookups across refinement boundaries;
- four-lobe eligibility evaluation;
- parallax reprojection;
- BSDF/product-guide construction;
- vMF sampling and PDF evaluation.

Half of all paths are selected for training by `PG_TRAIN_PROB = 0.5`. The backward pass walks up to eight records. A successful lobe deposit evaluates all four lobes and performs numerous atomics for directional sums, count, distance, spatial moments, visit statistics, and half-cell radiance statistics.

The guide is also a large, randomly accessed data structure. Its default capacity is 524,288 cells. Each cell contains 98 floats, or 392 bytes, so cell data alone can consume approximately 196 MiB. Hash keys, hash values, cell keys, counters, and allocator state add more memory. Only the leading 112 bytes are identified as hot sampling data, but sparse lookups and training updates can still put pressure on cache and memory bandwidth.

At the same time, the guide is intentionally conservative:

- lobes are refit every four training frames;
- subdivision runs every 30 training frames;
- guide confidence ramps with maturity up to 32 units;
- source comments describe maturity accumulating over roughly 50 refits in sparse regions.

This makes the guide primarily a long-run convergence mechanism. It cannot normally make the first one to three frames dramatically cleaner because those frames are producing the evidence that later guide proposals consume.

## Finding 5: SpectraPBR Still Traces Full Paths and NEE Shadow Rays

For every non-delta surface vertex, the renderer traces a continuation ray and normally performs one next-event-estimation shadow ray. Balanced mode permits up to eight bounces, High permits sixteen, and Accurate permits thirty-two, with Russian roulette beginning earlier.

That is substantially different from systems that terminate most rendering paths into a cache, trace sparse long paths for training, reuse paths across pixels and frames, or reconstruct a full-resolution result from lower-resolution radiance.

World-space path guiding changes the proposal distribution for future samples. It does not reuse the radiance estimate from an already successful path across nearby pixels or later frames. Consequently, even a well-trained guide may improve variance without giving the same first-frame effective sample multiplication as ReSTIR-style reuse.

## Finding 6: The Optional Denoiser Adds Work Without Temporal Reuse

The current OptiX denoiser uses `OPTIX_DENOISER_MODEL_KIND_AOV` with albedo and normal guides. Each enabled frame performs:

1. HDR average-color computation;
2. denoiser invocation;
3. a full-resolution `float4` device-to-device output copy.

At 1920x1080, one RGBA32F image is approximately 31.6 MiB. The denoiser is useful for presentation, but this path has no motion vectors, previous denoised output, or temporal denoiser state. It therefore costs GPU time without providing the temporal reconstruction responsible for much of the stability and rapid legibility in modern real-time renderers.

Measure the denoiser separately. Do not fold it into the path-tracing time.

## Finding 7: Editor and Interop Overhead Is Real but Probably Secondary

The scene display uploads an RGBA32F PBO into an OpenGL texture each displayed frame. The UI PBO is also mapped and unmapped every frame even when `UIRenderer::renderIfChanged()` decides no UI raster work is necessary. Texture-preview changes explicitly synchronize the UI stream.

These operations can contribute:

- CUDA/OpenGL ownership-transition overhead;
- approximately 31.6 MiB of scene texture transfer at 1080p;
- another full-resolution UI transfer when the UI changes;
- occasional UI-stream synchronization spikes.

The UI is cached and uses a separate stream, while the scene path is triple buffered. These costs should be measured, but they are unlikely by themselves to explain a large convergence or frame-time gap.

## Why Research Results Can Look So Much Faster

### Hardware

High-end hardware is common. The 2021 neural radiance caching paper and the 2022 ReSTIR PT/GRIS paper report 1920x1080 results on an RTX 3090. More recent caustics and bidirectional ReSTIR work commonly reports RTX 4090 results.

Hardware must therefore be normalized before comparing timings. Record GPU model, clock/power state, driver, OptiX version, CUDA version, resolution, scene, path depth, SPP, denoising, and whether presentation/UI work is included.

### Specialized estimators

The more important advantage is that the research systems are designed around low-sample reuse:

- ReSTIR PT shades one new path per pixel but resamples paths from spatial neighbors and prior frames.
- Neural radiance caching combines ReSTIR direct lighting with short render paths, sparse long training paths, and a fully fused Tensor-Core cache.
- Denoisers and reconstructors often use temporal history, motion vectors, and auxiliary buffers.
- Some systems render at a lower internal resolution and upscale.
- Results are commonly evaluated on selected scenes with method-specific parameter choices.

The neural radiance caching paper reports satisfactory quality after about eight frames, approximately 70 ms, at 1080p on an RTX 3090. It also reports that terminating paths into the cache reduced path-tracing work by roughly 25%. In several equal-error comparisons, PT plus ReSTIR required roughly 67–431 ms while PT plus ReSTIR plus NRC required roughly 8–15 ms. This was not simply a leaner implementation of the same estimator; the cache and sample reuse changed how much information each traced ray produced.

The ReSTIR PT paper also shows that real-time figures are scene dependent. Equal-time comparisons span budgets around 25–80 ms, not a universal 15 ms. A 2026 ReSTIR PT follow-up reports an additional 2–3x speedup from algorithmic and engineering improvements, demonstrating how aggressively these research pipelines are tuned.

Primary references:

- [Generalized Resampled Importance Sampling / ReSTIR PT](https://research.nvidia.com/labs/rtr/publication/lin2022generalized/)
- [Real-time Neural Radiance Caching for Path Tracing](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing)
- [ReSTIR PT Enhanced](https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/)
- [ReSTIR Path Guiding](https://research.nvidia.com/labs/rtr/publication/zeng2025restirpg/)

## Controlled Experiment

### Fixed conditions

Use a representative static scene and camera. Record the scene name and exact camera transform.

- Release build only.
- VSync off.
- 1920x1080 internal and output resolution.
- One SPP per frame.
- Balanced quality mode and its default depth/Russian-roulette behavior.
- Fixed seed sequence and fixed camera for comparable accumulated images.
- No guide visualization, cell inspector, verbose logging, screenshots, or scene edits.
- No unrelated GPU applications.
- Warm the process and GPU before collecting steady-state timing.
- Collect at least 200 measured frames per configuration after warmup.

Repeat the experiment on one easy scene and one difficult indirect-lighting scene if time permits. Do not combine their results.

### A/B matrix

Run these configurations independently:

| ID | Guiding | Training | Denoiser | Purpose |
|---|---|---|---|---|
| A | Disabled | Disabled | Off | Baseline megakernel and interop cost |
| B | Enabled with a mature frozen guide | Paused | Off | Isolate guide lookup, sampling, and PDF cost |
| C | Enabled | Running | Off | Isolate backward training, atomics, refits, and subdivision |
| D | Disabled | Disabled | On | Isolate denoiser cost and visual benefit |
| E | Enabled with a mature frozen guide | Paused | On | Measure the best current presentation path without training |
| F | Enabled | Running | On | Measure the full interactive guided configuration |

For B and E, train the guide using a documented fixed procedure, then pause it without changing the camera. Preserve the guide and reset only image accumulation before measurement.

Useful differences:

- `B - A`: guide lookup/sampling/PDF overhead.
- `C - B`: guide training plus periodic maintenance overhead.
- `D - A`: denoiser overhead.
- `F - C`: denoiser overhead when sharing the full guided workload.

Because refit and subdivision are periodic, report median, mean, p95, p99, and maximum frame time. Also report regular-frame and maintenance-frame populations separately.

### GPU timing output

For every configuration, report milliseconds for:

- guide refit;
- guide subdivision;
- OptiX path launch;
- denoiser average-color calculation;
- denoiser invocation;
- denoiser output copy;
- total render stream;
- PBO map/unmap ownership interval, if it can be measured correctly;
- CPU wait for the displayed render buffer;
- OpenGL texture update and display submission;
- total CPU loop interval.

Also report achieved frames per second, but treat it as a derived value rather than the primary measurement.

### Time-to-legibility test

Frame time and convergence efficiency must be evaluated separately.

1. Produce a high-sample reference for the same camera and scene.
2. Reset accumulation and any temporal image history.
3. For each configuration, save the raw and displayed result after frames 1, 2, 4, 8, 16, 32, and 64.
4. Compute an image error metric against the reference, preferably FLIP plus a numerical metric such as relative MSE.
5. Record the first frame and elapsed GPU time at which each configuration crosses predetermined error thresholds.
6. Repeat with identical seeds if determinism is available; otherwise run multiple trials and report dispersion.

Evaluate both equal-frame and equal-time comparisons. A slower guided frame may still win at equal time if its variance reduction is sufficiently large. Conversely, a guide that wins after 64 frames may still be unsuitable for an interactive first-response target.

Do not compare a denoised SpectraPBR image against a raw reference method or vice versa without labeling the difference.

### Profiler captures

Capture at least configurations A, B, and C on the same representative frame:

- A reveals the base megakernel's resource use and divergence.
- B shows the incremental cost of guide lookups and vMF/product sampling.
- C shows training atomics and backward-pass effects.

Retain the profiler reports and summarize the top limiting factors rather than only reporting aggregate kernel duration.

### Interpretation guide

- **A is already slow, with high local traffic or low occupancy:** prioritize reducing megakernel live state and spills; compare against the wavefront path.
- **B is much slower than A:** guide lookup, cache locality, vMF/product evaluation, or extra branch divergence is expensive.
- **C is much slower than B:** backward records and atomic training traffic are the primary guide cost.
- **Maintenance frames dominate p95/p99:** refit or subdivision cadence/work distribution needs attention even if average cost is low.
- **D is expensive:** denoise less frequently, eliminate avoidable copies, or evaluate a temporally amortized reconstruction path—but only after measuring quality.
- **GPU work is fast but `syncRdr` or display is slow:** investigate CUDA/OpenGL ownership transitions, texture copies, queue depth, and presentation separately.
- **Frames are fast but the image becomes legible slowly:** the central problem is estimator efficiency, not frame plumbing. Screen-space temporal/spatial reuse or radiance caching is the relevant direction.
- **The frozen guide improves equal-time error but active training loses:** separate guide training from the steady render path or reduce training frequency/contention.
- **The wavefront path has more launches but lower local traffic and better occupancy:** optimize queue/compaction overhead rather than returning immediately to the megakernel.

## Expected Conclusion Before Measurement

The current leading hypothesis is:

1. **Per-frame GPU cost** is most likely dominated by full-path ray count plus megakernel state/divergence.
2. **Additional guided-frame cost** is most likely dominated by sparse hash access, vMF/product-guide work, the per-thread training array, and contended backward-pass atomics.
3. **Time to initial legibility** is most constrained by the absence of screen-space temporal/spatial sample reuse; world-space guiding improves future proposals but does not multiply already discovered samples across pixels and frames.
4. **Interaction latency** includes approximately two queued display frames by design.
5. **Editor and interop work** should be measured but is unlikely to be the main explanation for the gap with recent research systems.

The experiment should replace these hypotheses with measured costs before any optimization is selected.

## Deliverables

The experiment is complete when it produces:

- the exact commit, build configuration, GPU, driver, CUDA, and OptiX versions;
- the scene, camera, resolution, SPP, quality mode, and feature state;
- CPU and GPU timing tables for configurations A–F;
- median, mean, p95, p99, and maximum timings;
- raw and displayed convergence images at the requested frame counts;
- FLIP and numerical error curves versus both frames and milliseconds;
- Nsight captures for A, B, and C;
- a short conclusion identifying the measured dominant latency source;
- a recommendation that distinguishes throughput, time-to-legibility, and interaction latency.

Do not optimize during the first measurement pass. Preserve the baseline so subsequent changes can be evaluated against identical conditions.
