# Prismatic ReSTIR Phase 1 Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic offline experiment that validates wavelength-aware pupil reconnection, its change-of-variables Jacobian, its support failures, and its potential for path reuse before changing SpectraPBR's production renderer.

**Architecture:** Add a standalone C++17 research target under `research/prismatic/` with no CUDA, OptiX, OpenGL, or application dependencies. Phase 1 models a dispersive paraxial thin lens, maps a source film/pupil/wavelength sample to a target film/wavelength while preserving the first scene hit, verifies the mapping analytically and numerically, and emits repeatable CSV/JSON evidence. The production renderer remains unchanged until the experiment passes the decision gates in this document.

**Tech Stack:** C++17, standalone CMake 3.20+, CTest, standard library only, deterministic Hammersley sampling, double-precision experiment math.

**Spec:** `docs/superpowers/plans/2026-08-31-prismatic-restir-phase-1.md` — the Research Specification and Phase 1 Contract sections below are normative.

## Global Constraints

- Do not modify the production OptiX renderer, launch-parameter ABI, camera, materials, path guide, UI, or scene format in Phase 1.
- The research target must configure and run without CUDA, OptiX, OpenGL, GLFW, GLM, or network access.
- Use C++17 and CMake 3.20 or newer.
- Use millimetres for geometric distances, nanometres for public wavelength inputs, and micrometres only inside the Cauchy index calculation.
- Use `double` for all experiment geometry, PDFs, Jacobians, accumulated statistics, and reference calculations.
- Accept wavelengths only in the closed interval `[380, 780]` nm.
- Use one deterministic seed/sample sequence for every comparison. Do not use wall-clock seeding.
- Treat a failed shift as zero proposal support. Never clamp an invalid pupil back into the aperture.
- Keep a fresh canonical target sampler in every future estimator design; mapped samples alone cannot guarantee complete target support.
- Do not claim an unbiased ReSTIR estimator from Phase 1. Phase 1 validates a shift map and change of variables, not reservoir combination weights.
- Commit generated source and documentation, but do not commit build directories or generated experiment artifacts.

---

## Research Specification

### Research question

Can a reversible chromatic pupil-and-path reconnection turn one expensive dispersive path into a valid candidate over film position, aperture position, and wavelength, and does its support remain broad and numerically well-conditioned enough to justify integration with GRIS/ReSTIR?

The intended eventual integration domain is

\[
I_p =
\int_{A_p}
\int_{A_L}
\int_{\Lambda}
\int_{\mathcal P}
f_p(u,s,\lambda,\bar{x})
\,d\mu(\bar{x})\,d\lambda\,ds\,du,
\]

where:

- \(u\) is a two-dimensional film/subpixel coordinate;
- \(s\) is a two-dimensional pupil/aperture coordinate;
- \(\lambda\) is a hero wavelength;
- \(\bar{x}\) is the remaining light-transport path.

The eventual reservoir sample is therefore conceptually

\[
z=(u,s,\lambda,\bar{x}).
\]

Phase 1 isolates only the camera-domain mapping \((u,s,\lambda)\mapsto(u',s',\lambda')\) and preserves the first scene-plane hit. That is the smallest experiment capable of rejecting the central idea before expensive renderer work begins.

### Fit with the current SpectraPBR renderer

The current production camera is an RGB pinhole camera. [`CameraParams`](../../../src/core/shared_types.h) contains position, basis vectors, field of view, aspect ratio, and clip planes, while [`raygen.cu`](../../../optix_programs/raygen.cu) samples a subpixel film location but launches every primary ray from the same camera position. Neither pupil coordinates nor wavelength are part of the path state. The host and device camera/launch structures are deliberately mirrored and guarded by static assertions, so adding experimental fields there would create immediate ABI and pipeline risk.

The current path guide is a world-space directional vMF mixture. It can eventually become wavelength-conditioned, but it solves a different problem from camera-domain film/pupil reuse. Phase 1 therefore stays outside the production executable and does not depend on whether the megakernel or optional wavefront path is selected.

### Prior-art boundary

The implementation agent must preserve the following distinction in code comments and any results report. No individual row is claimed as novel.

| Existing work | Established component | Boundary relevant to this project |
|---|---|---|
| [Area ReSTIR](https://graphics.cs.utah.edu/research/projects/area-restir/) | Reservoir integration over 2D film and 2D aperture domains; lens-copy and primary-hit-reconnection shifts with Jacobians | Does not formulate joint wavelength-aware camera/path shifts |
| [Practical Aspects of Spectral Data in Digital Content Production](https://www.wetafx.co.nz/assets/Uploads/PDFs/sig-course-2022.pdf) | Experimental GRIS/ReSTIR reuse across receiver position and hero wavelength for spectral direct illumination | Uses a unit wavelength-change Jacobian in the described prototype and leaves full-path evaluation open |
| [Spectral Gradient Sampling for Path Tracing](https://diglib.eg.org/bitstreams/a2498da7-720d-412d-a0bf-20642bba35e0/download) | Wavelength-dependent shifts through dispersive refractive paths, with local Jacobians | Uses the shifts for gradient-domain reconstruction rather than joint lens-space reservoirs |
| [Sample Space Partitioning and Spatiotemporal Resampling for Specular Manifold Sampling](https://graphics.cs.utah.edu/research/projects/psms-restir/) | ReSTIR coupled to specular-manifold sampling for difficult caustic paths | Does not target a joint film/pupil/wavelength camera domain |
| [ToF ReSTIR](https://arxiv.org/abs/2605.11536) | ReSTIR reuse in an additional path-length dimension using a constraint-preserving reversible shift | Demonstrates the importance of preserving the new dimension's physical constraint, but not the chromatic lens constraint used here |

The potentially novel contribution is not the tuple `(film, pupil, wavelength)` by itself. A defensible contribution would require all of the following:

1. A useful reversible mapping that couples a wavelength-dependent camera lens to downstream dispersive transport.
2. A correct Jacobian and explicit support/invertibility treatment.
3. A GRIS/MIS proposal portfolio that remains valid when mappings fail.
4. Equal-time variance reduction in scenes where defocus and spectral dispersion interact.

This plan records a plausible unpublished formulation, not proof of novelty. A formal literature and patent review remains necessary before publication claims.

### Proposed shift portfolio after Phase 1

Do not force one mapping to handle every path family. The eventual estimator should combine distinct proposals:

1. **Canonical proposal:** trace a new target-domain sample; this guarantees target support.
2. **Area proposal:** change film and/or pupil coordinates while keeping wavelength fixed, using Area ReSTIR mappings.
3. **Spectral proposal:** change wavelength while holding the camera sample fixed, shifting downstream dispersive vertices.
4. **Composed chromatic proposal:** reconnect through the chromatic camera, then shift downstream wavelength-dependent path vertices.
5. **Manifold proposal:** solve camera-lens and scene specular constraints together for paths on which the composed mapping fails.

GRIS/MIS must combine these as separate proposal techniques. A failed mapping contributes no candidate; it must not silently fall back to an unaccounted mutation.

### Phase 1 paraxial camera model

Use a coordinate system with:

- the thin lens in the plane `z = 0`;
- the film plane at `z = -sensorDistanceMm`;
- the scene test plane at `z = targetDistanceMm > 0`;
- two-dimensional film coordinate \(u\) and pupil coordinate \(s\), both in millimetres.

For wavelength \(\lambda\), compute a refractive index with Cauchy's two-term equation:

\[
n(\lambda)=A+\frac{B}{\lambda_{\mu m}^{2}}.
\]

Compute thin-lens focal length with the zero-thickness lensmaker equation:

\[
\frac{1}{f(\lambda)}=
(n(\lambda)-1)
\left(\frac{1}{R_1}-\frac{1}{R_2}\right).
\]

Use the paraxial ray state independently on x and y. A film/pupil sample reaches the target plane at

\[
h(u,s,\lambda,z)
=
s+z\left(\frac{s-u}{d}-\frac{s}{f(\lambda)}\right)
=
B(z)u+A(\lambda,z)s,
\]

where

\[
B(z)=-\frac{z}{d},
\qquad
A(\lambda,z)=1+\frac{z}{d}-\frac{z}{f(\lambda)}.
\]

For a source sample \((u,s,\lambda)\), first calculate \(h\). Given target film coordinate \(u'\) and target wavelength \(\lambda'\), solve

\[
s'=
\frac{h-B(z)u'}{A(\lambda',z)}.
\]

Reject the mapping when:

- either wavelength lies outside `[380, 780]` nm;
- the source pupil lies outside the source aperture;
- `abs(A(sourceLambda, z)) < 1e-10`;
- `abs(A(targetLambda, z)) < 1e-10`;
- any result is non-finite;
- `length(targetPupil) > apertureRadiusMm`.

Because the mapping is an isotropic affine transformation in pupil coordinates,

\[
\frac{\partial s'}{\partial s}
=
\frac{A(\lambda,z)}{A(\lambda',z)}I_2,
\]

and its area Jacobian is

\[
J_{\mathrm{lens}}
=
\left|
\frac{A(\lambda,z)}{A(\lambda',z)}
\right|^2.
\]

The implementation must calculate this value directly and also verify it with central finite differences. Do not replace it with one merely because typical visible-wavelength differences can be small.

### Full-path extension after Phase 1

If the camera reconnection succeeds, a later phase may shift the downstream path from \(\lambda\) to \(\lambda'\):

- replay wavelength-independent diffuse and glossy events under their normal path mapping;
- reproduce projected generalized half vectors at dispersive refractions;
- reconnect after compatible non-specular vertices;
- reject total internal reflection, topology changes, occlusion, and zero-spectral-support transitions;
- retain the canonical proposal for target regions not covered by the shifted source.

For an explicitly ordered composition,

\[
J_T =
J_{\mathrm{film}}
J_{\mathrm{chromatic\ lens}}
J_{\mathrm{spectral\ path}}
J_{\mathrm{ordinary\ path}}.
\]

This product follows from the chain rule for the defined conditional composition. It must not be justified by claiming lens and wavelength variables are independent.

### Manifold extension after the composed mapping

The ambitious formulation treats lens surfaces and scene specular vertices as a single constraint:

\[
C(q;u,s,\lambda)=0,
\]

where \(q\) contains the chosen free coordinates of lens and scene-interface vertices. Continuation in \((u,s,\lambda)\) gives

\[
\frac{dq}{d\theta}
=
-\left(\frac{\partial C}{\partial q}\right)^{-1}
\frac{\partial C}{\partial\theta},
\qquad
\theta=(u,s,\lambda).
\]

This is not part of Phase 1. It becomes justified only when Phase 1 or the later composed-shift prototype shows that separable camera and scene mappings fail in important chromatic bokeh or caustic configurations.

### Intended production data, not Phase 1 data

Do not add these structures to the renderer yet. Preserve the following conceptual split for later design:

```text
CameraSample
    filmPosition[2]
    pupilPosition[2]
    heroWavelengthNm
    spectralPacketSeed
    sensorResponseWeight

PathState
    replaySeed or compact vertices
    scatteringTopology
    dispersiveEventMask
    sourceProposalDensities
    contribution

ShiftMetadata
    mappingFamily
    forwardValid
    reverseValid
    jacobian
    supportReason
    reservoirStatistics
```

Do not allocate a dense pixel × pupil × wavelength grid. Use one or a small number of continuous-domain reservoirs per pixel, with candidate mappings evaluated lazily. A later physical-lens implementation may add a lens-transfer cache keyed by film radius, pupil coordinate, and wavelength, returning an exit ray and derivatives.

The world-space path guide remains a separate estimator component. A later spectral guide may factor as

\[
q(\lambda,\omega\mid x)
=q_\lambda(\lambda\mid x)
q_\omega(\omega\mid x,\lambda),
\]

but pupil coordinates should not be inserted into the world-space incident-radiance guide.

### Why the first implementation is offline

An offline progressive experiment has useful properties beyond convenience:

- successive progressions estimate the same still image, so reuse is not complicated by motion, disocclusion, or stale visibility;
- large candidate neighborhoods and exact support/visibility reevaluation are affordable;
- wavelength bands and path-topology signatures can later be sorted into coherent wavefront queues;
- expensive lens or manifold solutions can be tested against several compatible target domains;
- a spectral result can remain sensor-independent until final integration into XYZ or a camera response.

This does not change Monte Carlo's asymptotic convergence by itself. Its potential advantage is lowering the time required to discover and reuse rare high-energy paths, especially chromatic bokeh and dispersive caustics. Phase 1 measures whether the camera portion of that reuse has enough valid support to justify a complete estimator.

---

## Phase 1 Contract

### Deliverable

A standalone executable and three CTest tests under `research/prismatic/` that:

- implement the dispersive paraxial lens transfer exactly as specified;
- reconnect source samples into target film/wavelength domains;
- classify support failures without clamping;
- validate the analytic Jacobian against finite differences;
- validate forward/reverse reciprocity;
- validate change of variables over a covered target aperture;
- run a fixed wavelength-pair sweep and emit machine-readable evidence.

### Default experiment configuration

Use these defaults exactly:

```text
Cauchy A                    1.500000
Cauchy B                    0.004000 um^2
Front radius R1            +50.0 mm
Back radius R2             -50.0 mm
Sensor distance d           52.0 mm
Aperture radius             12.5 mm
Target distance             2000.0 mm
Source wavelength           550.0 nm
Target wavelengths          420, 460, 500, 550, 600, 650, 700 nm
Source film coordinate      (0.0, 0.0) mm
Target film offsets         (0.0, 0.0), (0.25, 0.0), (0.0, 0.25) mm
Samples per sweep cell      262144
Finite-difference epsilon   1e-6 mm
Singularity epsilon         1e-10
```

### Required metrics

For every `(target wavelength, target film offset)` cell, write:

```text
source_wavelength_nm
target_wavelength_nm
target_film_x_mm
target_film_y_mm
sample_count
valid_count
support_rate
mean_hit_error_mm
max_hit_error_mm
mean_abs_reverse_pupil_error_mm
max_abs_reverse_pupil_error_mm
analytic_jacobian
finite_difference_jacobian
relative_jacobian_error
mean_abs_reciprocal_jacobian_error
log_jacobian_p50
log_jacobian_p95
log_jacobian_p99
```

The executable must also write a JSON summary containing configuration values, aggregate pass/fail gates, and the minimum/median/maximum support rate over all non-identity cells.

### Decision gates

Phase 1 passes only when all correctness gates hold:

- identity reconnection pupil error is at most `1e-10` mm;
- identity Jacobian differs from one by at most `1e-12`;
- valid mapped samples preserve the target-plane hit to at most `1e-9` mm maximum error;
- forward-then-reverse pupil error is at most `1e-8` mm maximum;
- analytic and finite-difference Jacobians have relative error at most `1e-7` away from singular configurations;
- forward and reverse Jacobian products differ from one by at most `1e-10`;
- the covered-domain change-of-variables area estimate has relative error at most `5e-3`;
- out-of-aperture and singular mappings are classified by distinct failure codes;
- all CTest tests pass on Linux or macOS without the production renderer's dependencies.

The research direction advances to renderer integration only if the correctness gates pass and the results show useful non-identity support. Do not encode an arbitrary minimum support percentage as a correctness test. Instead, record support by wavelength and film offset and make the go/no-go decision from those distributions. Low support is a valid negative experimental result.

### Explicit non-goals

Phase 1 does not implement:

- a reservoir or GRIS weight;
- spatial or temporal reuse;
- spectral material evaluation;
- downstream scene-path shifts;
- visibility reuse;
- a compound lens;
- Newton/manifold solving;
- an OptiX or CUDA kernel;
- UI controls;
- production image output.

---

## File Structure

Create the following isolated tree:

```text
research/prismatic/
├── CMakeLists.txt
├── README.md
├── include/prismatic/
│   ├── lens_model.h
│   ├── reconnection.h
│   └── sample_sequence.h
├── src/
│   ├── lens_model.cpp
│   ├── reconnection.cpp
│   ├── sample_sequence.cpp
│   └── phase1_experiment.cpp
└── tests/
    ├── test_support.h
    ├── lens_model_test.cpp
    ├── reconnection_test.cpp
    └── change_of_variables_test.cpp
```

Responsibilities:

- `lens_model.*`: units, wavelength validation, Cauchy IOR, focal length, paraxial coefficients, target-plane transfer.
- `reconnection.*`: support classification, target-pupil solve, analytic Jacobian, reverse validation.
- `sample_sequence.*`: deterministic Hammersley samples uniformly mapped to a disk.
- `phase1_experiment.cpp`: fixed sweep, aggregate metrics, CSV and JSON serialization, process exit status.
- `tests/*`: analytic correctness and regression gates, with no third-party test framework.
- `README.md`: exact build/run commands, model limits, output schema, and interpretation.

---

### Task 1: Standalone Research Target and Test Skeleton

**Files:**

- Create: `research/prismatic/CMakeLists.txt`
- Create: `research/prismatic/src/lens_model.cpp`
- Create: `research/prismatic/src/reconnection.cpp`
- Create: `research/prismatic/src/sample_sequence.cpp`
- Create: `research/prismatic/src/phase1_experiment.cpp`
- Create: `research/prismatic/tests/test_support.h`
- Create: `research/prismatic/tests/lens_model_test.cpp`
- Create: `research/prismatic/tests/reconnection_test.cpp`
- Create: `research/prismatic/tests/change_of_variables_test.cpp`

**Interfaces:**

- Consumes: no production SpectraPBR targets or libraries.
- Produces: executables `prismatic_phase1`, `prismatic_lens_model_test`, `prismatic_reconnection_test`, and `prismatic_change_of_variables_test`; CTest names matching the three test executable names.

- [ ] **Step 1: Write the standalone CMake target definitions**

Use this target structure:

```cmake
cmake_minimum_required(VERSION 3.20)
project(PrismaticReSTIRPhase1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(prismatic_phase1_lib
    src/lens_model.cpp
    src/reconnection.cpp
    src/sample_sequence.cpp
)
target_include_directories(prismatic_phase1_lib PUBLIC include)

if(MSVC)
    target_compile_options(prismatic_phase1_lib PRIVATE /W4 /permissive-)
else()
    target_compile_options(prismatic_phase1_lib PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(prismatic_phase1 src/phase1_experiment.cpp)
target_link_libraries(prismatic_phase1 PRIVATE prismatic_phase1_lib)

include(CTest)
foreach(test_name IN ITEMS lens_model reconnection change_of_variables)
    add_executable(prismatic_${test_name}_test tests/${test_name}_test.cpp)
    target_link_libraries(prismatic_${test_name}_test PRIVATE prismatic_phase1_lib)
add_test(NAME prismatic_${test_name}_test COMMAND prismatic_${test_name}_test)
endforeach()
```

- [ ] **Step 2: Add compileable source stubs**

Create empty `lens_model.cpp`, `reconnection.cpp`, and `sample_sequence.cpp` translation units containing only a namespace declaration:

```cpp
namespace prismatic {
}  // namespace prismatic
```

Create `phase1_experiment.cpp` with a temporary executable entry point:

```cpp
int main() {
    return 2;
}
```

These stubs exist only so the standalone target can configure before each interface is introduced test-first.

- [ ] **Step 3: Add a minimal assertion helper**

`test_support.h` must expose exact-value, near-value, relative-value, and boolean assertions that print the source line and return a nonzero process status on failure. Use an `int failures` counter passed by reference; do not abort on the first failure.

```cpp
inline void expectTrue(bool value, const char* expression, int line, int& failures);
inline void expectNear(double actual, double expected, double tolerance,
                       const char* expression, int line, int& failures);
inline void expectRelativeNear(double actual, double expected, double tolerance,
                               const char* expression, int line, int& failures);

#define EXPECT_TRUE(expr) \
    expectTrue((expr), #expr, __LINE__, failures)
#define EXPECT_NEAR(actual, expected, tolerance) \
    expectNear((actual), (expected), (tolerance), #actual, __LINE__, failures)
#define EXPECT_RELATIVE_NEAR(actual, expected, tolerance) \
    expectRelativeNear((actual), (expected), (tolerance), #actual, __LINE__, failures)
```

Relative comparison must divide by `max(abs(expected), 1e-300)` so a zero expected value does not divide by zero.

- [ ] **Step 4: Add intentionally failing test mains**

Each test source must initially include `test_support.h`, declare `int failures = 0`, contain `EXPECT_TRUE(false)`, and return `failures == 0 ? 0 : 1`.

- [ ] **Step 5: Configure, build, and prove the test skeleton is red**

Run:

```bash
cmake -S research/prismatic -B build/prismatic -DCMAKE_BUILD_TYPE=Release
cmake --build build/prismatic --parallel
ctest --test-dir build/prismatic --output-on-failure
```

Expected: configure and compilation succeed; all three tests fail because of the deliberate `EXPECT_TRUE(false)` assertions.

- [ ] **Step 6: Commit the red test skeleton**

```bash
git add research/prismatic/CMakeLists.txt research/prismatic/src research/prismatic/tests
git commit -m "test(research): scaffold prismatic phase one validation"
```

---

### Task 2: Dispersive Paraxial Lens Transfer

**Files:**

- Create: `research/prismatic/include/prismatic/lens_model.h`
- Modify: `research/prismatic/src/lens_model.cpp`
- Modify: `research/prismatic/tests/lens_model_test.cpp`

**Interfaces:**

- Consumes: standard `<cmath>`, `<cstdint>`, `<stdexcept>`, and `<string_view>` only.
- Produces: `prismatic::Vec2`, `prismatic::LensModel`, `prismatic::defaultLensModel()`, `refractiveIndex()`, `focalLengthMm()`, `paraxialA()`, `paraxialB()`, and `hitOnTargetPlane()`.

- [ ] **Step 1: Define the public lens interface**

Use these exact declarations:

```cpp
namespace prismatic {

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct LensModel {
    double cauchyA;
    double cauchyBUm2;
    double radius1Mm;
    double radius2Mm;
    double sensorDistanceMm;
    double apertureRadiusMm;
};

constexpr double kMinWavelengthNm = 380.0;
constexpr double kMaxWavelengthNm = 780.0;

LensModel defaultLensModel();
bool isFinite(Vec2 value);
double lengthSquared(Vec2 value);
bool isValidWavelength(double wavelengthNm);
double refractiveIndex(const LensModel& lens, double wavelengthNm);
double focalLengthMm(const LensModel& lens, double wavelengthNm);
double paraxialA(const LensModel& lens, double wavelengthNm,
                  double targetDistanceMm);
double paraxialB(const LensModel& lens, double targetDistanceMm);
Vec2 hitOnTargetPlane(const LensModel& lens, Vec2 filmMm, Vec2 pupilMm,
                      double wavelengthNm, double targetDistanceMm);

}  // namespace prismatic
```

Invalid wavelengths, non-positive target distance, zero sensor distance, equal/zero lens radii producing zero optical power, and non-finite inputs must throw `std::invalid_argument`. Use the exact defaults from the Phase 1 Contract.

- [ ] **Step 2: Replace the deliberate lens-test failure with specification tests**

Cover:

1. Default values match the contract exactly.
2. `refractiveIndex(defaultLens, 450) > refractiveIndex(defaultLens, 650)`.
3. `focalLengthMm(defaultLens, 450) < focalLengthMm(defaultLens, 650)` for the positive biconvex lens.
4. A zero film/pupil sample hits `(0,0)`.
5. The closed-form hit equals the direct expression `pupil + z * ((pupil-film)/d - pupil/f)` on both axes to `1e-12` mm.
6. Wavelengths `379.999` and `780.001` throw; `380` and `780` succeed.
7. Non-finite input and non-positive target distance throw.

- [ ] **Step 3: Run the lens test and confirm it fails to compile**

Run:

```bash
cmake --build build/prismatic --target prismatic_lens_model_test --parallel
```

Expected: compilation fails because `prismatic/lens_model.h` or its symbols do not yet exist.

- [ ] **Step 4: Implement the minimal lens transfer**

Implement the equations from the Research Specification without matrix or vector dependencies. Convert wavelength with:

```cpp
const double wavelengthUm = wavelengthNm * 1e-3;
```

Use:

```cpp
const double opticalPower = (n - 1.0) *
    (1.0 / lens.radius1Mm - 1.0 / lens.radius2Mm);
```

and reject `abs(opticalPower) < 1e-15`.

- [ ] **Step 5: Run the lens test**

Run:

```bash
cmake --build build/prismatic --target prismatic_lens_model_test --parallel
ctest --test-dir build/prismatic -R prismatic_lens_model_test --output-on-failure
```

Expected: `prismatic_lens_model_test` passes.

- [ ] **Step 6: Commit the lens model**

```bash
git add research/prismatic/include/prismatic/lens_model.h \
        research/prismatic/src/lens_model.cpp \
        research/prismatic/tests/lens_model_test.cpp
git commit -m "feat(research): add dispersive paraxial lens model"
```

---

### Task 3: Chromatic Pupil Reconnection and Jacobian

**Files:**

- Create: `research/prismatic/include/prismatic/reconnection.h`
- Modify: `research/prismatic/src/reconnection.cpp`
- Modify: `research/prismatic/tests/reconnection_test.cpp`

**Interfaces:**

- Consumes: `prismatic::LensModel`, `Vec2`, `hitOnTargetPlane()`, and `paraxialA()`.
- Produces: `ShiftFailure`, `ReconnectRequest`, `ReconnectResult`, `reconnectPupil()`, `reverseRequest()`, and `finiteDifferenceJacobian()`.

- [ ] **Step 1: Define the reconnection interface**

Use these exact declarations:

```cpp
namespace prismatic {

enum class ShiftFailure : std::uint8_t {
    None,
    InvalidInput,
    SingularSource,
    SingularTarget,
    NonFiniteResult,
    OutsideAperture,
};

struct ReconnectRequest {
    Vec2 sourceFilmMm;
    Vec2 sourcePupilMm;
    double sourceWavelengthNm;
    Vec2 targetFilmMm;
    double targetWavelengthNm;
    double targetDistanceMm;
};

struct ReconnectResult {
    bool valid = false;
    ShiftFailure failure = ShiftFailure::InvalidInput;
    Vec2 sourceHitMm{};
    Vec2 targetPupilMm{};
    double jacobian = 0.0;
};

ReconnectResult reconnectPupil(const LensModel& lens,
                               const ReconnectRequest& request,
                               double singularityEpsilon = 1e-10);
ReconnectRequest reverseRequest(const ReconnectRequest& forward,
                                const ReconnectResult& result);
double finiteDifferenceJacobian(const LensModel& lens,
                                const ReconnectRequest& request,
                                double epsilonMm = 1e-6,
                                double singularityEpsilon = 1e-10);

}  // namespace prismatic
```

`finiteDifferenceJacobian()` must central-difference each output pupil component with respect to each source pupil component and return the absolute 2×2 determinant. It must throw `std::invalid_argument` if the base or any perturbed mapping is invalid.

`reconnectPupil()` must translate validation exceptions from the lens helpers into `ShiftFailure::InvalidInput`; invalid requests must not throw through this interface. `reverseRequest()` must return the exact field swap below and must reject an invalid forward result:

```cpp
ReconnectRequest reverse;
reverse.sourceFilmMm = forward.targetFilmMm;
reverse.sourcePupilMm = result.targetPupilMm;
reverse.sourceWavelengthNm = forward.targetWavelengthNm;
reverse.targetFilmMm = forward.sourceFilmMm;
reverse.targetWavelengthNm = forward.sourceWavelengthNm;
reverse.targetDistanceMm = forward.targetDistanceMm;
```

- [ ] **Step 2: Replace the deliberate reconnection-test failure with behavior tests**

Add tests for:

1. Identity film/wavelength mapping returns the original pupil within `1e-10` mm and Jacobian one within `1e-12`.
2. A wavelength and film shift preserves `sourceHitMm` within `1e-9` mm.
3. The analytic and finite-difference Jacobians agree within relative error `1e-7`.
4. A valid forward map followed by `reverseRequest()` recovers the original pupil within `1e-8` mm.
5. Forward Jacobian × reverse Jacobian equals one within `1e-10`.
6. A target film offset large enough to move the solved pupil outside the aperture returns `OutsideAperture` and `jacobian == 0`.
7. `targetDistanceMm = 1.0 / (1.0 / focalLengthMm(lens, targetLambda) - 1.0 / lens.sensorDistanceMm)` makes the target coefficient singular and returns `SingularTarget`.
8. The same construction at the source wavelength returns `SingularSource`.
9. NaN inputs and a source pupil outside the aperture return `InvalidInput` rather than escaping as NaN output.

- [ ] **Step 3: Run the reconnection test and confirm it fails to compile**

Run:

```bash
cmake --build build/prismatic --target prismatic_reconnection_test --parallel
```

Expected: compilation fails because `prismatic/reconnection.h` or its symbols do not yet exist.

- [ ] **Step 4: Implement the support checks, solve, and analytic Jacobian**

The implementation order must be:

1. Validate the request and lens inputs.
2. Compute the source hit.
3. Compute source and target `A` coefficients.
4. Reject source and target singularities with their distinct failure values.
5. Solve target pupil independently on x and y.
6. Reject non-finite results.
7. Reject target pupils outside the circular aperture.
8. Compute `jacobian = abs((sourceA / targetA) * (sourceA / targetA))`.
9. Return `valid = true` and `ShiftFailure::None`.

Do not use the aperture radius in the Jacobian; the aperture is a support test, not part of the local affine derivative.

- [ ] **Step 5: Run the reconnection test**

Run:

```bash
cmake --build build/prismatic --target prismatic_reconnection_test --parallel
ctest --test-dir build/prismatic -R prismatic_reconnection_test --output-on-failure
```

Expected: `prismatic_reconnection_test` passes.

- [ ] **Step 6: Commit the mapping**

```bash
git add research/prismatic/include/prismatic/reconnection.h \
        research/prismatic/src/reconnection.cpp \
        research/prismatic/tests/reconnection_test.cpp
git commit -m "feat(research): add chromatic pupil reconnection"
```

---

### Task 4: Deterministic Sampling and Change-of-Variables Validation

**Files:**

- Create: `research/prismatic/include/prismatic/sample_sequence.h`
- Modify: `research/prismatic/src/sample_sequence.cpp`
- Modify: `research/prismatic/tests/change_of_variables_test.cpp`

**Interfaces:**

- Consumes: `Vec2`, `LensModel`, `ReconnectRequest`, and `reconnectPupil()`.
- Produces: `radicalInverseBase2()`, `hammersleyUnitSquare()`, and `concentricDiskSample()`.

- [ ] **Step 1: Define the deterministic sample interface**

Use these declarations:

```cpp
namespace prismatic {

double radicalInverseBase2(std::uint32_t bits);
Vec2 hammersleyUnitSquare(std::uint32_t index, std::uint32_t count);
Vec2 concentricDiskSample(Vec2 uniformSample);

}  // namespace prismatic
```

`hammersleyUnitSquare(i,n)` must return `((i + 0.5) / n, radicalInverseBase2(i))`. `concentricDiskSample()` must use the Shirley–Chiu concentric square-to-disk map and return `(0,0)` for the exact center.

- [ ] **Step 2: Replace the deliberate change-of-variables failure with deterministic tests**

Test sequence range, repeatability, and disk containment. Then validate change of variables using a centered film sample and a wavelength direction whose affine source-to-target pupil scale has magnitude greater than one, so the mapped source disk covers the target aperture.

Choose the direction at runtime:

```cpp
const double a450 = std::abs(paraxialA(lens, 450.0, targetDistanceMm));
const double a650 = std::abs(paraxialA(lens, 650.0, targetDistanceMm));
const double sourceLambda = a450 >= a650 ? 450.0 : 650.0;
const double targetLambda = a450 >= a650 ? 650.0 : 450.0;
```

For `262144` source samples uniformly distributed on the source aperture disk:

1. Reconnect each source pupil to the target wavelength.
2. For valid mappings, add the analytic Jacobian to `weightedSum`.
3. Estimate target-aperture area as `pi * r * r * weightedSum / sampleCount`.
4. Compare with `pi * r * r` at relative error at most `5e-3`.
5. Separately verify the mean analytic/finite-difference relative Jacobian error over the first 1024 non-boundary samples is at most `1e-7`.

This test verifies the local measure conversion only in a configuration where mapped support covers the target disk. It does not establish a complete ReSTIR estimator.

- [ ] **Step 3: Run the test and prove it is red**

Run:

```bash
cmake --build build/prismatic --target prismatic_change_of_variables_test --parallel
```

Expected: compilation fails because the sample-sequence symbols do not yet exist.

- [ ] **Step 4: Implement the deterministic sample sequence**

Use bit reversal for the base-two radical inverse and the standard branch form of the Shirley–Chiu mapping. Clamp neither unit-square inputs nor output disk coordinates; tests must expose invalid generation.

- [ ] **Step 5: Run all three tests**

Run:

```bash
cmake --build build/prismatic --parallel
ctest --test-dir build/prismatic --output-on-failure
```

Expected: all three tests pass.

- [ ] **Step 6: Commit deterministic sampling and measure validation**

```bash
git add research/prismatic/include/prismatic/sample_sequence.h \
        research/prismatic/src/sample_sequence.cpp \
        research/prismatic/tests/change_of_variables_test.cpp
git commit -m "test(research): validate chromatic shift measure"
```

---

### Task 5: Wavelength/Film Sweep and Machine-Readable Evidence

**Files:**

- Modify: `research/prismatic/src/phase1_experiment.cpp`

**Interfaces:**

- Consumes: every Phase 1 library interface and the fixed configuration in this document.
- Produces: CLI `prismatic_phase1 --output-dir <directory>`, `<directory>/sweep.csv`, and `<directory>/summary.json`; returns `0` only when all correctness gates pass.

- [ ] **Step 1: Add a failing CLI smoke check to CTest**

Add this to `research/prismatic/CMakeLists.txt` after defining `prismatic_phase1`:

```cmake
add_test(
    NAME prismatic_phase1_smoke
    COMMAND prismatic_phase1
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/phase1-smoke
)
```

Run:

```bash
cmake -S research/prismatic -B build/prismatic -DCMAKE_BUILD_TYPE=Release
cmake --build build/prismatic --target prismatic_phase1 --parallel
ctest --test-dir build/prismatic -R prismatic_phase1_smoke --output-on-failure
```

Expected: build or test fails because `phase1_experiment.cpp` is not implemented.

- [ ] **Step 2: Implement strict CLI parsing and output setup**

Accept exactly `--output-dir <directory>` and `--help`. Unknown arguments, missing values, or output-directory creation failures must print a diagnostic to stderr and return `2`. Use `std::filesystem::create_directories`.

- [ ] **Step 3: Implement the fixed sweep**

For every target wavelength and film offset in the Phase 1 Contract:

1. Generate `262144` Hammersley disk samples and scale by the aperture radius.
2. Reconnect from the fixed source domain.
3. Count support failures by `ShiftFailure` value.
4. For valid mappings, recompute the target hit and accumulate hit error.
5. Construct and evaluate the reverse request.
6. Accumulate reciprocal Jacobian error.
7. Evaluate finite-difference Jacobians for the first 1024 valid samples whose source and target pupils are at least `1e-3` mm inside their aperture boundaries.
8. Store `log(max(jacobian, 1e-300))` for valid samples, sort once, and select nearest-rank p50/p95/p99.

Use a compensated sum, such as Neumaier accumulation, for all large floating-point reductions.

- [ ] **Step 4: Serialize CSV and JSON without external dependencies**

Write the Required Metrics as the first CSV row in the listed order. Use `std::setprecision(17)` for numeric output.

The JSON root must contain:

```json
{
  "schema_version": 1,
  "experiment": "prismatic-restir-phase1",
  "configuration": {},
  "aggregate": {
    "correctness_passed": true,
    "cell_count": 21,
    "minimum_support_rate": 0.0,
    "median_support_rate": 0.0,
    "maximum_support_rate": 0.0
  },
  "failure_counts": {
    "invalid_input": 0,
    "singular_source": 0,
    "singular_target": 0,
    "non_finite_result": 0,
    "outside_aperture": 0
  },
  "gates": []
}
```

Populate every numeric field from the run. Each `gates` entry must contain `name`, `threshold`, `observed`, and `passed`. Do not emit the zero example values above unless the measured value is actually zero.

- [ ] **Step 5: Make gate failures fail the process**

Print a compact table to stdout and return:

- `0` when every correctness gate passes;
- `1` when the experiment runs but any correctness gate fails;
- `2` for invocation or I/O errors.

Low support alone must not return `1`; support is the experimental outcome being measured.

- [ ] **Step 6: Run the complete experiment and tests**

Run:

```bash
cmake -S research/prismatic -B build/prismatic -DCMAKE_BUILD_TYPE=Release
cmake --build build/prismatic --parallel
ctest --test-dir build/prismatic --output-on-failure
rm -rf build/prismatic-results
build/prismatic/prismatic_phase1 --output-dir build/prismatic-results
```

Expected:

- four CTest tests pass;
- the CLI returns zero;
- `build/prismatic-results/sweep.csv` contains 21 data rows plus its header;
- `build/prismatic-results/summary.json` parses as JSON;
- `correctness_passed` is true.

- [ ] **Step 7: Commit the experiment executable**

```bash
git add research/prismatic/CMakeLists.txt \
        research/prismatic/src/phase1_experiment.cpp
git commit -m "feat(research): add prismatic phase one sweep"
```

---

### Task 6: Reproducibility Guide and Final Phase 1 Review

**Files:**

- Create: `research/prismatic/README.md`
- Modify: `.gitignore` only if `build/` is not already ignored.

**Interfaces:**

- Consumes: the built targets and output schemas from Tasks 1–5.
- Produces: a human and agent-readable execution guide and an explicit recommendation based on recorded support distributions.

- [ ] **Step 1: Document the exact reproduction commands**

The README must include:

```bash
cmake -S research/prismatic -B build/prismatic -DCMAKE_BUILD_TYPE=Release
cmake --build build/prismatic --parallel
ctest --test-dir build/prismatic --output-on-failure
build/prismatic/prismatic_phase1 --output-dir build/prismatic-results
```

It must explain units, lens defaults, output files, correctness gates, support-rate interpretation, and the limitations of a paraxial thin lens.

- [ ] **Step 2: Record the three possible Phase 1 conclusions**

Use these exact decision categories:

- **Advance:** correctness passes and support remains useful across non-identity wavelength/film cells; proceed to an offline camera-domain GRIS prototype.
- **Narrow:** correctness passes but support collapses in large wavelength or film shifts; restrict proposals adaptively and test smaller neighborhoods before renderer integration.
- **Stop or redesign:** correctness fails after numerical issues are ruled out, or support is negligible even for small shifts; investigate a different invariant or a coupled manifold parameterization.

The README must state that Phase 1 does not prove unbiased reservoir reuse and that canonical target samples remain mandatory.

- [ ] **Step 3: Run the plan's placeholder and hygiene scans**

Run:

```bash
rg -n "T[B]D|T[O]DO|implement l[a]ter|fill i[n]|appropriate error handlin[g]|similar to Tas[k]" \
    research/prismatic docs/superpowers/plans/2026-08-31-prismatic-restir-phase-1.md
git status --short
git diff --check
```

Expected: the placeholder scan has no matches in the implemented research tree or this plan; `git diff --check` reports no whitespace errors. Pre-existing unrelated working-tree files may appear in status and must not be added.

- [ ] **Step 4: Run final verification from a clean research build directory**

Remove only the experiment's dedicated build directory and rebuild:

```bash
rm -rf build/prismatic
cmake -S research/prismatic -B build/prismatic -DCMAKE_BUILD_TYPE=Release
cmake --build build/prismatic --parallel
ctest --test-dir build/prismatic --output-on-failure
build/prismatic/prismatic_phase1 --output-dir build/prismatic-results-final
```

Expected: configure, build, all four tests, and the final experiment pass.

- [ ] **Step 5: Review the measured support distribution**

Read `build/prismatic-results-final/summary.json` and `sweep.csv`. Select exactly one of **Advance**, **Narrow**, or **Stop or redesign**, cite the minimum/median/maximum non-identity support rates and the worst Jacobian/reversibility errors, and add that evidence to the implementation task's final report. Do not commit generated results unless the user explicitly requests a recorded benchmark artifact.

- [ ] **Step 6: Commit the reproducibility guide**

```bash
git add research/prismatic/README.md .gitignore
git commit -m "docs(research): document prismatic phase one experiment"
```

Only include `.gitignore` in the commit if it actually required a change.

---

## Phase 2 Decision, Not Authorization

Passing Phase 1 authorizes design work for an offline camera-domain GRIS prototype; it does not by itself authorize production integration. The next plan should add:

1. A canonical target proposal and one mapped proposal.
2. Generalized source/target proposal densities with explicit support.
3. Reservoir combination weights verified against an independent direct sampler.
4. An offline progressive mode with no temporal motion or disocclusion.
5. Equal-sample and equal-time comparisons on defocused chromatic scenes.

Only after that estimator is validated should SpectraPBR's production camera state grow film/pupil/wavelength fields or its OptiX path state carry spectral replay metadata.

The most informative later scenes are:

- a defocused prism viewed through a large aperture;
- a dispersive glass caustic outside the focal plane;
- chromatic bokeh from a bright compact emitter;
- an achromatic in-focus control scene;
- a scene with lens-only chromatic aberration and no dispersive scene material.

Measure spectral RMSE before sensor integration, XYZ/linear-RGB error after integration, equal-time error, mapping success by proposal family, visibility/topology failure rates, Jacobian tails, and effective reservoir sample size.

## Publication Hypothesis

If the complete approach works, the strongest research claim is:

> A reversible chromatic pupil-and-path reconnection can reuse difficult dispersive paths across film, aperture, and wavelength domains without bias, providing disproportionate variance reduction for chromatic defocus and caustic transport.

The claim is only supported if the final system includes rigorous GRIS weights, a clear invertibility/support analysis, canonical coverage, and equal-time evidence against independent spectral path tracing and the relevant one-dimension-at-a-time ablations.
