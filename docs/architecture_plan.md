# HikariEngine — Architecture Report & Target Structure

**Date:** 14/08/2026

**Scope:** Full read of `src/` (70 C++ files), `src/shaders/` (7 files), `CMakeLists.txt`,
`CMakePresets.json`, `.github/workflows/ci.yml`, `vcpkg.json`, build scripts, `scenes/`.

**Goal:** Describe the structure as it exists today, then define a target structure that
supports (a) headless + automated runtime testing, (b) data-oriented performance,
(c) scalability as features are added.

**Status:** Stage 6 and the cleanup work between the stages are complete. Next: Stage 7.

> Companion document: `suggested_work.md` covers *correctness bugs* and
> localised fixes. This document deliberately does **not** repeat them. Where a bug is
> load-bearing for the architecture (e.g. per-upload `queue.waitIdle()`), it is
> referenced, not re-explained.

---

## Table of Contents

**Part I — Starting state**
1. [Inventory](#1-inventory)
2. [Module map as it exists today](#2-module-map-as-it-exists-today)
3. [What `main.cpp` owns](#3-what-maincpp-owns)
4. [Current frame data flow](#4-current-frame-data-flow)
5. [Structural findings](#5-structural-findings)
6. [Testability blockers](#6-testability-blockers-ranked)

**Part II — Target architecture**
7. [Design principles](#7-design-principles)
8. [Target module & target graph](#8-target-module--target-graph)
9. [Target directory layout](#9-target-directory-layout)
10. [Headless architecture](#10-headless-architecture)
11. [Data-oriented core](#11-data-oriented-core)
12. [Render graph and passes](#12-render-graph-and-passes)
13. [Asset system](#13-asset-system)
14. [Configuration, CLI and determinism](#14-configuration-cli-and-determinism)

**Part III — Testing & delivery**
15. [Test strategy](#15-test-strategy)
16. [Test harness components](#16-test-harness-components)
17. [CI plan](#17-ci-plan)
18. [Migration roadmap](#18-migration-roadmap)
19. [Conventions & tooling](#19-conventions--tooling)
20. [Risks and open decisions](#20-risks-and-open-decisions)

**Part IV — Incremental work order**
23. [How to use Part IV](#23-how-to-use-part-iv)
24. [Stage 0 — Verification harness](#stage-0--verification-harness-steps-16)
25. [Stage 1 — Build hygiene](#stage-1--build-hygiene-steps-711)
26. [Stage 2 — Header self-containment](#stage-2--header-self-containment-steps-1214)
27. [Stage 3 — Core library](#stage-3--core-library-steps-1519)
28. [Stage 4 — Platform library](#stage-4--platform-library-steps-2023)
29. [Stage 5 — RHI extraction](#stage-5--rhi-extraction-steps-2434)
30. [Stage 6 — Headless capability](#stage-6--headless-capability-steps-3540a)
31. [Stage 7 — Engine shell & dependency injection](#stage-7--engine-shell--dependency-injection-steps-40b47)
32. [Stage 8 — Passes & frame graph](#stage-8--passes--frame-graph-steps-4856)
33. [Stage 9 — Data-oriented rewrite](#stage-9--data-oriented-rewrite-steps-5768)
34. [Stage 10 — Scalability features](#stage-10--scalability-features-steps-6976)
35. [Dependency summary](#dependency-summary)

**Appendices**
21. [Appendix A — File relocation table](#appendix-a--file-relocation-table)
22. [Appendix B — Frame budget model](#appendix-b--frame-budget-model)

---

# Part I — Starting state

## 1. Inventory

| Metric | Value |
|---|---|
| C++ files | 70 (43 `.h`, 27 `.cpp`) |
| C++ lines | ≈ 7,100 |
| `src/main.cpp` | 2,453 lines (**≈ 35% of all C++**) |
| Slang shader files | 7 (5 `.slang`, 2 `.slangh`), ≈ 860 lines |
| Build targets | 1 executable (`HikariEngine`) + 1 shader custom target + 1 fetched dep |
| Test targets | **0** |
| CI jobs | 7 (configure + build only, 3 OS × Debug/Release + Linux ASan) |
| Directory nesting under `src/` | 1 level (`src/shaders/`); everything else is flat |

Largest translation units:

| File | Lines |
|---|---|
| `main.cpp` | 2,453 |
| `CloudSystem.cpp` | 400 |
| `XmlParser.cpp` | 338 |
| `Utility.h` | 339 |
| `PipelineBuilder.cpp` | 193 |
| `CubemapLoader.cpp` | 160 |
| `Barrier.h` | 156 |
| `PBRMaterial.cpp` | 143 |

The distribution is the headline problem: one file is larger than the next seven combined,
and it is the file that owns every cross-cutting decision.

## 2. Module map as it exists today

There are no directories, but there *are* de facto modules. Grouping by responsibility:

| De facto module | Files | State |
|---|---|---|
| **Core / utility** | `Log.h`, `Timer.h`, `ThreadPool.*`, `SwapbackArray.h`, `MyMacros.h`, `Common.h`, `pch.h` | Reasonable. `Log.h` is good. `ThreadPool` is a singleton. |
| **RHI (Vulkan wrapping)** | `Utility.h`, `AllocatedBuffer.*`, `AllocatedImage.*`, `VulkanAllocator.h`, `Barrier.h`, `PipelineBuilder.*`, `ComputePipelineBuilder.*`, `Texture.*`, `Cubemap.*` | Partially extracted. `Barrier.h` + `PipelineBuilder` are recent and good. Device/instance/swapchain are **not** extracted — they live in `main.cpp`. |
| **Renderer** | `FrameData.h`, `InstanceData.h` | Essentially non-existent as a module. All pass logic is in `main.cpp`. |
| **Passes** | `CloudSystem.*` | The *only* pass that is a class. Proves the pattern works; nothing else follows it. |
| **Scene** | `Entity.*`, `Component.h`, `SceneComponent.*`, `LogicComponent.h`, `SceneGraph.h`, `Transform.*`, `Camera.*`, `Lights.h`, `Node.*` | OOP inheritance + `dynamic_cast` lookup. `SceneGraph` is a bare struct of 3 vectors. |
| **Assets** | `ResourceManager.*`, `ResourceCache.h`, `TextureLoader.*`, `CubemapLoader.*`, `ModelLoader.*`, `Model.*`, `ModelData.*`, `Mesh.*`, `Material.*`, `PBRMaterial.*`, `MaterialFactory.*` | Recently improved: `ResourceCache<T>` with `shared_ptr`/`weak_ptr` is the right shape. Loaders are singletons holding `vk::raii::X&` references. |
| **Render-scene bridge** | `ModelManager.*`, `Drawable.h` | This is the DOD hot path. Currently full rebuild per frame. |
| **Serialization** | `XmlParser.*` | Static-function class, `SceneGraph` in/out. Testable today with modest work. |
| **Editor / UI** | *(inside `main.cpp`)* | ~120 lines of ImGui in `App::DrawImGuiFrame`. |
| **Entry point** | `main.cpp` (bottom 50 lines) | Fine. |

### Actual dependency edges (problematic ones)

```
main.cpp ───────────────► everything (23 direct engine includes)
Model.cpp ──────────────► ModelManager::Get()      (asset → renderer, via singleton)
Model.cpp ──────────────► ResourceManager::Get()
XmlParser.cpp ──────────► Model, Lights, SceneGraph  (serialization → scene → assets)
ModelManager.cpp ───────► ModelData, Mesh, Material  (renderer → assets)
CloudSystem.h ──────────► vk::raii::Device&, DescriptorSetLayout&, CommandPool&, Queue&
                          (pass holds references into App's members)
*.h ────────────────────► pch.h (implicitly, via /FI force-include)
```

Three edges are worth calling out because they will fight every refactor:

1. **`Model`'s constructor registers itself with `ModelManager` and its destructor
   unregisters.** An asset-side type reaches into a renderer singleton. Creating a `Model`
   in a unit test requires a live `ModelManager` *and* a live `ResourceManager` *and*
   therefore a `VkDevice`.
2. **`CloudSystem` stores `vk::raii::Device&` and three more references** to `App`
   members. It cannot outlive or be constructed without an `App`.
3. **Every header depends on the PCH force-include.** `target_precompile_headers(HikariEngine
   PRIVATE src/pch.h)` means headers compile only inside this one target. A second target
   (test, tool, headless app) will not compile them without duplicating the PCH.

## 3. What `main.cpp` owns

`class App` is 2,200 lines and holds 45 members. Its responsibilities:

| Responsibility | Functions / lines |
|---|---|
| SDL init, window creation, shutdown | `InitSDL`, `CreateSDLWindow`, `ShutdownSDL` |
| Vulkan instance, layers, extensions, portability | `CreateInstance` (~120 lines) |
| Debug messenger + validation callback | `DebugCallback`, `SetupDebugMessenger` |
| Surface creation | `CreateSurface` |
| Physical device selection + feature gating | `IsPhysicalDeviceSuitable`, `PickPhysicalDevice` |
| Logical device + queue selection | `CreateLogicalDevice` |
| Swapchain + image views + recreate | `CreateSwapchain`, `CreateSwapchainImageViews`, `RecreateSwapchainAndRenderImages` |
| Format selection | `FindSupportedFormat`, `FindDepthFormat`, `HasStencilComponent` |
| 3 graphics pipelines | `CreateOpaquePipeline`, `CreateTransparentPipeline`, `CreateCompositePipeline` |
| Descriptor layouts / pools / sets (3 of each) | `CreateDescriptorSetLayouts`, `CreateDescriptorPool`, `CreateDescriptorSets`, `UpdateCompositeDescriptorSet`, `UpdateDepthDescriptorSets` |
| 7 command pools × 2 frames, 7 command buffers × 2 frames | `CreateCommandPools`, `CreateCommandBuffers` (~200 lines) |
| 5 command-buffer recorders + 2 layout-transition recorders | `RecordOpaque/Transparent/Clouds/Composite/ImGui/SwapImageTo{Draw,Present}Layout` |
| Render targets, depth, quad buffers, instance buffers, global UBO | `CreateRenderTargets`, `CreateDepthResources`, `CreateQuadBuffers`, `CreateInstanceBuffers`, `CreateGlobalBuffers` |
| GPU struct layouts (`GlobalBuffer`, `CameraData`, `LightData`) | lines 73–102 |
| Sync objects | `CreateSyncObjects` |
| Frame loop, timing, FPS smoothing | `Run` |
| Input: mouse, keys, movement, cursor capture | `HandleMouse`, `HandleKey`, `HandleMovement`, `Show/HideCursor` |
| Editor UI (lights, load/save scene, quit, stats) | `DrawImGuiFrame` |
| ImGui backend init/shutdown/pipeline recreate | `InitImGui`, `ShutdownImGui` |
| Submit + present + frame index advance | `DrawFrame` |
| Scene ownership and scene switching | `m_SceneGraph`, inline in `DrawImGuiFrame` |
| `main()` + exception handling | lines 2407–2453 |

That is roughly **fourteen** distinct subsystems in one class with one lifetime. Every one
of them is unreachable from a test.

## 4. Current frame data flow

```
Run()
 ├─ compute dt / runtime / smoothed FPS
 ├─ SDL_PollEvent loop ──► ImGui, camera rotate, resize, focus, keys
 ├─ m_Camera->Tick()            (recompute view matrix)
 ├─ HandleMovement()            (poll keyboard state, integrate camera pos)
 ├─ if (cursor visible) DrawImGuiFrame()
 ├─ ModelManager::GenerateBatches()      ◄── FULL REBUILD, EVERY FRAME
 │    ├─ clear 4 vectors
 │    ├─ for each Model: GetDrawables() → heap-allocates vector<Drawable>, copies out
 │    ├─ std::sort(m_Drawables)          (80-byte elements, 3-branch comparator)
 │    └─ linear scan → MeshBatch[] + InstanceData[]
 │         └─ per instance: glm::transpose(glm::inverse(mat4))
 └─ DrawFrame()
      ├─ waitForFences(frame.DrawFence)
      ├─ acquireNextImage
      ├─ UpdateGlobalBuffer(frameIndex)   (memcpy into persistently mapped UBO)
      ├─ UpdateInstanceBuffer(frameIndex) (memcpy; THROWS if > 1024 instances)
      ├─ record 7 command buffers (2 on thread pool, 5 on main thread)
      ├─ submit all 7 in one vkQueueSubmit
      └─ presentKHR
```

Observations that drive Part II:

- **Nothing is cached between frames.** The scene is static in practice; the CPU rebuilds
  the entire draw list, re-sorts it, and recomputes every normal matrix, every frame.
- **`Drawable` is 96 bytes** (`Mesh*` 8, `Material*` 8, `BlendMode` 1 + 7 pad, `mat4` 64,
  plus alignment). Sorting moves 96-byte objects. The sort key is 17 bytes of it.
- **Instance data is 128 bytes** (two `mat4`) and consumes **8 of the 16** guaranteed vertex
  input attribute slots.
- **There is no culling of any kind.**
- **7 command buffers, 14 command pools, 1 submit.** The pass set is hardcoded in three
  places (`FrameData`, `CreateCommandPools`/`CreateCommandBuffers`, and the
  `std::array<vk::CommandBuffer, 7>` in `DrawFrame`). Adding a shadow pass is a 5-file,
  ~10-site change.
- **Barriers are correct but hand-placed.** `Barrier.h`'s named factories are a good
  intermediate step; the placement logic is still spread across 7 recorders.
- **Resize fans out manually**: `RecreateSwapchainAndRenderImages` must remember to
  recreate depth, 3 render targets, cloud outputs, 2 descriptor set groups, the camera
  projection and the ImGui pipeline. Adding a render target means editing this function
  and hoping.

## 5. Structural findings

Numbered for reference. Severity: **S1** blocks testing/scaling now, **S2** will block soon,
**S3** friction.

### S1-1 — There is no library. There is only an executable.
Nothing in the codebase can be linked into a second binary. This is the root cause of
"cannot test": a test binary has nothing to link against. Everything else in this document
is downstream of fixing this.

### S1-2 — Windowing and presentation are hard requirements, not options.
`App` takes an `SDL_Window*` in its constructor. `CreateInstance` unconditionally requires
`SDL_Vulkan_GetInstanceExtensions`. `CreateLogicalDevice` requires a queue family with
`getSurfaceSupportKHR`. `IsPhysicalDeviceSuitable` requires `VK_KHR_swapchain`.
`DrawFrame` acquires and presents a swapchain image and returns early if it can't.
`Run` calls `SDL_ShowWindow` and `SDL_PollEvent`.

There is no seam at which a headless run could be inserted. Adding one is not a matter of
an `if (headless)` — it needs the presentation path behind an interface (§10).

### S1-3 — Global singletons with implicit initialisation order.
`ResourceManager`, `MaterialFactory`, `ModelManager`, `ThreadPool`, `TextureLoader`,
`CubemapLoader`, `ModelLoader` are all `static Get()` singletons initialised in a specific
order inside `InitVulkan`/`Init`. Consequences:
- Tests cannot instantiate one subsystem in isolation.
- Tests cannot run in parallel (shared mutable global state).
- Two scenes / two worlds / an editor preview viewport are impossible.
- `Model`'s ctor/dtor talking to `ModelManager::Get()` means asset lifetime is coupled to
  renderer global state.

### S1-4 — `Entity`/`Component` uses `dynamic_cast` polymorphism with pointer-chasing.
`Entity` owns `vector<unique_ptr<SceneComponent>>` + `vector<unique_ptr<LogicComponent>>`.
Component lookup is a linear scan with `dynamic_cast` per element. `SceneComponent` walks a
parent pointer chain in `GetAccumulatedTransform()`. Each component is an individual heap
allocation.

This is the opposite of data-oriented: for N entities you get N+ allocations, random access
patterns, virtual dispatch and RTTI in the query path. It works at 1 Sponza; it will not
work at 10,000 entities, and it cannot be vectorised.

### S1-5 — No separation between simulation state and render state.
`ModelManager` reads `Model`s (which are `SceneComponent`s in the scene graph) directly at
record time. There is no immutable per-frame snapshot. Therefore:
- Simulation and rendering cannot overlap (no render thread).
- Rendering cannot be tested without a live scene graph.
- Frame N-1's data is not available for temporal effects (TAA, motion vectors, temporal
  cloud reprojection — the last of which §3.7 of the companion doc wants).

### S2-1 — GPU struct layouts are declared twice, by hand.
`main.cpp:73-102` and `src/shaders/common.slangh:3-45` declare the same five structs. The
only guard is `sizeof(GlobalBuffer) % 16 == 0`. A field-order divergence passes that check.

### S2-2 — Hard limits that fail loudly.
`MAX_INSTANCE_COUNT = 1024` (throws), `s_MAX_MATERIAL_SET_COUNT = 100`,
`s_MAX_TEXTURE_COUNT_PER_MAT = 3`, `MAX_POINT_LIGHTS = 4` (silently clamped),
`MAX_DIR_LIGHTS = 1`. Each is a wall a growing scene hits. Each is also a *test* that will
start failing when test scenes grow.

### S2-3 — Descriptor model does not scale.
One descriptor set per material, one fixed-size pool, three texture slots, per-batch
`bindDescriptorSets` + `pushConstants`. `descriptorBindingPartiallyBound` is *already
enabled* on the device (`main.cpp:1019`) but unused. Bindless is available and would remove
S2-2's material ceiling, the per-batch descriptor bind, and the 3-texture cap in one change.

> **Correction, September 2026.** Two halves of that have not aged well. The flag is no longer
> unused — it moved to `VulkanDevice.cpp:980` and now backs the partially-bound material set
> that lets an untextured material render (`MaterialFactory.cpp:76`). And "bindless is
> available" overstated it: partial binding is a different feature, and bindless additionally
> needs `runtimeDescriptorArray` and `shaderSampledImageArrayNonUniformIndexing` at minimum,
> neither of which is enabled. The material ceiling was lifted separately by a growable pool
> (R13); the 3-texture cap remains. See D14 in `backend_readiness_plan.md`.

### S2-4 — Assets load synchronously on the frame thread, with a full GPU drain per resource.
`EndSingleTimeCommand` does `queue.waitIdle()` per texture/buffer. Scene load runs inside
the ImGui frame. For headless CI this matters twice over: load time dominates a short
render test, and a wedged load is indistinguishable from a hang.

### S3-1 — Asset paths are CWD-relative.
`"shaders/opaque.spv"`, `"models/sponza/Sponza.gltf"`, `"textures/skybox/right.jpg"`,
`"scenes/"`. Works when launched from the source dir. A CI test binary in `build/bin` will
not find anything.

### S3-2 — No content/config layer.
Window size, frames in flight, sky colour, near/far planes, cloud parameters, the skybox
path and the validation severity threshold are `constexpr` in `main.cpp`. A test cannot
vary them.

### S3-3 — Logging goes to `stdout`/`stderr` with no sink abstraction, and validation
errors are logged, not counted. A headless test cannot detect "the frame rendered but
produced 40 validation errors" — which is exactly the failure mode automated GPU tests
exist to catch.

## 6. Testability blockers (ranked)

In the order they must be removed to reach "CI launches a scene headless and asserts on it":

| # | Blocker | Minimum fix |
|---|---|---|
| 1 | No linkable library | Split into static libs; `main.cpp` becomes a thin `apps/` target |
| 2 | Headers need the PCH force-include | Make headers self-contained; keep PCH as an optimisation only |
| 3 | Vulkan init requires a surface | `IPresentTarget` with `Swapchain` and `Offscreen` implementations (§10.2) |
| 4 | Device selection requires present support | Split "required" from "required for presentation" feature sets |
| 5 | Frame loop requires SDL events | `IPlatform` with `SdlPlatform` and `HeadlessPlatform` |
| 6 | Loop runs until user quits | `Engine::Run(RunSpec)` with `MaxFrames` / `MaxWallClock` and a returned `RunReport` |
| 7 | Singletons | Constructor-injected subsystems owned by an `Engine` instance |
| 8 | CWD-relative assets | `Paths` service resolving an explicit content root |
| 9 | Validation errors only logged | Error counter + policy (`Ignore` / `Count` / `FailFast`) |
| 10 | No result to assert on | `RunReport` with counters (draw calls, instances, validation errors, frame times, peak VRAM) |
| 11 | Non-determinism (mailbox present, smoothed dt, thread pool ordering) | Deterministic mode: fixed dt, FIFO present, serial job executor |
| 12 | `Model` ctor touches renderer globals | Assets produce data; a system registers it |

---

# Part II — Target architecture

## 7. Design principles

1. **Layers are enforced by the build system, not by discipline.** If `Core` does not link
   `RHI`, then `Core` cannot include Vulkan headers. The compiler is the reviewer.
2. **Everything hardware-facing sits behind an interface with at least two
   implementations.** Two implementations is the minimum count at which an abstraction is
   actually an abstraction. For us: real + headless/null. This is what makes CI possible
   and it is *also* what will make a D3D12/Metal backend possible later.
3. **Ownership is explicit and injected.** No singletons. `Engine` owns subsystems; they
   receive their dependencies by reference/handle in their constructor. A test constructs
   exactly the subsystems it needs.
4. **Simulation state and render state are separate, connected by a one-way per-frame
   snapshot.** This buys the render thread, temporal effects, determinism and testability
   simultaneously.
5. **Hot data is stored as arrays of scalars, not arrays of objects.** Sort keys are
   integers. Identities are 32-bit handles. Per-frame scratch memory comes from an arena.
6. **Data flows through explicit stages: Extract → Cull → Sort → Batch → Encode.** Each
   stage takes spans in and produces spans out, so each stage is independently testable and
   independently parallelisable.
7. **Every output that a human currently judges by eye should also be a number.** Draw
   calls, instances, culled fraction, validation errors, allocation counts, image hash.
   Numbers can be asserted in CI; screenshots cannot (reliably).

## 8. Target module & target graph

Nine CMake targets. Arrows point to allowed dependencies; nothing else may be linked.

```
                        ┌──────────────┐
                        │   Platform   │  windowing, input, files, clock, args
                        │  (SDL / null)│
                        └──────┬───────┘
                               │
          ┌──────────┐         │
          │   Core   │◄────────┘   logging, jobs, containers, math, memory,
          └────┬─────┘             handles, time, profiling, reflection-lite
               │
        ┌──────┴───────┐
        │     RHI      │  instance, device, queues, allocator, swapchain,
        │ (Vulkan/Null)│  images, buffers, pipelines, descriptors, barriers,
        └──────┬───────┘  upload, command encoding, present target
               │
        ┌──────┴───────┐        ┌──────────────┐
        │    Render    │◄───────┤    Assets    │  loaders, cache, GPU upload,
        │              │        └──────┬───────┘  mesh/material/texture data
        └──────┬───────┘               │
               │                ┌──────┴───────┐
               │                │    Scene     │  ECS world, components,
               │                └──────┬───────┘  systems, serialization
               │                       │
        ┌──────┴───────────────────────┴───┐
        │             Engine               │  subsystem registry, frame loop,
        └──────┬──────────────────┬────────┘  config, RunSpec/RunReport
               │                  │
        ┌──────┴─────┐    ┌───────┴────────┐
        │   Editor   │    │  apps/*        │  thin mains
        └────────────┘    └────────────────┘
```

### Target responsibilities and rules

| Target | Type | May link | Must NOT know about |
|---|---|---|---|
| `Core` | static lib | *(nothing but std + glm)* | Vulkan, SDL, assimp, ImGui |
| `Platform` | static lib | `Core` | Vulkan, rendering |
| `RHI` | static lib | `Core`, `Platform`, Vulkan, VMA | assimp, ImGui, scene concepts |
| `Asset` | static lib | `Core`, `Platform`, `RHI`, assimp, stb | ECS, render passes |
| `Scene` | static lib | `Core`, `Platform`, `Asset`, pugixml | `RHI`, Vulkan |
| `Render` | static lib | `Core`, `RHI`, `Asset` | ECS internals (receives snapshots) |
| `Engine` | static lib | all of the above | ImGui |
| `Editor` | static lib | `Engine`, ImGui, ImGuiFileDialog | — |
| `HikariEngine` | exe | `Engine`, `Editor` | — |
| `HikariEngineHeadless` | exe | `Engine` | ImGui, `Editor` |
| `Tests*` | exe | whichever layer is under test | — |

Note the two important non-dependencies:

- **`Scene` does not link `RHI`.** The ECS, transforms, hierarchy, serialization and game
  logic are testable with zero Vulkan. That is the largest and cheapest test surface in the
  project, and today it is unreachable.
- **`Render` does not link `Scene`.** It consumes a `FrameSnapshot` of POD arrays. That
  means renderer tests build snapshots by hand — no scene graph, no assimp, no files.

Enforcement helper:

```cmake
# cmake/EngineModule.cmake
function(engine_module name)
  cmake_parse_arguments(M "" "" "SOURCES;PUBLIC_DEPS;PRIVATE_DEPS" ${ARGN})
  add_library(${name} STATIC ${M_SOURCES})
  target_include_directories(${name}
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(${name} PUBLIC ${M_PUBLIC_DEPS} PRIVATE ${M_PRIVATE_DEPS})
  engine_set_warnings(${name})           # one place for /W4 -Wall -Wextra …
  set_target_properties(${name} PROPERTIES FOLDER "Engine")
endfunction()
```

Public headers live in `include/<module>/`, private ones next to the `.cpp`. Includes become
`#include <core/Log.h>` — self-documenting, and impossible to write if the layering is wrong.

## 9. Target directory layout

```
HikariEngine/
├── CMakeLists.txt                  # top-level: options, deps, add_subdirectory only
├── CMakePresets.json
├── cmake/
│   ├── EngineModule.cmake          # engine_module(), warnings, folders
│   ├── Sanitizers.cmake            # ASan/UBSan, MSVC-aware, AFTER project()
│   ├── Shaders.cmake               # slang compilation + spirv-val + reflection
│   └── Testing.cmake               # catch2 discovery, CTest labels, fixtures
│
├── engine/
│   ├── core/
│   │   ├── include/core/
│   │   │   ├── Assert.h  Log.h  LogSink.h
│   │   │   ├── Handle.h            # typed 32-bit index+generation handle
│   │   │   ├── HandlePool.h        # dense storage + free list + generations
│   │   │   ├── SparseSet.h  SwapbackArray.h  RingBuffer.h  FixedVector.h
│   │   │   ├── Arena.h             # frame linear allocator + scoped scratch
│   │   │   ├── Jobs.h              # JobSystem interface, Serial + WorkStealing
│   │   │   ├── ParallelFor.h
│   │   │   ├── Clock.h             # IClock: RealClock, FixedStepClock
│   │   │   ├── Math.h  Aabb.h  Frustum.h  Simd.h
│   │   │   ├── RadixSort.h
│   │   │   ├── Counters.h          # named counters, dumped into RunReport
│   │   │   └── Profiler.h          # scoped CPU zones; Tracy-compatible later
│   │   └── src/…
│   │
│   ├── platform/
│   │   ├── include/platform/
│   │   │   ├── IPlatform.h         # window, input, events, message box
│   │   │   ├── SdlPlatform.h
│   │   │   ├── HeadlessPlatform.h  # synthesises events from a script
│   │   │   ├── FileSystem.h  Paths.h   # content root resolution
│   │   │   └── CommandLine.h
│   │   └── src/…
│   │
│   ├── rhi/
│   │   ├── include/rhi/
│   │   │   ├── RhiTypes.h          # formats, usage, handles — no vk:: in signature
│   │   │   ├── Device.h            # instance+physical+logical+queues+features
│   │   │   ├── DeviceFeatures.h    # required vs required-for-present split
│   │   │   ├── Allocator.h
│   │   │   ├── Buffer.h  Image.h  Sampler.h  TextureView.h
│   │   │   ├── IPresentTarget.h    # ◄── the headless seam
│   │   │   ├── SwapchainTarget.h
│   │   │   ├── OffscreenTarget.h
│   │   │   ├── CommandEncoder.h    # thin, records into vk cmd buffer
│   │   │   ├── ICommandSink.h      # ◄── the null/recording seam
│   │   │   ├── Barrier.h  BarrierBatcher.h
│   │   │   ├── PipelineBuilder.h  ComputePipelineBuilder.h  PipelineCache.h
│   │   │   ├── DescriptorAllocator.h   # growable pool set
│   │   │   ├── BindlessTable.h
│   │   │   ├── UploadContext.h     # batched staging, one fence
│   │   │   ├── QueryPool.h         # GPU timestamps
│   │   │   └── Diagnostics.h       # validation callback → counters + policy
│   │   └── src/vulkan/… , src/null/…
│   │
│   ├── assets/
│   │   ├── include/assets/
│   │   │   ├── AssetId.h  AssetCache.h  AssetRegistry.h
│   │   │   ├── MeshData.h          # CPU: vertices, indices, submeshes, AABBs
│   │   │   ├── MaterialData.h  TextureData.h  CubemapData.h
│   │   │   ├── GpuMeshRegistry.h   # mega vertex/index buffer + allocations
│   │   │   ├── MaterialRegistry.h  # material params SSBO + bindless indices
│   │   │   ├── TextureRegistry.h
│   │   │   ├── importers/{ModelImporter,ImageImporter,CubemapImporter}.h
│   │   │   └── AsyncLoader.h       # CPU decode on jobs, GPU upload on transfer queue
│   │   └── src/…
│   │
│   ├── scene/
│   │   ├── include/scene/
│   │   │   ├── Entity.h            # {index:24, generation:8}
│   │   │   ├── World.h             # component storages + systems + queries
│   │   │   ├── components/{Transform,Hierarchy,MeshRenderer,Light,Camera,Tag}.h
│   │   │   ├── systems/{TransformSystem,LightSystem,CameraSystem}.h
│   │   │   ├── SceneDesc.h         # serialization-facing plain description
│   │   │   └── serialization/{SceneReader,SceneWriter,XmlBackend}.h
│   │   └── src/…
│   │
│   ├── render/
│   │   ├── include/render/
│   │   │   ├── FrameSnapshot.h     # ◄── the sim/render boundary (SoA, POD)
│   │   │   ├── RenderScene.h       # persistent GPU-side mirror + dirty tracking
│   │   │   ├── View.h  ViewUniforms.h
│   │   │   ├── Culling.h
│   │   │   ├── DrawListBuilder.h   # keys → radix sort → batches
│   │   │   ├── FrameGraph.h  PassBuilder.h  ResourceRegistry.h
│   │   │   ├── passes/{DepthPrepass,Opaque,Transparent,Clouds,Composite,Skybox,
│   │   │   │           Shadow,PostProcess,ImGuiPass}.h
│   │   │   ├── Renderer.h          # orchestrates graph + submission
│   │   │   ├── FrameResources.h    # per-frame-in-flight ring resources
│   │   │   └── shared/ShaderTypes.h   # ◄── included by C++ AND Slang
│   │   └── src/…
│   │
│   ├── engine/
│   │   ├── include/engine/
│   │   │   ├── Engine.h            # owns subsystems, runs the loop
│   │   │   ├── EngineConfig.h      # everything currently constexpr in main.cpp
│   │   │   ├── RunSpec.h           # headless?, scene, frames, size, screenshot, seed
│   │   │   ├── RunReport.h         # ◄── what tests assert on
│   │   │   ├── ISubsystem.h
│   │   │   └── ScriptedInput.h     # deterministic input playback
│   │   └── src/…
│   │
│   └── editor/
│       ├── include/editor/{EditorLayer,panels/*}.h
│       └── src/…
│
├── apps/
│   ├── hikariengine/main.cpp       # ~40 lines: parse args, Engine, Editor, Run
│   └── headless/main.cpp           # ~30 lines: parse args, Engine, Run, print report
│
├── shaders/                         # SOURCE (moved out of src/)
│   ├── include/{common,pbr,brdf,tonemap,phase,surface}.slangh
│   ├── opaque.slang  weightedBlendedOIT.slang  composite.slang  skybox.slang
│   └── compute/{clouds,bakePerlinWorley}.comp.slang
│
├── tests/
│   ├── CMakeLists.txt
│   ├── support/                     # shared test utilities (a real library)
│   │   ├── TestPaths.h              # locates tests/data regardless of CWD
│   │   ├── RhiTestFixture.h         # headless device, or SKIP if unavailable
│   │   ├── ValidationGuard.h        # fails the test on any validation error
│   │   ├── LayoutTracker.h          # asserts image layout/barrier legality
│   │   ├── RecordingSink.h          # ICommandSink capturing the command stream
│   │   ├── SnapshotBuilder.h        # build FrameSnapshots without a World
│   │   └── ImageCompare.h           # perceptual diff + PNG write on failure
│   ├── unit/                        # label: unit — no GPU, no files
│   │   ├── core/{handles,arena,jobs,radixsort,math,frustum,containers}_test.cpp
│   │   ├── scene/{transform_hierarchy,world,queries,serialization}_test.cpp
│   │   ├── render/{culling,drawlist,batching,keys}_test.cpp
│   │   └── assets/{cache,ids,meshdata}_test.cpp
│   ├── contract/                    # label: contract — null RHI, no GPU
│   │   ├── framegraph_ordering_test.cpp
│   │   ├── barrier_legality_test.cpp
│   │   └── descriptor_binding_test.cpp
│   ├── gpu/                         # label: gpu — needs a Vulkan device
│   │   ├── device_creation_test.cpp
│   │   ├── pipeline_compilation_test.cpp   # builds EVERY pipeline
│   │   ├── upload_roundtrip_test.cpp
│   │   ├── scene_launch_test.cpp            # ◄── the headline runtime test
│   │   └── resize_and_reload_test.cpp
│   ├── golden/                      # label: golden — image comparison
│   │   ├── golden_test.cpp
│   │   └── references/*.png
│   ├── perf/                        # label: perf — counter budgets
│   │   └── frame_budget_test.cpp
│   └── data/
│       ├── scenes/{empty,single_cube,two_materials,lights_only,stress_5k}.map
│       └── models/{cube,two_material_quad}.gltf     # tiny, committed
│
├── content/                         # runtime content root (was models/, textures/, scenes/)
│   ├── models/  textures/  scenes/  shaders/   ← compiled .spv output lands here
│
└── docs/
    ├── architecture.md   (this document, once adopted)
    ├── testing.md
    └── conventions.md
```

Rationale for the moves that aren't obvious:

- **`shaders/` out of `src/`.** Shader source is not C++ source; it has its own compiler,
  its own dependency graph and its own test (`spirv-val`, pipeline creation). Compiled
  `.spv` lands beside the executable, in `<exe dir>/shaders/`, and is reached through
  `Paths::Shader()` rather than `Paths::Content()`. It is a build output, not an asset: the
  same sources compile differently per configuration, so a shared directory under the content
  root had debug and release silently overwriting each other.
- **`content/` as a single root.** One env var / one CLI flag (`--content <dir>`) makes
  every asset path resolvable from any CWD, which is prerequisite #8 in §6.
- **`tests/support/` is a library, not headers copy-pasted between tests.** The validation
  guard and layout tracker are the two components that turn "it rendered" into "it rendered
  correctly", and they need to be shared.

## 10. Headless architecture

This is the section that unlocks the CI goal. Three seams are needed.

### 10.1 Platform seam

**Built, and not in the shape this section originally drew.** Read
`engine/platform/include/platform/IPlatform.h` for the interface; what follows is why it
differs, because the difference changes what step 40 has to do.

The original sketch gave `IPlatform` three Vulkan-facing members —
`GetRequiredInstanceExtensions`, `CreateSurface(VkInstance, VkSurfaceKHR*)` and
`PumpEvents(EventQueue&)`. The first two are gone: **surface creation and instance-extension
enumeration moved into the RHI**, where `VulkanDevice::CreateSurface` does the work and
`DeviceRequirements::NativeWindowHandle` carries the only thing it needs from the platform.
That is a better split than the sketch — Platform links no Vulkan at all now, so the type
`CreateSurface` would have returned cannot even be named there — and it means
`HeadlessPlatform` has nothing to stub for either.

What shipped instead is `IsHeadless`, `GetFramebufferExtent`, `Show`, `SetWindowMode`,
`SetRelativeMouseMode`, `WarpMouse` and a temporary `GetNativeWindowHandle`. `SetWindowMode`
and the `WindowMode` enum (windowed / borderless / exclusive fullscreen) arrived with the
window-mode work and were never in this plan.

**`PumpEvents` did not ship, and nothing replaced it.** `App::Run` still calls
`SDL_PollEvent` directly, and `HandleMovement` still reads `SDL_GetKeyboardState`. So there
is no event seam, and §14's scripted input (`frame 15 key.up W`) has nowhere to attach. That
is now the larger half of step 40 rather than an incidental part of it — see the step, which
splits accordingly.

`GetNativeWindowHandle` is the one member marked temporary: Vulkan surface creation and the
ImGui SDL3 backend both still need a concrete `SDL_Window*`, and it disappears when the
second of those moves into `engine/editor`.

### 10.2 Presentation seam

**Built in step 35.** `engine/rhi/include/rhi/IPresentTarget.h` is the interface and carries
the reasoning for each member; the sketch that used to sit here has been removed rather than
maintained in parallel, because a second copy of a signature is a second thing to get wrong.
Two ways the built version differs from what this section drew:

- **`Recreate` returns `[[nodiscard]] bool`, not `void`.** False means it cannot rebuild yet
  and touched nothing — the state a minimised window leaves a surface in, which a caller
  cannot test for itself without knowing it holds a swapchain. It is `[[nodiscard]]` where
  `Present` is not because the two fail differently when dropped: a dropped `Present` costs
  one frame and the loop corrects itself, while a dropped `Recreate` leaves the caller
  rebuilding per-image state against a target still at the old extent, and nothing asks
  again.
- **`PresentTargetDesc` exists**, carrying the creation extent and `FramesInFlight` — the
  latter sizes the ring of acquire semaphores and is deliberately *not* the image count,
  which the target chooses and reports.

Everything here is spelled in the neutral vocabulary, because
`cmake/RhiBoundaryCheck.cmake` rejects a `vk::` type under `include/rhi/` outright — this
section predates the Stage 5 neutralisation and originally named `vk::Image`,
`vk::Semaphore` and `vk::Queue`. Three consequences worth keeping:

- **The target owns its semaphores.** `Acquire()` takes none and returns the one it
  signalled; `Present()` takes no queue. `SemaphoreHandle` (added to `rhi/Handles.h` in
  step 35) exists only so a caller can name one for its own submit, which is what
  `Handles.h` had reserved the shape for. While the frame loop still records its own
  submit it resolves both through `Rhi::Vulkan::GetSemaphore`; that goes away in Stage 8.
- **Out-of-date is a value, not an error.** `bNeedsRecreate` and `Present()`'s `bool`
  cover suboptimal and out-of-date alike, because a resize races every frame.
- **`Readback` is not here.** It arrives with `OffscreenTarget` in step 39 rather than
  being a member `SwapchainTarget` would only stub.

Two implementations:

- **`SwapchainTarget`** — ✅ done. The old `CreateSwapchain` / `acquireNextImage` /
  `presentKHR`, extracted. Fixes S1-15 from the companion doc naturally, because
  `GetImageCount()` changing on `Recreate` becomes an explicit, observable event that the
  renderer reacts to rather than something it forgets.
- **`OffscreenTarget`** — ✅ done, steps 38 and 39. N color images
  (`eColorAttachment | eTransferSrc | eSampled`), no surface, no swapchain, no
  `VK_KHR_swapchain`. N is `FramesInFlight`; `Acquire` returns `(acquireCount % N)` and
  never reports `bNeedsRecreate`; `Present` records only that the caller signalled that
  image's render-complete semaphore, which the next `Acquire` of the same image hands back
  as its wait. `Readback` arrived in step 39 as a member of this class rather than of the
  interface, for the reason given above, and consumes that same signal to order its copy.

**Which target a device hands back is already decided by `IDevice::CreatePresentTarget`,**
not by the caller — the header says so, and today it throws when there is no surface. That
throw is where `OffscreenTarget` attaches, so step 38 needs no new entry point.

Two decisions step 38 had to settle, both of which the sketch above left open:

- **The acquire wait is a span — ✅ done in step 38.** `AcquiredImage::Available`
  (one `SemaphoreHandle`, documented as null only when `bNeedsRecreate` is set) became
  `std::span<const SemaphoreHandle> WaitSemaphores`, empty when there is nothing to wait on.
  An empty set is not a sentinel, and the submit already takes a count and an array, so the
  empty case costs no branch. The offscreen target does turn out to need a wait, and fills
  the span like any other target — see step 38 for why it is not optional.
- **Readback is format-driven, and the screenshot writer now is too — ✅ done in step 39.**
  `WriteScreenshot` hardcoded a BGRA→RGBA swizzle, correct only because
  `ChooseSwapchainFormat` asks for `BGRA8Unorm` first. It reads `GetFormat()` for both the
  texel size and the channel order, and refuses a format `stbi_write_png` cannot take rather
  than reinterpreting the bytes.

`tests/support/GpuReadback.h` already records the copy, submits it, fences and reads the
mapped bytes — but only part of it is genuinely shareable with step 39. See that step for
which part, and why copying the rest would be the wrong move.

`Renderer` talks only to `IPresentTarget`. It never sees `vk::SwapchainKHR`.

### 10.3 Device-creation seam

**✅ Done, during Stage 5 rather than Stage 6 — this is what step 37 asked for, delivered
early.** See `engine/rhi/include/rhi/DeviceDesc.h`.

The shipped `DeviceRequirements` is far smaller than the struct this section drew: two
fields, `bPresent` and an opaque `NativeWindowHandle`. Everything the sketch wanted to
express as caller-supplied policy — API version, extension lists, the required feature set —
stayed *inside* the backend instead, because none of it can be spelled in a backend-neutral
header without leaking Vulkan's vocabulary into it, and a caller has no basis on which to
vary it. That is the D0–D12 rule from `rhi_extraction_plan.md` applied: the seam says what is
wanted, not how it is spelled.

When `bPresent` is false the backend creates no surface, requests neither the surface
instance extensions nor `VK_KHR_swapchain`, and drops the `getSurfaceSupportKHR` term from
queue-family selection. Everything else stays required, which is what makes the headless test
*meaningful*: it exercises the same feature set the real app does. Queue families are
enumerated properly (graphics / compute / copy), which resolved the two `main.cpp` `TODO`s
and unblocked the batched `UploadContext`.

The gpu tests already run this way — `RhiTestFixture` creates **every** device with
`bPresent = false`, across four device shapes — so the surfaceless path is exercised
continuously rather than by one case.

**One idea from the sketch has no home yet and is still needed:** `bPreferDiscreteGpu =
false`, "for CI: prefer the software ICD".

Worth being precise about the starting point, because both this plan and the obvious reading
of the code overstate it. **There is no device preference implemented at all.**
`PickPhysicalDevice` is `std::ranges::find_if(devices, IsPhysicalDeviceSuitable)` — the first
suitable device in enumeration order, no scoring. So `bPreferDiscreteGpu` is not a field that
went missing; the behaviour it names was never written. On a machine with an integrated and a
discrete GPU you get whichever the loader lists first, which is why §14 counts device
selection as a source of non-determinism.

**Decided: CI pins the ICD rather than the engine picking a device.** `VK_DRIVER_FILES` makes
lavapipe the only device the loader offers, which is deterministic without any selection code,
and no mechanism is written until a case needs one — `--gpu <name-substring>` is a `backlog.md`
row, not part of step 47. When that row is picked up, note that `vkEnumeratePhysicalDevices`
order is not guaranteed stable across driver updates or between machines, so a bare index is a
poor persistent identifier and is not the answer.

Independently of that, **log the selected device name and driver version into the run report** —
two reports cannot be compared without knowing what produced them. Also a backlog row.

### 10.4 CI device availability

| Runner | Vulkan device | Use |
|---|---|---|
| `ubuntu-latest` + **Mesa lavapipe** (`mesa-vulkan-drivers`) | software, Vulkan 1.3, dynamic rendering + sync2 supported | Primary headless GPU job |
| `ubuntu-latest` + lavapipe + ASan/UBSan | same | The highest-value job in the whole matrix |
| `windows-latest` | none by default | Unit + contract tests only |
| Self-hosted with a real GPU (optional, later) | real | Golden images, perf budgets |

Practical notes for lavapipe:
- Install: `sudo apt-get install -y mesa-vulkan-drivers vulkan-tools`.
- Select explicitly: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`
  (newer loaders: `VK_DRIVER_FILES`). Verify with `vulkaninfo --summary` as a CI step so a
  driver regression is diagnosable.
- It is slow. Test scenes must be small (`tests/data/models/cube.gltf`, 256×256 render
  target, 3–10 frames). Do **not** put Sponza in a CI test.
- It does not implement everything. Gate optional features and have the fixture
  **skip with a clear message** rather than fail when a required feature is missing — but
  fail loudly if it's a feature the app hard-requires, because that is real information.

### 10.5 Headless `main`

```cpp
// apps/headless/main.cpp
int main(int argc, char** argv)
{
    const RunSpec spec = ParseRunSpec(argc, argv);   // Platform::CommandLine

    Engine engine(EngineConfig::FromRunSpec(spec));
    const RunReport report = engine.Run(spec);

    report.WriteJson(spec.ReportPath);               // machine-readable
    report.PrintSummary();                           // human-readable

    return report.bSucceeded && report.ValidationErrorCount == 0
             ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

The same `Engine` runs windowed and headless. `apps/hikariengine/main.cpp` differs only in
that it constructs `SdlPlatform` and attaches `EditorLayer`.

## 11. Data-oriented core

### 11.1 Handles instead of pointers

```cpp
// core/Handle.h
template <typename Tag>
struct Handle
{
    uint32_t Value = kInvalid;                 // index:24 | generation:8
    static constexpr uint32_t kInvalid = 0xFFFFFFFF;

    constexpr uint32_t Index()      const { return Value & 0x00FFFFFF; }
    constexpr uint32_t Generation() const { return Value >> 24; }
    constexpr bool     IsValid()    const { return Value != kInvalid; }
    constexpr auto operator<=>(const Handle&) const = default;
};

using MeshHandle     = Handle<struct MeshTag>;
using MaterialHandle = Handle<struct MaterialTag>;
using TextureHandle  = Handle<struct TextureTag>;
using EntityHandle   = Handle<struct EntityTag>;
```

Why this matters concretely:
- `Drawable`'s `Mesh*`/`Material*` become 4 bytes each and **fit inside a sort key**.
- Use-after-free becomes a detectable generation mismatch rather than UB — this directly
  addresses the `ModelData::RegisterMesh` pointer-invalidation class of bug.
- Registries can reallocate, defragment and hot-reload freely.
- Tests can fabricate handles without constructing any GPU object.

### 11.2 ECS replacing `Entity`/`Component`

Sparse-set component storage (simpler than archetypes, ~90% of the benefit, far less code):

```cpp
// scene/World.h  (sketch)
class World
{
public:
    EntityHandle Create();
    void         Destroy(EntityHandle);
    bool         IsAlive(EntityHandle) const;

    template <typename C> C&   Add(EntityHandle, C&& value);
    template <typename C> C*   TryGet(EntityHandle);
    template <typename C> void Remove(EntityHandle);

    // Contiguous access — this is the point of the whole exercise.
    template <typename C> std::span<C>                  Components();
    template <typename C> std::span<const EntityHandle> Owners() const;

    template <typename... Cs> auto View();   // sorted-intersection iteration
};
```

Components are POD:

```cpp
struct LocalTransform { glm::vec3 Position; glm::quat Rotation; glm::vec3 Scale; };
struct WorldTransform { glm::mat4 Matrix; };          // derived, recomputed
struct Hierarchy      { EntityHandle Parent; uint32_t Depth; };
struct MeshRenderer   { MeshHandle Mesh; MaterialHandle Material; uint8_t Layers; };
struct Bounds         { glm::vec3 Center; float Radius; glm::vec3 Extent; float Pad; };
struct PointLightC    { glm::vec3 Color; float Intensity; float Radius; };
struct DirLightC      { glm::vec3 Color; float Intensity; glm::vec3 Direction; };
struct CameraC        { float FovY, Aspect, Near, Far; };
```

Notes:
- **Rotation as a quaternion**, not Euler `vec3`. The current `Transform` stores Euler
  angles and rebuilds a matrix each query; quaternions compose cleanly, interpolate, and
  remove a whole class of gimbal/order bugs. Keep Euler purely as an *editor/serialization*
  representation, converted at the boundary.
- **`Hierarchy::Depth`** lets `TransformSystem` sort entities by depth once and then
  compute world matrices in **a single linear forward pass** — parents are always already
  done. No recursion, no parent-pointer chasing, cache-friendly. This replaces
  `SceneComponent::GetAccumulatedTransform()` walking a pointer chain per query.
- **Dirty tracking**: a bitset over the transform array. `TransformSystem` processes only
  dirty entities and marks their subtrees. A static scene costs ~0.
- **No inheritance, no `dynamic_cast`, no virtuals in the query path.**

Migration note: `Camera`, `PointLight`, `DirectionalLight` and `Model` all currently inherit
`SceneComponent`. In the target they are components + systems. `Camera`'s matrix code moves
to `CameraSystem`; `Light::GetData()` becomes a packing function in `LightSystem` that
writes straight into the GPU light array.

### 11.3 The simulation → render boundary

```cpp
// render/FrameSnapshot.h — POD, SoA, arena-allocated, double-buffered.
struct FrameSnapshot
{
    uint64_t FrameIndex = 0;
    float    DeltaTime  = 0.f;
    float    TotalTime  = 0.f;

    ViewInfo View;                       // view, proj, invViewProj, pos, near, far, frustum

    // Renderable instances — parallel arrays, index i describes one instance.
    std::span<const MeshHandle>     Meshes;
    std::span<const MaterialHandle> Materials;
    std::span<const uint8_t>        BlendModes;
    std::span<const Mat3x4>         WorldTransforms;    // 48 B, not 64
    std::span<const Mat3x4>         NormalTransforms;   // 48 B, not 64
    std::span<const BoundingSphere> Bounds;             // 16 B

    std::span<const GpuPointLight>  PointLights;
    std::span<const GpuDirLight>    DirLights;

    CloudParams Clouds;
    EnvParams   Environment;
};
```

Properties that matter:
- **POD spans only.** A renderer test constructs one with `std::vector`s and a
  `SnapshotBuilder` helper. No `World`, no assimp, no `VkDevice`.
- **Double-buffered** in the arena: the render thread reads snapshot N while simulation
  writes N+1. This is what makes a render thread possible without locks.
- **`Mat3x4` instead of `mat4`** for transforms: affine transforms never need the last row.
  48 bytes instead of 64 is 25% less to write, upload and read, and it lets three rows pack
  into a single `float4x3` in the shader.

### 11.4 The frame pipeline

```
Simulation (main thread)
  Input → Scripted/SDL events → World systems → TransformSystem (dirty-only)
                                                         │
                                                    Extract  ── parallel_for over
                                                         │      MeshRenderer view
                                                    FrameSnapshot[N]
                                                         │
Render (render thread)                                   ▼
  Cull ──► VisibleIndices        parallel_for, SIMD sphere/AABB vs 6 planes
     │
  BuildKeys ──► (uint64 key, uint32 instanceIndex)[]
     │
  RadixSort ──► sorted pairs                3 × 8-bit passes, 12 B/element
     │
  Batch ──► DrawBatch[] + write InstanceData straight into the mapped ring buffer
     │
  FrameGraph::Compile ──► pass order + barriers + transient aliasing
     │
  Encode ──► parallel_for over passes, one command pool per thread per frame
     │
  Submit ──► one vkQueueSubmit2 with timeline semaphores
     │
  Present ──► IPresentTarget::Present
```

### 11.5 Sort keys and batching

Replace `std::sort` over 96-byte `Drawable`s with a radix sort over 12-byte pairs:

```cpp
// render/DrawListBuilder.h
// [63:62] blend/layer  [61:50] pipeline  [49:32] material
// [31:16] mesh         [15:0]  depth bucket (front-to-back opaque, back-to-front alpha)
constexpr uint64_t MakeDrawKey(uint8_t layer, uint16_t pipeline,
                               MaterialHandle mat, MeshHandle mesh,
                               uint16_t depthBucket)
{
    return (uint64_t(layer)          << 62)
         | (uint64_t(pipeline  & 0xFFF)   << 50)
         | (uint64_t(mat.Index()  & 0x3FFFF) << 32)
         | (uint64_t(mesh.Index() & 0xFFFF)  << 16)
         |  uint64_t(depthBucket);
}
```

Then batching is one branch-free linear scan comparing adjacent keys masked to the
"batch identity" bits. Instance data is written directly into the persistently mapped
ring buffer during that scan — the intermediate `std::vector<InstanceData>` disappears.

Expected effect at 10k instances (see [Appendix B](#appendix-b--frame-budget-model)):
sorting drops from ~1.5 ms of comparison + 96-byte moves to ~0.15 ms of 3 radix passes,
and the per-instance `glm::inverse` (≈100 flops × 10k ≈ 1 M flops/frame) disappears
entirely because normal matrices are cached and only recomputed on transform change.

### 11.6 Memory

| Allocation class | Mechanism |
|---|---|
| Per-frame scratch (snapshots, visible lists, key arrays, batch arrays) | `Arena` — bump pointer, reset at frame start, 2 arenas for double buffering |
| Per-frame GPU writes (instances, view uniforms, light arrays) | One large `HOST_ACCESS_SEQUENTIAL_WRITE` buffer per frame-in-flight, sub-allocated by offset. Replaces today's separate global/instance buffers *and* the `TODO` about allocating 3 times. |
| Mesh vertex/index data | One growable device-local mega-buffer + `BufferAllocation{offset,size}`. Reduces per-batch `bindVertexBuffers`/`bindIndexBuffer` from *per batch* to **once per frame**. |
| Material parameters | One SSBO, indexed by `MaterialHandle::Index()` from instance data. Removes per-batch `pushConstants`. |
| Textures | Bindless `Sampler2D[]` descriptor array (`descriptorBindingPartiallyBound`, already enabled). Removes per-batch `bindDescriptorSets` and the 100-material / 3-texture ceilings. |
| Long-lived GPU resources | VMA, unchanged. |
| Staging | `UploadContext` with a staging ring; one submit + one fence per batch, not per resource. |

With bindless + mega-buffer + material SSBO, the inner draw loop collapses from
`5 binds + 1 push + 1 draw` per batch to just `drawIndexed` per batch — and from there to a
single `drawIndexedIndirectCount` once GPU-driven batching is worth doing.

### 11.7 Jobs

```cpp
// core/Jobs.h
class IJobSystem
{
public:
    virtual ~IJobSystem() = default;
    virtual uint32_t WorkerCount() const = 0;
    virtual JobHandle Submit(std::function<void()>) = 0;
    virtual void      Wait(JobHandle) = 0;
    virtual void      ParallelFor(uint64_t count, uint64_t grain,
                                  std::function<void(uint64_t, uint64_t)>) = 0;
};
```

- `WorkStealingJobSystem` for the app (fixes the `hardware_concurrency() <= 1` underflow by
  construction: `WorkerCount = max(1, hc - 1)`).
- **`SerialJobSystem` for tests and `--jobs 0`.** `ParallelFor` runs inline. Every
  parallel algorithm is then testable deterministically, and any parallel-vs-serial result
  divergence is itself a test (`ASSERT(serialResult == parallelResult)`), which is a very
  effective way to catch data races without a race detector.
- Injected, not a singleton. Two `Engine`s can coexist.

## 12. Render graph and passes

The current 7-hardcoded-command-buffers design is the main scalability brake on the
rendering side: pass identity is duplicated across `FrameData`, two creation functions and
the submit array, and every barrier is placed by hand in the recorder that happens to need
it.

Target: a small frame graph. Not a research-grade one — ~600–900 lines total.

```cpp
class Pass
{
public:
    virtual ~Pass() = default;
    virtual const char* Name() const = 0;
    // Declare reads/writes/creates. No Vulkan calls here.
    virtual void Setup(PassBuilder&) = 0;
    // Resources are real, layouts are already correct.
    virtual void Execute(PassContext&) = 0;
};
```

What the graph gives us, mapped to concrete pain today:

| Graph feature | Replaces |
|---|---|
| Declared reads/writes | Hand-placed `RecordImageBarrier` calls in 7 recorders; the missing composite→ImGui barrier and the missing `eComputeShader` depth barrier become **impossible to write**, not bugs to remember |
| Automatic layout tracking | `Barriers::XToY()` factories chosen by hand at each site |
| Transient resource creation from a description | `CreateRenderTargets`, `CreateDepthResources` and their manual re-creation in `RecreateSwapchainAndRenderImages` |
| Resize = recompile the graph with a new extent | The 7-step manual fan-out in `RecreateSwapchainAndRenderImages` |
| Pass culling | ImGui pass being recorded and submitted even when hidden |
| Topological order + parallel encode | The manual `MULTITHREADED_COMMAND_RECORDING` split and the fixed submit array |
| A named, inspectable graph | Nothing — this is new, and it is directly testable (§16.4) |

Pass registration becomes declarative:

```cpp
graph.Add<DepthPrepass>();        // new features are one line + one file
graph.Add<ShadowPass>(cascades);
graph.Add<OpaquePass>();
graph.Add<SkyboxPass>();
graph.Add<CloudPass>();
graph.Add<TransparentPass>();
graph.Add<CompositePass>();
if (editor) graph.Add<ImGuiPass>();
```

Also fold in from the companion doc, since the graph is where they belong:
`PipelineCache` seeded from disk, `BarrierBatcher` collecting into one `pipelineBarrier2`,
and `QueryPool` timestamps per pass (which feed `RunReport`, which feeds §15.6 perf tests).

### Shared GPU struct types

One header, included from both C++ and Slang, kills S2-1:

```cpp
// render/shared/ShaderTypes.h
#ifdef __cplusplus
  #include <glm/glm.hpp>
  namespace shader {
  using float2 = glm::vec2; using float3 = glm::vec3;
  using float4 = glm::vec4; using float4x4 = glm::mat4;
  using uint = uint32_t;
#endif

struct GpuPointLight { float3 Color; float Intensity; float3 Pos; float Pad; };
struct GpuDirLight   { float3 Color; float Intensity; float3 Dir; float Pad; };
struct GpuCamera     { float4x4 View, Proj, InvViewProj; float3 Pos; float Near;
                       float3 Pad; float Far; };
struct GpuFrame      { GpuLights Lights; GpuCamera Camera; float3 SkyColor; float Time; };

#ifdef __cplusplus
  }  // namespace shader
  static_assert(sizeof(shader::GpuFrame) % 16 == 0);
  static_assert(offsetof(shader::GpuFrame, Camera) == /* expected */);
  // …one offsetof assertion per field. A divergence becomes a compile error.
#endif
```

## 13. Asset system

Split each asset type into three distinct things — this is what makes assets testable and
async-able:

| Stage | Type | Depends on |
|---|---|---|
| **Source** | file bytes | filesystem only |
| **CPU data** | `MeshData`, `TextureData`, `MaterialData` — plain arrays | nothing (POD) |
| **GPU residency** | `GpuMeshRegistry` allocation, bindless texture slot | RHI |

Consequences:
- `ModelImporter::Import(path) → MeshData` is a **pure function testable with no GPU**.
  Today, importing a model requires a device, a queue, a command pool and an allocator, and
  it performs ~70 full GPU drains.
- Async loading falls out: import on jobs → `MeshData` → hand to `UploadContext` on the
  transfer queue → publish handle when the fence signals. The frame loop never blocks.
  Scene switching becomes "build the new registry entries, then swap the snapshot source",
  which is also what makes `resize_and_reload_test` meaningful.
- Mip generation, tangent generation, index optimisation (meshoptimizer later) and
  bounding-box computation all become CPU-side transformations on `MeshData` with
  exact, cheap unit tests.
- `Model`-registers-itself-with-`ModelManager` disappears: importing produces data; a
  `MeshRenderer` component references handles; `Extract` reads components. No back-pointers.

`AssetId` should be a hash of a normalised path + import settings, not a raw path string.
This fixes the cubemap key collision (keyed on `RightPath` alone) and makes "same file,
different format/sRGB" cache correctly.

## 14. Configuration, CLI and determinism

### `RunSpec` / `RunReport`

```cpp
struct RunSpec
{
    bool        bHeadless        = false;
    std::string ScenePath;                  // empty = empty scene
    std::string ContentRoot;                // resolved: flag → env → exe/.. → source
    uint32_t    Width = 1920, Height = 1080;
    uint64_t    MaxFrames        = 0;       // 0 = unbounded
    double      MaxWallClockSecs = 0.0;     // hard timeout → non-zero exit
    bool        bDeterministic   = false;   // fixed dt, FIFO present, serial jobs
    float       FixedDeltaTime   = 1.f / 60.f;
    uint64_t    Seed             = 0;
    int32_t     JobWorkers       = -1;      // -1 auto, 0 serial
    ValidationPolicy Validation  = ValidationPolicy::Count;  // Ignore|Count|FailFast
    std::string ScreenshotPath;             // write final frame here
    std::string ReportPath;                 // write RunReport JSON here
    std::string InputScriptPath;            // scripted events
    CameraOverride Camera;                  // fixed pos/rot for golden images
};

struct RunReport
{
    bool     bSucceeded         = false;
    uint64_t FramesRendered     = 0;
    uint64_t ValidationErrors   = 0;
    uint64_t ValidationWarnings = 0;

    // Content
    uint64_t EntitiesLoaded = 0, MeshesLoaded = 0, MaterialsLoaded = 0, TexturesLoaded = 0;

    // Per-frame counters (last frame + min/max/mean over the run)
    uint64_t DrawCalls = 0, Instances = 0, InstancesCulled = 0, Batches = 0;
    uint64_t PipelineBinds = 0, DescriptorBinds = 0, BarrierCount = 0;

    // Timing
    double   CpuFrameMsMean = 0, CpuFrameMsP99 = 0;
    std::unordered_map<std::string, double> GpuPassMs;   // from QueryPool

    // Memory
    uint64_t PeakDeviceMemoryBytes = 0, PeakHostMemoryBytes = 0;
    uint64_t TransientArenaPeakBytes = 0;

    std::vector<std::string> Errors, Warnings;

    void WriteJson(const std::string& path) const;
    void PrintSummary() const;
};
```

`RunReport` is the single most important testing artefact in this plan. It converts
"the engine ran" into a set of assertable numbers, and it is equally useful for local
profiling and for CI regression detection.

### Determinism requirements

For a headless test to be a *test* rather than a coin flip:

| Source of non-determinism | Fix |
|---|---|
| `high_resolution_clock` delta time | `FixedStepClock` when `bDeterministic` |
| Smoothed FPS/frame time feedback loop | Report raw values; smoothing is a UI concern. **The current code overshoots this**: under `--fixed-dt` it records the *timestep* rather than measured cost, so the reported frame times are a constant. Measure wall clock separately from simulation dt — see `backlog.md` |
| `eMailbox` present mode | **Not a determinism problem — struck.** Present mode changes timing, not pixels, and forcing FIFO would cap frame times at the refresh interval and hide real regressions. See step 45 |
| Thread pool completion order affecting sort/batch order | Serial job system, plus: keys must be **total orders** (include instance index as a tiebreaker) so the sort is stable regardless of algorithm |
| `unordered_map` iteration order (`ModelData::m_MeshLocalTransforms`, caches) | Never iterate a hash map to produce ordered output; use sorted vectors in the hot path |
| Pointer values used as sort keys (`pMesh < other.pMesh` in `Drawable::operator<`) | **This is a real determinism bug today**: batch order depends on heap addresses. Handles fix it. |
| Physical device selection order | Deterministic tie-break (prefer by `deviceID`, or explicit `--gpu-index`) |
| Uninitialised reads | ASan/UBSan/MSan job |
| Random seeds (cloud noise bake, future particles) | Explicit `Seed` in `RunSpec` |

The `Drawable::operator<` point deserves emphasis: today, two runs of the same scene can
produce different draw orders, which means golden-image tests would be flaky *for a real
reason* and the flakiness would be blamed on the GPU. Handles must land before golden
images.

### Scripted input

```
# tests/data/input/orbit.txt
frame 0   camera.set pos=0,2,8 rot=0,0,0
frame 5   key.down W
frame 15  key.up   W
frame 20  window.resize 320x240
frame 30  screenshot
frame 40  quit
```

`HeadlessPlatform` replays this through the same event path as SDL, so input handling,
resize handling and swapchain/target recreation are all exercised headless — which is where
a large share of real crashes live.

---

# Part III — Testing & delivery

## 15. Test strategy

Framework: **Catch2 v3** via vcpkg (`"catch2"`), driven by **CTest** with labels.
Rationale over GoogleTest: header/lib footprint is smaller, `SECTION`s suit the
"same fixture, many assertions" shape of render tests, and `catch_discover_tests`
integrates cleanly with CTest labels. Either is fine; pick one and don't mix.

Six layers, fastest and most numerous first.

### 15.1 `unit` — no GPU, no files, no threads (target: < 2 s total)

Runs on all three platforms, on every push. This is where the bulk of the tests live.

| Area | Representative cases |
|---|---|
| `core/Handle`, `HandlePool` | index/generation packing; reuse bumps generation; stale handle rejected; exhaustion |
| `core/Arena` | alignment; reset; high-water mark; scoped scratch nesting |
| `core/RadixSort` | matches `std::sort` on 10k random keys; stable; empty; single element; all-equal keys |
| `core/Frustum` | sphere fully in/out/straddling each of 6 planes; degenerate radius 0; frustum from a known viewProj matches hand-computed planes |
| `core/Aabb` | transform by rotation/scale; merge; degenerate |
| `core/Jobs` (serial) | ParallelFor covers exactly `[0,count)` once; grain > count; count 0 |
| `core/SwapbackArray` | `RemoveAt` last element; `Erase` missing value; iterator behaviour |
| `scene/TransformSystem` | 3-level hierarchy world matrices; depth-sorted single pass == recursive reference; reparenting; dirty propagation to subtree only; 1000-node chain |
| `scene/World` | create/destroy/reuse; `TryGet` after `Remove`; view intersection of 3 component types; component array stays contiguous after destroy |
| `scene/serialization` | round-trip `SceneDesc` → XML → `SceneDesc` is identity; missing attributes fall back with a warning (not a crash); malformed XML returns an error rather than throwing through the frame loop; **entity + model transform is not double-applied** (an open question in the companion doc — make it a test and the question is answered forever) |
| `render/Culling` | 0 of N visible; N of N visible; exactly the expected index set for a hand-built frustum; parallel result == serial result |
| `render/DrawListBuilder` | key ordering respects blend > pipeline > material > mesh; batching merges adjacent identical keys and splits on any difference; instance indices within a batch are contiguous; empty input; 1 instance; 10k instances all-distinct materials |
| `assets/MeshData` | tangent generation matches reference for a known quad; AABB computation; index-buffer offsets across submeshes |
| `assets/AssetId` | same path different separators/case → same id (platform-appropriate); different sRGB flag → different id; six cubemap faces contribute to the key |

### 15.2 `contract` — null RHI, no GPU (target: < 5 s)

Uses `RecordingSink` (an `ICommandSink` that appends structured commands to a vector) so
the *encoding logic* is tested without a driver. Runs on all platforms.

- Frame graph produces a valid topological order; a cycle is reported as an error;
  an unused pass is culled; a pass reading a resource nobody writes is an error.
- Barrier legality: `LayoutTracker` replays the recorded command stream, maintaining
  per-image/per-subresource layout state, and asserts that every use matches the declared
  layout and that every layout change is covered by a barrier with compatible
  stage/access masks. **This catches, in a 20 ms CPU-only test, the exact class of bug that
  `suggested_work.md` items 1.10 and 1.11 describe** — and it catches them for
  every future pass too.
- Encoding: given a known snapshot, assert exact draw-call count, instance count and bind
  count. A regression that doubles descriptor binds is visible immediately.
- Resize: recompiling the graph at a new extent recreates exactly the size-dependent
  resources and no others.

### 15.3 `gpu` — headless device required (target: < 60 s on lavapipe)

Linux CI + any developer machine. Skipped with a clear message where no device exists.

| Test | Asserts |
|---|---|
| `device_creation` | Headless device is created without `VK_KHR_swapchain`; all hard-required features present; queue families enumerated; a dedicated transfer queue is found *or* the fallback path is taken |
| `pipeline_compilation` | **Every** pipeline in the project builds — graphics and compute. This alone catches shader/layout/push-constant mismatches, which are currently only found by running the app and looking at it |
| `upload_roundtrip` | Buffer upload → readback is byte-identical; texture upload → readback matches; **all 6 cubemap faces** differ as expected (turns companion-doc bug 1.1 into a permanent regression test); mip chain has expected level count and plausible content |
| `offscreen_target` | Acquire/present/readback across N frames-in-flight; readback dimensions and stride correct |
| `scene_launch` | **The headline test.** Load `tests/data/scenes/single_cube.map` headless at 256×256, render 5 frames, assert: `bSucceeded`, `ValidationErrors == 0`, `FramesRendered == 5`, `DrawCalls > 0`, `EntitiesLoaded == expected`, no device-lost, exits within the timeout |
| `scene_variants` | Same for `empty` (0 draw calls, still no validation errors — the degenerate case that usually crashes), `two_materials` (batch count == 2), `lights_only`, `transparent_only` (OIT path), `stress_5k` (exceeds the old 1024 instance limit — proves the growable buffer works) |
| `resize_and_reload` | Scripted resize 256→64→512 mid-run with no validation errors and no leaks; load scene A → scene B → scene A, asserting `AssetCache` live counts return to baseline (a direct leak test for the `shared_ptr`/`weak_ptr` cache) |
| `validation_clean_run` | Runs with `ValidationPolicy::FailFast` and sync-validation + best-practices enabled (both already enabled in `CreateInstance`), so the first validation error aborts with a message |

### 15.4 `golden` — image comparison

Deliberately **not** in the default CI gate initially; software rasterisers and drivers
differ in ULP-level ways, and a flaky visual test destroys trust in the suite faster than
having no visual test at all.

Plan:
1. Land it as a **non-blocking** job that uploads diffs as artefacts.
2. Compare with a perceptual metric and a tolerance (mean absolute error over a
   tonemapped LDR image; a small percentage of outlier pixels allowed), never bit-exact.
3. Pin: fixed camera via `RunSpec::Camera`, fixed extent, fixed frame index, deterministic
   mode, FIFO present, serial jobs.
4. Reference images are per-driver-class (`lavapipe/`, `nvidia/`, …), committed as small
   PNGs (256×256).
5. `--update-golden` regenerates; the diff is reviewed in the PR like any other change.
6. Promote to blocking only once it has been green for a few weeks.

Highest-value goldens: opaque PBR lit sphere, OIT overlapping quads, cloud dispatch,
tonemap/composite output, skybox once implemented.

### 15.5 `perf` — counter budgets, not wall clock

CI wall-clock timings are too noisy to gate on. Gate on **counters**, which are exact:

```cpp
REQUIRE(report.DrawCalls        <= 3);        // 5k cubes, 1 material → must batch
REQUIRE(report.DescriptorBinds  <= 2);        // bindless: not per-batch
REQUIRE(report.Instances        == 5000);
REQUIRE(report.InstancesCulled  >= 4000);    // camera outside the cluster
REQUIRE(report.TransientArenaPeakBytes < 4 * 1024 * 1024);
REQUIRE(report.HeapAllocationsDuringSteadyState == 0);   // ◄── the big one
```

That last assertion is worth its own note. A steady-state allocation counter (an
`operator new` counting hook active only in tests, or a policy allocator on the hot paths)
turns "the renderer allocates every frame" from a code-review observation into a build
failure. Given that `ModelManager::GenerateBatches` currently heap-allocates once per model
per frame, this is the assertion that will keep the DOD work from regressing.

Wall clock and GPU pass timings are still *recorded* into `RunReport` and can be tracked
over time (upload the JSON as an artefact; chart it later), just not asserted.

### 15.6 Sanitizer runtime job

Currently ASan/UBSan is **built but never executed**. That is most of the cost with none of
the benefit. The single highest-value addition to CI is running
`scene_launch` and the unit suite under ASan+UBSan on Linux with lavapipe.

The companion document lists several P0s (uninitialised `float opacity`, wrong memory
property flags, pointers invalidated by `resize`, unchecked `mNormals` dereference). A
sanitized headless scene launch would likely surface several of them automatically on the
first run.

## 16. Test harness components

These live in `tests/support/` and are the reason the above is cheap to write.

### 16.1 `RhiTestFixture`
Creates one headless `Device` per test binary (device creation on lavapipe is slow; do it
once), exposes `Device&`, `UploadContext&`, a `SerialJobSystem`, and a scratch `Arena`.
Reports "SKIPPED: no Vulkan device available" rather than failing when there is no ICD, but
**fails** if a device exists and lacks a hard-required feature.

### 16.2 `ValidationGuard`
RAII. Installs a counting/aborting validation callback for the scope of a test and asserts
zero errors on destruction, printing every captured message. Every `gpu` test gets one.
This is what makes "it rendered" mean "it rendered correctly".

### 16.3 `LayoutTracker`
Consumes a recorded command stream, models per-subresource image layout and last
stage/access, and flags:
- use in a layout that doesn't match the declared one,
- a layout transition with insufficient src/dst stage or access masks,
- a read-after-write with no intervening barrier,
- a missing `layerCount`/`mipCount` (i.e. transitioning 1 of 6 cubemap faces).

A ~250-line class that provides a meaningful subset of sync-validation's value on **every
platform, with no GPU, in milliseconds**.

### 16.4 `RecordingSink`
`ICommandSink` implementation that appends `{BindPipeline, BindDescriptorSet, SetViewport,
Draw, Barrier, BeginRendering, …}` structs to a vector, with helpers like
`CountOf<DrawCmd>()` and `FindFirst<BarrierCmd>(pred)`. Enables exact assertions on
encoding without a driver.

### 16.5 `SnapshotBuilder`
Fluent construction of `FrameSnapshot`s:
```cpp
auto snap = SnapshotBuilder(arena)
              .View(Camera::LookAt({0,0,5}, {0,0,0}), 1.f, 0.1f, 100.f)
              .AddInstances(5000, MeshHandle{0}, MaterialHandle{0}, GridPositions(5000))
              .AddDirLight({1,1,1}, 3.f, {0,-1,0})
              .Build();
```
This is what makes renderer unit tests possible at all.

### 16.6 `ImageCompare`
Perceptual diff, per-channel statistics, and on failure: writes `actual.png`,
`expected.png` and an amplified `diff.png` into the CTest output directory so CI can upload
them as artefacts.

### 16.7 `TestPaths`
Locates `tests/data` via a compile-time-injected absolute path
(`target_compile_definitions(... TEST_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/data")`), so
tests are CWD-independent — the test-side counterpart of the `Paths` content root.

### 16.8 CTest wiring

```cmake
# cmake/Testing.cmake
include(Catch)
function(engine_test name)
  cmake_parse_arguments(T "" "LABEL;TIMEOUT" "SOURCES;DEPS" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE Catch2::Catch2WithMain TestSupport ${T_DEPS})
  catch_discover_tests(${name}
    PROPERTIES LABELS "${T_LABEL}"
               TIMEOUT ${T_TIMEOUT}
               ENVIRONMENT "VK_LAYER_PATH=$ENV{VULKAN_SDK}/bin")
endfunction()
```

Then: `ctest --preset ci-fast -L "unit|contract"` and
`ctest --preset ci-gpu -L "gpu" --output-on-failure`.

Every test gets a **timeout**. A headless render test that hangs must fail, not block the
runner for six hours.

## 17. CI plan

Keep the existing 7 build jobs (they are genuinely useful cross-platform coverage) and add
five. Test presets are added to `CMakePresets.json` alongside the existing ones.

```yaml
jobs:
  build:              # unchanged: 2 OS × Debug/Release/ASan (6 jobs)

  unit-tests:
    strategy: { matrix: { os: [ubuntu-latest, windows-latest] } }
    steps:
      - configure/build preset  (…-debug-…, -DENGINE_BUILD_TESTS=ON)
      - run: ctest --preset ... -L "unit|contract" --output-on-failure
    # Fast (< 1 min after cache). Blocking.

  headless-gpu:
    runs-on: ubuntu-latest
    steps:
      - run: sudo apt-get install -y mesa-vulkan-drivers vulkan-tools
      - run: vulkaninfo --summary       # diagnosable driver info in the log
      - configure/build ninja-debug-linux -DENGINE_BUILD_TESTS=ON
      - run: ctest -L gpu --output-on-failure --timeout 300
        env:
          VK_ICD_FILENAMES: /usr/share/vulkan/icd.d/lvp_icd.x86_64.json
      - uses: actions/upload-artifact@v4      # if: failure()
        with: { name: gpu-reports, path: build/**/run_report*.json }
    # Blocking.

  headless-asan:
    runs-on: ubuntu-latest
    steps:
      - install mesa-vulkan-drivers
      - configure/build ninja-asan-linux -DENGINE_BUILD_TESTS=ON
      - run: ctest -L "unit|gpu" --output-on-failure --timeout 900
        env:
          VK_ICD_FILENAMES: /usr/share/vulkan/icd.d/lvp_icd.x86_64.json
          ASAN_OPTIONS: detect_leaks=1:abort_on_error=1:strict_string_checks=1
          UBSAN_OPTIONS: print_stacktrace=1:halt_on_error=1
          LSAN_OPTIONS: suppressions=ci/lsan.supp    # driver-internal leaks
    # Highest-value job in the matrix. Blocking once green.

  golden-images:
    runs-on: ubuntu-latest
    continue-on-error: true            # non-blocking until proven stable
    steps: [ …, ctest -L golden, upload actual/expected/diff PNGs ]

  static-analysis:
    runs-on: ubuntu-latest
    steps:
      - clang-format --dry-run -Werror on all sources
      - clang-tidy on changed files (compile_commands.json from the Ninja preset)
      - spirv-val on every compiled .spv
      - build the HeaderSelfContainment target (no PCH) — enforces §6 blocker 2
```

Additional CI notes:

- **Shader validation** (`spirv-val`, and `slangc` warnings-as-errors) belongs in the build
  job, not a test — it is a compile-time property.
- **Artefact the `RunReport` JSON on every gpu run**, pass or fail. Historical counters are
  how you notice "draw calls tripled" three weeks before anyone opens a profiler.
- **Concurrency group + cancel-in-progress** on PRs to avoid burning runner minutes.
- Total added wall-clock: unit ≈ 1 min, gpu ≈ 3–5 min on lavapipe, asan ≈ 8–12 min. Run
  `unit-tests` on every push; `headless-asan` on PRs and `main` only if minutes are tight.

## 18. Migration roadmap

> **This section describes the six phases at a strategic level — what each phase achieves and
> why it is sequenced where it is. For the actual implementation order, use
> [Part IV](#part-iv--incremental-work-order), which breaks these phases into 76 steps that
> each end in a runnable, verifiable build.** Read §18 for the *why*, Part IV for the *how*.

Each phase is independently shippable and ends with CI green. No phase requires a
"stop-the-world" rewrite; `main.cpp` shrinks monotonically.

### Phase 0 — Foundations (2–3 days)
- `.clang-format`, `.clang-tidy`, `.editorconfig`, `.clangd`.
- `cmake/` helpers: `EngineModule.cmake`, `Warnings.cmake`, `Sanitizers.cmake` (moved after
  `project()` with the MSVC branch), `Shaders.cmake`.
- Add `catch2` to `vcpkg.json`; add `ENGINE_BUILD_TESTS` option; add a trivial test and the
  `unit-tests` CI job so the pipeline exists before there is anything in it.
- Add the `HeaderSelfContainment` object library (no PCH) — it will fail; that failure list
  *is* the Phase 1 work list.

**Exit:** CI runs `ctest` and it passes with one placeholder test.

### Phase 1 — Make it a library (3–5 days)
- Create `engine/core` and `engine/platform`. Move `Log.h`, `Timer.h`, `ThreadPool`,
  `SwapbackArray`, `MyMacros`, and the SDL init/window/shutdown functions.
- Make every moved header self-contained.
- `main.cpp` links `Core` + `Platform`.
- First real unit tests: `SwapbackArray`, `ThreadPool` (including the
  `hardware_concurrency() <= 1` case), `Log` formatting.

**Exit:** two targets exist; `HeaderSelfContainment` passes for `Core`/`Platform`.

### Phase 2 — Extract the RHI (1–2 weeks)
- `engine/rhi`: `Device` (instance + debug messenger + physical + logical + queues +
  features, with the present/no-present split), `Allocator`, `Buffer`, `Image`, `Sampler`,
  plus the existing `Barrier.h`, `PipelineBuilder`, `ComputePipelineBuilder`, and the
  free functions from `Utility.h`.
- `IPresentTarget` + `SwapchainTarget` + `OffscreenTarget`.
- `DescriptorAllocator` (growable) — removes the 100-material ceiling.
- `UploadContext` — removes per-resource `waitIdle`.
- `PipelineCache` on disk.
- `Diagnostics`: validation callback → counter + policy.
- Enumerate graphics/compute/transfer queue families properly.

**Exit:** `device_creation` and `upload_roundtrip` gpu tests pass headless on lavapipe. This
is the moment headless becomes real, and it is the highest-leverage phase in the plan.

### Phase 3 — Engine shell, `RunSpec`/`RunReport` (1 week)
- `engine/engine`: `Engine` owning injected subsystems, the frame loop, `EngineConfig`,
  `RunSpec`, `RunReport`, `MaxFrames`/timeout.
- `Paths` content root; move `models/`, `textures/`, `scenes/` under `content/`; shader
  output to `content/shaders/`.
- `HeadlessPlatform` + scripted input.
- `apps/hikariengine` and `apps/headless`; `main.cpp` drops to ~40 lines.
- Delete the singletons in favour of constructor injection (mechanical but touches many
  files — do it here, while the call sites are already moving).

**Exit:** `HikariEngineHeadless --scene content/scenes/test_scene.map --frames 5 --headless`
runs and prints a `RunReport`. Add `scene_launch` to CI. **The user's stated CI goal is met
at the end of this phase**, before any DOD work.

### Phase 4 — Renderer extraction + frame graph (2–3 weeks)
- `engine/render`: `FrameSnapshot`, `View`, `FrameResources`, `Renderer`.
- Frame graph; convert the 7 recorders into `Pass` classes one at a time, keeping the app
  running after each.
- `ShaderTypes.h` shared with Slang + `static_assert`s.
- `surface.slangh` de-duplication of the two surface shaders.
- Move ImGui into `engine/editor` + `ImGuiPass`.
- `contract` tests: graph ordering, barrier legality, encoding counts.

**Exit:** adding a pass is one file + one `graph.Add<>()` line. Resize is one recompile.

### Phase 5 — DOD scene and render data (2–3 weeks)
- `Handle`/`HandlePool`; convert `Mesh*`/`Material*` to handles (also removes the
  pointer-value-dependent sort order).
- `World` sparse-set ECS; components; `TransformSystem` with depth-sorted single-pass
  world matrices and dirty tracking.
- `Extract` → `FrameSnapshot`; `Culling`; radix-sorted `DrawListBuilder`; `Arena`.
- Cached normal matrices; `Mat3x4` transforms; instance data to an SSOB indexed by
  `SV_InstanceID`, freeing 8 vertex attribute slots.
- Port `XmlParser` to `SceneReader`/`SceneWriter` over `SceneDesc` (keeps the `.map` format;
  makes round-trip unit-testable).
- The large `unit` test suite from §15.1 lands here.

**Exit:** steady-state heap allocations per frame == 0 (asserted). Static scenes cost ~0
CPU to re-batch.

### Phase 6 — Scalability features (ongoing)
- Bindless textures + material SSBO → removes per-batch binds and the material/texture
  ceilings.
- Mega vertex/index buffer → one bind per frame.
- Async asset loading on the job system + transfer queue.
- Mipmaps, reverse-Z, depth prepass.
- Indirect draws, then GPU culling.
- Render thread (the snapshot boundary from Phase 5 is the prerequisite).
- Golden images and perf budgets promoted to blocking.

### Sequencing rationale

The order is chosen so that **testing infrastructure lands before the risky refactors**.
Phases 2–3 give a headless, assertable engine while the code is still recognisably the
current code. Phases 4–6 are then large refactors performed *with* a safety net, rather
than the usual "big rewrite, hope it still looks right". Doing DOD first would mean
rewriting the hot path with no way to prove equivalence.

## 19. Conventions & tooling

The codebase is already consistent (4-space, Allman, 80 columns, `m_`/`s_`/`g_`/`b`
prefixes, `p` for raw pointers). Codify rather than change it:

- **`.clang-format`** matching the existing style; `format` and `format-check` targets;
  `clang-format --dry-run -Werror` in the `static-analysis` job. Fixes the tab/space drift
  in ~8 files.
- **`.clang-tidy`**: start narrow (`bugprone-*`, `performance-*`,
  `cppcoreguidelines-pro-type-*`, `modernize-use-nullptr`) so it is adoptable; expand later.
  Run on changed files only.
- **`.editorconfig`**: trailing whitespace, final newline, UTF-8, LF.
- **`.clangd`**: `CompileFlags: { Remove: [/Yu*, /Fp*, /Yc*, /FI*] }` so clangd works with
  the `cl.exe` compilation database.
- **Naming additions** for the new concepts: interfaces `IThing`, POD GPU structs `GpuThing`
  in `namespace shader`, systems `ThingSystem`, passes `ThingPass`, handles `ThingHandle`.
- **One class per file**, filename == class name, keep the current convention.
- **Assertions**: `ENGINE_ASSERT` (debug, breaks) vs `ENGINE_VERIFY` (always evaluates) vs
  `ENGINE_CHECK` (returns an error). Today `assert` is used for things that should be
  runtime checks (`Mesh.cpp`'s `mNormals` dereference), which means Release builds crash
  where Debug builds abort.
- **Errors**: exceptions for unrecoverable init failures (as today, and it works well);
  `std::expected`-style returns for asset loading and parsing, so a bad file logs and skips
  instead of unwinding through the frame loop.
- **`docs/testing.md`**: how to add a test at each layer, how to run each label locally,
  how to update goldens. Undocumented test infrastructure decays.

## 20. Risks and open decisions

| # | Risk / decision | Assessment |
|---|---|---|
| 1 | **Scope.** This is 8–12 weeks of part-time work. | Mitigated by phase ordering: value lands at the end of each phase, and the CI goal is met at Phase 3, not Phase 6. |
| 2 | **lavapipe fidelity.** It is not a real driver; some bugs won't reproduce, some lavapipe quirks aren't real bugs. | Treat it as a *smoke and correctness* device, not a conformance oracle. Golden images per-driver-class. Add a self-hosted GPU runner later if it becomes worthwhile. |
| 3 | **ECS rewrite risk.** Replacing `Entity`/`Component` touches scene, serialization, editor and renderer. | Sequenced *after* headless tests exist (Phase 5, not Phase 1). Sparse sets over archetypes to limit complexity. Keep `SceneDesc`/`.map` stable so scenes don't need re-authoring. |
| 4 | **Frame graph complexity.** Easy to over-engineer into 3k lines. | Constrain: no async compute in v1, no aliasing in v1, no multi-queue in v1. Add only when a pass needs it. Target < 900 lines. |
| 5 | **Bindless portability.** Older and mobile-class drivers have weaker descriptor-indexing support. | **Resolved by deferral** — see D14 in `backend_readiness_plan.md`. This row and D7 pointed in opposite directions: D7 deferred the binding model *because* bindless would replace it, while this row's mitigation ("keep a non-bindless fallback behind a device-capability flag; the `gpu` suite runs both") required building that model regardless. Bindless now waits until after the D3D12 backend and the binding model is neutralised in Stage 7.5, so there is no bindless path to fall back from and no flag to carry. Revisit this row when step 70 is scheduled. Note also that the assurance below it was overstated: only `descriptorBindingPartiallyBound` is enabled (`VulkanDevice.cpp:980`), and it serves partially-bound material sets, not bindless — which additionally needs `runtimeDescriptorArray` and `shaderSampledImageArrayNonUniformIndexing` at minimum. |
| 6 | **Build time** with 9 targets and no PCH sharing. | Per-module PCHs, `ccache` (already wired), unity builds for the leaf modules if needed. Note that splitting *reduces* rebuild cost: touching a pass no longer rebuilds a 2,453-line TU. |
| 7 | **Golden-image flakiness eroding trust.** | Non-blocking until stable; perceptual tolerance; determinism fixes (esp. the pointer-value sort order) land first. |
| 8 | **Catch2 vs GoogleTest.** | Recommendation: Catch2 v3. Decide once, in Phase 0. |
| 9 | **`SceneGraph`/`.map` format.** Currently no versioning. | Add a `version` attribute in Phase 3 (cheap now, painful later). |
| 10 | **Editor coupling.** ImGui currently draws directly from live scene objects. | `EditorLayer` mutates the `World` through commands; it must not be on the render data path. Keeps headless free of ImGui entirely. |
| 11 | **Content root move.** Breaks existing `.map` paths and the VS debugger working directory. | Do it in one commit with a path-rewrite pass over the 3 scene files; `Paths` keeps a source-dir fallback so nothing breaks locally. |

---

# Part IV — Incremental work order

## 23. How to use Part IV

Part II describes destinations. Part IV describes **landing sites**: 76 numbered steps, each
of which ends with a **compiling, running application** that you can confirm before starting
the next one. Nothing here requires holding a broken build across multiple sessions.

Each step lists:

| Field | Meaning |
|---|---|
| **Do** | The change. |
| **Verify** | The specific, mechanical check that it worked. If a step has no meaningful verification, it is too big — split it. |
| **Size** | XS < 1h · S 1–3h · M ½–1 day · L 2–4 days · XL 1–2 weeks |
| **Needs** | Steps that must be done first. "—" means it can be done at any time, starting today. |

### The one rule that makes this work

> **Stage 0 comes first, even though it is not architecture.**

Steps 1–6 add `--frames N`, `--screenshot`, a fixed camera, a fixed timestep and a validation
error counter **to the existing `main.cpp`**, with no restructuring at all. They take about a
day. After them, "did my refactor change anything?" is answered by:

```
HikariEngine --scene content/scenes/test_scene.map --camera-preset 0 \
          --fixed-dt --frames 30 --screenshot after.png --report after.json
```

…and comparing `after.png` / `after.json` against the baseline you captured before the
change. That converts every one of the remaining 70 steps from "build it and squint at it"
into a mechanical pass/fail.

Without Stage 0, steps 15–56 are all verified by eye. That is where refactors silently
introduce a wrong barrier, a lost `setCullMode`, a dropped light or a subtly different
matrix — the exact bugs that are cheapest to catch immediately and most expensive to catch
three weeks later. **Do not skip Stage 0.** It is the highest return-on-time in this entire
document.

### Reading the stage structure

| Stage | Steps | Ends with | Cumulative size |
|---|---|---|---|
| 0 | 1–6 | Objective before/after regression checking | ~1 day |
| 1 | 7–11 | Formatting, sanitizers, CTest pipeline alive | ~1 day |
| 2 | 12–14 | Every header compiles standalone | ~2 days |
| 3 | 15–19 | `Core` static library + first real unit tests | ~3 days |
| 4 | 20–23 | `Platform` library, CWD-independent assets | ~3 days |
| 5 | 24–34 | `RHI` library, batched uploads, growable descriptors | ~2 weeks |
| 6 | 35–40 | **Headless device + offscreen rendering + readback** | ~1 week |
| 7 | 41–47 | **`--headless --frames N` in CI. Stated goal met.** | ~1.5 weeks |
| 8 | 48–56 | Passes as classes, frame graph, automatic barriers | ~2.5 weeks |
| 9 | 57–68 | Handles, snapshots, radix sort, culling, ECS | ~3 weeks |
| 10 | 69–76 | Bindless, mega-buffers, async loading, indirect | open-ended |

Stage **7.5** is inserted between 7 and 8 and is not in that table because it is not part of
Part IV: `docs/backend_readiness_plan.md` carries it as B1–B6. It neutralises the RHI's frame
API — submission, rendering scope, binding, pipelines, draw and dispatch — which Stage 5 left
undone and which a D3D12 backend needs before it can be written. It also reorders this table's
Stage 8 and defers its step 70.

**The stated CI goal — "launch a scene headless and assert on it" — is complete at step 47.**
Steps 48–76 are performance and scalability, and they are much safer to attempt once 47 is
done, because from then on every change is regression-tested by CI rather than by hand. The
**backend goal** — "a D3D12 backend is a matter of implementing interfaces rather than
designing them" — is complete at B6.

---

## Stage 0 — Verification harness (steps 1–6)

Purpose: make every later step objectively checkable. All changes are additive and local to
`main.cpp` plus a tiny helper; no files move.

### 1. Minimal command-line parsing
- **Do:** Add a ~60-line `ParseArgs(argc, argv)` returning a plain `Options` struct
  (`ScenePath`, `Frames`, `bFixedDt`, `ScreenshotPath`, `ReportPath`, `CameraPreset`,
  `bHeadless` — accepted but unused for now). Pass it into `App`. Add `--help`.
  `main()` currently takes no arguments at all (`int main()`), so this also fixes that.
- **Verify:** `HikariEngine --help` prints the options and exits 0. `HikariEngine` with no
  arguments behaves exactly as before.
- **Size:** S · **Needs:** —

### 2. Frame limit and clean exit
- **Do:** In `App::Run`, increment a frame counter and set `g_bShouldClose` when
  `Options::Frames != 0 && count >= Frames`.
- **Verify:** `HikariEngine --frames 30` renders 30 frames, exits with code 0, and prints no
  validation errors. `echo $LASTEXITCODE` / `echo %ERRORLEVEL%` is 0.
- **Size:** XS · **Needs:** 1

### 3. Load a scene from the command line
- **Do:** If `Options::ScenePath` is non-empty, call `XmlParser::LoadScene` during `Init()`
  instead of requiring the ImGui file dialog. This is the same code path the dialog already
  uses at `main.cpp:691`.
- **Verify:** `HikariEngine --scene scenes/sponza_scene.map --frames 30` shows Sponza without
  any UI interaction.
- **Size:** XS · **Needs:** 1

### 4. Deterministic time and camera
- **Do:** When `--fixed-dt` is set, replace the `high_resolution_clock` delta with a constant
  `1/60` and derive `m_RunTime` as `frameIndex / 60.f`. Add 2–3 hardcoded camera presets
  (position + rotation) selected by `--camera-preset N`, applied after `Init()` and with
  input ignored while a preset is active.
- **Why:** `m_RunTime` feeds `GlobalBuffer.Time`, which drives cloud wind animation. Without
  this, two runs never produce the same image and screenshot comparison is worthless.
- **Verify:** Run the same command twice; the two `--screenshot` outputs (after step 5) are
  **byte-identical**. Until step 5 exists, verify that logged camera position at frame 30 is
  identical across two runs.
- **Size:** S · **Needs:** 1, 2

### 5. Screenshot capture
- **Do:** Add `vk::ImageUsageFlagBits::eTransferSrc` to the swapchain `imageUsage`
  (`main.cpp:1079`). Add `WriteScreenshot(path)`: `waitIdle`, single-time command buffer
  transitioning the last presented swapchain image `ePresentSrcKHR → eTransferSrcOptimal`,
  `copyImageToBuffer` into a host-visible buffer, transition back, then write PNG with
  `stb_image_write.h`. Call it on the final frame when `--screenshot` is set.
  **stb is already a dependency** (`find_package(Stb REQUIRED)`), so no new package is needed.
  Remember the swapchain format is BGRA (`eB8G8R8A8Unorm`) — swizzle to RGBA before writing.
- **Verify:** `HikariEngine --scene scenes/test_scene.map --camera-preset 0 --fixed-dt --frames 30 --screenshot base.png`
  produces a PNG that looks like the app. Run twice → the files are byte-identical.
- **Size:** M · **Needs:** 1, 2, 4

### 6. Validation counter and frame counters
- **Do:** In `DebugCallback`, increment `std::atomic<uint64_t>` error/warning counters. Count
  draw calls, batches and instances in the two record loops. Add `--report <path>` writing a
  small JSON blob: frames, validation errors/warnings, draw calls, batches, instances, mean
  and P99 CPU frame ms. Make the process exit non-zero if validation errors > 0 **and**
  `--strict-validation` is passed.
- **Verify:** `--report r.json` produces plausible numbers for `test_scene.map`. Deliberately
  break something (e.g. comment out one `RecordImageBarrier`) and confirm the error count
  goes above zero and `--strict-validation` returns a non-zero exit code. **Then revert.**
- **Size:** M · **Needs:** 1, 2

> ### ✅ Checkpoint: capture your baseline
> ```
> HikariEngine --scene scenes/test_scene.map  --camera-preset 0 --fixed-dt --frames 30 \
>           --screenshot baseline_test.png  --report baseline_test.json
> HikariEngine --scene scenes/sponza_scene.map --camera-preset 1 --fixed-dt --frames 30 \
>           --screenshot baseline_sponza.png --report baseline_sponza.json
> ```
> Commit these four files. **From here on, "Verify: output unchanged" means these two
> commands still produce byte-identical PNGs and equal counters in the JSON.** Where a step
> legitimately changes output, the step says so explicitly and you re-baseline.

---

## Stage 1 — Build hygiene (steps 7–11)

Independent of everything else. Can be done before, during or after Stage 0 — but the
`clang-format` pass (step 8) produces a huge diff, so land it while the tree is otherwise
quiet.

### 7. `.clang-format`, `.editorconfig`, `.clangd`
- **Do:** Author a `.clang-format` matching the existing style (4-space, Allman, 80 columns).
  `.editorconfig` for trailing whitespace / final newline / LF. `.clangd` with
  `CompileFlags: { Remove: [/Yu*, /Fp*, /Yc*, /FI*] }` so clangd tolerates the MSVC PCH flags
  in `compile_commands.json`.
- **Verify:** `clang-format --dry-run src/Camera.cpp` reports only the known tab/space drift,
  not wholesale reformatting — i.e. the config genuinely matches your style.
- **Size:** S · **Needs:** —

### 8. Apply formatting; add `format` / `format-check` targets
- **Do:** Run `clang-format -i` over all of `src/`. Add `format` and `format-check` CMake
  targets. Commit the reformat **alone**, with no other changes, so `git blame` damage is
  isolated to one reviewable commit.
- **Verify:** `cmake --build . --target format-check` passes. Output unchanged (Stage 0
  baseline).
- **Size:** XS · **Needs:** 7

### 9. Fix `ENABLE_SANITIZERS`
- **Do:** Move the block from before `project()` (currently `CMakeLists.txt:25-31`, where
  `MSVC` is not yet defined) to after it, and branch: `/fsanitize=address /Zi` +
  `/INCREMENTAL:NO` for MSVC, `-fsanitize=address,undefined …` otherwise.
- **Verify:** `cmake --workflow --preset ninja-asan-linux` builds **and the binary actually
  runs**: `HikariEngine --frames 30`. Expect it to report real bugs — that is the point. Triage
  them via `suggested_work.md`; do not fix them in this step.
- **Size:** S · **Needs:** —

### 10. CMake helper modules
- **Do:** Create `cmake/Warnings.cmake` (`engine_set_warnings(target)`) and
  `cmake/EngineModule.cmake` (`engine_module(...)`, unused for now). Route the existing
  `HikariEngine` warning flags through `engine_set_warnings`. Remove the trailing space in
  `$<$<CONFIG:Release>:/DEBUG >`.
- **Verify:** Build with zero new warnings; `ninja -t commands` shows the same flags as
  before.
- **Size:** S · **Needs:** —

### 11. Catch2, `ENGINE_BUILD_TESTS`, and the CI test job
- **Do:** Add `"catch2"` to `vcpkg.json`. Add `option(ENGINE_BUILD_TESTS "" OFF)`,
  `enable_testing()`, `cmake/Testing.cmake` with an `engine_test()` helper using
  `catch_discover_tests`. Add `tests/unit/placeholder_test.cpp` containing one trivial
  assertion. Add a `unit-tests` CI job running `ctest -L unit`.
- **Verify:** `ctest --output-on-failure` reports `1/1 tests passed` locally and the new CI
  job is green.
- **Why now:** The pipeline must exist before there is anything to put in it, otherwise the
  first real test also has to debug CI.
- **Size:** M · **Needs:** —

---

## Stage 2 — Header self-containment (steps 12–14)

This is the true prerequisite for *any* second target. Currently
`target_precompile_headers(HikariEngine PRIVATE src/pch.h)` force-includes `pch.h` into every
TU, so headers compile only inside this one target.

### 12. Add the `HeaderSelfContainment` check target
- **Do:** Add an `OBJECT` library, `EXCLUDE_FROM_ALL`, that compiles every `src/*.h` as a TU
  with `LANGUAGE CXX`, links the same dependencies, and **deliberately has no
  `target_precompile_headers`**.
- **Verify:** `cmake --build . --target HeaderSelfContainment` **fails**, and the error list
  is your step-13 worklist. Save it. `HikariEngine` still builds.
- **Size:** S · **Needs:** —

### 13. Make headers self-contained, one at a time
- **Do:** Work down the step-12 list adding the missing includes. Known offenders:
  `Utility.h` (`<algorithm>`, `<ranges>`, `<format>`, `SDL3/SDL.h`, `vulkan/vulkan_raii.hpp`),
  `SwapbackArray.h` (`<ranges>` for `std::ranges::find`), `Mesh.h` (`<vector>`),
  `Entity.h` (`<memory>`, `<vector>`, `<string>`), `Material.h` / `Texture.h` /
  `FrameData.h` / `Barrier.h` (all use `vk::` with no Vulkan include),
  `ThreadPool.h` (`<future>`, `<queue>`, `<mutex>`, `<condition_variable>`, `<functional>`,
  `<thread>`, `<vector>`, `<cstdint>`), `Log.h` (`<string_view>`, `<format>`, `<cstdio>`,
  `<cstdint>`), `InstanceData.h` / `Vertex.h` (`<array>`).
- **Verify:** After each header: `HeaderSelfContainment` has one fewer error and `HikariEngine`
  still builds and produces unchanged output. This step is ~25 independently verifiable
  micro-commits.
- **Size:** M · **Needs:** 12

### 14. Turn `HeaderSelfContainment` on in CI
- **Do:** Build it as part of the default target set (or add it to the `static-analysis` job).
- **Verify:** CI green. From now on, a header that depends on the PCH cannot be merged.
- **Size:** XS · **Needs:** 13

---

## Stage 3 — Core library (steps 15–19)

First real library. Deliberately starts with the leaf-most files so the blast radius is
minimal.

### 15. Create `engine/core` with the zero-dependency files
- **Do:** `engine_module(Core ...)` with `MyMacros.h`, `Timer.h`, `SwapbackArray.h`.
  Public headers under `engine/core/include/core/`. Change include sites to
  `#include <core/Timer.h>`. `HikariEngine` links `Core`.
- **Verify:** Two targets build; output unchanged. `Core` does **not** link Vulkan or SDL —
  confirm by temporarily adding `#include <vulkan/vulkan.hpp>` to a Core `.cpp` and checking
  that it fails to compile.
- **Size:** S · **Needs:** 13

### 16. Split `Log.h`
- **Do:** `LogCategory`, `LogSeverity`, `LogMsg` and the ANSI helpers move to
  `core/Log.h`. `ShowMessageBox` uses `SDL_ShowSimpleMessageBox`, so it stays behind in
  `src/` for now (it becomes `Platform`'s in step 20). `XmlParser.cpp` and `main.cpp` are the
  two callers.
- **Verify:** Output unchanged; log formatting and colours identical. `Core` still has no SDL
  dependency.
- **Size:** S · **Needs:** 15

### 17. First real unit tests
- **Do:** `tests/unit/core/` covering `SwapbackArray` (`RemoveAt` on last element, `Erase`
  of a missing value → throws, iteration after removal) and `Log` (severity filtering,
  format-string forwarding). Add `tests/support/TestPaths.h` with
  `TEST_DATA_DIR` injected via `target_compile_definitions`.
- **Verify:** `ctest -L unit` runs ~15 assertions and passes on all three platforms in CI.
  Delete the placeholder test from step 11.
- **Size:** M · **Needs:** 11, 15

### 18. `IJobSystem` + `SerialJobSystem` + `WorkStealingJobSystem`
- **Do:** Move `ThreadPool` to `core/`. Introduce `IJobSystem` with `Submit`, `Wait`,
  `ParallelFor`, `WorkerCount`. `WorkStealingJobSystem` wraps the existing implementation;
  fix the `hardware_concurrency()` underflow via `WorkerCount = max(1u, hc - 1u)`.
  Add `SerialJobSystem` running everything inline. Keep the `ThreadPool::Get()` singleton as
  a thin shim for now so `main.cpp` doesn't have to change yet.
- **Verify:** Output unchanged. Unit tests: `ParallelFor` visits `[0,count)` exactly once for
  both implementations; `count == 0` and `grain > count` are no-ops; a job that throws does
  not deadlock the pool.
- **Size:** M · **Needs:** 15, 17

### 19. Inject the job system into `App`
- **Do:** `App` takes an `IJobSystem&`. `main()` constructs `WorkStealingJobSystem` (or
  `SerialJobSystem` when `--jobs 0`). Delete the `ThreadPool::Get()` shim and the
  `MULTITHREADED_COMMAND_RECORDING` macro — `--jobs 0` replaces it.
- **Verify:** `--jobs 0` and default both produce **identical screenshots**. If they don't,
  you have found a real race in command recording, which is exactly what this step is for.
- **Size:** S · **Needs:** 18

---

## Stage 4 — Platform library (steps 20–23)

### 20. Create `engine/platform` with `SdlPlatform`
- **Do:** `IPlatform` interface (per §10.1). Move `InitSDL`, `CreateSDLWindow`,
  `ShutdownSDL`, `ShowMessageBox` and `Utility.h`'s `ChooseSwapchainExtent` window query into
  `SdlPlatform`. `App` holds `IPlatform&` instead of `SDL_Window*`. Event polling stays in
  `App` for now — only *creation* and *teardown* move.
- **Verify:** Output unchanged. Window still starts hidden then shows, is borderless and
  resizable, and mouse capture on Escape still works.
- **Size:** M · **Needs:** 15, 16

### 21. `platform/FileSystem.h` and `Paths`
- **Do:** Move `Utility.h`'s `ReadFile` into `platform/FileSystem.h`. Add `Paths` resolving a
  content root in priority order: `--content` flag → `HIKARI_CONTENT` env var →
  `<exe dir>/content` → `<source dir>/content`. Add `Paths::Content("shaders/opaque.spv")`.
- **Verify:** Unit-test the resolution priority with a temp directory. App still runs.
- **Size:** M · **Needs:** 20

### 22. Move assets under `content/` and route all paths through `Paths`
- **Do:** `git mv models content/models`, same for `textures` and `scenes`. Point shader
  output at `content/shaders/`. Replace every hardcoded relative path: `"shaders/*.spv"`
  (3 pipeline creators + `CloudSystem`), `"textures/skybox/*.jpg"` (`main.cpp:306-312`),
  `"scenes/"` (2 ImGui dialog configs), and the model paths inside the 3 `.map` files.
- **Verify:** **`cd build/ninja-debug-windows && ./HikariEngine --frames 30` works.** This is
  the key check — it fails today and is a hard blocker for any CI test binary. Also confirm
  the VS debugger launch still works.
- **Size:** M · **Needs:** 21

### 23. Route `RunSpec`-ish options through `platform/CommandLine.h`
- **Do:** Move step 1's ad-hoc parser into `Platform` as a small reusable parser; keep the
  `Options` struct where it is for now (it becomes `RunSpec` at step 41).
- **Verify:** All flags from Stage 0 still work; `--help` output unchanged; unit tests for
  the parser (unknown flag → error, missing value → error, `--flag=value` and `--flag value`
  both accepted).
- **Size:** S · **Needs:** 20

---

## Stage 5 — RHI extraction (steps 24–34)

The longest stage. Each step keeps a running app. Steps 27–31 deliver real, measurable wins
independently of the headless goal, so this stage is worth doing even if you stop here.

### 24. Move the RHI leaf types
- **Do:** `engine/rhi` containing `AllocatedBuffer`, `AllocatedImage`, `VulkanAllocator`,
  `VMAImpl.cpp`, `Barrier.h`, `Texture`, `Cubemap`, `PipelineBuilder`,
  `ComputePipelineBuilder`. No logic changes.
- **Verify:** Output unchanged.
- **Size:** M · **Needs:** 20

### 25. Dissolve `Utility.h`
- **Do:** Split its 339 lines by concern: `rhi/DebugNames.h` (`SetVkDebugName`),
  `rhi/Buffer.h` (`CreateBuffer`, `CopyBuffer`, `CreateStagedBuffer`),
  `rhi/Image.h` (`CreateImage`, `CreateImageView`, `CopyBufferToImage`,
  `CreateRenderTexture`), `rhi/Barrier.h` (`RecordImageBarrier`),
  `rhi/SwapchainSupport.h` (`ChooseSwapchainFormat`, `ChoosePresentMode`,
  `ChooseSwapchainExtent`, `ChooseSwapMinImageCount`),
  `rhi/CommandBuffer.h` (`BeginSingleTimeCommand`, `EndSingleTimeCommand`),
  and `FindMemoryType` deleted (dead since VMA).
- **Verify:** Output unchanged. `Utility.h` no longer exists — a good sign for a codebase.
- **Size:** M · **Needs:** 24

### 26. Extract `rhi::Device`
- **Do:** Move `CreateInstance`, `SetupDebugMessenger`, `IsPhysicalDeviceSuitable`,
  `PickPhysicalDevice`, `CreateLogicalDevice` into `rhi::Device`. It owns context, instance,
  messenger, physical device, device, queue and queue-family index. Surface creation stays in
  `App` for now (`Device` receives a `vk::SurfaceKHR`).
- **Verify:** Output unchanged. Compare the startup log line-for-line against a saved
  baseline: same physical device, same queue index, same swapchain image count, same
  validation output.
- **Size:** L · **Needs:** 25

### 27. Enumerate all queue families
- **Do:** In `Device`, find graphics+present, dedicated compute and dedicated transfer
  families. Log all of them. **Keep using the graphics queue everywhere** — this step only
  discovers and reports.
- **Verify:** Log shows the families your GPU exposes. Output unchanged. Resolves the two
  `TODO`s at `main.cpp:400` and `main.cpp:2173` in terms of information, not yet usage.
- **Size:** S · **Needs:** 26

### 28. `rhi::Diagnostics` — validation policy
- **Do:** Promote step 6's counters into `Diagnostics`, owned by `Device`, with
  `ValidationPolicy{Ignore, Count, FailFast}` and message capture.
- **Verify:** `--strict-validation` still exits non-zero on an injected error; `FailFast`
  aborts at the first one with the message printed.
- **Size:** S · **Needs:** 26

### 29. `UploadContext` — batch transfers
- **Do:** Replace the per-resource `EndSingleTimeCommand` → `queue.waitIdle()`
  (`Utility.h:182`, now `rhi/CommandBuffer.h`) with a context that records many copies into
  one command buffer, submits once, waits on one fence, then releases staging buffers.
  Route `TextureLoader`, `CubemapLoader`, `ModelLoader` and `CreateQuadBuffers` through it.
- **Verify:** **Time a Sponza load before and after** with the existing `Timer` class. Sponza
  performs ~70 full GPU drains today; expect a large reduction. Output unchanged.
- **Size:** L · **Needs:** 26, 27

### 30. Use the dedicated transfer queue
- **Do:** Point `UploadContext` at the transfer family when one exists, with its own command
  pool, plus a queue-family ownership transfer barrier (or `eExclusive` → `eConcurrent`, or
  simply release/acquire pairs) before first graphics use.
- **Verify:** Output unchanged, **zero validation errors** — queue ownership transfer is easy
  to get wrong and sync validation is already enabled (`main.cpp:833`), so this is a genuine
  test. Load time improves further on discrete GPUs.
- **Size:** M · **Needs:** 29

### 31. Growable `DescriptorAllocator`
- **Do:** `std::vector<vk::raii::DescriptorPool>`; on `eErrorOutOfPoolMemory`, allocate
  another pool of ~1.5× the size and retry. Use it in `MaterialFactory`, replacing
  `s_MAX_MATERIAL_SET_COUNT`.
- **Verify:** **Temporarily set the initial pool size to 4**, load Sponza (~25 materials),
  and confirm it loads correctly with pool growth logged. Then restore a sensible initial
  size. Output unchanged.
- **Size:** M · **Needs:** 26

### 32. Growable instance buffer
- **Do:** Replace the `throw` in `UpdateInstanceBuffer` (`main.cpp:2198`) with: `waitIdle`,
  reallocate to `max(needed, capacity * 2)`, remap, log once.
- **Verify:** Author `content/scenes/stress.map` with > 1024 instances and confirm it renders
  instead of throwing. `MAX_INSTANCE_COUNT` becomes an initial capacity, not a ceiling.
- **Size:** S · **Needs:** 26

### 33. `PipelineCache`
- **Do:** One `vk::raii::PipelineCache` created at startup, seeded from
  `<user data dir>/pipeline_cache.bin`, passed to all 5 pipeline creations (currently all
  pass `nullptr`) and to `ImGui_ImplVulkan_InitInfo::PipelineCache`. Write on shutdown.
- **Verify:** Log pipeline-creation time; second launch is measurably faster. Delete the file
  and confirm it regenerates without error. Corrupt the file and confirm it is rejected
  gracefully rather than crashing.
- **Size:** M · **Needs:** 26

### 34. First GPU tests
- **Do:** `tests/support/RhiTestFixture.h` (one `Device` per binary, SKIP if no ICD) and
  `ValidationGuard.h`. Tests: device creation reports the required features;
  buffer upload → readback round-trips byte-exactly; image upload → readback matches;
  **all six cubemap faces differ as expected**.
- **Verify:** `ctest -L gpu` passes locally. The cubemap test may **fail** — that is
  companion-doc bug 1.1 (`CopyBufferToImage`/`TransitionImageLayout` hardcode
  `layerCount = 1`). Fix it here; the test locks it down permanently.
- **Size:** L · **Needs:** 11, 26, 29

---

## Stage 6 — Headless capability (steps 35–40a)

**Done.** **37 was delivered early, during Stage 5** — the gpu tests needed a surfaceless
device before Stage 6 did — and step 40's event half became 40b and moved to Stage 7, for the
reason recorded there. The engine now runs with no window: `--headless --frames N
--screenshot` renders into an `OffscreenTarget` and writes a PNG, ImGui included, with zero
validation errors.

### 35. `IPresentTarget` + `SwapchainTarget` — ✅ done
- **Do:** Define the interface from §10.2. Move `CreateSwapchain`,
  `CreateSwapchainImageViews` and the swapchain half of
  `RecreateSwapchainAndRenderImages` into `SwapchainTarget`. `App` calls
  `Acquire`/`Present`/`Recreate`.
- **Verify:** Output unchanged. **Exercise resize hard**: drag-resize, maximise, minimise and
  restore, alt-tab, and toggle fullscreen, all with zero validation errors. This is the
  riskiest step in Stage 6.
- **Size:** L · **Needs:** 26

### 36. Recreate sync objects on image-count change — ✅ done
- **Do:** Step 35 shrank this. The per-image semaphores moved into `SwapchainTarget`, which
  rebuilds them with the images they order access to, so the `m_RenderCompleteSemaphores`
  half of this step and the `CreateSyncObjects` re-entrancy bug are already gone. What
  remains is `App` reacting to a changed `GetImageCount()` across `Recreate` — updating
  `ImGui_ImplVulkan_SetMinImageCount`, and auditing anything else sized to the old count.
- **Verify:** Enter and leave fullscreen (where drivers commonly change image count) with
  zero validation errors and no out-of-bounds indexing. Under ASan this is now a hard check.
- **Size:** S · **Needs:** 35

### 37. Split device requirements: present vs non-present — ✅ done, during Stage 5
- **Do:** Introduce `DeviceRequirements` (§10.3) with `bNeedsPresent`. When false, drop
  `VK_KHR_swapchain` from the required extensions and drop the `getSurfaceSupportKHR` term
  from queue-family selection. Keep every other feature required.
- **Verify:** Windowed app output unchanged. Add a gpu test that constructs a device with
  `bNeedsPresent = false` and **no surface at all**, asserting it succeeds. First genuinely
  headless artefact.
- **Size:** M · **Needs:** 26, 35
- **What actually happened:** the RHI extraction needed this before Stage 6 did — a test
  binary has no window, so the gpu tests could not create a device at all until presentation
  was optional. It landed as `DeviceRequirements::bPresent`, and the test this step asks for
  exists as *"A headless device is created and reports no presentation"*. `RhiTestFixture`
  goes further than the step required and creates every device surfaceless. Nothing remains;
  §10.3 records the one loose end (choosing a software ICD in CI), which belongs to step 47.

### 38. `OffscreenTarget` — ✅ done
- **Do:** Implement `IPresentTarget` over N owned color images
  (`eColorAttachment | eTransferSrc | eSampled`), no surface, no swapchain. `Acquire` returns
  `frame % N`; `Present` is a no-op; `Recreate` reallocates. Return it from
  `IDevice::CreatePresentTarget` where that currently throws for a device with no surface —
  the interface already says the device decides which kind of target comes back, so there is
  no new entry point to add.
- **Decided:** `AcquiredImage::Available` becomes a span, not a single handle:

  ```cpp
  // The semaphores the caller's submit must wait on before writing this image.
  // Empty when nothing needs waiting on. Points into the target; valid until the
  // next Acquire() or Recreate().
  std::span<const SemaphoreHandle> WaitSemaphores;
  ```

  An empty set is not a sentinel, so nothing has to be explained and there is no "invalid
  means fine" to trip over. `std::optional<SemaphoreHandle>` was the alternative and is worse
  here: `SemaphoreHandle` is a `Handle<Tag>`, which already has `kInvalid`, so an optional
  would give the struct two ways to spell "nothing" and a reader would have to work out
  whether they differ.

  The call site decides it. The submit already takes a count and an array, so a span becomes
  `waitSemaphoreCount = waits.size()` and the empty case needs **no branch at all** — where a
  null handle would need an `if`. It also generalises for free if a target ever needs more
  than one wait, which makes the next point cheap rather than a redesign.
- **Check before assuming there is nothing to wait on.** With N images and `frame % N`, the
  previous write to image *i* must have completed before it is written again. The frame loop's
  `DrawFence` covers that only while N equals `FramesInFlight`. If this step lets those
  differ, the offscreen target has a real dependency to express and should signal a genuine
  semaphore — which the span carries exactly like a swapchain's, with no special case.
- **Verify:** A gpu test renders a known clear colour through the real
  `CompositePass`-equivalent path into an `OffscreenTarget` for 3 frames with zero validation
  errors.
- **Size:** M · **Needs:** 37
- **What was built:** `engine/rhi/src/vulkan/OffscreenTarget.{h,cpp}`, returned from
  `VulkanDevice::CreatePresentTarget` where it used to throw. `AcquiredImage::Available`
  became `WaitSemaphores`, as decided above. `tests/gpu/rhi/PresentTargetTests.cpp` covers
  it, including three overlapping frames whose pixels are read back and compared exactly.
- **The wait turned out to be real, and for a reason the step did not anticipate.** N *is*
  `FramesInFlight`, so the caller's own fence ring already covers the write-after-write
  hazard the step was worried about — but the semaphore has to be waited on regardless.
  `GetRenderCompleteSemaphore` is signalled by the caller's submit every frame, and with no
  presentation engine nothing consumes it; a binary semaphore must be unsignalled when a
  signal operation reaches the device (`VUID-vkQueueSubmit-pSignalSemaphores-00067`), so the
  second frame to reach a given image would signal an already-signalled semaphore. The
  target therefore hands that semaphore back as the next `Acquire`'s wait, which both
  unsignals it and states the real dependency. Deleting the wait fails the gpu test with
  that exact VUID, which is what makes the case worth having.
- **`BarrierPresets::AcquiredSwapchainToRenderTarget` is now `AcquiredImageToRenderTarget`.**
  Both targets hand back an acquired image on the same terms, and the preset was only ever
  named after one of them. Its `SrcStage` was already `RenderTarget` rather than `None`, and
  that is load-bearing — a layout transition is ordered after a semaphore wait only if the
  barrier's source stage covers the stage waited at — so the reason is now written down
  beside it.
- **Left for 40a: what layout a finished offscreen frame ends in.** The renderer records
  `RenderTargetToPresent()` unconditionally, and an offscreen image can never be in
  `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`: the enum belongs to `VK_KHR_swapchain`, which a device
  with no surface does not enable. Nothing breaks today because nothing yet builds an
  offscreen target from `App` — that starts with `HeadlessPlatform`. The fix is one query on
  `IPresentTarget` (the layout the target requires at `Present`: `Present` for a swapchain,
  `ShaderResource` for an offscreen target) plus a parameter on the two `…ToPresent` presets.
  It is listed under 40a rather than done here because 40a is its first caller.

### 39. `OffscreenTarget::Readback` — ✅ done
- **Do:** `CopyTextureToBuffer` into a host-visible staging buffer, fence, return tightly
  packed bytes. Reuse step 5's PNG writer, now shared by both targets.
- **Prerequisite:** the PNG writer is not shareable as written — `WriteScreenshot` hardcodes
  a BGRA→RGBA swizzle. Drive it from `IPresentTarget::GetFormat()` first. Worth doing
  independently of this step; see `backlog.md` for why it is already a latent
  bug.
- **Share, do not copy:** `tests/support/GpuReadback.h` already does record → submit → fence
  → read mapped bytes against the neutral `ICommandList`. Exactly one piece of it should move
  rather than be duplicated:
  - **`Detail::BytesPerTexel` becomes real RHI API.** Its own comment says why it is stuck in
    a test — "the RHI has no texel-size query, and inventing one here would be a public API
    decision made in a test". This step *is* that decision: sizing a tightly packed readback
    buffer is impossible without it. Promote it to a neutral free function beside `Format`
    (`Rhi::BytesPerTexel(Format)`), covered by a unit test, and have the test helper call it.
  - **The rest stays in `tests/support/`, and is not what step 39 builds on.** It reads
    `REQUIRE` from Catch2, which production cannot link; it creates a command pool and fence
    per call, which its comment justifies specifically because tests run it a handful of
    times; and `ReadTextureLayers` returns a vector per array layer and assumes the texture
    arrives in `ShaderResource` — the layout `IUploadContext` leaves it in. `Readback` needs
    one mip of one layer of a *render target*, in a different layout, packed into one buffer.
    The shapes only look alike.
  - Moving the whole file into `engine/rhi/src/vulkan/` would therefore relocate test-only
    code into the production module to be used by nobody but the tests, which is worse than
    where it is. What *does* shrink is the other direction: once `Readback` exists, the
    offscreen cases call it instead of the hatch, and the
    `tests/support/GpuReadback.h|VulkanNative.h` allowlist entry may be able to go.
  - `OffscreenTarget` lives inside the module, so its implementation needs no escape hatch at
    all — it has `VulkanDevice` directly. That is the other reason the test helper's body is
    not reusable: most of it exists to get *through* a boundary that step 39 is on the far
    side of.
- **Verify:** gpu test asserts exact pixel values for a solid clear colour, and correct
  dimensions/stride for a non-square, non-power-of-two extent (e.g. 253×101).
- **Size:** M · **Needs:** 38
- **What was built:**
  - `Rhi::BytesPerTexel(Format)` in `rhi/RhiTypes.h`, `constexpr` and total, covered by
    `tests/unit/rhi/FormatTests.cpp`. It returns **0** for `Undefined` and for both combined
    depth/stencil formats, because those have a size per *aspect* and the two differ — a
    caller that sizes a buffer has to reject 0 rather than allocate nothing. `GpuReadback.h`
    calls it and its private copy is gone.
  - `WriteScreenshot` reads `IPresentTarget::GetFormat()` for both the texel size and the
    channel order, so the BGRA swizzle now happens because the target said BGRA. A format
    that is not 8-bit four-channel is refused with a message rather than reinterpreted.
  - `OffscreenTarget::Readback(index, currentLayout)`, returning tightly packed bytes.
- **`currentLayout` is a parameter, and that is the interesting part.** The target records no
  commands during a frame, so it cannot know what the caller's last barrier left the image
  in; naming it is what keeps the transition correct rather than plausible. The image is left
  in `CopySrc`, which costs nothing — the next `Acquire` transitions from `Undefined`.
- **The wait reuses step 38's chain rather than a `WaitIdle`.** `Readback` consumes the same
  render-complete signal `Acquire` would have, so the copy is ordered after the rendering and
  the semaphore is left unsignalled for the next frame to signal. A readback between two
  frames therefore stands in for one link of the chain instead of breaking it, and the host
  fence wait covers the next write to that image.
- **The allowlist entry did not shrink after all.** This step hoped
  `tests/support/GpuReadback.h|VulkanNative.h` could go once the offscreen cases stopped
  reaching through the hatch. They have — `PresentTargetTests.cpp` calls `Readback` and no
  longer includes `GpuReadback.h` — but `UploadRoundTripTests.cpp` still needs `ReadBuffer`
  and `ReadTextureLayers`, which is what that entry is really for. It goes when upload
  verification has a neutral way to read bytes back, not here.
- **`Readback` is on `OffscreenTarget`, not on `IPresentTarget`** — §10.2's reasoning held up
  under implementation, and the spec is what settles it: *"Use of a presentable image must
  occur only after the image is returned by `vkAcquireNextImageKHR`, and before it is
  presented by `vkQueuePresentKHR`"*, and *"presented images must not be used again before
  they have been reacquired"* (Vulkan 1.4, *Presenting Images*). A standalone readback is by
  definition outside that window, and `WriteScreenshot` runs after the loop, when nothing is
  acquired at all. The two implementations would therefore differ in *when they may be
  called*, which is the one thing an interface exists to make impossible. The gpu tests reach
  it by `dynamic_cast`, which also asserts that a surfaceless device really does hand back an
  offscreen target.
- **A correction this step turned up.** `BarrierPresets::AcquiredImageToRenderTarget` claimed
  the contents of an acquired image are "undefined by definition". They are not — the spec
  guarantees a reacquired image still holds what was presented, unless something outside
  Vulkan modified the window. The `Undefined` old layout is right regardless, because the
  composite pass clears the whole render area, but the reason is a deliberate discard rather
  than an absence of contents. Comment corrected.

### 40a. `HeadlessPlatform` (no events) — ✅ done

Everything below is decided. The options that were weighed and rejected are recorded at the
bottom, so that a fresh session can start work without re-opening them.

- **Do:** Implement `IPlatform` in `engine/platform/` with `IsHeadless() == true`,
  `GetFramebufferExtent` returning the configured offscreen extent, `GetNativeWindowHandle`
  → `nullptr`, and `Show`, `SetWindowMode`, `SetRelativeMouseMode` and `WarpMouse` as no-ops.
  **Much smaller than this step originally was**: surface creation and instance-extension
  enumeration moved into the RHI (§10.1), so there is nothing Vulkan-shaped left to stub.
- **It needs no SDL at all**, which is worth knowing before writing it. `SdlPlatform` calls
  `SDL_Vulkan_LoadLibrary`, but nothing consumes SDL's loader: the RHI reaches Vulkan through
  vulkan.hpp's own dispatcher. The proof is already in the tree — the gpu tests create real
  devices with no SDL anywhere. So `HeadlessPlatform` is a plain class with no window-system
  dependency, and the unit test this step asks for needs no SDL subsystem. That call turns out
  to be redundant in `SdlPlatform` too, for a second reason — see `backlog.md`.
- **Takes a `WindowDesc`, exactly like `SdlPlatform`**, so `main()` builds one description
  and hands it to whichever implementation it picked — which is the swap the seam exists to
  make possible. `Title`, `bResizable` and `bBorderless` are ignored; say so on the class.
  They are ignored only in states they can reach, since `--headless` with a window-mode flag
  is rejected below.
- **Extent when `--resolution` is absent:** 1280×720, resolved *inside* `HeadlessPlatform` so
  that the zero-means-default rule stays where `SdlPlatform` keeps it rather than splitting
  across `main()`. There is no display to size against, so it has to be a documented constant;
  that one matches `SdlPlatform`'s existing `kFallbackWindowSize`. `--resolution` overrides it
  and is honoured rather than rejected. A zero *component* cannot arrive —
  `RequireExtent2D` rejects `1920x0` at parse time (`CommandLine.cpp:88`).

#### Option validation

Both checks belong in `ParseOptions`' post-parse block, beside the existing
`--strict-validation` one, **not** in the flag loop — they have to hold whichever order the
flags arrive in.

- **Reject `--headless` with `--borderless` or `--fullscreen`.** A window mode for a run with
  no window asks for nothing coherent.
- **Reject `--headless` without `--frames`.** Of the frame loop's three exits, two need a
  window: `SDL_EVENT_QUIT` and ImGui's Quit button. The third is the frame counter, which sets
  `g_bShouldClose` only when `Frames != 0`. So headless with `--frames 0` ends only on a
  signal — and headless exists for CI and the baseline, where nobody is there to send one.
  The job then burns its whole timeout and dies to `SIGTERM`, which nothing handles, so it
  writes neither screenshot nor report.

  Note that Ctrl-C *does* work, contrary to what this entry used to claim: `HandleSIGINT`
  sets `g_bShouldClose` (`main.cpp:91`), so an interactive headless run leaves the loop the
  ordinary way and still writes its artefacts. That makes an unbounded run defensible at a
  terminal — a soak test, say — which is why this is a rejection rather than an impossibility.
  It stays rejected because the failure it prevents is silent and expensive in the place this
  mode is actually for, and `Frames` is a `uint64_t` if someone wants to soak. Lift it the day
  that use case is real.

#### The present-layout seam (deferred here from step 38)

Every frame ends with a barrier to `TextureLayout::Present` — `CopySrcToPresent()` when
capturing a screenshot and `RenderTargetToPresent()` otherwise, both in
`RecordSwapImageToPresentLayout`. `Present` maps to `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`, a
`VK_KHR_swapchain` enumerator that a surfaceless device never enabled, so a headless frame
raises a validation error on its last barrier. `App` has to stop hardcoding it.

**Decided: add the query, and let it answer "no requirement".**

- `TextureLayout GetRequiredFinalLayout() const` on `IPresentTarget`. `SwapchainTarget`
  returns `Present`, which is a hard requirement of the present call itself
  (VUID-VkPresentInfoKHR-pImageIndices-01430). `OffscreenTarget` returns
  `TextureLayout::Undefined`, meaning it requires no particular layout, and `App` records no
  trailing barrier at all when it gets that answer.
- **`Undefined` is safe to mean "none" because it is illegal as a barrier destination**
  (VUID-VkImageMemoryBarrier2-newLayout-01198). A caller who forgets to check and transitions
  *to* it raises a validation error immediately, which the headless verify already gates on —
  the mistake cannot be made quietly.
- **Why not a real layout.** An offscreen target genuinely requires none: it owns its images,
  nothing reads them between frames, and `Acquire` discards to `Undefined` on the way back in.
  Every real value is therefore an invented requirement, and none is even free — `RenderTarget`
  costs a pointless transition on the capturing frame, which has already moved the image to
  `CopySrc`, and `CopySrc` costs one on every other frame. `ShaderResource` was the earlier
  answer here, justified by the images' `Sampled` usage and by "where a readback wants to
  start from". The second half is wrong — `Readback` takes `currentLayout` and transitions
  itself (`OffscreenTarget.cpp:232`) — and the first has no consumer: nothing in Stages 8–10
  samples the frame result. Steps 50–56 write *into* the present image; step 56's layout
  tracker consumes this query rather than the image, and "no terminal constraint" is a more
  useful thing to tell it than an invented one.
- **Naming.** `GetRequiredFinalLayout`, not `GetPresentLayout`: the word *required* is what
  makes "none" a legal answer instead of a hole, where "the present layout is `Undefined`"
  reads as an uninitialised value. The two `…ToPresent` presets become
  `RenderTargetToFinal(layout)` and `CopySrcToFinal(layout)`; `RenderTargetToCopySrc` is
  untouched. `TextureLayout::Present`'s own spelling still stands — it is the neutral
  vocabulary's word and D3D12 calls the same idea `D3D12_RESOURCE_STATE_PRESENT`.
- **Skipping the barrier leaves the command buffer empty, and it is still submitted.** The
  frame loop submits a fixed array of seven, and the ImGui one already goes out empty when
  the panel is hidden (the `TODO` at `main.cpp:1078`). Step 56 culls both properly; doing one
  of them here would leave the other looking like an oversight.
- `App`'s existing in-frame screenshot copy needs no other change: an offscreen image carries
  `CopySrc` exactly like a swapchain's.

#### ImGui runs headless

The ImGui panel is **in the committed baseline screenshot** — `m_bCursorVisible` defaults to
`true`, so `DrawImGuiFrame()` runs every frame and the panel occupies a 423×443 block in the
top-left.

**Decided: it renders, through the Vulkan backend only.** The reason is coverage: a headless
CI run should be exercising ImGui's bring-up and drawing, not skipping a fifth of the frame.
It is explicitly *not* about making the headless capture comparable to a windowed one —
step 46 compares captures taken with `--no-ui`, because the panel carries a hover highlight
that no headless run can reproduce.

- Skip `ImGui_ImplSDL3_InitForVulkan`, `ImGui_ImplSDL3_ProcessEvent`,
  `ImGui_ImplSDL3_NewFrame` and `ImGui_ImplSDL3_Shutdown`.
- Set `io.DisplaySize` and `io.DeltaTime` by hand in their place — that is what the SDL3
  backend was supplying, and ImGui with no platform backend is a supported configuration.
  `io.DeltaTime` is the load-bearing one: `NewFrame` asserts it is positive, where
  `DisplaySize` need only be non-negative.
- `ImGui_ImplVulkan_*` needs no window and is used unchanged. Everything surface-shaped in
  that backend lives in the optional `ImGui_ImplVulkanH_*` helper family, which this app has
  never called — it records ImGui's draws into a dynamic-rendering pass it opens itself,
  against whatever image the present target handed back and whatever format that target
  chose.
- **`MinImageCount` is fed `GetImageCount()`, and ImGui documents it as `>= 2`.** An
  `OffscreenTarget` creates one image per frame in flight, so `NUM_FRAMES_IN_FLIGHT = 2` meets
  that exactly, with no margin. Step 41 makes that count runtime and its verify asks for
  `--frames-in-flight 1`, which would violate it — deal with it there.

#### Wiring

- `main()` holds `std::unique_ptr<IPlatform>`, not `std::unique_ptr<SdlPlatform>`, and picks
  the implementation from `options.bHeadless`. `SdlPlatform::ShowErrorMessageBox` is static,
  so the `SDLException` handler is unaffected.
- `MakeDeviceDesc`: `desc.Requirements.bPresent = !m_Platform.IsHeadless()`. **This is the
  line that makes the whole thing work** — `bPresent = false` means no surface, which is what
  makes `CreatePresentTarget` hand back an `OffscreenTarget`.
- Guard the two direct SDL calls left in `App`: `SDL_PollEvent` in `Run` and
  `SDL_GetKeyboardState` in `HandleMovement`. Both run against an SDL that was never
  initialised. This is not step 40b's event seam arriving early — it is the minimum that
  stops a headless run touching an uninitialised subsystem.

#### Order of work

Land the layout seam first and confirm the windowed baseline is pixel-identical before
touching anything headless. That is the checkpoint; it needs no renumbering of 40b, and the
seam half on its own is not a milestone worth recording as a step.

- **Verify:** Unit test constructs `HeadlessPlatform` with no SDL video subsystem
  initialised, asserting `IsHeadless()`, the configured extent, the 1280×720 default, a null
  native handle, and that the no-ops are callable. Windowed app output unchanged — baseline
  screenshot pixel-identical and report unchanged, which is the checkpoint the layout seam
  has to clear on its own. Then a real run: `--headless --frames 5 --screenshot` exits 0,
  writes a PNG at the requested extent, and reports zero validation errors.
- **Size:** M, not the S this step used to carry — the layout seam and the headless wiring
  both arrived here after it was written. **Needs:** 20, 37

#### Rejected, and why — do not re-open without new information

- **`RecordCapture(ICommandList&, index, BufferHandle)` on `IPresentTarget`,** moving the
  screenshot copy behind the seam. It was recommended here on the claim that it subsumed the
  layout query. It does not: it covers only the capturing branch, and the ordinary frame
  still ends with a transition to a layout `App` has to know. So the query is needed either
  way, and `RecordCapture` is a second interface member with real behaviour in both
  implementations plus a gpu test — for a refactor Stage 7 revisits anyway when it dismantles
  `App`. Reconsider it there, not here.
- **Skipping ImGui headless.** Cheapest, and it no longer costs step 46 anything now that
  that comparison runs with `--no-ui` on both sides. Rejected on coverage alone: headless CI
  is the only place ImGui's bring-up and drawing get exercised automatically, and a mode that
  skips it proves less than it appears to.
- **A warning when a headless run writes no screenshot and no report.** That invocation is a
  smoke test, not a mistake — with `--strict-validation` the exit code carries the whole
  result, and it is the shape step 47's `scene_launch_test` is built around. A warning would
  fire on the most common headless run in CI and train everyone past it.
- **Splitting this step in two and renumbering 40b → 40c.** The natural cut is after the
  layout seam, but that half is a seam change whose only observable effect is that the
  windowed baseline still matches — not a milestone. 40b is referenced from the Stage 7
  header, the dependency table and this entry, so the renumber is doc churn for no gain. The
  ordering above gives the same checkpoint without it.

---

## Stage 7 — Engine shell & dependency injection (steps 40b–47)

Step 40b (the event seam) joins this stage from Stage 6 — it is loop-shaped work and belongs
next to `IClock` and `RunSpec` rather than next to `HeadlessPlatform`.

### 40b. Event seam + scripted input
- **Do:** The member this plan called `PumpEvents` never shipped, and nothing replaced it —
  `App::Run` still calls `SDL_PollEvent` directly and `HandleMovement` reads
  `SDL_GetKeyboardState` (§10.1). Define the event abstraction, move SDL's polling behind
  `SdlPlatform`, and give `HeadlessPlatform` an in-memory queue fed by §14's scripted input.
- **Verify:** Windowed input unchanged by hand. A headless run replays
  `tests/data/input/orbit.txt` and produces the frame count, resizes and screenshot the
  script asks for.
- **Size:** M–L · **Needs:** 40a
- **Note:** deliberately split out of step 40 and **deferred here from Stage 6**, next to the other
  loop-shaped work (`IClock`, `RunSpec`). Nothing in steps 41–47 needs it except the scripted
  half of 47's tests: `--frames N --screenshot` requires no input at all, so 40a alone is
  enough for the first genuinely headless artefact. Keeping them together would have made the
  smallest remaining step in Stage 6 into its largest.

### 41. `Engine` + `EngineConfig` + `RunSpec` + `RunReport`
- **Do:** Rename `App` → `Engine` in `engine/engine`. Promote the surviving `constexpr`s
  (`INITIAL_INSTANCE_CAPACITY`, `NUM_FRAMES_IN_FLIGHT`, `SKY_COLOR`) into `EngineConfig`.
  Promote step 1's `Options` into `RunSpec` and step 6's JSON into `RunReport`, returned from
  `Engine::Run(RunSpec)`.
- **Verify:** Output unchanged. `NUM_FRAMES_IN_FLIGHT` is now runtime, so test
  `--frames-in-flight 1` and `3` and confirm both render correctly — a good latent-bug
  detector, since it's currently a compile-time constant baked into array sizes.
- **Size:** L · **Needs:** 23, 35
- **Landed (2026-09-05), with four deviations worth keeping:**
  - **The `Engine` class stays in `src/main.cpp`**, in `namespace Hikari::Engine`, and moves to
    `engine/engine` at step 46. It reaches for sixteen `src/` headers, which an engine module
    cannot include, and step 43 requires `MaterialFactory` to stay in `src/` while `Engine`
    constructs it — so the class cannot leave before the things it holds do. The module was
    created now and holds `EngineConfig`, `RunSpec`, `RunReport`, `CapturedFrame`, `RunResult`,
    `CameraPresetData`/`kCameraPresets`, and the option parsing. Same pattern this step's own
    §41 note already applies to `WriteRunReport`: name the eventual home, move it when the step
    that owns the move arrives.
  - **`ParseRunSpec` is `ParseEngineOption` + `PrintEngineUsage`**, taking one option at a time
    and filling `EngineConfig` alongside `RunSpec`. An app has flags of its own to interleave,
    so it walks the options once and offers each to the engine first; and `--frames-in-flight`
    configures the engine rather than the run, so one entry point fills both.
  - **`RunSpec` carries `bRecordTimings`** as well as `bCaptureFinalFrame`. The frame-timing
    samples were collected only when `--report` was passed, and the engine can no longer see
    that flag — without the field it would either lose the condition or accumulate two floats
    per frame forever in an unbounded run.
  - **ImGui's `ImageCount` is `max({2u, GetImageCount(), FramesInFlight})`**, not
    `max(2u, GetImageCount())`. The ring is reused every `ImageCount` frames, so a two-image
    swapchain with three frames in flight would be overwritten while an earlier frame still
    read it. `--screenshot` and `--report` stayed app flags, since the paths are the app's.
- **Note:** this step used to name `WIDTH` and `HEIGHT` as well. They no longer exist — the
  window-mode work replaced them with `WindowDesc` plus `--resolution`, where zero means "ask
  the display". `RunSpec` should carry the resolution and the window mode rather than
  reintroducing fixed dimensions, and `MAX_INSTANCE_COUNT` is now
  `INITIAL_INSTANCE_CAPACITY` — a starting size rather than the ceiling this plan was
  written against.

### 42. Inject `ResourceManager` and the loaders
- **Do:** Delete the `Get()` singletons for `ResourceManager`, `TextureLoader`,
  `CubemapLoader`, `ModelLoader`. `Engine` owns an `AssetRegistry`; loaders receive their
  dependencies in constructors.
- **Verify:** Output unchanged. Unit test constructs **two** independent registries and
  confirms their caches are separate (impossible today).
- **Size:** L · **Needs:** 41
- **Landed (2026-09-05), with two deviations:**
  - **`engine/asset` holds `AssetCache` only; `AssetRegistry` stays in `src/`.** The registry
    caches `Texture`, `Cubemap` and `ModelData`, and its implementation needs those complete —
    they are `src/` types until Stage 8 splits them into `TextureData`/`MeshData`. Same
    constraint that kept `Engine` in `src/` at step 41, and it resolves the same way: the
    registry follows its types into the module rather than dragging them in early.
  - **The two-registries test is a two-caches test.** `AssetRegistry` is not linkable from
    `tests/` while it lives in `src/` and needs a device, so the independence the step is about
    — no process-wide table behind the caches — is asserted against `AssetCache` directly, in
    `tests/unit/asset/AssetCacheTests.cpp`. The registry-level test arrives with the registry.

  What did land in full: the `Get()` singletons for `ResourceManager`, `TextureLoader`,
  `CubemapLoader` and `ModelLoader` are gone, the registry owns the three loaders and is
  constructor-injected through `XmlParser::LoadScene` → `Model` and through `ModelLoader` →
  `MaterialFactory::CreatePBRMaterial` → `PBRMaterial`, and `counters` split into
  `counters.frame` and `counters.run` with `uploadSubmissions` in the latter. `ModelManager`'s
  `Init`/`Shutdown` moved to `Engine`, where they wait for step 44.

### 43. Inject `MaterialFactory`
- **Do:** Same treatment; owned by `AssetRegistry`.
- **Verify:** Output unchanged.
- **Size:** M · **Needs:** 42
- **Landed (2026-09-05), owned by `Engine` rather than by `AssetRegistry`.** The factory has
  callers on both sides of the asset/renderer line: `ModelLoader` builds materials with it while
  loading, and the opaque and transparent pipeline layouts are built against its descriptor set
  layout. Registry ownership would put a `vk::DescriptorSetLayout` on `engine/asset`' public
  surface — and an `rhi/vulkan/` allowlist entry on a module that otherwise needs none — for the
  sake of the renderer's use of it. `Engine` builds it before the registry and passes it through
  to `ModelLoader`, the one loader that needs it; it is destroyed after the registry, since the
  materials that hold descriptor sets from its allocator die with the models. It stays in `src/`:
  it is not an assets type, and Stage 8 relocates materials and pipelines together, which is when
  the materials-versus-pipeline-layouts question has to be answered anyway.

### 44. Inject `ModelManager` and break `Model`'s back-pointer
- **Do:** The awkward one. `Model`'s constructor currently calls `ModelManager::Get()->
  RegisterModel(this)` and its destructor unregisters (`Model.cpp:10,13`). Change to: the
  scene owns models; a `CollectRenderables` pass walks the scene each time the scene changes
  and populates the (now injected) `ModelManager`. `Model` becomes inert data.
- **Verify:** Output unchanged. Scene load/unload/reload still correct; `AssetCache`
  `LiveCount()` returns to baseline after unloading (a leak check). Unit test constructs a
  `Model` with **no** `ModelManager` in existence.
- **Size:** L · **Needs:** 42
- **Landed (2026-09-05).** `ModelManager` is a value member of `Engine` with no `Get()`, no
  `Init`/`Shutdown` and no model list: `GenerateBatches(const SceneGraph&)` walks the scene it is
  handed through a private `CollectRenderables`, every frame, as the registration it replaced
  effectively did. `Model` no longer includes `ModelManager.h` at all — that absence is what the
  planned "construct a `Model` with no `ModelManager`" test was there to assert, and it is not a
  test that can be written yet: `Model` lives in `src/` and needs an `AssetRegistry`, so it is
  reachable from neither a unit nor a GPU test until Stage 8 splits it. The runtime scene swap
  got more robust rather than less — batches are rebuilt from whatever the scene is now, so a
  replaced scene needs nothing to have unregistered itself. Of the verify list, output-unchanged
  and the leak check are covered (`AssetRegistry`'s destructor asserts every cache is empty, on
  every debug run); **reload is not reachable without input scripting**, which arrives at 40b.

### 45. Deterministic clock
- **Do:** `IClock` with `RealClock` and `FixedStepClock`. In `bDeterministic` mode force
  `SerialJobSystem` and fixed dt. Add `--seed`.
- **Verify:** Two deterministic runs produce byte-identical screenshots **and** identical
  `RunReport` counters, on both windowed and offscreen targets.
- **Size:** M · **Needs:** 41
- **Caveat:** Batch order still depends on pointer values via `Drawable::operator<`
  (`Drawable.h:16-22`), so ordering is only stable within a single process. Fully
  deterministic ordering arrives at step 58.
- **Landed (2026-09-05), at the size `stage7_plan.md` cut it to: `IClock` only.**
  `engine/core/include/core/Clock.h` holds `IClock`, `RealClock` and `FixedStepClock`, where
  §9's target layout already put them. `Engine` owns one, chosen by `--fixed-dt`, and reads it
  for the simulation timestep alone — the run report's timings still read the steady clock
  directly, because a measurement a fixed step silently rewrote would be worthless.
  `Elapsed()` reports the time as of the last `Tick()` rather than as of now, which is both what
  the old code did and what two readers within one frame need. No `SerialJobSystem` forcing:
  nothing today can make job order affect output. No `--seed`: nothing in `src/` or `engine/`
  draws a random number, so the flag would have had nothing to seed.
- **Verified across processes, and the caveat above still stands.** Two windowed runs and two
  offscreen runs each produced identical counters and pixel-identical captures, and all four
  match the committed baseline. That is stronger than the caveat promises — the same build
  allocates in the same order, so the pointer-value sort happens to agree run to run — but it is
  an observation about this build, not a guarantee. Step 58 is what turns it into one.
- **"Force FIFO in deterministic mode" was dropped from this step, deliberately.** It does
  not do what it looks like it does. Present mode affects *timing*, not pixels — the loop
  renders every frame either way — so it buys nothing for byte-identical screenshots. And for
  the counters it would actively hurt: FIFO blocks on vblank, capping frame time at the
  refresh interval, so every regression faster than 16.67 ms becomes invisible. A measurement
  harness wants vsync **off**, not on. Once step 38 lands the deterministic path has no
  swapchain at all, which settles it. Choosing a present mode is still worth having on its own
  merits and lives in `backlog.md` as `--present-mode`.

### 46. `apps/` split
- **Do:** `apps/hikariengine/main.cpp` (SDL + Editor, ~40 lines) and
  `apps/headless/main.cpp` (`HeadlessPlatform` + `OffscreenTarget`, ~30 lines).
  `src/main.cpp` ceases to exist.
- **Verify:** **`HikariEngineHeadless --scene content/scenes/test_scene.map --frames 5
  --report r.json --screenshot h.png` runs with no window and exits 0.** Compare `h.png`
  against a windowed capture of the same scene, **both taken with `--no-ui`**, and require
  them to be pixel-identical rather than merely close: with the panel suppressed the only
  thing left that can differ is the composite path, which is exactly what this comparison
  exists to detect. Compare decoded pixels, never bytes — PNG encoding is not reproducible.

  The panel has to be out of both captures, and suppressing it is the only way to get there:
  nothing warps the cursor at startup, so a windowed run carries a hover highlight on
  whichever widget the mouse was last over, while a headless run has no mouse at all. That
  difference is guaranteed, and it says nothing about the renderer. ImGui still *renders*
  headless — see 40a — so this costs no CI coverage of it; step 47's assertions run against a
  build that drew the panel.
- **Size:** M · **Needs:** 40a, 41, 44 (`--no-ui` already exists)
- **Landed (2026-09-05), and it was not M.** This step inherited the move step 41 deferred: the
  engine could not be shared by two binaries while it lived in `src/main.cpp`. What happened:
  - **`src/` no longer exists.** Everything in it moved to `engine/engine/src/`, private to the
    module, exactly as `stage7_plan.md`'s cross-cutting note anticipated — renderer internals
    live in `engine/engine` until Stage 8 draws the Render/Scene boundary deliberately. Stage 8
    then promotes files out of that directory rather than migrating them out of an application.
    The shaders went with them, to `engine/engine/src/shaders/`, because `common.slangh`
    includes `../Common.h` and that relative path had to keep resolving.
  - **The engine class lives wholly in `Engine.cpp`**, with no header. `namespace_check` forbids
    using-directives in headers, and a header carrying the class would have meant qualifying
    every name across 2,000 lines of code scheduled for demolition. Nothing else in the module
    needs the class: apps see `IEngine`, `CreateEngine` and `EngineDesc`.
  - **`RunApp` in `engine/engine` holds what both binaries do** once they have a platform —
    diagnostics, the job system, the content root, the engine, the files a run leaves behind,
    and the exit code. That is what keeps each `main.cpp` to its own flags and its own platform.
    `WriteRunReport` moved here and PNG encoding to `engine/asset`, as §41 said they would.
  - **`engine/editor` holds `VulkanUiBackend` only.** `IUiBackend` is declared in
    `engine/engine`, because §8's layering puts Editor *above* Engine: the app builds the
    backend and hands it over. The panels stay in the engine — they read the camera, the scene
    graph and the model manager — and become `EditorLayer` when Stage 8 moves what they read.
  - **One new escape-hatch accessor**, `Rhi::Vulkan::GetNative(ICommandList&)`, because ImGui's
    backend takes a `VkCommandBuffer` by value. B5 retires it with the rest of the recording
    path. The allowlist went from sixteen sites to eighteen: `main.cpp`'s three became
    `Engine.cpp`'s three plus the editor's one.
- **Verified:** an editor capture and a headless capture of the same scene at 1920x1080, both
  `--no-ui`, are pixel-identical; the committed baseline's counters, run block and pixels all
  match through `HikariEditor`; `precommit.sh` green, including the header self-containment
  check repointed from `src/` to the module's private headers.

### 47. Wire headless tests into CI
**How this step is built is settled in `docs/stage7_plan.md`**, which supersedes the Do and
Verify below where they differ: device selection is `VK_DRIVER_FILES` (§10.3), the two new jobs
became steps inside the existing ubuntu matrix jobs, and the scene set gained an instancing case
and a real-content one.
- **Do:** `tests/data/scenes/{empty,single_cube,two_materials,lights_only,transparent_only}.map`
  plus a tiny committed `cube.gltf`. Write `scene_launch_test.cpp` asserting `bSucceeded`,
  `ValidationErrors == 0`, expected `FramesRendered`, `DrawCalls > 0`, and expected batch
  counts. Add the `headless-gpu` CI job (lavapipe, per §17) and the `headless-asan` job.
- **Verify:** **CI is green with a headless scene launched and asserted under ASan.**
  Deliberately break a barrier in a PR and confirm CI catches it.
- **Size:** L · **Needs:** 39, 46
- **Landed (2026-09-05), as `stage7_plan.md` specified it.** Eight cases under a new `scene`
  CTest label, in `tests/scene/SceneLaunchTests.cpp`: seven run the engine **in-process** —
  `CreateEngine`, `Run`, assert on the returned `RunReport`, no file and no JSON parser — and
  one launches the real `HikariHeadless` with `--strict-validation` to cover argument parsing,
  the platform and UI it builds, and the files it writes. Every in-process case runs its scene
  **twice and compares** the counters and the captured pixels, which is the determinism check;
  the panel is suppressed with `bNoUi` because it prints the frame time and the FPS, and the
  pass still records either way, so the counters remain those of a shipping frame. Test data is
  `tests/data/` as its own content root: two hand-authored ~2.7 KB glTF cubes with the buffer
  base64-embedded, and six scenes built from them, plus one case running the shipped
  `content/scenes/test_scene.map`. CI installs `mesa-vulkan-drivers` and pins lavapipe with
  `VK_DRIVER_FILES`; `ctest -L gpu` runs in all three Linux jobs and `ctest -L scene` in the
  debug and ASan ones.
- **Two defects the suite found on its first run, both fixed here:**
  - **The dynamic cull mode was never set for a single-sided material.** The opaque recorder
    tracked it starting from an assumed `eBack`, but a command buffer begins with no dynamic
    cull mode at all, so a batch that wanted `eBack` never called `vkCmdSetCullMode` and its
    draw was invalid (`VUID-vkCmdDrawIndexed-None-07840`). Every material shipped today is
    two-sided, which is exactly why nothing had noticed: the first batch wanted `eNone`, which
    differed from the assumption and so was set. Now set unconditionally per batch: one extra
    state token per batch costs nothing measurable, and it removes the rule about when the set
    may be skipped along with the bug that rule caused.
  - **The stop flag survived a run.** `g_bShouldClose` is a global because a signal handler has
    to reach it, and nothing reset it, so a second `Engine::Run` in one process ended before
    its first frame. Invisible to a binary that runs the engine once; fatal to a test that runs
    it eight times. `Run` now clears it on entry.
  - Also made non-fatal: a missing skybox cubemap, which threw and killed the run. The error
    convention asks asset loading to log and skip, nothing renders the skybox yet, and a
    content root without one is now a scene that looks unfinished rather than a run that cannot
    start.

> ### 🎯 Checkpoint: stated goal complete
> CI now builds on 3 platforms, runs unit tests everywhere, and launches real scenes headless
> on Linux under ASan+UBSan with zero-validation-error assertions. **Everything from here is
> optimisation and scalability, protected by that net.** This is a reasonable place to pause
> for weeks and ship features instead.

---

## Stage 8 — Passes & frame graph (steps 48–56)

> **Stage 7.5 comes first, and changes this stage.** `docs/backend_readiness_plan.md` inserts
> six steps (B1–B6) that neutralise submission, rendering scope, binding, pipelines and
> draw/dispatch recording — the prerequisites for a D3D12 backend, none of which this stage
> was written to provide. Three consequences here:
>
> - **50–54 follow B1–B6, not the reverse** (D18). Each recorder is then moved onto the neutral
>   API once and into a `Pass` class once, rather than having `Pass::Execute` take a
>   `vk::CommandBuffer` and be edited again later.
> - **Step 56, the frame graph, is deferred past the D3D12 backend.** A second backend needs a
>   neutral command list, not a graph; building the graph against one backend bakes in its
>   assumptions. 55 folds into 56 as this stage already suggests.
> - **48 and 49 are unaffected**, and 48 is worth pulling early — two backends mean two shader
>   targets and double the ways a hand-mirrored GPU struct layout drifts silently.

### 48. `ShaderTypes.h` shared with Slang
- **Do:** Move `GlobalBuffer`/`CameraData`/`LightData` (`main.cpp:73-102`) and
  `shaders/common.slangh:3-45` into one header included by both, with `static_assert`s on
  `sizeof` and `offsetof` for every field.
- **Verify:** Output unchanged. Deliberately reorder two fields in the shared struct and
  confirm you get a **compile error**, not a rendering artefact. Revert.
- **Size:** M · **Needs:** 41

### 49. Table-driven `FrameResources`
- **Do:** Replace `CreateCommandPools` + `CreateCommandBuffers` (~200 lines of seven
  near-identical blocks) with a loop over a `{name}` table. Same for the three identical
  blocks in `CreateRenderTargets`.
- **Verify:** Output unchanged; debug names in RenderDoc unchanged.
- **Size:** M · **Needs:** 41

### 50–54. Convert each recorder to a `Pass` class, one per step
- **Do:** Define `Pass` with `Name()`, `Setup()`, `Execute()`. Keep a plain
  `std::vector<std::unique_ptr<Pass>>` invoked in the current fixed order — **no graph yet**.
  Convert one pass per step: **50** `OpaquePass`, **51** `TransparentPass`,
  **52** `CloudPass` (rework `CloudSystem` to stop holding `vk::raii::Device&` and three
  other references to `Engine` members), **53** `CompositePass`,
  **54** `ImGuiPass` + the two layout-transition passes.
- **Verify (each):** Output unchanged after every single step. Five independent, individually
  revertable commits.
- **Size:** M each · **Needs:** 49 (then sequential)

### 55. `BarrierBatcher` — half already built
- **Do:** Collect barriers and emit one `pipelineBarrier2` per batch instead of one per
  barrier (the `TODO` at `rhi/Barrier.h`, formerly `Utility.h:234`).
- **Already done by Stage 5:** `ICommandList::Barrier(std::span<const TextureBarrier>)` emits
  a batch as one command and returns `BarrierCounts`, and the run report already carries
  `barriers` and `barrierCalls` (14 in 9 calls on the baseline scene). The *mechanism* and the
  *measurement* both exist. What remains is a collector that groups automatically instead of
  each call site choosing — and the frame graph in step 56 does exactly that when it inserts
  barriers from declared reads and writes.
- **Therefore:** consider folding this into 56 rather than building a `BarrierBatcher` that
  56 would then replace. Keep it separate only if 56 slips.
- **Verify:** Output unchanged; `barriers`/`barrierCalls` in the report move as expected;
  zero validation errors.
- **Size:** S (was M) · **Needs:** 54

### 56. Frame graph with declared reads/writes
- **Do:** `PassBuilder` declarations, topological sort, automatic layout tracking and barrier
  insertion, transient resource creation from descriptions, pass culling. Delete all
  hand-placed `RecordImageBarrier` calls and the manual fan-out in
  `RecreateSwapchainAndRenderImages`.
- **Verify:** Output unchanged. Resize still clean. Add `contract` tests: cycle detection,
  unused-pass culling, and `LayoutTracker` barrier-legality replay. The ImGui pass is now
  culled when hidden — confirm `RunReport` shows one fewer submitted command buffer, which
  resolves the `TODO` at `main.cpp:581`.
- **Size:** XL · **Needs:** 55

---

## Stage 9 — Data-oriented rewrite (steps 57–68)

### 57. `Handle` / `HandlePool` — ✅ done, during Stage 5
- **Do:** Implement both plus unit tests. Not yet used anywhere.
- **Verify:** `ctest -L unit` covers packing, generation bump on reuse, stale-handle
  rejection, exhaustion. Zero effect on the app.
- **Size:** M · **Needs:** 17
- **What actually happened:** the RHI's handle-based resource model needed these four stages
  early, so they live in `engine/core/include/core/{Handle,HandlePool}.h` with
  `HandleTests.cpp` covering exactly the cases above. Note the free list is FIFO, which is
  what makes eight generation bits enough — a high-churn per-frame user would invalidate that
  reasoning and want a wider field. **Step 58 is untouched:** `Drawable` still holds `Mesh*`
  and `Material*`, so the determinism caveat on step 45 stands in full.

### 58. Convert `Mesh*`/`Material*` to handles
- **Do:** `MeshHandle`/`MaterialHandle` in `Drawable`, `MeshBatch` and `InstanceData`.
  `Drawable::operator<` compares handle indices instead of pointer values.
- **Verify:** Output unchanged **and now reproducible across processes** — run twice from
  scratch and compare `RunReport` batch counts and screenshot bytes. This removes the
  determinism caveat from step 45 and is the prerequisite for golden images.
- **Size:** L · **Needs:** 57

### 59. `Arena`
- **Do:** Frame linear allocator with scoped scratch; two instances for double buffering.
  Report `TransientArenaPeakBytes`.
- **Verify:** Unit tests for alignment, reset and high-water mark. App unchanged.
- **Size:** M · **Needs:** 17

### 60. `FrameSnapshot` + `Extract`
- **Do:** Define the SoA POD snapshot (§11.3). Build it from the *existing* scene graph.
  `ModelManager` consumes the snapshot instead of reading `Model`s directly.
- **Verify:** Output unchanged. Add `tests/support/SnapshotBuilder.h` and the first renderer
  unit test that needs no scene and no GPU.
- **Size:** L · **Needs:** 56, 59

### 61. Cached normal matrices
- **Do:** Compute `transpose(inverse(m))` when a transform changes, not per instance per
  frame (`ModelManager.cpp:67-68`).
- **Verify:** Output unchanged (bit-identical — same maths, fewer times). CPU frame time in
  `RunReport` drops measurably on Sponza.
- **Size:** M · **Needs:** 60

### 62. Radix-sorted `DrawListBuilder`
- **Do:** 64-bit sort keys (§11.5), 3-pass radix sort over `(key, index)` pairs, batching in
  one linear scan writing instance data straight into the mapped buffer.
- **Verify:** Unit test asserts the radix result **equals** a `std::sort` reference on 10k
  random keys. `RunReport` batch and draw counts unchanged. Screenshot unchanged.
- **Size:** L · **Needs:** 58, 60

### 63. Dirty-flag batch regeneration
- **Do:** Regenerate only when a renderable is added/removed or a transform changes.
- **Verify:** Add a `BatchRebuilds` counter: a static scene reports **0 rebuilds after frame
  1**; moving a light or model reports exactly 1. Screenshot unchanged.
- **Size:** M · **Needs:** 62

### 64. `Mat3x4` transforms
- **Do:** 48-byte affine transforms in the snapshot and instance data; shader reads
  `float4x3`.
- **Verify:** Screenshot unchanged or within golden tolerance; instance upload bytes in
  `RunReport` drop ~25%.
- **Size:** M · **Needs:** 62

### 65. Frustum culling
- **Do:** Per-mesh AABBs from assimp (`aiProcess_GenBoundingBoxes`), world-space bounding
  spheres in the snapshot, SIMD sphere-vs-6-planes, `ParallelFor`.
- **Verify:** `InstancesCulled == 0` when the whole scene is in view; `> 0` when facing away;
  **`Instances == InstancesCulled` when the camera faces fully away**. Move the camera slowly
  through Sponza and confirm nothing pops. Unit tests for the 6-plane cases.
- **Size:** L · **Needs:** 62
- **Note:** This is the first step that *legitimately* changes the screenshot only if culling
  is wrong — a strong self-check.

### 66. Steady-state zero-allocation assertion
- **Do:** Counting `operator new` hook active in test builds; assert 0 heap allocations
  between frame 2 and frame N.
- **Verify:** The perf test passes. It will initially **fail** and show you exactly what
  still allocates per frame — that list is the remainder of the DOD work.
- **Size:** M · **Needs:** 63

### 67. ECS `World` and components
- **Do:** Sparse-set `World`, POD components, `TransformSystem` with depth-sorted single-pass
  world matrices and dirty propagation. Port `Camera`, lights and `Model` from
  `SceneComponent` inheritance to components + systems.
- **Verify:** Large unit suite (§15.1 scene rows) including "depth-sorted single pass ==
  recursive reference" on a randomised 1000-node hierarchy. Scenes load and render unchanged.
- **Size:** XL · **Needs:** 60

### 68. `SceneReader`/`SceneWriter` over `SceneDesc`
- **Do:** Port `XmlParser` to read/write a `SceneDesc` decoupled from runtime types. Add a
  `version` attribute to the `.map` format.
- **Verify:** Round-trip unit test is identity on all 3 existing scenes. **Settle the
  open question** from the companion doc: assert that entity + model transforms are not
  double-applied. Malformed XML returns an error instead of throwing through the frame loop.
- **Size:** L · **Needs:** 67

---

## Stage 10 — Scalability features (steps 69–76)

Now genuinely independent of each other — pick by what you want to build. Each is verified by
"screenshot unchanged (or intentionally improved) + `RunReport` counters move in the expected
direction + CI green".

| # | Work | Verify | Size |
|---|---|---|---|
| 69 | Instance data → SSBO indexed by `SV_InstanceID` (`shaderDrawParameters` already enabled at `main.cpp:1016`) | Screenshot unchanged; vertex attributes drop 12 → 4 | L |
| 70 | Bindless texture array + material params SSBO — **deferred until after the D3D12 backend** (D14). It is not a prerequisite for one: Stage 7.5's B3 neutralises the binding model instead, and 70 then lands behind a stable seam where it can be verified on both backends. Until it does, the 3-texture cap stays, and raising it is a contained change (`TextureBinding`, the layout, `PBRMaterial`'s writes, both surface shaders) rather than one entangled with an XL step | Screenshot unchanged; `DescriptorBinds` per frame drops to ~2; the 3-textures-per-material cap is gone | XL |
| 71 | Mega vertex/index buffer | Screenshot unchanged; vertex/index binds drop to 1 per frame | L |
| 72 | Mipmaps + `maxLod = VK_LOD_CLAMP_NONE` (`main.cpp:2109` currently clamps to 0) | Screenshot **changes for the better**; re-baseline. Sponza floor stops shimmering. Aniso finally does something | L |
| 73 | Async asset loading on the job system | Window stays responsive during a Sponza load; a progress counter advances; `resize_and_reload` test passes | XL |
| 74 | Reverse-Z | Screenshot changes slightly; re-baseline. Add a z-fighting test scene at 5000 units | M |
| 75 | Depth prepass | Screenshot unchanged; GPU opaque-pass time drops on Sponza | M |
| 76 | Indirect draws, then GPU culling | `DrawCalls` collapses toward 1; `InstancesCulled` still correct vs the CPU reference | XL |

---

## Dependency summary

The long pole, and the only strictly serial chain:

```
13 → 15 → 20 → 24 → 25 → 26 → 35 → 37 → 38 → 39 ┐
└───────────────── all done ──────────────────┘ ├→ 46 → 47   ← CI goal
                                                │        │
                     41 → 42 → 44 ──────────────┘        │
                     ↑                                   │
                    next                                 ↓
                          B1 → B2 → B3 → B4 → B5 → B6 → D3D12   ← backend goal
```

The chain continues past the CI goal into **Stage 7.5** (`backend_readiness_plan.md`), which
is the serial run-up to a second backend. Step 47 is a genuine prerequisite rather than a
convenient predecessor: it is the instrument that tells you whether the D3D12 backend renders
what the Vulkan one does, and without it every B-step is verified by eye.

The spine starts much further along than it reads: **35, 37, 38 and 39 are all done**, so
the next link in the chain is 41. Step 40a is off the spine entirely — 46 needs it, and
nothing on the spine does.

Everything else hangs off that spine. Notably parallel:

- **Steps 7–11** (build hygiene) share nothing with the spine — do them any time, or hand
  them to someone else. ✅ done.
- **Steps 29–33** (uploads, descriptors, instance buffer, pipeline cache) only need step 26.
  They are the best value-per-hour in the document and are worth doing even if you never go
  headless. ✅ done.
- **Steps 57, 59** (handles, arena) only need step 17 and can be written and unit-tested
  long before they are wired in. **57 is done**; 59 is not.
- **Steps 48, 49** only need 41 and are independent of the pass conversions.
- **Stage 10** items are mutually independent.

Steps you should **not** attempt out of order:

| Don't | Because |
|---|---|
| Any refactor before Stage 0 | No way to prove you didn't change behaviour |
| A second target before step 13 | Headers won't compile outside the PCH |
| Golden images before step 58 | Batch order depends on heap addresses → real flakiness |
| Frame graph (56) before passes are classes (50–54) | Two large refactors superimposed |
| ECS (67) before headless CI (47) | The largest-blast-radius change with no safety net |
| Bindless (70) before growable descriptors (31) | You'd rewrite the descriptor layer twice |
| Bindless (70) before the D3D12 backend | Same reason, one layer up: you'd design the binding seam against one backend, then discover on the second what it assumed (D14) |
| Frame graph (56) before the D3D12 backend | You'd bake one backend's assumptions into the graph, and the backend needs a neutral command list rather than a graph |
| Any B-step before step 47 | Cross-backend verification is a screenshot and a counter diff; 47 is what makes those a test rather than an eyeball |
| Pass classes (50–54) before B1–B6 | `Pass::Execute` would take a `vk::CommandBuffer` and every pass would be edited again (D18) |

---

## Appendix A — File relocation table

| Current | Target | Notes |
|---|---|---|
| `main.cpp` (SDL init/window) | `engine/platform/src/SdlPlatform.cpp` | behind `IPlatform` |
| `main.cpp` (instance/debug/device/queues) | `engine/rhi/src/vulkan/Device.cpp` | present/no-present split |
| `main.cpp` (swapchain + views + recreate) | `engine/rhi/src/vulkan/SwapchainTarget.cpp` | behind `IPresentTarget` |
| `main.cpp` (format selection) | `engine/rhi/src/vulkan/FormatSupport.cpp` | |
| `main.cpp` (3 pipeline creators) | `engine/render/src/passes/*.cpp` | each pass owns its pipeline |
| `main.cpp` (descriptor layouts/pools/sets) | `engine/rhi/src/DescriptorAllocator.cpp` + per-pass | growable |
| `main.cpp` (command pools/buffers) | `engine/render/src/FrameResources.cpp` | table-driven, per-thread pools |
| `main.cpp` (5 `Record*` + 2 layout) | `engine/render/src/passes/*.cpp` | `Pass::Execute` |
| `main.cpp` (`GlobalBuffer`/`CameraData`/`LightData`) | `engine/render/include/render/shared/ShaderTypes.h` | shared with Slang |
| `main.cpp` (render targets, depth, quad, instance, UBO) | `engine/render/src/FrameGraph.cpp` + `FrameResources.cpp` | transient resources |
| `main.cpp` (`Run`, timing) | `engine/engine/src/Engine.cpp` | `IClock`, `RunSpec` |
| `main.cpp` (input handling) | `engine/engine/src/InputSystem.cpp` | + `ScriptedInput` |
| `main.cpp` (`DrawImGuiFrame`) | `engine/editor/src/EditorLayer.cpp` + `panels/` | |
| `main.cpp` (ImGui backend init) | `engine/editor/src/ImGuiBackend.cpp` + `render/passes/ImGuiPass.cpp` | |
| `main.cpp` (`main()`) | `apps/hikariengine/main.cpp` | ~40 lines |
| `Log.h` `Timer.h` `MyMacros.h` | `engine/core/include/core/` | + `LogSink`, `Assert.h` |
| `ThreadPool.*` | `engine/core/src/WorkStealingJobSystem.cpp` | + `SerialJobSystem` |
| `SwapbackArray.h` | `engine/core/include/core/SwapbackArray.h` | + `#pragma once` fix |
| `Common.h` | `engine/render/include/render/shared/Limits.h` | |
| `pch.h` | per-module PCHs | optimisation only, never a dependency |
| `Utility.h` | split → `rhi/{Buffer,Image,FormatSupport,DebugNames}.h`, `platform/FileSystem.h` | 339 lines → 5 focused files |
| `AllocatedBuffer.*` `AllocatedImage.*` `VulkanAllocator.h` `VMAImpl.cpp` | `engine/rhi/src/vulkan/` | |
| `Barrier.h` | `engine/rhi/include/rhi/Barrier.h` | + `BarrierBatcher` |
| `PipelineBuilder.*` `ComputePipelineBuilder.*` | `engine/rhi/` | merge the shared parts |
| `Texture.*` `Cubemap.*` | `engine/rhi/{Image,TextureView}.h` + `assets/TextureData.h` | split GPU vs CPU |
| `TextureLoader.*` `CubemapLoader.*` `ModelLoader.*` | `engine/asset/src/importers/` | pure `→ *Data` functions |
| `ResourceManager.*` `ResourceCache.h` | `engine/asset/{AssetRegistry,AssetCache}.h` | keyed by `AssetId` |
| `Model.*` `ModelData.*` `Mesh.*` | `assets/MeshData.h` + `assets/GpuMeshRegistry.h` + `scene/components/MeshRenderer.h` | one class → three concerns |
| `Material.*` `PBRMaterial.*` `MaterialFactory.*` | `assets/MaterialData.h` + `assets/MaterialRegistry.h` | params → SSBO |
| `Node.*` | `engine/asset/src/importers/ModelImporter.cpp` | import-time only; not runtime state |
| `Entity.*` `Component.h` `SceneComponent.*` `LogicComponent.h` | `engine/scene/{World,Entity}.h` + `components/` | ECS |
| `SceneGraph.h` | `engine/scene/World.h` + `scene/SceneDesc.h` | |
| `Transform.*` | `scene/components/Transform.h` + `systems/TransformSystem.cpp` | quaternion rotation |
| `Camera.*` | `scene/components/Camera.h` + `systems/CameraSystem.cpp` + `render/View.h` | |
| `Lights.h` | `scene/components/Light.h` + `systems/LightSystem.cpp` + `shared/ShaderTypes.h` | |
| `ModelManager.*` `Drawable.h` | `render/{RenderScene,DrawListBuilder,FrameSnapshot}.h` | the DOD rewrite |
| `InstanceData.h` | `render/shared/ShaderTypes.h` (`GpuInstance`) | `Mat3x4`, SSBO |
| `FrameData.h` | `render/FrameResources.h` | pass-agnostic |
| `CloudSystem.*` | `render/passes/CloudPass.*` + `render/CloudNoiseBaker.*` | no `Device&` members |
| `XmlParser.*` | `scene/serialization/{SceneReader,SceneWriter,XmlBackend}.*` | `SceneDesc` in/out |
| `src/shaders/*` | `shaders/` (+ `shaders/include/`) | output to `<exe dir>/shaders/` |
| `models/` `textures/` `scenes/` | `content/` | single content root |

## Appendix B — Frame budget model

Rough CPU cost of the current design vs the target, at 10,000 instances / 500 batches on
one core. Indicative, not measured — the point is the *shape*, and these are the numbers
the `perf` tests should watch.

| Stage | Current | Target | Mechanism |
|---|---|---|---|
| Gather drawables | ~0.35 ms + 1 heap alloc per model | ~0 ms (cached) | Persistent `RenderScene` + dirty flags |
| World transforms | recursive parent walk per query | ~0.02 ms dirty-only | Depth-sorted single linear pass |
| Normal matrices | ~0.9 ms (10k × `inverse`+`transpose`) | ~0 ms (cached) | Recompute on transform change only |
| Culling | *none* | ~0.08 ms | SIMD sphere/frustum, `parallel_for` |
| Sort | ~1.4 ms (`std::sort`, 96 B elements) | ~0.15 ms | Radix sort, 12 B pairs |
| Batch + instance write | ~0.30 ms + `vector` growth | ~0.06 ms | Single scan, direct write to mapped ring |
| Encode | 5 binds + 1 push + 1 draw per batch | 1 draw per batch | Bindless + mega-buffer + material SSBO |
| Steady-state heap allocations | ≥ 1 per model per frame | **0** | Arena + persistent arrays |
| Instance upload bandwidth | 128 B × 10k = 1.28 MB/frame | 96 B × visible | `Mat3x4` × 2 |
| Vertex attribute slots used | 12 of 16 (4 vertex + 8 instance) | 4 of 16 | Instances via SSBO + `SV_InstanceID` |

Two additional hard limits removed along the way: `MAX_INSTANCE_COUNT = 1024` (currently
`throw`s — replaced by a growable ring buffer) and `s_MAX_MATERIAL_SET_COUNT = 100`
(replaced by bindless + a growable descriptor allocator).

---

## Summary

The codebase has good bones: `vulkan-hpp` RAII throughout, dynamic rendering + sync2, debug
names on everything, an `ImageBarrierDesc` abstraction, a real `PipelineBuilder`, a
`weak_ptr` resource cache, per-pass command pools and parallel recording. The problem is not
quality — it is **shape**. One 2,453-line file owns fourteen subsystems, there is exactly one
build target, and nothing hardware-facing has a seam.

The critical path to the stated goals is short and specific:

1. **Split into libraries** (nothing is testable until something is linkable).
2. **Put presentation behind `IPresentTarget`** and split present-only device requirements
   out (this single change is what makes headless possible).
3. **Add `RunSpec`/`RunReport` and a bounded loop** (this is what makes a headless run
   *assertable* rather than merely possible).

Those three are Phases 1–3, roughly 3–4 weeks part-time, and they deliver the CI goal —
`ctest -L gpu` launching a scene headless under ASan with zero validation errors — while the
rendering code is still substantially as it is today. The data-oriented rewrite (Phases 4–6)
is then done with a regression net underneath it, which is the only way a hot-path rewrite of
this size is safe.
