# HikariEngine — Codebase Review & Suggested Work

**Date:** 04/08/2026

**Scope:** Full review of `src/`, `src/shaders/`, `CMakeLists.txt`, `CMakePresets.json` and the build scripts.

**Status:** In progress; completed tasks are removed as they land. Last reconciled against
the tree on 25/08/2026, after Stage 5 and Part IV steps 35-36.

---

## Table of Contents

1. [How to read this document](#1-how-to-read-this-document)
2. [What the project does well](#2-what-the-project-does-well)
3. [Part 1 — Correctness bugs](#part-1--correctness-bugs)
4. [Part 2 — Architecture & code structure](#part-2--architecture--code-structure)
5. [Part 3 — Rendering & performance](#part-3--rendering--performance)
6. [Part 4 — Shaders](#part-4--shaders)
7. [Part 5 — Build system & tooling](#part-5--build-system--tooling)
8. [Part 6 — Prioritised work order](#part-6--prioritised-work-order)

---

## 1. How to read this document

Each item has:

- **Where** — file and line references (as of this review).
- **What** — the problem.
- **Why it matters** — the observable consequence.
- **Fix** — a concrete suggested change.

Severity tags:

| Tag | Meaning |
|---|---|
| **P0** | Actively wrong / undefined behaviour / crash risk. Fix first. |
| **P1** | Wrong output or a latent bug that will bite soon. |
| **P2** | Design/architecture debt. Fix before the codebase grows further. |
| **P3** | Performance or polish. |

---

## 2. What the project does well

Worth stating explicitly, because these are the parts you should *not* rewrite:

- **`vulkan-hpp` + RAII throughout.** Almost no manual `vkDestroy*`. Very few leak
  opportunities in the Vulkan layer itself.
- **Dynamic rendering + `synchronization2`** rather than `VkRenderPass`/`VkFramebuffer`
  boilerplate. This is the right modern choice and it keeps the pass code readable.
- **`SetVkDebugName` on essentially every object.** Debugging in RenderDoc/NSight will be
  dramatically easier than in most hobby renderers. The `[[maybe_unused]]` +
  `#ifdef DEBUG` pattern is clean.
- **Weighted-blended OIT** is implemented correctly against the Casual Effects reference,
  including the `isinf` guard and the revealage blend factors.
- **Per-pass command pools per frame-in-flight**, with parallel recording of the two
  expensive passes on the thread pool. This is a real design, not an accident.
- **Batching by mesh+material with instanced `drawIndexed`** and a persistently mapped
  instance buffer. Correct instinct.
- **Platform abstraction is honest.** The macOS/MoltenVK comments in `CMakeLists.txt`,
  `pch.h` and `CreateSurface()` explain *why*, not *what*. Keep writing comments like that.
- **Volumetric clouds with a GPU-baked Perlin–Worley 3D texture** at quarter resolution,
  depth-aware. Ambitious and structurally sound.

---

# Part 1 — Correctness bugs

## 1.6 — **P0** `ModelData::Init` will throw or crash on sparse mesh indices

**Where** `src/ModelData.cpp:11-36`

**What**

```cpp
for (const Mesh& mesh : m_Meshes)
{
    Mesh* pMesh = const_cast<Mesh*>(&mesh);
    uint32_t meshIndex = mesh.GetMeshIndex();
    for (const glm::mat4& transform : m_MeshLocalTransforms.at(meshIndex))
    {
        m_Drawables.push_back(
            Drawable{.pMesh = pMesh,
                     .pMat  = mesh.GetMaterial(),
                     .blendMode = mesh.GetMaterial()->GetBlendMode(),  // nullptr deref
                     .Transform = transform});
    }
}
```

Two problems:

1. `resize(meshIndex + 1)` creates *default-constructed* `Mesh` objects for any index gap.
   Those have `m_bIsValid == false`, `m_MeshIndex == 0` and `m_Material == nullptr`.
   The loop does not skip them. `mesh.GetMaterial()->GetBlendMode()` is a null dereference.
2. Every invalid mesh reports `GetMeshIndex() == 0`, so it also duplicates mesh 0's
   drawables. If mesh 0 was itself never registered, `m_MeshLocalTransforms.at(0)` throws
   `std::out_of_range`.

The `const_cast` is also a smell: iterate by non-const reference instead.

**Fix**

```cpp
for (Mesh& mesh : m_Meshes)
{
    if (!mesh.IsValid())
        continue;

    const auto it = m_MeshLocalTransforms.find(mesh.GetMeshIndex());
    if (it == m_MeshLocalTransforms.end())
        continue;

    for (const glm::mat4& transform : it->second)
    {
        m_Drawables.push_back(Drawable{.pMesh = &mesh,
                                       .pMat = mesh.GetMaterial(),
                                       .blendMode = mesh.GetMaterial()->GetBlendMode(),
                                       .Transform = transform});
    }
}
```

Item 1.5 (`RegisterMesh` handing out pointers that `resize` invalidates) is already fixed,
which removes the "gap" case at its source — but keep the `IsValid()` guard as defence in
depth: an assimp scene can legitimately contain a mesh that no node references.

---

## 1.14 — **P2** The skybox is loaded but never rendered

**Where** `src/main.cpp:256-268` (load), `364-367` (unload)

**What**

`m_pSkybox` is loaded from `textures/skybox/*.jpg`, stored, and unloaded at shutdown. It
is never bound to a descriptor, never sampled, and there is no skybox pipeline. The
background is `SKY_COLOR` as a clear colour instead (`main.cpp:1675-1676`).

**Why it matters**

~6 JPEGs of VRAM and load time for nothing. It also meant the since-fixed
face-0-only cubemap upload bug went unnoticed for as long as it did — nothing sampled the
result. That one now has a gpu test ("Every cubemap face lands on its own layer").

**Fix**

Either delete the load, or finish the feature. Finishing it is cheap and high-impact:

1. Add the cubemap to the global/composite descriptor set.
2. Draw it in the opaque pass with a full-screen triangle, reconstructing the view ray
   from `InvViewProj` (`clouds.comp.slang:63-69` already does exactly this — reuse the
   code), depth test `eLessOrEqual`, depth write off, drawn last.
3. Use the same cubemap for the ambient/reflection terms currently stubbed as
   `globalBuffer.SkyColor` in `opaque.slang:162` and `weightedBlendedOIT.slang:173`
   (which is already marked `// TODO: replace with environment map`).

---

# Part 2 — Architecture & code structure

## 2.1 — **P2** `main.cpp` is the whole engine

~2,600 lines, down from 2,765 at the time of the review — SDL init, device creation and
swapchain management have since moved out to `Engine::Platform` and `Engine::RHI`. What is
left is still most of an engine: three pipeline builders, descriptor layout/pool/set
management, all seven command-buffer recorders, render-target management, the global uniform
buffer layout, the ImGui editor UI, the frame loop, input handling, and `main()`.

The suggested layout below predates `architecture_plan.md` §8-§9, which supersedes it — nine
layered CMake targets rather than folders under `src/`, with the layering enforced by the
build system. Read that instead; this is kept for the extraction *order*, which still holds.

This is the single biggest brake on the project's velocity. Everything else in this
document is easier after this split.

**Suggested target layout** (incremental — do it one box at a time, keep it compiling):

```
src/
  Core/        Log, Timer, ThreadPool, MyMacros, SwapbackArray, Common
  RHI/         VulkanContext (instance/device/queues/debug messenger)
               Swapchain     (swapchain + views + recreate + format choice)
               Image, Buffer (wrap the free functions in Utility.h — since done)
               PipelineBuilder
               DescriptorAllocator
  Renderer/    Renderer      (frame loop, submit, present)
               OpaquePass, TransparentPass, CompositePass, CloudPass, ImGuiPass
               FrameData, GlobalBuffer
  Scene/       Entity, SceneComponent, SceneGraph, Transform, Node, Camera, Lights
  Assets/      ResourceManager, *Loader, Model, ModelData, Mesh, Material, Texture, Cubemap
  Editor/      EditorUI (everything currently in DrawImGuiFrame)
  main.cpp     ~40 lines: init SDL, construct App, Run, catch
```

**Suggested order of extraction**, cheapest and safest first:

1. `DrawImGuiFrame` → `Editor/EditorUI.cpp`. Pure UI, almost no coupling. Immediately
   removes ~120 lines and gives you somewhere to put the debug controls you will want for
   the cloud parameters and tonemapping.
2. `GlobalBuffer` / `CameraData` / `LightData` structs → `Renderer/GlobalBuffer.h`. These
   must stay byte-compatible with `src/shaders/common.slangh` — see
   [4.1](#41--p2-share-one-source-of-truth-for-gpu-struct-layouts).
3. Pipeline creation → `RHI/PipelineBuilder`. Done — the three near-identical builders
   were merged during Stage 5.
4. Instance/device/surface/debug-messenger → `RHI/VulkanContext`.
5. Swapchain + views + `RecreateSwapchainAndRenderImages` → `RHI/Swapchain`.
6. The five `Record*CommandBuffer` functions → one class per pass, behind a common
   `IRenderPass { void Record(FrameContext&); }` interface. This is what makes adding
   shadow maps or a depth prepass a bounded change instead of another 200 lines in
   `main.cpp`.

## 2.6 — **P2** Fixed limits that are too low, and fail loudly rather than gracefully

**Three of the four are fixed.** The instance buffer grows (`App::GrowInstanceBuffers` — the
wait before the swap is the load-bearing part, not the reallocation); the material descriptor
pool grows (`DescriptorAllocator::Grow`, with `kInitialMaterialSetCapacity` now documented as
"a starting size, not a ceiling"); and exceeding the light limits logs a warning instead of
dropping silently.

**What is left:**

| Limit | Where | Value | Problem |
|---|---|---|---|
| `TextureBinding::COUNT` | `engine/engine/src/Texture.h` | 3 | Albedo / Normal / MetallicRoughness only — the ceiling on emissive, occlusion or clearcoat maps |

**Fix**

The three-texture cap is no longer a descriptor-pool ceiling, so it will not abort — it is
now purely an expressiveness limit in the material model. Adding a fourth slot means touching
the enum, the descriptor set layout, `PBRMaterial`'s writes and both surface shaders in step.
That argued for doing it as part of **step 70 (bindless texture array + material params
SSBO)** rather than on its own, since 70 removes the cap entirely rather than raising it by
one.

> **Superseded, 6 September 2026.** Two things changed. Stage 7.5's D14 defers bindless until
> after the D3D12 backend, so step 70 is much further out than it was when this was written —
> and the argument "wait for 70" weakens as 70 recedes. And the premise above is wrong about
> what the cap costs: emissive and occlusion are not parsed by the loader, are absent from
> `MaterialData`, and are named by no shader, so nothing is currently being *held back* — the
> cap bounds a feature that does not exist rather than blocking one that half does. Raising it
> is therefore new feature work, which is why Stage 7.5 excluded it under its inclusion test
> and why it is now its own `backlog.md` row rather than a rider on step 70. It also gets
> cheaper after Stage 7.5's step 5, which rewrites the material set and splits combined image
> samplers: after that, a fourth map is an enum value, a layout entry, a `PBRMaterial` write
> and two shaders, with no descriptor-pool consequences. It **changes the baseline
> deliberately**, so it is scheduled rather than opportunistic.

## 2.7 — **P3** Smaller code-quality items

- **`main.cpp:531` / `530`** — the `// TODO: even when ImGui is not showing, it's being
  submitted` comment is correct. `RecordImGui` always begins/ends a render pass and is
  always submitted. Either skip the command buffer in the submit array when
  `!m_bCursorVisible`, or fold ImGui into the composite pass.
- **`main.cpp:1553-1655` (`CreateCommandBuffers`)** — 100 lines of seven identical
  seven-line blocks. Loop over a small table of `{pool, &FrameData::member, name}`.
- **`main.cpp:2479-2551` (`CreateRenderTargets`)** — three identical 22-line blocks. A
  `CreateRenderTarget(format, name)` helper returning a `Texture` collapses it to 6 lines.
- **`CubemapLoader.cpp:51-90`** — the 6-case `switch` mapping index → path is more code
  than the data. Give `CubemapCreateInfo` a `std::array<std::string, 6> FacePaths` (or a
  `GetFace(size_t)` accessor) and loop.
- [DONE] **`Camera.h`** — `m_MoveSpeed` and `m_LookSens` now have in-class initialisers.
- [DONE] **`Entity.h:76`, `Model.h:19`** — `static constexpr Transform GetDefaultTransform()`
  was duplicated and just returned `Transform{}`. Deleted both; `Transform{}` is clearer.
- **`Entity.h:22-71`** — the four `GetComponents`/`GetFirstComponent` overloads use
  `dynamic_cast` in a loop. Fine at current scale; if the scene grows, a type-id keyed
  map is the usual next step. Superseded by the ECS at step 67 rather than worth fixing
  separately.
- [DONE] **Tab/space mixing** — `.clang-format` is in place with a pinned version, enforced
  by `tests/scripts/format_check.sh` and by CI on all nine configurations.
- **`XmlParser.cpp:281-282, 291-292`** — `append_attribute(...) = Vec3ToString(...)`
  passes a `std::string` where the sibling `WriteTransform` (lines 257-261) passes
  `.c_str()`. Make them consistent.
- **`XmlParser`: round-tripping transforms.** `SaveScene` writes a transform on both the
  `<entity>` and each `<model>` (lines 310, 268), and `LoadScene` applies both
  (lines 192, 84). Since `Model::GetDrawables()` multiplies by
  `GetAccumulatedTransform()`, confirm that save→load is idempotent and doesn't
  double-apply the entity transform.

---

# Part 3 — Rendering & performance

## 3.1 — **P1** No mipmaps anywhere

**Where** `rhi/TextureDesc.h` (`MipLevels = 1`), `rhi/TextureViewDesc.h` (`MipCount = 1u`),
`rhi/SamplerDesc.h` (`MaxLod = 0.f`) and `App::CreateTextureSampler`, which takes that default

`Utility.h` no longer exists; these moved into the RHI during Stage 5, and the descs carry the
fields — nothing generates a chain or raises `MaxLod`. Anisotropy *is* enabled on the sampler,
so it is currently paid for and does nothing.

Every texture is created with a single mip level and the sampler is clamped to LOD 0,
despite `mipmapMode = eLinear` and full anisotropy being enabled.

**Why it matters**

- Severe aliasing and shimmering on any surface viewed at an angle or at distance —
  Sponza's floor and curtains will crawl badly as the camera moves.
- Anisotropic filtering does almost nothing without a mip chain.
- Every sample is a full-resolution texture fetch, so texture cache hit rates are poor.
  This is often a large fraction of fragment cost in a scene like Sponza.

**Fix**

In `TextureLoader::Load`:

1. `mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1`.
2. Add `vk::ImageUsageFlagBits::eTransferSrc` to the image usage.
3. After the buffer→image copy, generate the chain with `cmd.blitImage` in a loop,
   transitioning mip *i* to `eTransferSrcOptimal` and mip *i+1* to `eTransferDstOptimal`.
   Check `eSampledImageFilterLinear` in `getFormatProperties(format).optimalTilingFeatures`
   first.
4. `CreateImageView` needs `levelCount = mipLevels` (add a parameter).
5. Sampler `maxLod = VK_LOD_CLAMP_NONE`.

Do the same for cubemap faces — the upload path itself is already correct.

Alternative worth considering: a compute-shader downsample pass, which avoids the
`blitImage` format-support caveat and is faster for large atlases. The `blit` route is
simpler and correct — start there.

## 3.3 — **P3** Asset loading blocks the frame loop

**Where** `src/main.cpp:623-649` (the `LoadSceneDlg` handler)

```cpp
m_Device.waitIdle();
std::unique_ptr<SceneGraph> tempSceneGraph = XmlParser::LoadScene(path);
```

Assimp import, image decode, staging, upload and ~70 `waitIdle`s all happen inside the
ImGui frame. The window is frozen (and marked "not responding" by Windows) for the
duration.

The `ThreadPool` already exists and is currently used only for command-buffer recording.

**Fix**

1. Dispatch `XmlParser::LoadScene`'s CPU work (assimp parse, `stbi_load`) to the thread
   pool, returning a `std::future<std::unique_ptr<SceneGraph>>`.
2. Poll the future in the frame loop; keep rendering the old scene meanwhile, with a
   progress indicator.
3. Perform GPU uploads on the main thread from a queue of completed CPU work (or on a
   dedicated transfer queue with its own command pool — Vulkan command pools are not
   thread-safe, so one pool per thread).

The existing comment at `main.cpp:634-639` about loading-before-unloading is good
thinking and stays valid.

## 3.4 — **P3** Batches, instance data and the global buffer are rebuilt from scratch every frame

**Where** `src/main.cpp:232` (`ModelManager::Get()->GenerateBatches()`),
`src/ModelManager.cpp:21-80`, `src/Model.cpp:19-29`

Every single frame, unconditionally:

```cpp
m_Drawables.clear();  m_InstanceDatas.clear();
m_OpaqueBatches.clear();  m_TransparentBatches.clear();

for (Model* pModel : m_Models)
{
    const std::vector<Drawable> drawables = pModel->GetDrawables();  // allocates a vector
    m_Drawables.insert(...);                // copies it
}
std::sort(m_Drawables.begin(), m_Drawables.end());                // full re-sort
// then a full pass computing glm::transpose(glm::inverse(transform)) per instance
```

Costs: one heap allocation per model per frame, a full copy of every drawable, a full
`std::sort` of every drawable, and a 4×4 matrix inverse per instance per frame — all for a
scene that is usually static.

**Fix**

1. **Dirty flag.** `ModelManager` regenerates only when a `Model` is added, removed, or
   its transform changes. `SceneComponent` transforms already go through
   `GetTransform()`, so a `MarkDirty()` there is straightforward.
2. **`Model::GetDrawables()` should not allocate.** Change it to
   `void AppendDrawables(std::vector<Drawable>& out) const` and `reserve` once in
   `GenerateBatches`.
3. **Cache the normal matrix.** `glm::transpose(glm::inverse(m))`
   (`ModelManager.cpp:63-64`) is expensive and only changes when the transform does.
   Compute it in `ModelData`/`Model` when the transform is set. Better: upload a `3x3`
   (padded) instead of a full `4x4` — the shader already casts to `float3x3`
   (`opaque.slang:70`) and there is a `// TODO: only upload 3x3 matrix` at
   `opaque.slang:48`. That halves the per-instance vertex-attribute bandwidth and frees
   4 of the 8 instance attribute slots.
4. **Frustum culling.** There is none. Add AABBs per mesh (assimp gives you
   `aiMesh::mAABB` with `aiProcess_GenBoundingBoxes`), transform to world, and test
   against the six frustum planes extracted from `viewProj` before emitting a drawable.
   With Sponza this alone is usually a large win when the camera is inside the atrium.

## 3.6 — **P3** Missing rendering features, in rough order of visual impact

These are features, not bugs — listed so the priorities are explicit.

| Feature | Why | Rough size |
|---|---|---|
| **Shadow maps** | Biggest single visual upgrade. Directional light → cascaded shadow maps; the OIT and opaque passes both just need a shadow lookup. | Large |
| **IBL / environment lighting** | `ambient = 0.1f * SkyColor * albedo * ao` (`opaque.slang:162`) is a flat hack. A prefiltered cubemap + BRDF LUT makes PBR materials actually read as metal/dielectric. The skybox cubemap is already loaded ([1.14](#114--p2-the-skybox-is-loaded-but-never-rendered)). | Medium-large |
| **Reverse-Z depth** | `NEAR_PLANE = 0.1f`, `FAR_PLANE = 10000.f` (`main.cpp:29-30`) is a 100,000:1 ratio — severe z-fighting at distance. Flip to `eGreater` + clear to 0 + swap near/far in `glm::perspective`. `GLM_FORCE_DEPTH_ZERO_TO_ONE` is already set, so this is a small, contained change. | Small |
| **Depth prepass** | Sponza has heavy overdraw and the PBR fragment shader is expensive. A depth-only prepass with `eEqual` in the main pass is a straightforward win. Also gives the cloud pass complete depth earlier. | Small-medium |
| **AO (SSAO/GTAO)** | `pcMatData.AO` is a per-material constant; there is no screen-space AO. | Medium |
| **Bloom** | You already render to `eR16G16B16A16Sfloat` and tonemap. Bloom is cheap to add and makes HDR lighting read correctly. | Small-medium |
| **Anti-aliasing** | `rasterizationSamples = e1` everywhere and no post-process AA. FXAA in the composite pass is ~40 lines; TAA needs motion vectors. | Small (FXAA) |
| **Transparent sorting hint** | Weighted-blended OIT is order-independent, which is the point — but it is an approximation. If specific objects need correctness, a per-object flag routing them to a sorted back-to-front pass is a useful escape hatch. | Medium |

## 3.7 — **P3** Cloud shader cost

**Where** `src/shaders/clouds.comp.slang:114-150`

Two issues in the march loop:

```cpp
for (uint i = 0u; i < pc.viewStepCount; i++)
{
    float3 samplePos = rayOrigin + rayDir * t;
    float density = SampleDensity(samplePos);        // 2 texture fetches

    float sunTNear, sunTFar;
    IntersectHeightSlab(samplePos, sunDir, ...);      // computed unconditionally
    sunTNear = max(sunTNear, 0.f);
    float sunStepSize = (sunTFar - sunTNear) / float(pc.sunStepCount);
    float sunT = sunTNear;

    if (density > 0.f) { /* sun march */ }
    ...
}
```

1. **The sun-slab setup runs even when `density == 0`.** Move
   `IntersectHeightSlab` and the `sunStepSize` computation inside the
   `if (density > 0.f)` block. Free win.
2. **`SampleDensity` costs two 3D texture fetches** — one for the density, one for the
   boundary perturbation (lines 35and 39). The boundary noise only depends on `pos.xz`,
   so it is constant along a vertical ray and could be hoisted out of the march for
   near-vertical rays, or baked into a separate 2D texture.

Additional notes:

- **`DirLights[0]` is read unconditionally** (lines 106-108) without checking
  `globalBuffer.Lights.DirLightCount > 0`. With no directional light in the scene,
  `sunDir = normalize(-float3(0))` is a NaN, which propagates into `phase` and then into
  the output image. Guard it, or early-out the whole dispatch.
- **`sunTFar` can be `-inf`/`+inf`** when `sunDir.y` is near zero
  (`IntersectHeightSlab` divides by `dir.y`, lines 24-25). The viewray has an
  `if (... || rayDir.y == 0.f)` guard (line 96) but the sun ray has none, and `== 0.f` is
  too strict anyway — use an epsilon on `abs(dir.y)` for both.
- **The quarter-res output is upsampled with a plain bilinear `Sample`**
  (`composite.slang:35`) with no depth-aware filtering, so clouds bleed across geometry
  silhouettes. A bilateral/nearest-depth upsample fixes the halos.
- **Temporal reprojection** with a 4- or 16-frame Bayer offset would let you cut
  `viewStepCount` substantially. This is how production volumetric clouds are affordable.
- The push-constant parameters (`windVelocity`, `minHeight`, `maxHeight`, `coverage`,
  `anisotropy`, `boundaryDisplacement`, `viewStepCount`, `sunStepCount`) are never
  exposed in the ImGui panel. `m_CloudData` is pushed from
  `CloudSystem::RecordDispatch` (line 353-354) but nothing writes it. Add sliders — you
  cannot tune clouds without them.

## 3.8 — **P3** Document the matrix convention (it is currently correct but inconsistent)

**Where** `src/main.cpp:2180`, `2195`, `2197-2198`; `src/shaders/opaque.slang:1-3, 66-67`;
`src/shaders/clouds.comp.slang:63-64`

Two different conventions are in use:

```cpp
// main.cpp — View and Proj are transposed on upload
m_GlobalBuffer.CamData.Proj = glm::transpose(colMajProj);
m_GlobalBuffer.CamData.View = glm::transpose(view);

// ...but InvViewProj is not
m_GlobalBuffer.CamData.InvViewProj =
    glm::inverse(glm::transpose(m_GlobalBuffer.CamData.Proj) * view);
```

and correspondingly in the shaders:

```hlsl
// opaque.slang — vector-first
o.Pos = mul(mul(worldPos, globalBuffer.Camera.View), globalBuffer.Camera.Proj);

// clouds.comp.slang — matrix-first
float4 nearPoint = mul(globalBuffer.Camera.InvViewProj, float4(ndc, 0.f, 1.f));
```

The two combinations (transpose-on-upload + `mul(v, M)`) and (no-transpose +
`mul(M, v)`) are mathematically equivalent, so **this is not a bug** — but it means a
reader has to re-derive the convention for every matrix, and the `Node.cpp:26`
`// TODO: this is already transposed somehow` comment suggests it has already cost you
time.

**Fix**

Pick one convention, apply it to all matrices, and state it once. The file header comment
in `opaque.slang:1-3` is the right place; extend it and reference it from
`common.slangh`. Then either transpose `InvViewProj` too and use `mul(v, M)` in
`clouds.comp.slang`, or stop transposing `View`/`Proj` and use `mul(M, v)` everywhere.
Consider passing `-matrix-layout-row-major` (or column-major) to `slangc` explicitly in
`CMakeLists.txt` so the layout does not depend on the compiler default.

---

# Part 4 — Shaders

## 4.1 — **P2** Share one source of truth for GPU struct layouts

`src/shaders/common.slangh:3-45` declares `PointLight`, `DirLight`, `LightData`,
`CameraData` and `GlobalBuffer`. `src/main.cpp:49-78` declares the same five structs
again in C++. They are kept in sync by hand.

`common.slangh` already does `#include "../Common.h"` for `MAX_POINT_LIGHTS` /
`MAX_DIR_LIGHTS`, so the mechanism exists.

**Why it matters**

The comment at `main.cpp:69-71` explains std140/std430 alignment rules, and there is a
runtime guard (`main.cpp:2144-2146`) that `sizeof(GlobalBuffer) % 16 == 0`. But a
mismatch in *field order* or a differently-sized padding member passes that check and
produces garbage lighting that looks like a shader bug.

**Fix**

Move the shared structs into a header included by both, using a small compatibility
shim:

```c
// src/shaders/SharedTypes.h— included from C++ and from Slang
#ifdef __cplusplus
    #include "glm/glm.hpp"
    using float2 = glm::vec2;
    using float3 = glm::vec3;
    using float4 = glm::vec4;
    using float4x4 = glm::mat4;
    using uint = uint32_t;
#endif

struct PointLightData { float3 Color; float Intensity; float3 Pos; float Padding; };
// ... etc
```

Then add `static_assert(sizeof(GlobalBuffer) == <expected>)` and `offsetof` assertions on
the C++ side so a divergence is a compile error.

## 4.2 — **P2** `opaque.slang` and `weightedBlendedOIT.slang` duplicate ~130 lines

The two files are byte-identical from line 7 (`struct MaterialPushConstant`) through
line ~165: the same push-constant struct, the same `VS_In`/`VS_Out`, the same `vertMain`,
the same three texture bindings, the same albedo/metallic/roughness sampling, the same
TBN normal mapping, the same two-sided normal flip, and the same two light loops.

They diverge only in the fragment output: `opaque` returns `float4(color, albedo.a)`,
`weightedBlendedOIT` computes the weight and writes `Accum`/`Revealage`.

**Why it matters**

Every material or lighting change has to be made twice, correctly. This is exactly the
kind of duplication that produces "transparency looks different from opaque" bugs later.

**Fix**

Extract `src/shaders/surface.slangh`:

```hlsl
// surface.slangh
struct MaterialPushConstant { /* ... */ };
[[vk::push_constant]] MaterialPushConstant pcMatData;

[[vk::binding(0, 1)]] Sampler2D albedoTex;
[[vk::binding(1, 1)]] Sampler2D normalTex;
[[vk::binding(2, 1)]] Sampler2D metallicRoughnessTex;

struct VS_In  { /* ... */ };
struct VS_Out { /* ... */ };

VS_Out TransformVertex(VS_In v);

struct SurfaceSample
{
    float4 Albedo;
    float3 N;
    float3 V;
    float  Metallic;
    float  Roughness;
    float  AO;
};
SurfaceSample SampleSurface(VS_Out f);
float3 ShadeSurface(SurfaceSample s);   // both light loops + ambient
```

`opaque.slang` becomes ~20 lines and `weightedBlendedOIT.slang` ~35.

Also worth cleaning while you are there:

- [DONE] `VS_Out::Color : TEXCOORD1` was declared and interpolated in both shaders but never
  written or read. Deleted, and the remaining semantics renumbered to close the gap.
- `weightedBlendedOIT.slang:178-181`: `transmit` is a hardcoded `float3(0,0,0)`, so
  `premultipliedReflect.a *= 1.f - clamp(0, 0, 1)` is a no-op multiply by 1. Either wire
  up per-material transmission or delete the dead arithmetic and the `transmit`
  declaration.
- `weightedBlendedOIT.slang:188`: the commented-out `b /= sqrt(1e4 * abs(csZ));` refers to
  `csZ`, which does not exist in this shader (the reference implementation's camera-space
  Z). If you ever enable it you will need to pass view-space depth through `VS_Out`. Note
  that in the comment so future-you doesn't chase it.
- `clouds.comp.slang` includes `pbr.slangh` (line 2) but only uses
  `HenyeyGreenstein`. `composite.slang` includes it (line 2) for `HillACES` only. Consider
  splitting `pbr.slangh` into `brdf.slangh` (Fresnel/GGX/Smith) and
  `tonemap.slangh` + `phase.slangh`, so a change to the BRDF doesn't force a recompile of
  the clouds and composite shaders.

# Part 5 — Build system & tooling

**Nothing outstanding.** Every item this part raised has been done: the `compile_commands.json`
guard, the MSVC debug-information split, the Ninja presets for Windows, a shared
`VCPKG_INSTALLED_DIR`, the `/DEBUG` argument typo, CWD-relative asset paths (replaced by
`Paths` and a content root), and tests + CI (Catch2, CTest labels, a nine-configuration
matrix). Shader compilation ergonomics moved into `cmake/Shaders.cmake` with a real
dependency graph, `-warnings-as-errors all` and a `spirv-val` pass.

New build-system work is tracked in `backlog.md` rather than reopened here — the one open
item, `rhi_boundary_check` running in precommit but not in CI, lives there.

# Part 6 — Prioritised work order

> **Superseded.** This ordering was written before `architecture_plan.md` Part IV existed,
> and Part IV's 76-step work order is what the project actually follows. Most of what
> follows is done; the rest has been absorbed into numbered steps there. Kept only as the
> record of what the original review thought the sequence should be.

What remains from this part, and where it now lives:

| Original item | Now |
|---|---|
| Guard `ModelData::Init` against invalid meshes ([1.6](#16--p0-modeldatainit-will-throw-or-crash-on-sparse-mesh-indices)) | Still open, still **P0**. Independent-work table |
| Mip generation + `maxLod` + `levelCount` ([3.1](#31--p1-no-mipmaps-anywhere)) | Part IV step 72 |
| Extract `Editor/EditorUI` out of `main.cpp` ([2.1](#21--p2-maincpp-is-the-whole-engine)) | Part IV steps 53–54, Stage 8 |
| `surface.slangh` de-duplication ([4.2](#42--p2-opaqueslang-and-weightedblendedoitslang-duplicate-130-lines)) | Independent-work table |
| `SharedTypes.h` + `static_assert`s ([4.1](#41--p2-share-one-source-of-truth-for-gpu-struct-layouts)) | Part IV step 48 |
| Dirty flags, cached normal matrices, non-allocating `AppendDrawables` ([3.4](#34--p3-batches-instance-data-and-the-global-buffer-are-rebuilt-from-scratch-every-frame)) | Part IV steps 61, 63 |
| Frustum culling with per-mesh AABBs | Part IV step 65 |
| Reverse-Z | Part IV step 74 |
| Instance data → SSBO + `SV_InstanceID` | Part IV step 69 |
| Cloud shader: expose parameters in ImGui ([3.7](#37--p3-cloud-shader-cost)) | Independent-work table (the other three cloud items are done) |
| Async scene loading ([3.3](#33--p3-asset-loading-blocks-the-frame-loop)) | Part IV step 73 |
| Skybox + IBL ([1.14](#114--p2-the-skybox-is-loaded-but-never-rendered)) | Independent-work table |
| Shadow maps; depth prepass, SSAO, bloom, FXAA | Part IV step 75 and Stage 10; see [3.6](#36--p3-missing-rendering-features-in-rough-order-of-visual-impact) |

Done since the review: sync-object recreation on image-count change, `UploadContext` batching
with a dedicated copy queue, the growable `DescriptorAllocator` and instance buffer, the
pipeline cache, and tests + CI.
