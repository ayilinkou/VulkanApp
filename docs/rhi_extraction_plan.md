# Stage 5 — RHI Extraction: Implementation Plan

> **Retained past its stage.** It was written to be deleted when Stage 5 completed, and kept
> instead: R1–R17 are history, but the design decisions in §2 still govern what the RHI's
> public seam is allowed to say. See [§10 Retirement](#10-retirement) for what a future
> retirement would promote and what it would throw away —
> `backend_readiness_plan.md`'s B6 proposes itself as the moment.

**Created:** 16 August 2026 · **Supersedes:** `architecture_plan.md` Part IV,
steps 24–34 · **Superseded in part:** D7 and D8, by `backend_readiness_plan.md`'s D14 and
D15 · **Status:** Stage 5 complete — see the [progress table](#progress)

---

## Table of contents

1. [Purpose and authority](#1-purpose-and-authority)
2. [Design decisions](#2-design-decisions)
3. [Module layout](#3-module-layout)
4. [How the boundary is enforced](#4-how-the-boundary-is-enforced)
5. [The step sequence](#5-the-step-sequence) — [progress table](#progress)
6. [Mapping back to Part IV](#6-mapping-back-to-part-iv)
7. [Out of scope](#7-out-of-scope)
8. [D3D12 readiness checklist](#8-d3d12-readiness-checklist)
9. [Risks](#9-risks)
10. [Retirement](#10-retirement)

---

## 1. Purpose and authority

Stage 5 turns the Vulkan code currently spread across `src/main.cpp`, `src/Utility.h` and the
resource classes into `Engine::RHI`, a library the rest of the engine talks to without
including a Vulkan header.

Part IV's steps 24–34 describe that extraction for a Vulkan-only engine: `IPresentTarget`
returns `vk::Image`, `PipelineBuilder` takes `vk::Format`, and resources stay RAII objects
holding Vulkan handles. Since a D3D12 backend is now an explicit goal, this document
re-plans the stage so that the public API is backend-neutral **from the start**, rather than
being written twice.

**For the duration of Stage 5 this document is the authority.** Where it disagrees with
Part IV steps 24–34, this document wins. Everything outside steps 24–34 — the target module
graph (§8), the directory layout (§9), the headless seams (§10), the test strategy
(Part III) — is unchanged and still governed by the architecture plan.

**Cost of the change of course:** Part IV budgeted ~2 weeks for Stage 5's 11 steps. This plan
has 17 steps and is closer to **~3 weeks**. The extra week buys a public API that Stages 6–10
can be written against once instead of twice.

---

## 2. Design decisions

Each decision records what was chosen, why, and what it costs. `D` numbers are referenced by
the steps in §5.

### D0 — Namespace `Rhi`, directory `rhi/`, include as `<rhi/Device.h>`

The architecture plan writes `rhi::Device`. This codebase spells namespaces `Log`,
`JobSystemDetail` — PascalCase — so the C++ namespace is `Rhi` while the include directory
stays lowercase to match `engine_module`'s convention (`<core/Timer.h>`, `<platform/Paths.h>`).

A namespace is mandatory here rather than optional: `Rhi::Texture`, `Rhi::Format` and
`Rhi::Device` are names that would otherwise collide with `src/Texture.h` and with anything
D3D12's headers drag in later.

### D1 — Neutral public API; Vulkan confined to the backend

`engine/rhi/include/rhi/*.h` contains no Vulkan type, no VMA type and no Vulkan header
include. Vulkan lives in:

- `engine/rhi/src/vulkan/` — the implementation, invisible outside the module.
- `engine/rhi/include/rhi/vulkan/` — a **transitional, explicitly-scoped** area of headers
  that do expose Vulkan. During Stage 5 this is where moved code lands before it is
  converted; at the end of the stage it retains only the deliberate escape hatch (D9).

That gives a mechanically checkable end state: `src/` may include `rhi/vulkan/...` only from
the ImGui backend glue. See §4.

**Cost:** every enum used at an API boundary needs a neutral counterpart and a conversion
table. That's ~1 day of typing plus the tests that keep the tables honest.

### D2 — Resources are 32-bit handles, not RAII objects

`Device::CreateBuffer` returns a `Rhi::BufferHandle` — an index + generation packed into a
`uint32_t`, following the `Handle<Tag>` template already specified in architecture plan
§11.1. The device owns the backing storage in a `HandlePool`; `Destroy(handle)` bumps the
slot's generation.

Why, in order of weight:

1. Once Vulkan types are confined to the backend (D1), a public RAII object cannot hold a
   `VkBuffer` member. The remaining options — `unique_ptr<IBuffer>` (vtable + heap allocation
   per resource), pImpl (allocation + indirection), or an opaque fixed-size blob (per-backend
   size constant) — all pay a real cost to hide the backend. A handle hides it for free.
2. Use-after-free of a GPU resource becomes a detected generation mismatch that can be
   logged, instead of undefined behaviour. That matters right now: `ResourceCache` holds
   `weak_ptr`s and `PBRMaterial` holds `shared_ptr<Texture>`, so texture lifetime is already
   refcounted, and `ResourceManager` / `ModelManager` / `MaterialFactory` are still singletons
   with unspecified destruction order relative to the device until Stage 7.
3. `core/Handle.h` and `core/HandlePool.h` are already in the target layout for `Core`
   (architecture plan §9), and Stages 9–10 assume 32-bit identities for `FrameSnapshot`, sort
   keys and bindless indices. Building them now is work pulled forward, not work added — and
   it is pure CPU code, so it lands in the `unit` tier.

**Handle layout** follows §11.1 exactly — `index:24 | generation:8`, `kInvalid = 0xFFFFFFFF`.
Eight generation bits wrap after 256 reuses of one slot; the free list is FIFO, so a slot is
not reused until every other free slot has been, which makes an aliasing collision require
256 full cycles of the pool. Acceptable, and worth a comment in `HandlePool.h` saying so.

**Cost, and the mitigation:** handles are worse for scope-local resources (staging buffers,
readback buffers) because every exit path must call `Destroy`. `Rhi::UniqueHandle<H>` — a
~30-line move-only wrapper holding `IDevice*` + handle — covers those cases. Handles as the
ABI with RAII sugar on top works; the reverse does not.

Note that RAII is *not* banished: `VulkanBuffer` inside the backend can hold `vk::raii`
members and be destroyed by its pool. D2 is about what crosses the seam.

### D3 — Virtual interfaces at object granularity, not compile-time backend selection

`Rhi::IDevice`, `Rhi::ICommandList` and (Stage 6) `Rhi::IPresentTarget` are abstract; a free
function `Rhi::CreateDevice(const DeviceDesc&)` returns `std::unique_ptr<IDevice>`.

Alternatives considered: a compile-time `using Device = VulkanDevice;` typedef removes the
vtable but makes it impossible for a null/recording backend to coexist with Vulkan in one
test binary, which the contract test tier (architecture plan §15.2) needs; and it leaves a
future runtime `--rhi d3d12` flag impossible without another rewrite.

The overhead is a vtable dispatch on calls that are already crossing into a driver. Resource
creation happens hundreds of times per run, not per frame. `ICommandList` calls are the only
hot ones, and Stage 8's frame graph moves recording toward per-batch and eventually indirect
draws, which collapses the call count regardless.

**This is the decision most worth revisiting if profiling later disagrees** — it is contained
to the RHI boundary and does not affect D1 or D2.

### D4 — Barriers are the neutral (Stage, Access, Layout) triple

Vulkan `VK_KHR_synchronization2` splits a barrier into `VkPipelineStageFlags2`,
`VkAccessFlags2` and `VkImageLayout`. D3D12 Enhanced Barriers splits it into
`D3D12_BARRIER_SYNC`, `D3D12_BARRIER_ACCESS` and `D3D12_BARRIER_LAYOUT` — "three enums
operating independently, replacing the monolithic `D3D12_RESOURCE_STATE`"
([DirectX-Specs, Enhanced Barriers](https://microsoft.github.io/DirectX-Specs/d3d/D3D12EnhancedBarriers.html)).

So the RHI exposes the same three-way split:

```cpp
// rhi/Barrier.h
namespace Rhi
{
enum class PipelineStage : uint32_t   // → VkPipelineStageFlags2 / D3D12_BARRIER_SYNC
{
    None = 0, Draw = 1 << 0, VertexStage = 1 << 1, PixelStage = 1 << 2,
    ComputeStage = 1 << 3, DepthStencil = 1 << 4, RenderTarget = 1 << 5,
    Copy = 1 << 6, Resolve = 1 << 7, AllGraphics = 1 << 8, All = 1 << 9,
};

enum class AccessFlags : uint32_t     // → VkAccessFlags2 / D3D12_BARRIER_ACCESS
{
    None = 0, VertexBufferRead = 1 << 0, IndexBufferRead = 1 << 1,
    ConstantBufferRead = 1 << 2, ShaderRead = 1 << 3, UnorderedAccess = 1 << 4,
    RenderTargetWrite = 1 << 5, DepthStencilRead = 1 << 6, DepthStencilWrite = 1 << 7,
    CopySrc = 1 << 8, CopyDst = 1 << 9,
};

enum class TextureLayout : uint32_t   // → VkImageLayout / D3D12_BARRIER_LAYOUT
{
    Undefined, Common, RenderTarget, ShaderResource, UnorderedAccess,
    DepthStencilWrite, DepthStencilRead, CopySrc, CopyDst, Present,
};
}
```

Today's `src/Barrier.h` already has the right *shape* — named preset functions such as
`UndefinedToTransferDst()` and `TransferDstToShaderRead()` returning an `ImageBarrierDesc`.
Those presets survive verbatim, re-expressed in neutral terms in `rhi/BarrierPresets.h`; only
the field types change. This is the single cheapest portability win in the stage, and it is
cheap precisely because the existing code already used sync2 rather than legacy barriers.

Two caveats to record honestly:

- The DirectX spec does **not** claim Vulkan parity, and the enumerators are not
  interchangeable. The two conversion tables are hand-written, per-backend, and each one has
  to be checked against its own specification. The neutral enum above is a *superset shape*,
  not a proof of equivalence.
- `D3D12_BARRIER_LAYOUT` has queue-type-specific variants (`DIRECT_QUEUE_COMMON`,
  `COMPUTE_QUEUE_COMMON`, …) that "may only be used within a compatible command queue", and
  copy-queue resources must be in `COMMON`. That interacts with D6; the neutral `Common`
  layout exists specifically to express it.

### D5 — CPU/GPU synchronization is fence + value; present sync stays behind the seam

D3D12 has exactly one synchronization primitive: `ID3D12Fence` with a monotonically
increasing value. Vulkan's equivalent is a timeline semaphore (core since 1.2). The RHI
therefore models queue and CPU waits as `FenceHandle` + `uint64_t Value`, not as binary
semaphores.

Binary semaphores cannot be eliminated, though — the swapchain requires them:

- `VUID-vkAcquireNextImageKHR-semaphore-03265`: the semaphore "**must** have a
  `VkSemaphoreType` of `VK_SEMAPHORE_TYPE_BINARY`".
- `VUID-vkQueuePresentKHR-pWaitSemaphores-03267`: all wait semaphores "**must** be created
  with a `VkSemaphoreType` of `VK_SEMAPHORE_TYPE_BINARY`".

(Both verbatim from `$VULKAN_SDK/share/vulkan/registry/validusage.json`, SDK 1.4.341.1.)

So binary semaphores are an implementation detail of the present path and never appear in a
neutral header. `IPresentTarget` (Stage 6) owns them. Stage 5 leaves today's per-frame
`PresentCompleteSemaphore` / `RenderCompleteSemaphores` in `App` untouched.

### D6 — Queues are `QueueType`, and ownership transfer is a backend concern

Neutral: `enum class QueueType : uint8_t { Graphics, Compute, Copy }` — chosen to match
D3D12's `DIRECT` / `COMPUTE` / `COPY` command list types, which have no notion of a queue
family index. Vulkan's family indices stay inside `VulkanDevice`.

The queue-family ownership transfer that step R12a adds is a Vulkan-only mechanism, and one
R12b then makes conditional: `VK_KHR_maintenance9` removes the requirement for buffers, linear
images and most optimal images, so the backend asks per resource whether it is needed. D3D12
has no equivalent; it requires copy-queue resources to be in the `COMMON` layout instead.
The RHI expresses the intent — "this resource was written on the copy queue and will next be
read on the graphics queue" — and each backend does whatever its API requires. Concretely
that means `UploadContext` returns an explicit "acquire" record rather than the caller
issuing raw release/acquire barriers.

### D7 — The descriptor/binding model is *not* abstracted in Stage 5

> **Superseded by D14** (`backend_readiness_plan.md`). The half that held — Stage 5 leaves the
> binding model alone, isolated rather than abstracted — is history now that Stage 5 is done.
> The half that did not is the reasoning below that bindless makes the question moot: the
> architecture plan's §20 row 5 requires a non-bindless fallback anyway, only
> `descriptorBindingPartiallyBound` is actually enabled, samplers and per-frame constants stay
> conventional even under bindless, and the D3D12 convergence needs SM6.6. Bindless is
> therefore deferred until after the D3D12 backend, and the binding model is neutralised in
> Stage 7.5 instead. Kept because the *cost* it names is real and D14 accepts it.

Vulkan descriptor sets/layouts/pools and D3D12 root signatures + descriptor heaps have no
cheap common denominator; every portable RHI that tries pays for it in complexity. It is also
the part of the design that a later step makes largely moot: bindless (step 69) converges
Vulkan descriptor indexing — which this app **already enables**
(`VK_EXT_descriptor_indexing`, `main.cpp:1143`) — with D3D12 SM6.6 `ResourceDescriptorHeap`.

So `DescriptorAllocator` (R13) is written as a Vulkan-side component under `rhi/vulkan/`,
used by `MaterialFactory`, and is deliberately *not* given a neutral interface. The
requirement on Stage 5 is only that it stays isolated, so replacing it later is a contained
change.

Related: prefer push constants for per-draw data. They map 1:1 onto D3D12 root constants,
and the code already uses them for material data (`main.cpp:1680`).

### D8 — Pipelines stay Vulkan-side; the *cache* is a neutral opaque blob

> **First half superseded by D15** (`backend_readiness_plan.md`). Pipelines stayed Vulkan-side
> only because the binding model was not neutral; D14 neutralises it, so the reason expires and
> pipeline creation moves behind the RHI in Stage 7.5 (B4). **The rest of this decision
> stands** — the cache is still a neutral opaque blob and `IPipelineCache` does not change
> shape, and the dynamic-rendering note below is reaffirmed as D17.

`PipelineBuilder` and `ComputePipelineBuilder` keep taking `vk::Format` and friends under
`rhi/vulkan/` for the whole of Stage 5. Neutralizing pipeline creation means neutralizing
the binding model (D7), so it waits.

`PipelineCache` (R15) is neutral, because it can be: create at startup, seed from a file,
hand to pipeline creation, serialize on shutdown. D3D12's equivalent is a cached PSO blob or
`ID3D12PipelineLibrary`; both fit "opaque bytes on disk that may be rejected as stale".

One favourable accident worth recording: the renderer uses **dynamic rendering**
(`vk::RenderingInfo`, `main.cpp:1641`) rather than `VkRenderPass`/`VkFramebuffer` objects.
That is much closer to D3D12's `OMSetRenderTargets` model, and
`vk::PipelineRenderingCreateInfo`'s colour formats correspond to a PSO's `RTVFormats`.
Do not reintroduce render pass objects.

### D9 — One documented native escape hatch, for ImGui

ImGui's Vulkan backend needs raw `VkInstance`, `VkPhysicalDevice`, `VkDevice`, queue family
index, `VkQueue`, `VkDescriptorPool` and the swapchain format. Pretending otherwise would
mean wrapping ImGui, which is not Stage 5's job.

```cpp
// rhi/vulkan/VulkanNative.h — the ONLY sanctioned leak. Editor/ImGui glue only.
namespace Rhi::Vulkan
{
struct NativeDevice
{
    VkInstance       Instance;
    VkPhysicalDevice PhysicalDevice;
    VkDevice         Device;
    VkQueue          GraphicsQueue;
    uint32_t         GraphicsQueueFamily;
};

NativeDevice GetNative(IDevice& device);
VkImage      GetNativeImage(IDevice& device, TextureHandle handle);
VkImageView  GetNativeView(IDevice& device, TextureViewHandle handle);
VkBuffer     GetNativeBuffer(IDevice& device, BufferHandle handle);
}
```

The rule is that the escape hatch is *listed*, not *available*: R17 checks that no file in
`src/` includes `rhi/vulkan/` except the ImGui glue. When the editor is extracted in Stage 7,
this header goes with it.

### D10 — Clip-space handedness gets exactly one site

Vulkan NDC is Y-down; D3D12 NDC is Y-up. Both use depth 0..1, and
`GLM_FORCE_DEPTH_ZERO_TO_ONE` is already set, so the depth half is done. The Y half is
currently one line — `proj[1][1] *= -1.f` at `main.cpp:2076`.

Keep it one line, and make it conditional on a capability rather than on the build:
`DeviceCaps::bFlipClipSpaceY`. Anything that recomputes a projection matrix elsewhere later
must read that flag rather than repeating the constant.

### D11 — Formats: a curated neutral enum, not a mirror of `VkFormat`

`Rhi::Format` contains only formats that have both a `VkFormat` and a `DXGI_FORMAT`
equivalent. Everything the renderer uses today qualifies: `R8Unorm`, `RGBA8Unorm`,
`RGBA8Srgb`, `BGRA8Unorm`, `RGBA16Float`, `D32Float`, `D24UnormS8Uint`.

Adding a format means adding it to the enum *and* the conversion table in the same commit;
§4 explains the switch-without-`default` trick that makes forgetting a compile error.

### D12 — Shaders are already portable; keep them that way

`slangc` lists `dxil` and `hlsl` as first-class targets alongside `spirv`
(`$VULKAN_SDK/share/doc/slang/command-line-slangc-reference.md:1157`), so the shader
language is not a portability problem — the shader *build* is. Stage 5 changes nothing here,
but two rules start applying now:

- Do not add SPIR-V-specific workarounds to `.slang` sources without an `#ifdef` on the
  target profile.
- Binding annotations should stay expressible for both targets; avoid hand-assigned
  `[[vk::binding]]` where Slang's automatic layout would do.

The `add_slang_shader_target` function in the root `CMakeLists.txt` will need a target
parameter when D3D12 lands. Not now.

### D13 — Where the two APIs disagree on a name, use D3D12's

The neutral vocabulary has to pick one word for every concept the two backends name
differently. It picks D3D12's, consistently, including for the Vulkan-side helpers.

The reason is asymmetry of harm. Vulkan is the backend that exists today, so its terms are
the ones already in the codebase and in the reader's head — a Vulkan name in neutral code
reads as natural and is therefore easy to leave in place by accident, right up until a second
backend makes it wrong. A D3D12 name in neutral code is mildly unfamiliar, which is exactly
what makes it obvious that the surrounding code is *meant* to be backend-neutral. The
unfamiliarity is doing work.

Consequences already in the tree:

| Concept | Vulkan | D3D12 | Chosen |
|---|---|---|---|
| Copy queue role | transfer | copy | `QueueType::Copy` |
| Recorded command container | command buffer | command list | `CommandListUtil.h` |
| Fragment shader stage | fragment | pixel | `PipelineStage::PixelStage` |
| Read-write shader resource | storage | unordered access | `AccessFlags::UnorderedAccess` |

This applies to *names*, not semantics. `FamilySupports` still encodes Vulkan's queue-family
capability rules because that is what the Vulkan backend has to obey; only the word "Copy" is
borrowed. Where Vulkan has a concept D3D12 lacks entirely, or vice versa, there is nothing to
reconcile and the owning API's term stands.

Utility headers in `rhi/vulkan/` take a `Util` suffix uniformly (`BufferUtil.h`, `ImageUtil.h`,
`BarrierUtil.h`, `CommandListUtil.h`, `SwapchainUtil.h`) so that "is this a type or a bag of
free functions" is answerable from the filename.

---

## 3. Module layout

```
engine/rhi/
├── CMakeLists.txt
├── include/rhi/                    # NEUTRAL. No Vulkan, no VMA, no vk:: — enforced (§4)
│   ├── RhiTypes.h                  # Format, QueueType, MemoryAccess, Extent2D/3D, SampleCount
│   ├── Handles.h                   # BufferHandle, TextureHandle, TextureViewHandle,
│   │                               #   SamplerHandle, FenceHandle  (Handle<Tag> from Core)
│   ├── UniqueHandle.h              # RAII sugar over a handle + IDevice*
│   ├── BufferDesc.h  TextureDesc.h  SamplerDesc.h
│   ├── Barrier.h                   # PipelineStage / AccessFlags / TextureLayout  (D4)
│   ├── BarrierPresets.h            # today's Barriers:: presets, neutral
│   ├── IDevice.h                   # creation, destruction, mapping, caps
│   ├── DeviceDesc.h                # DeviceRequirements, DeviceCaps
│   ├── ICommandList.h              # barriers + copies in Stage 5; draws in Stage 8
│   ├── UploadContext.h             # batched staging, one fence
│   ├── PipelineCache.h             # opaque blob on disk
│   └── Diagnostics.h               # DiagnosticSeverity, ValidationPolicy, counters, capture
│
├── include/rhi/vulkan/             # TRANSITIONAL + the sanctioned escape hatch (D9)
│   ├── VulkanNative.h              # survives Stage 5 — ImGui/editor only
│   ├── PipelineBuilder.h           # stays Vulkan-shaped through Stage 5 (D8)
│   ├── ComputePipelineBuilder.h
│   └── DescriptorAllocator.h       # deliberately Vulkan-only (D7)
│
├── src/                            # backend-neutral implementation
│   └── Diagnostics.cpp             # counting/policy/capture: no backend in it (R7)
│
└── src/vulkan/
    ├── VulkanDevice.{h,cpp}        # instance, messenger, physical, logical, queues, VMA
    ├── VulkanCommandList.{h,cpp}
    ├── VulkanConversions.{h,cpp}   # every ToVk()/FromVk() table lives here and nowhere else
    ├── VulkanBuffer.h  VulkanTexture.h     # pool payloads; may use vk::raii freely
    ├── VulkanUploadContext.cpp
    ├── VulkanPipelineCache.cpp
    ├── VMAImpl.cpp                 # moved from src/
    └── PipelineBuilder.cpp  ComputePipelineBuilder.cpp  DescriptorAllocator.cpp
```

`engine/core` gains two headers this stage: `core/Handle.h` and `core/HandlePool.h`
(architecture plan §9 already lists both).

CMake wiring:

```cmake
# engine/rhi/CMakeLists.txt
engine_module(RHI
  SOURCES src/vulkan/VulkanDevice.cpp ...          # explicit list, no globbing
  LINK_LIBRARIES Engine::Core Engine::Platform Vulkan::Vulkan
                 GPUOpen::VulkanMemoryAllocator)
```

Root `CMakeLists.txt`: `add_subdirectory(engine/rhi)` after `engine/platform`, then
`Engine::RHI` added to `HikariEngine`'s `target_link_libraries` **and** to the
`engine_header_self_containment(App ... LINK_LIBRARIES ...)` list — a header under `src/`
that includes `<rhi/...>` fails the header check otherwise.

---

## 4. How the boundary is enforced

Discipline does not hold a boundary for three weeks. Three mechanisms do:

**1. A neutral-header check target.** `engine_module` already creates
`HeaderSelfContainment_RHI`, which links the module itself and therefore *can* see Vulkan. A
second, stricter target compiles only `include/rhi/*.h` (excluding `include/rhi/vulkan/`)
while linking **only `Engine::Core` and `Engine::Platform`**:

```cmake
# engine/rhi/CMakeLists.txt, after engine_module(RHI ...)
file(GLOB neutral_headers CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/rhi/*.h)
engine_header_self_containment(RHI_Neutral
  HEADERS ${neutral_headers}
  LINK_LIBRARIES Engine::Core Engine::Platform
  INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include)
```

A neutral header that includes `vulkan/vulkan.hpp` then fails to compile. `HeaderSelfContainment.cmake`
already documents the caveat that applies here: a dependency also present on the default
system include path is found regardless of what a target links, and on a typical Arch box
that can cover Vulkan. So this is a strong net, not a proof — hence:

**2. A grep gate in `scripts/precommit.sh`.** Cheap, exact, and immune to include paths:

```bash
# scripts/rhi_boundary_check.sh
! grep -rn --include='*.h' -E 'vulkan|vk_mem_alloc|\bvk::|\bVk[A-Z]' \
    engine/rhi/include/rhi --exclude-dir=vulkan
```

Plus, from R17 onward, the `src/` side of the same rule (only ImGui glue may include
`rhi/vulkan/`).

**3. Exhaustive switches with no `default:` label** in `VulkanConversions.cpp`. With
`-Wall -Wextra` and `CMAKE_COMPILE_WARNING_AS_ERROR ON`, `-Wswitch` turns "added an
enumerator, forgot the mapping" into a build failure on all nine CI configs. Do not add a
`default:` case to a conversion switch; throw after the switch instead.

**Per-step verification, every step, no exceptions:**

```bash
scripts/precommit.sh                  # configure + build + header check + tests + format
tests/scripts/baseline_test.sh        # writes tests/screenshots/ + tests/reports/
```

then diff the report against the one in `tests/baseline/` (referred to by directory rather
than filename, since re-capturing a baseline changes its timestamped name).
`validationErrors` must be 0 and `drawCalls` / `batches` / `instances` must be identical
unless the step's **Verify** line says otherwise. "It still builds" is not evidence.

---

## 5. The step sequence

Sizes use Part IV's scale: XS < 1h · S 1–3h · M ½–1 day · L 2–4 days.
Every step ends with a compiling, running application.

### Progress

**This table is the authority on what is done.** Update it in the same commit as the step.
`CLAUDE.md`'s stage table is coarse (Stage 5 as a whole) and deliberately does not repeat
per-step status, so there is only one place to change.

Where the as-built result differs from the step's **Do** text — because implementing it
revealed something the plan got wrong — the step carries an **As built** note. Those notes
are the ones later steps need to read.

| Step | Status |
|---|---|
| R1 — `core/Handle.h` + `core/HandlePool.h` | ✅ done |
| R2 — `engine/rhi` skeleton and the neutral vocabulary | ✅ done |
| R3 — Move the RHI leaf types | ✅ done |
| R4 — Dissolve `Utility.h` | ✅ done |
| R5 — Extract `Rhi::Device` | ✅ done |
| R6 — Enumerate all queue families | ✅ done |
| R7 — `Rhi::Diagnostics` | ✅ done |
| R8 — `Rhi::ICommandList` and the neutral barrier API | ✅ done (barriers here; `ICommandList` landed in R10 — see **As built**) |
| R9 — Buffers become handles | ✅ done |
| R10 — Textures, views and samplers become handles | ✅ done |
| R11 — `UploadContext` — batch transfers | ✅ done |
| R12a — Use the dedicated transfer queue | ✅ done (the acquire is submitted by the context, not returned — see **As built**) |
| R12b — Adopt `VK_KHR_maintenance8` and `VK_KHR_maintenance9` | ✅ done (exposed a barrier that was invalid on a copy-only queue — see **As built**) |
| R13 — Growable `DescriptorAllocator` | ✅ done (grows before the pool is overrun, not after — see **As built**) |
| R14 — Growable instance buffer | ✅ done (the wait, not the reallocation, is the load-bearing part — see **As built**) |
| R15 — `PipelineCache` | ✅ done (the header check is the load-bearing part, not the file I/O — see **As built**) |
| R16 — First GPU tests | ✅ done (the cubemap bug was already gone; headless device creation was the real work — see **As built**) |
| R17 — Seal the boundary and update the docs | ✅ done (one header could move; the boundary became a ratchet instead of a ceiling — see **As built**) |

### R1 — `core/Handle.h` + `core/HandlePool.h`

- **Do:** Add `Handle<Tag>` exactly as specified in architecture plan §11.1
  (`index:24 | generation:8`, `kInvalid`, `Index()`, `Generation()`, `IsValid()`,
  `operator<=>`). Add `HandlePool<T, Tag>`: dense storage, FIFO free list, generation bump on
  release, `Get()` returning `nullptr` on a stale handle, `Size()`/`Capacity()`. Add
  `tests/unit/core/HandleTests.cpp` to the `core_tests` target: create/get/destroy, stale
  handle rejected, generation wrap behaviour, reuse ordering, capacity growth.
- **Why now:** D2. It is CPU-only, so it is testable before any GPU code depends on it.
- **Verify:** `ctest -L unit` passes with the new tests. Application untouched, so the
  headless report must be byte-identical to the baseline.
- **As built:** `Handle` gained `FromIndexAndGeneration()` so `HandlePool` never open-codes
  the bit layout, and `kMaxIndex = kIndexMask - 1` so that no valid handle can collide with
  `kInvalid` (index `0xFFFFFF` at generation `0xFF` *is* `kInvalid`). §11.1's `MeshHandle` /
  `MaterialHandle` / `TextureHandle` / `EntityHandle` aliases were **not** added: a global
  `TextureHandle` in `Core` is the collision D0 warns about, and R2/R9/R10 declare those
  under `Rhi::` instead.
- **Size:** S · **Needs:** —

### R2 — `engine/rhi` skeleton and the neutral vocabulary

- **Do:** Create the module with `engine_module(RHI ...)`, `add_subdirectory`, and both
  header checks from §4 including `rhi_boundary_check.sh` wired into `scripts/precommit.sh`.
  Add `RhiTypes.h`, `Handles.h`, `Barrier.h`, `BufferDesc.h`, `TextureDesc.h`,
  `SamplerDesc.h`, and `src/vulkan/VulkanConversions.{h,cpp}` with `ToVk`/`FromVk` for every
  enumerator. Add `tests/unit/rhi/ConversionTests.cpp` in a new `rhi_tests` target
  (CPU-only — no device is created) asserting round-trips.
- **Note:** nothing uses any of this yet. That is intentional: the vocabulary is in place
  before the first type moves, so moved code can be converted in the same step it lands.
- **Verify:** `HeaderSelfContainment` and `rhi_boundary_check` pass. `ctest -L unit` passes.
  Headless report identical to baseline.
- **As built:** four departures, each with a consequence for a later step:
  - **`FromVk` exists only where the mapping is one-to-one.** `PipelineStage`, `AccessFlags`,
    `TextureLayout` and the usage/aspect flag enums are `ToVk`-only, because one neutral value
    can expand to several Vulkan values (`DepthStencil` is both fragment-test stages) or two
    neutral values can share one (`Common` and `UnorderedAccess` are both `eGeneral`). Where
    `FromVk` does exist it is *derived* from `ToVk` by search, not hand-written twice.
  - **`Format` omits `D16UnormS8Uint`.** D11 requires a DXGI equivalent and there is none —
    `dxgiformat.h` offers stencil only with 24-bit unorm or 32-bit float depth. It is
    currently the last candidate in `FindDepthFormat` (`main.cpp:2278`), so **R10 must drop it
    from that list or accept a promotion to `D24UnormS8Uint`.** Vertex-attribute formats are
    also absent, since vertex input stays Vulkan-side until Stage 8 (D8).
  - **`QueueType` is tested with a predicate, not a bit mapping.** `FamilySupports(flags, role)`
    replaces what would have been `ToVk(QueueType)`. The spec lets a graphics or compute family
    omit `VK_QUEUE_TRANSFER_BIT` while still being able to copy, so `Copy` is satisfied by any
    of `eTransfer`/`eGraphics`/`eCompute` — an "any of" test, which a caller handed a raw mask
    would likely write as "all of". There is no reverse mapping: a universal family serves all
    three roles, so "which role is this family" has no single answer.
  - **The boundary check is `cmake/RhiBoundaryCheck.cmake`** with thin `.sh`/`.bat` wrappers
    in `tests/scripts/`, not a shell grep in `scripts/`. It strips comments before matching,
    because §4's literal pattern would reject this document's own specimen comments
    (`// → VkPipelineStageFlags2`). It bans a *dependency*, not a mention.

  Also: §4's claim that the exhaustive-switch mechanism "fails the build on all nine CI
  configurations" was **false as written** — MSVC's C4062 is a level-4 warning that `/W3`
  leaves off, so the RHI target now sets `/w14062` explicitly. Verified by adding an unmapped
  enumerator and watching MSVC fail.
- **Size:** M · **Needs:** R1

### R3 — Move the RHI leaf types

- **Do:** Move `AllocatedBuffer`, `AllocatedImage`, `VulkanAllocator`, `VMAImpl.cpp`,
  `Barrier.h`, `Texture`, `Cubemap`, `PipelineBuilder`, `ComputePipelineBuilder` from `src/`
  into `engine/rhi`, initially under `include/rhi/vulkan/` + `src/vulkan/`. Update includes
  in `src/`, the `SOURCES` list in the root `CMakeLists.txt`, and the module's `SOURCES`.
  **No logic changes.**
- **Note:** `Texture` and `Cubemap` are conceptually Assets-layer types (a cache key plus a
  GPU image). Stage 7 moves them; leaving them in RHI for now matches Part IV step 24 and
  avoids pre-empting that decision.
- **Verify:** Headless report and screenshot identical to baseline. `src/` no longer contains
  those files.
- **As built:** the move exposed one ordering problem and one silent-breakage hazard.
  - **`SetVkDebugName` had to come along, out of R4.** Both pipeline builders call it, and it
    lived in `src/Utility.h` — so moving them into the module would have left module code
    depending on a header in `src/`, which is the dependency direction the whole stage exists
    to prevent. R4's first bullet already specifies `rhi/vulkan/DebugNames.h` for exactly this
    function, so that one piece was pulled forward verbatim; `src/Utility.h` now includes it
    and R4 has correspondingly less to do. Nothing else in the moved set needed anything from
    `Utility.h`.
  - **The RHI target now defines `DEBUG` PUBLIC in Debug configs.** `SetVkDebugName` is a
    template whose body is `#ifdef DEBUG`, so it is instantiated per calling translation unit.
    Before this step every caller was in `HikariEngine`, which defines `DEBUG` itself, so they all
    agreed. With callers now on both sides of the boundary, a module that did not define it
    would instantiate an empty body while the application instantiated a real one — an ODR
    violation whose only symptom is debug names going missing from some objects and not others,
    invisible to the baseline report. Verified present on the module's compile flags.
  - **There are now two files called `Barrier.h`**: `rhi/Barrier.h` (neutral, R2) and
    `rhi/vulkan/Barrier.h` (the moved Vulkan presets). Nothing includes both and no type names
    overlap, so confusing them is a compile error rather than a silent substitution. R8 deletes
    the latter. The moved file carries a comment saying so.
  - **`TextureBinding` (`Albedo`/`Normal`/`MetallicRoughness`) rode along inside `Texture.h`**
    and is a material concept, not an RHI one. Left in place because R3 moves files verbatim
    and Stage 7 relocates `Texture` wholesale anyway — but it should not acquire new users
    while it sits here.
- **Size:** M · **Needs:** R2 · **Was:** step 24

### R4 — Dissolve `Utility.h`

- **Do:** Split its remaining lines by concern:
  `rhi/vulkan/BufferUtil.h` (`CreateBuffer`, `CopyBuffer`, `CreateStagedBuffer`),
  `rhi/vulkan/ImageUtil.h` (`CreateImage`, `CreateImageView`, `CopyBufferToImage`,
  `CreateRenderTexture`),   `rhi/vulkan/BarrierUtil.h` (`RecordImageBarrier`),
  `rhi/vulkan/SwapchainUtil.h` (`ChooseSwapchainFormat`, `ChoosePresentMode`,
  `ChooseSwapchainExtent`, `ChooseSwapMinImageCount`), `rhi/vulkan/CommandListUtil.h`
  (`BeginSingleTimeCommand`, `EndSingleTimeCommand`).
  Delete `FindMemoryType` (dead since VMA, and its own comment says so).
- **Already done:** `rhi/vulkan/DebugNames.h` (`SetVkDebugName`) was pulled forward into R3,
  which could not move the pipeline builders without it.
- **Correction to Part IV step 25:** `EnsureParentDirectoryExists` and `EnsureExtension` are
  filesystem helpers with nothing to do with rendering. They go to
  `engine/platform/include/platform/FileSystem.h`, not to `rhi/`.
- **Verify:** Headless report identical. `src/Utility.h` no longer exists.
- **As built:** the split landed as specified, with the two filenames above corrected from what
  this section originally said (`SwapchainSupport.h` → `SwapchainUtil.h` for suffix consistency,
  `CommandBufferUtil.h` → `CommandListUtil.h` per D13). Three things worth recording.
  - **`FindMemoryType` was genuinely dead** — deleted rather than moved, as planned. Confirmed
    by search: the only occurrence in the tree was its own definition. Its comment ("shouldn't
    need to use this anymore since I am now using VMA") had been right for a while.
  - **Every consumer now includes what it uses.** `Utility.h` had become a de facto umbrella
    header: it pulled in `AllocatedBuffer.h`, `AllocatedImage.h`, `Barrier.h`, `Texture.h` and
    `DebugNames.h`, so files got those types without asking. Splitting it exposed that, and the
    seven former includers were given explicit includes for the types they name rather than the
    minimum needed to compile. Two of them — `PBRMaterial.cpp` and `MaterialFactory.cpp` —
    turned out to want nothing from `Utility.h` but `SetVkDebugName`.
  - **`BufferUtil` and `ImageUtil` still take a raw `VmaAllocator`.** Unchanged from before the
    split, and deliberately not tidied here: R5 moves allocator ownership into the device, at
    which point these become members rather than free functions and the parameter disappears.
    Changing the signature now would be churn against a shape that is about to be replaced.
- **Size:** M · **Needs:** R3 · **Was:** step 25

### R5 — Extract `Rhi::Device`

- **Do:** Move `CreateInstance`, `SetupDebugMessenger`, `IsPhysicalDeviceSuitable`,
  `PickPhysicalDevice`, `CreateLogicalDevice` and the VMA allocator out of `App` into
  `VulkanDevice`, behind the neutral `Rhi::IDevice` (D3). Add `DeviceDesc` /
  `DeviceRequirements` with the present vs non-present split already separated
  (architecture plan §10.3) even though both are still required — Stage 6 flips the flag.
  Add `DeviceCaps` including `bFlipClipSpaceY` (D10). Add `rhi/vulkan/VulkanNative.h` (D9)
  and route ImGui init through it. Surface creation stays in `App`; the surface is passed to
  `CreateDevice` as an opaque `uint64_t`/`void*`.
- **Verify:** Compare the startup log line-for-line against a saved baseline: same physical
  device, same queue index, same swapchain image count, same validation output. Headless
  report identical.
- **As built:** four departures from the plan above, two of them forced by ownership order,
  plus two notes on things that changed during implementation.
  - **The window handle crosses the boundary, not the surface.** The plan had `App` create the
    surface and pass it to `CreateDevice` as an opaque handle. That cannot work as written: the
    surface is created *from* the instance, and the instance is now owned by the device, so
    `App` has nothing to create a surface with until after the call it was supposed to feed.
    The alternatives were a two-phase device init or a surface-factory callback; both are more
    machinery than the problem deserves. Instead `DeviceRequirements` carries an opaque
    `void* NativeWindowHandle` and the device makes the surface itself, macOS branch and all.
    The Metal path is the reason this matters rather than being a coin toss — recreating it
    from a raw instance handle would need the plain (non-RAII) dispatcher, which `vk::raii`
    does not initialise, on the one platform this cannot be tested on. Stage 6's headless flip
    is `bPresent = false` plus a null handle, which is simpler than it would have been.
  - **Device creation moved into `App`'s constructor.** The renderer holds `vk::raii::Device&`
    and friends in about a hundred places, and the cheapest way to keep those call sites
    untouched is reference members — which must be bound in the initialiser list, which means
    the device has to exist by then. The consequence is visible in the log: the five device
    lines now print *before* `[main] Init()` rather than inside `InitVulkan()`. It also
    required moving `m_Platform`/`m_Paths`/`m_Options` to the top of the member list, since
    `MakeDeviceDesc()` reads them and members initialise in declaration order.
  - **`VulkanNative.h` has two tiers, not one.** D9's `NativeDevice` (raw handles, for ImGui)
    is there as specified, but the renderer also still *creates* Vulkan objects — swapchain,
    pipelines, descriptor sets — which raw handles cannot do. So the header also exposes
    `GetDevice`/`GetPhysicalDevice`/`GetSurface`/`GetGraphicsQueue`/`GetAllocator` returning
    the RAII wrappers. This is a much wider hole than D9 describes, and the honest framing is
    that it is a *staging area*: R9–R11 delete callers from it, and R17 should find only the
    ImGui tier left. Worth counting the callers at R17 rather than assuming.
  - **Validation counting is a callback, not a global.** `DebugCallback` moved into
    `VulkanDevice`, where it can no longer see `main.cpp`'s counters. `DeviceDesc` therefore
    carries `OnDiagnosticMessage`, and the app supplies a function that logs and increments.
    That is R7's seam arriving early in miniature, and it keeps the message text
    byte-identical.
  - **The log category for those five lines is now `[RHI]`, not `[Renderer]`.** Deliberate:
    they are no longer emitted by the renderer. It is the only intended difference in the
    startup log.
  - **The severity mapping was initially put in the wrong file.** `ToVkSeverity` first went into
    an anonymous namespace in `VulkanDevice.cpp`, which breaks the rule `VulkanConversions.h`
    states about itself: every neutral↔Vulkan mapping lives there and nowhere else, precisely so
    that the tests can reach it. Moved, with `FromVk` added alongside. It is the one mapping
    where `FromVk` is deliberately many-to-one — `eVerbose` and `eInfo` both collapse to `Info`,
    because the neutral scale has no verbose tier and dropping those messages instead would
    silently lose the driver's most detailed output.
- **Verified deliberately:** the baseline reports zero validation errors *and* zero warnings, so
  it does not exercise the diagnostic path at all — a miswired callback would have been
  invisible to every automated check. Confirmed by hand instead, by naming the graphics queue
  with `ObjectType::eBuffer`: the message arrived with the same `Type: {...}. Msg: ...` text
  under `[Validation Layer]`, the error counter reached the run report, and
  `--strict-validation` exited 1. Then reverted. Three unit tests now cover the enum mapping and
  the ascending-value ordering that the threshold comparison depends on, which is the part the
  hand check could not repeat cheaply.
- **Size:** L · **Needs:** R4 · **Was:** step 26

### R6 — Enumerate all queue families

- **Do:** In `VulkanDevice`, find graphics+present, dedicated compute and dedicated transfer
  families; log all of them; expose them as `QueueType` (D6). **Keep using the graphics queue
  everywhere** — this step discovers and reports only.
- **Use** `Rhi::Vulkan::FamilySupports(familyFlags, role)` to test a family rather than
  comparing `queueFlags` yourself. R2 found that the spec lets a graphics or compute family
  omit `VK_QUEUE_TRANSFER_BIT` while still being able to copy, so a plain
  `flags & eTransfer` test would reject a capable family on any driver that takes that option;
  `FamilySupports` encapsulates the "any of these capabilities" rule so the call site cannot
  get it wrong. "Dedicated transfer" is the narrower, separate test — supports `Copy` but not
  `Graphics` — and is R12a's concern, not this step's.
- **Verify:** Log lists the families the GPU exposes. Headless report identical. Resolves the
  information half of the dedicated-compute-queue `TODO` at `main.cpp:576` (Part IV cites
  `main.cpp:400` and `main.cpp:2173` for this; both line numbers are stale, and only the
  compute one still exists as a comment).
- **As built:** the selection rule is more than the step's one-line description, and the
  reason is that it cannot be checked on the machine it was written on.
  - **The rules are a pure function in `src/vulkan/QueueFamilies.{h,cpp}`, unit-tested.** The
    layouts that decide whether the rule is right — a transfer-only DMA family, a graphics
    family that omits compute, two graphics families of which only one presents — are all
    absent from this development machine, so testing on real hardware tests one arrangement
    out of many. `SelectQueueFamilies` therefore takes the family list as data and returns a
    `QueueFamilies`, with present support arriving as a `PresentSupportFn` callback because
    that is the one part needing a surface. `tests/unit/rhi/QueueFamilyTests.cpp` covers the
    arrangements this GPU does not have.
  - **"Dedicated" is resolved by preferring the narrowest family, not by R12a's one-line
    test.** R12a's note describes it as "supports `Copy` but not `Graphics`", which is
    ambiguous on any GPU exposing both an async compute family and a DMA family — both
    qualify, and taking the first would put uploads on the compute engine. The implemented
    rule adds a tie-break: among non-graphics candidates, prefer the one advertising the
    fewest of graphics/compute/copy. Ancillary bits (sparse binding, protected, video) are
    deliberately not counted, or a video family advertising transfer would outrank a
    transfer-only one. **R12a should use `GetQueueFamily(QueueType::Copy)` rather than
    re-deriving the family.**
  - **Every role always resolves to a usable family.** Compute and Copy fall back to the
    graphics family rather than reporting "absent", so R12a does not need a fallback at the
    call site; `IsDedicated()` is the separate question of whether that fallback happened.
    The one exception is Compute on a graphics family that omits compute, which stays
    `kInvalid` rather than pointing at a queue that would fail at submission.
  - **No queues are created for the new families.** `VkDeviceQueueCreateInfo` still asks for
    one queue from the graphics family only, because the step reports rather than uses, and
    an unused queue is one the driver schedules for nothing. That makes creating the queue
    part of R12a's work, not a detail it can assume: `vkGetDeviceQueue` on a family that was
    never requested is invalid, not merely useless.
  - **The neutral half is two `DeviceCaps` flags**, `bHasDedicatedComputeQueue` and
    `bHasDedicatedCopyQueue`, rather than a new `IDevice` virtual — `DeviceCaps` already is
    "what the device turned out to be able to do", and the flags describe the device rather
    than where work is submitted. Family indices stay backend-side:
    `VulkanDevice::GetQueueFamily(QueueType)` replaced `GetGraphicsQueueFamily()`, and the
    escape hatch did not grow — `VulkanNative.h` still exposes only the graphics family,
    which is all ImGui and the command pools ask for.
- **Size:** S · **Needs:** R5 · **Was:** step 27

### R7 — `Rhi::Diagnostics`

- **Do:** Promote step 6's global counters into a `Diagnostics` object owned by the device,
  with `ValidationPolicy { Ignore, Count, FailFast }` and message capture. Keep the interface
  neutral — D3D12's debug layer + `ID3D12InfoQueue` fits the same shape — and keep
  `g_ValidationErrorCount` working for the run report until Stage 7 moves it.
- **Verify:** `--strict-validation` still exits non-zero on an injected error. `FailFast`
  aborts at the first error with the message printed. Headless report identical.
- **As built:** three departures from the plan above, one of them a bug the step exposed.
  - **`Diagnostics` is injected by the application, not owned by the device.** The **Do**
    text says "owned by the device", which cannot work as written: `main` reads the error
    count for `--strict-validation` *after* `pApp.reset()` has destroyed the App and with it
    the device. So `main` owns a `Rhi::Diagnostics` declared before `pApp`, constructor-injects
    it into `App` (a fifth parameter, alongside `IPlatform`/`IJobSystem`), and `DeviceDesc`
    carries a non-owning `Diagnostics*`. The pointer may be null, in which case the device
    creates its own, so `IDevice::GetDiagnostics()` is never invalid — Stage 6's headless
    tests can create a device without caring about counters. Note this also means the
    `--strict-validation` check now covers teardown messages, which the run report does not:
    `WriteReport()` runs before `Shutdown()` and always did, which is what keeps the report
    comparable to the committed baseline.
  - **The counting/policy/capture code is neutral — `src/Diagnostics.cpp`, not
    `src/vulkan/VulkanDiagnostics.cpp`** as §3's tree had it. Nothing about incrementing a
    counter, comparing a threshold or keeping a ring of recent strings is backend-specific,
    and putting it outside `src/vulkan/` is what lets `tests/unit/rhi/DiagnosticsTests.cpp`
    reach it with no device and no ICD. That matters more here than the file location
    suggests: R5 already recorded that the baseline run produces zero errors *and* zero
    warnings, so a clean automated run proves nothing about this path, and the unit tests are
    the only repeatable coverage it has. §3's tree gained a neutral `src/` section
    accordingly. `VulkanDevice` keeps only the parts that are genuinely Vulkan — the
    messenger, and `DebugCallback` translating a `VkDebugUtilsMessengerCallbackDataEXT` into
    a neutral severity plus a string.
  - **The globals are gone, rather than kept until Stage 7.** The **Do** text says keep
    `g_ValidationErrorCount` working for the run report; there was no reason to, since the
    report's JSON keys are unchanged and `App` already holds the `Diagnostics` reference it
    needs. `CLAUDE.md`'s naming table cited `g_ValidationErrorCount` as its example of a
    global and now cites `g_bShouldClose`.
  - **The old callback was being invoked after its own destruction.** `m_OnDiagnosticMessage`
    was the *first* member of `VulkanDevice` declared, so the first destroyed, while
    `m_DebugMessenger` is declared second and destroyed second-to-last — after the allocator
    and the logical device. Any validation message raised during device teardown therefore
    called a destroyed `std::function`. It never misbehaved because the stored callable was a
    plain function pointer sitting in the small-object buffer, so the bytes outlived the
    object. The replacement members are declared *above* `m_Context` for exactly this reason,
    and the ordering comment there now says so — a fallback `unique_ptr<Diagnostics>` in the
    old position would have reproduced the same bug with a real destructor behind it.
  - **`--validation-policy <ignore|count|failfast>`** was added, defaulting to `count`, so
    `FailFast` is reachable without recompiling. `--strict-validation` combined with `ignore`
    is rejected at parse time: nothing would be counted, so the run would exit 0 with errors
    on the ground while reading in CI as "validation is enforced".
  - **The messenger's severity flags are derived from `MinSeverity`** so the driver filters
    before the callback. The callback keeps its own threshold check as well, because that is
    what avoids paying `std::format` for a message about to be discarded. Verbose is never
    requested: it collapses to `Info` on the neutral scale, so asking for it would multiply
    message volume with nothing a caller could distinguish.
- **Verified deliberately:** precommit green (121 unit tests, 9 of them new); the baseline
  report is byte-identical and the screenshot hash matches. Then, because none of that
  touches the diagnostic path, the R5 hand check was repeated — graphics queue named with
  `ObjectType::eBuffer` — confirming the message text is unchanged, the error reaches the run
  report, `--strict-validation` exits 1, `--validation-policy failfast` aborts at the first
  error with the message printed and before `Init()` completes, `ignore` reports nothing, and
  both malformed-flag cases are rejected. Then reverted.
- **Size:** S · **Needs:** R5 · **Was:** step 28

### R8 — `Rhi::ICommandList` and the neutral barrier API

- **Do:** Add `ICommandList` with, for now, only what Stage 5 needs: `Barrier(...)`,
  `CopyBuffer`, `CopyBufferToTexture`, `CopyTextureToBuffer`, plus begin/end. Implement
  `VulkanCommandList` over `vk::CommandBuffer`. Convert `src/Barrier.h`'s preset functions to
  neutral `BarrierPresets.h` and route every existing `RecordImageBarrier` call through
  `ICommandList::Barrier`. Batch multiple image barriers into one `pipelineBarrier2` call,
  which resolves the `TODO` at the top of the old `RecordImageBarrier`.
- **Boundary:** draw/bind/viewport recording stays on raw `vk::CommandBuffer` in `App` until
  Stage 8. Moving it now would drag the pipeline and descriptor model along (D7, D8).
- **Verify:** Headless report identical, **validation errors 0 with synchronization
  validation enabled** (`validate_sync` is already on at `main.cpp:1076`). Barrier count per
  frame logged and sane.
- **Size:** M · **Needs:** R5
- **As built:** the barrier half landed; `ICommandList` did not, because **this step is
  ordered too early for it**. Every method the step lists — `Barrier`, `CopyBuffer`,
  `CopyBufferToTexture`, `CopyTextureToBuffer` — has to name a resource, and the only
  neutral way to name one is the `BufferHandle`/`TextureHandle` that R9 and R10 introduce.
  A neutral interface written before them could only take `vk::Buffer`/`vk::Image`, which is
  the leak the interface exists to prevent; declaring handle-typed methods that nothing can
  call yet would be worse still. **The interface and its three copy methods belong at the end
  of R10**, where the call sites they replace are being rewritten anyway. Nothing else
  depends on the ordering: R11's `UploadContext` needs R9 and R10 regardless.

  What did land, and is the whole of the neutral vocabulary being *used* rather than merely
  declared:

  - `Rhi::TextureBarrier` in `rhi/Barrier.h` — the neutral (stage, access, layout) triple
    plus aspect and subresource range, deliberately without a resource. Separating the
    description from the resource is what lets the presets be constants; when R10 lands, the
    handle becomes a field on it.
  - `rhi/BarrierPresets.h` — all of the old `rhi/vulkan/Barrier.h` presets re-expressed
    neutrally and renamed per D13 (`UndefinedToCopyDst`, `RenderTargetToShaderResource`, …),
    plus `PreserveRenderTarget` for the load-op dependency that was written inline in
    `RecordImGui`, and `UndefinedToUnorderedAccess` / `UnorderedAccessToShaderResource` for
    the four barriers `CloudSystem` was hand-writing. `rhi/vulkan/Barrier.h` is deleted, and
    with it the near-namesake hazard its own header comment described.
  - `Rhi::Vulkan::RecordBarriers` (`rhi/vulkan/BarrierUtil.h`, now with a `.cpp` because the
    conversion tables are module-private) batches a span into one `pipelineBarrier2`,
    resolving the old `TODO`. Per frame: 13 barriers, in 8 calls rather than 13.
  - **`AccessFlags::RenderTargetRead` had to be added**, so D4's list is one enumerator short
    of as-built. A colour attachment is read whenever blending is on or a pass loads instead
    of clearing, and the ImGui pass does exactly that; there was no neutral way to say it.
    It collapses onto `D3D12_BARRIER_ACCESS_RENDER_TARGET` with `RenderTargetWrite`, the same
    many-to-one shape `UnorderedAccess` already has.
  - The neutral mapping moves two depth transitions from the separate depth-only layouts to
    the combined depth/stencil ones, while `beginRendering` and the depth descriptor still
    name the separate ones. That is correct, not tolerated: the specification defines the
    combined layouts as *equivalent* to the separate pair, which is why the pre-existing
    mismatch between `UndefinedToDepthAttachment` and `DepthAttachmentToShaderRead` was never
    a validation error either. Cited in `ToVk(TextureLayout)`.
  - The per-frame count the step asks for is **two** numbers, `barriers` and `barrierCalls`,
    both logged on change and both added to the run report. One without the other says
    nothing useful: a barrier count cannot distinguish one call carrying three barriers from
    three carrying one each, which is exactly what this step changed. `Rhi::BarrierCounts`
    carries the pair, `CloudSystem::RecordDispatch` returns it so the cloud pass is included,
    and the two threaded record functions keep their own rather than sharing one set.
  - **The committed baseline report is two fields short of what the app now writes.** Its
    other fields are unchanged and still compare exactly; regenerate
    `tests/baseline/report_*.json` when convenient so a plain `diff` is clean again.

### R9 — Buffers become handles

- **Do:** `BufferHandle` + `BufferDesc` + `IDevice::CreateBuffer/Destroy/Map/Unmap`, backed
  by a `HandlePool<VulkanBuffer>`. Add `UniqueHandle`. Convert every `AllocatedBuffer` user:
  quad buffers, instance buffers, global buffers, staging buffers, the screenshot staging
  buffer, and `ModelData`'s vertex/index buffers. `AllocatedBuffer` becomes the pool payload
  `VulkanBuffer` and stops being visible outside the module.
- **Verify:** Headless report identical. Live-buffer count logged at shutdown is 0 — the
  first thing the handle model buys.
- **Size:** L · **Needs:** R8
- **As built:** done, and the shutdown log reads `Device destroyed with 0 live buffers.` The
  report is byte-identical to the baseline, and a capture taken at the same resolution as a
  pre-R9 one is pixel-identical, so nothing about rendering moved.

  Five things differ from the **Do** text, four of them forced:

  - **`GetMappedData(BufferHandle)`, not `Map`/`Unmap`.** `ToVk(MemoryAccess)` already commits
    every host-visible allocation to `VMA_ALLOCATION_CREATE_MAPPED_BIT`, so the pointer is
    valid from creation to destruction whatever the API says. A Map/Unmap pair would be a
    fiction: `Unmap` would either do nothing or invalidate the pointer the per-frame uniform
    and instance buffers hold across frames. D3D12 maps UPLOAD/READBACK heaps persistently
    too, so nothing portable is given up.
  - **`Rhi::Vulkan::GetBuffer(IDevice&, BufferHandle)` had to be added** to `VulkanNative.h`.
    `vkCmdBindVertexBuffers`, `vkCmdBindIndexBuffer`, `vkCmdCopyBufferToImage` and
    `VkDescriptorBufferInfo` all take a `VkBuffer`, and all four still happen in `src/` — draw
    recording until Stage 8, descriptor writes until bindless. It is listed in the one file
    that counts the leaks, and it retires with those call sites rather than needing its own
    step.
  - **`MeshBatch` carries `BufferHandle`, not `vk::Buffer`** (`src/InstanceData.h`), so the
    resolve happens at the bind site rather than being stored. Unplanned, and a boundary win:
    it takes a Vulkan type out of a struct that Stage 9's `FrameSnapshot` is descended from.
  - **`CreateStagedBuffer` and `CopyBuffer` moved into `namespace Rhi::Vulkan`** and now speak
    handles, rather than being folded into `IDevice`. R11's `UploadContext` is what replaces
    them; designing that here would have meant writing it twice. They still drain the queue
    per call.
  - **Debug names fold into `BufferDesc::DebugName`.** Every creation site previously paired a
    `SetVkDebugName` with a `vmaSetAllocationName`; the device now does both, and the sites
    lost two lines each.

  `ModelLoader` no longer needs a `VmaAllocator` at all. `TextureLoader` and `CubemapLoader`
  keep theirs for image creation only, which is what R10 takes. `UniqueHandle` covers the
  scope-local staging buffers exactly as D2 predicted, and has six CPU-only tests
  (`tests/unit/rhi/UniqueHandleTests.cpp`) built on a recording `IDevice` stub — move
  transfer, overwrite-on-assign, `Release`, and double-`Reset`.

### R10 — Textures, views and samplers become handles

- **Do:** `TextureHandle`, `TextureViewHandle`, `SamplerHandle` + descs + pools. Convert
  `Texture`, `Cubemap`, the render targets, the depth resources and the texture sampler.
  `src/Texture.h` and `src/Cubemap.h` become thin asset-side wrappers holding a handle plus
  their name/path/create-info, so `ResourceCache` and `MaterialFactory` keep working
  unchanged until Stage 7.
- **Decide first:** `FindDepthFormat` (`main.cpp:2278`) currently ends its candidate list with
  `eD16UnormS8Uint`, which `Rhi::Format` deliberately does not carry — there is no DXGI
  equivalent (R2's as-built note). Either drop that candidate or map it to
  `D24UnormS8Uint`. Doing neither means the conversion throws on whatever hardware falls
  through to it.
- **Also, carried over from R8:** add `Rhi::ICommandList` — `Begin`/`End`, `Barrier` over a
  span of `TextureBarrier`, and `CopyBuffer` / `CopyBufferToTexture` / `CopyTextureToBuffer` —
  with `VulkanCommandList` over `vk::CommandBuffer` behind it. R8 deferred it because every
  one of those methods names a resource and handles did not exist yet; here they do. The
  handle moves into `TextureBarrier`, and `Rhi::Vulkan::ImageBarrier` and the free
  `RecordBarrier` functions in `rhi/vulkan/BarrierUtil.h` go away, since the barrier call
  sites in `App`, the loaders and `CloudSystem` are being rewritten for handles regardless.
  Draw/bind/viewport recording still stays on the raw `vk::CommandBuffer` until Stage 8.
- **Verify:** Headless report identical, screenshot identical. Live-texture count 0 at
  shutdown.
- **Size:** L · **Needs:** R9
- **As built:** done. The report is byte-identical to the baseline, a capture taken at the
  same resolution as a pre-R10 one is byte-identical too, and the shutdown log reads
  `Device destroyed with 0 live buffers, textures, texture views and samplers.` Validation
  errors stayed at 0 with synchronization validation on, which is the check that matters
  here: every barrier in the renderer was rewritten. The window was also resized twice
  during a separate run, because the fixed-frame baseline never exercises
  `RecreateSwapchainAndRenderImages` — and that is the one path that re-registers the
  swapchain's textures and move-assigns over live render targets. Clean, and still 0 live
  resources at shutdown.

  **The decision the step asked for: `D16UnormS8Uint` is dropped, not promoted.** It cost
  nothing, because it was unreachable. The specification's mandatory format table requires
  `VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT` to be "supported for at least one of
  `VK_FORMAT_D24_UNORM_S8_UINT` and `VK_FORMAT_D32_SFLOAT_S8_UINT`" (Vulkan 1.4,
  *Mandatory Format Support: Depth/Stencil*), and both sit above it in the candidate list —
  so no conformant device could ever fall through to a fourth candidate. Promoting it to
  `D24UnormS8Uint` would have been worse than dropping it: it would name a format the
  device had just been asked about and declined. The reasoning is in `FindDepthFormat`, and
  `Rhi::Format`'s note about the omission now records the outcome instead of the question.

  Seven things differ from the **Do** text:

  - **`ICommandList` is reached through `Rhi::Vulkan::WrapCommandList(IDevice&,
    vk::CommandBuffer)`,** not by constructing a public class. `VulkanCommandList` stays in
    `src/vulkan/` because R17's end state keeps only four headers in `include/rhi/vulkan/`,
    and a fifth would have to be removed again a step later. The wrapper is non-owning: the
    renderer still allocates and submits its own command buffers, records draws on the raw
    one, and uses the neutral list for barriers and copies.
  - **Swapchain images get handles too, via
    `Rhi::Vulkan::RegisterExternalTexture`.** `TextureBarrier` now names a `TextureHandle`,
    and the swapchain images are the one thing barriers touch that the device did not
    allocate — so without this there would have to be a second, image-typed barrier path
    purely for them. Registering gives the image a pool slot whose payload has a null
    `VmaAllocation`, which `VulkanTexture` reads as "not ours" and frees nothing. It is
    also the shape Stage 6 wants: `IPresentTarget` will hand out the handle itself and this
    retires with it.
  - **`TextureAspect` and `DefaultAspect` moved from `rhi/Barrier.h` to `rhi/RhiTypes.h`.**
    A view needs the aspect as much as a barrier does, and `TextureViewDesc` should not
    have to include the barrier vocabulary to say which half of a depth/stencil format it
    covers.
  - **`TextureViewDesc` has two sentinels**, both meaning "follow the texture": an
    `Undefined` format, and a `None` aspect resolving through `DefaultAspect`. The aspect
    one earns its keep — a colour default would silently produce a wrong-aspect view for
    every depth texture, and the call sites that would have to override it are exactly the
    ones most likely to forget.
  - **`TextureBarrier::On(handle)`** is what pairs a preset with a resource. The presets
    stay constants that name no texture, which is what R8's note said the handle would
    have to preserve; `On` returns a copy, so one preset can be applied to several
    textures in the same batch.
  - **`BufferTextureCopyRegion` carries no offset.** Every copy in the codebase covers a
    whole subresource from its origin, and an offset field would mean adding an `Offset3D`
    to the neutral vocabulary for a case that does not exist.
  - **`src/Texture.h` and `src/Cubemap.h` moved back out of `rhi/vulkan/`.** Holding two
    handles, they have no Vulkan in them and no business in the RHI; they are asset-side
    types that Stage 7 moves into `Assets`. `Cubemap` is now a `Texture` plus its create
    info, since the only thing that makes a cubemap one is the view dimension.
    `rhi/vulkan/ImageUtil.h`, `AllocatedImage.{h,cpp}` and `BarrierUtil.{h,cpp}` are
    deleted — every one of their callers now goes through `IDevice` or `ICommandList`.

  **The escape hatch grew, and that is worth reading rather than skimming** (§9's third
  risk). `VulkanNative.h` went from 8 functions to 13. `GetAllocator` left, because
  textures were the last VMA consumer outside the module. Six arrived, in five roles:

  | Added | Why | Retires with |
  |---|---|---|
  | `GetImageView` | `VkDescriptorImageInfo` and `vk::RenderingAttachmentInfo` take raw views | bindless (step 69) / Stage 8 |
  | `GetSampler` | same, for `VkDescriptorImageInfo::sampler` | bindless (step 69) |
  | `RegisterExternalTexture` | the swapchain's images, see above | Stage 6's `IPresentTarget` |
  | `WrapCommandList` | the renderer owns its command buffers until Stage 8 | Stage 8 |
  | `GetNativeFormat` / `FromNativeFormat` | `PipelineBuilder` takes `vk::Format` (D8), and the depth-format search queries `VkFormatProperties` | Stage 8 |

  There is deliberately **no** `GetImage`: nothing outside the module names an image any
  more, because barriers and copies both take a `TextureHandle`.

  Two smaller notes. `SamplerDesc::MaxAnisotropy == 0` means "the device maximum", which is
  what removed the last read of `maxSamplerAnisotropy` from the renderer. And `ModelLoader`
  lost its `vk::raii::Device` and `vk::raii::PhysicalDevice` members, which had been unused
  since R9 and were only still being passed because `ResourceManager::Init` fed all three
  loaders from one signature.

### R11 — `UploadContext` — batch transfers

- **Do:** Replace the per-resource `EndSingleTimeCommand` → `queue.waitIdle()` with a context
  that records many copies into one command buffer, submits once, waits on one fence, then
  releases staging buffers. Route `TextureLoader`, `CubemapLoader`, `ModelLoader` and
  `CreateQuadBuffers` through it. Interface is neutral and handle-based.
- **Verify:** **Time a Sponza load before and after** with `core/Timer.h`. Sponza performs
  ~70 full GPU drains today; expect a large reduction. Headless report identical.
- **Size:** L · **Needs:** R6, R10 · **Was:** step 29
- **As built:** done. Submissions for a Sponza load went **74 → 5**; the headless report is
  byte-identical to the baseline and the capture is pixel-identical.

  **The wall-clock expectation in the Verify line was wrong, and it is worth correcting
  rather than quietly meeting.** Before touching anything, the old path was instrumented to
  time its own drains: 71 submissions cost **44 ms of a 5.1 s load** — 0.9%. Sponza loads in
  5.1–5.2 s before and 5.1–5.3 s after, which is run-to-run noise. Nearly all of that 5 s is
  stb_image decoding and Assimp parsing, and no amount of batching touches either.

  So the value of this step is not the clock, and pretending otherwise would set up R12a to
  be judged against a number it also will not move. What it actually buys:

  - **A drain of the whole queue becomes a wait on one fence.** `vkQueueWaitIdle` is defined
    as a fence per outstanding submission plus `vkWaitForFences`, so this was always the
    bluntest available synchronisation. It only looked free because nothing else is running
    while a scene loads — and the ImGui "Load Scene" button already breaks that assumption
    once, and threaded loading would break it permanently.
  - **It is the step R12a needs.** Moving uploads to a dedicated transfer queue is a change
    to one submission site, not seventy.
  - **Peak staging memory is now bounded and visible.** Sponza stages 380 MiB in total; the
    old path never held more than one resource's worth at a time, so batching without a cap
    would have turned that into a 380 MiB spike. `UploadContextDesc::StagingBudget` (128 MiB)
    is what keeps both properties, and it is the reason a Sponza load is 5 submissions rather
    than 2.

  Four things worth recording beyond the **Do** text:

  - **A texture is uploaded whole or not at all.** `UploadTexture` takes a span of every
    subresource in one call, which looks like a convenience and is actually a correctness
    requirement: the context transitions a texture from `Undefined`, which permits the driver
    to discard its contents, so a cubemap whose faces straddled a budget flush would have the
    first batch's faces thrown away by the second. Taking them together makes that
    unexpressible. `TextureUpload` therefore carries the data and *no* staging offset — the
    caller does not own the staging buffer, so it cannot have one — which is why it is not
    `BufferTextureCopyRegion`.
  - **`ResourceManager` flushes at the outermost load, via a depth-counting RAII scope.** The
    nesting is what makes the batching work at all: loading a model loads its textures through
    the same class, so a flush per public call would put every texture back in its own
    submission. It also makes "a resource `ResourceManager` hands back is on the GPU" true by
    construction. Its destructor swallows and logs a failed flush, because throwing there
    during a failed load's unwind would terminate the process.
  - **No barrier is needed between an upload and the submission that reads it**, and this is
    a specification guarantee, not an assumption inherited from the old code. A fence signal's
    first access scope is "all memory access performed by the device" (Vulkan 1.4, *Fences*),
    which makes the copies available; the next queue submission's second access scope is
    likewise all device access (*Host Write Ordering Guarantees*), which makes them visible to
    everything in it. Cited at the submit site so nobody adds a redundant barrier later.
  - **The upload pool uses the graphics family**, because a command buffer may only be
    submitted to a queue of its pool's family and the graphics queue is still the only one the
    device creates. R12a changes the pool and the submit together, and brings the ownership
    transfer with it.

  `rhi/vulkan/BufferUtil.h` is deleted — `CreateStagedBuffer` and `CopyBuffer` were exactly
  what this replaces — and the three loaders lost their command pool and queue references,
  leaving them holding an `IDevice&` and an `IUploadContext&`. `rhi/vulkan/CommandListUtil.h`
  survives: `CloudSystem::BakeNoiseTexture` still uses it, and that is a dispatch rather than
  an upload, so it stays on the single-shot path until Stage 8.

  Verified beyond the baseline: the budget path was stressed by temporarily setting
  `StagingBudget` to 1 byte, which forces every upload into its own batch (22 submissions for
  22 resources) and exercises the split logic and the context's reuse across flushes — output
  stayed pixel-identical. Validation errors stayed at 0 with synchronization validation on,
  and the device still reports 0 live resources at shutdown.

  Not tested by unit tests: the context is backend code with no neutral logic to isolate, and
  R16 already plans the GPU tests that cover it — buffer and image upload round-trips, and the
  cubemap-face test. `UploadStats` exists so those can assert on submission counts rather than
  on log lines.

### R12a — Use the dedicated transfer queue

- **Do:** Point `UploadContext` at the transfer family when one exists, with its own command
  pool, plus queue-family ownership release/acquire before first graphics use — expressed as
  an acquire record returned by `UploadContext` rather than raw barriers at the call site
  (D6).
- **Read before writing:** the Vulkan spec's synchronization chapter on queue-family
  ownership transfer. A release in the source family and an acquire in the destination family
  must both be issued, with identical subresource ranges, and the acquire must be ordered
  after the release by a semaphore. This is the single easiest thing in the stage to get
  plausibly-but-wrongly right.
- **Verify:** Headless report identical, **zero validation errors** with synchronization
  validation on. Load time improves further on discrete GPUs.
- **Size:** M · **Needs:** R11 · **Was:** step 30
- **As built:** done. Uploads run on queue family 1 of this development machine's RX 580
  (RADV), each flush is a release on the copy queue and an acquire on the graphics queue
  ordered by a semaphore, validation errors stayed at 0 with synchronization validation on,
  and the render output is unchanged. Four things departed from the text above.

  **The acquire is submitted by `UploadContext`, not handed back.** D6 says the context
  "returns an explicit acquire record"; it does not, and returning one would not have worked.
  The callers are `TextureLoader`, `CubemapLoader` and `ModelLoader`, none of which owns a
  command list — R11 took their command pools away — and the renderer that does own one
  receives a finished resource with no idea which flush produced it. Every design that hands
  the record out therefore ends in the same place: some caller must remember to record it
  before first use, and forgetting is silent until it is a corrupt texture on somebody else's
  driver. Since `Flush()` already blocks until the GPU is finished, the context can submit
  the acquire itself and make "a resource a flush covered is ready to use" true by
  construction, which is the property the rest of the engine was already relying on.

  The neutral half of D6 is unaffected and is what actually matters for a second backend:
  `IUploadContext` still says nothing about queues, families or ownership. What changed is
  that the Vulkan side discharges the obligation instead of exporting it. D3D12 will have
  nothing to export in the first place — a copy-queue resource is left in `COMMON`, which
  costs a layout, not a submission.

  **A flush is two submissions, not one, and `UploadStats::Submits` counts both.** The
  headline R11 left behind — 74 → 5 for a Sponza load — becomes 5 copies and 5 acquires. The
  alternative was to keep counting batches and quietly stop counting submissions, which would
  make the number agree with the old one by measuring something else. The log line names both
  halves for the same reason.

  **Staging offsets are aligned to 4 bytes.** A queue family that supports transfer but
  neither graphics nor compute requires every `bufferOffset` to be a multiple of 4
  (`VUID-vkCmdCopyBufferToImage-commandBuffer-07737`), which the previous tight packing would
  violate for a `Format::R8Unorm` texture. No such family exists on this machine — RADV does
  not expose Polaris's SDMA engine even under `RADV_PERFTEST=transfer_queue`, so the copy role
  resolves to the async compute family, which advertises compute and is therefore exempt. The
  rule is honoured anyway rather than left as a trap for the first GPU that has one.

  **One best-practices message is muted by ID, and `CreateInstance` grew a
  `message_id_filter` list to do it.** Every transfer draws
  `BestPractices-PipelineBarrier-unneeded-QFOT` on a device that has a copy queue of its own
  *and* supports `VK_KHR_maintenance9` — 20 messages per load here, reported 10 times before
  the layer's duplicate limit cut it off. The advice is real: maintenance9 makes buffers,
  linear images, and optimal images that clear
  `VkQueueFamilyOwnershipTransferPropertiesKHR::optimalImageTransferToQueueFamilies` keep
  their contents through an implicit acquire, and the specification says explicit transfers in
  those cases have "no functional nor performance advantage" and are "not recommended for new
  applications".

  Taking that advice was considered and rejected *for now*, on testing grounds rather than
  technical ones. This GPU reports that property as every family and none of the uploaded
  textures carry an attachment usage, so with maintenance9 enabled every transfer this step
  performs would be skipped — leaving the path every other driver takes as dead code on the
  only machine that runs the application at all. Adopting maintenance9 is worth doing as its
  own piece of work, together with a way to force the transfer path so that both stay
  exercised; the mute is what should be deleted when that happens.

  The mute was one named ID with its reasoning at the call site, not a category, and it kept
  `validationWarnings` meaning "something needs looking at" — the run report stayed
  byte-identical to `tests/baseline/`, so nothing needed rebaselining. **R12b removed it**, by
  adopting maintenance9 so that the default configuration stops performing the transfers the
  message is about. A run that disables maintenance9 emits the message again, which is correct:
  it is a run deliberately pretending to be hardware that lacks the extension.

  Two spec details worth keeping, both from Vulkan 1.4 *Queue Family Ownership Transfer*.
  The release barrier's destination masks and the acquire barrier's source masks are ignored
  and are set to zero as the specification asks — writing the eventual reader's stage into the
  release would additionally be invalid on a transfer-only family, which has no pixel shader
  stage to name. And the acquire's destination stage is `AllCommands` because an acquire
  happens in no defined pipeline stage, so without `VK_KHR_maintenance8` nothing else can wait
  for one. Its destination *access* is `ShaderRead` for images and `MemoryRead` for buffers:
  an image's access mask has to agree with its layout, and the blanket `MemoryRead` is not
  among the flags `ShaderReadOnlyOptimal` permits.

  **Load time did not improve, and the Verify line above should not have expected it to.**
  R11 already established that a Sponza load is 5 s of `stb_image` and Assimp with 44 ms of
  GPU waiting inside it; moving that 44 ms to another engine cannot show up. Nor can the two
  queues overlap yet — loading is synchronous and `Flush()` blocks on the fence — so what this
  step buys is the same thing R11 bought: the mechanism threaded loading needs, in place and
  exercised, before anything depends on it.

  Verified beyond the baseline: `StagingBudget` temporarily set to 1 byte puts each of the 22
  resources in its own batch and produces exactly 44 submissions with the output unchanged;
  forcing `m_bOwnershipTransfer` false reproduces the pre-R12a path. Not covered by unit tests,
  for R11's reason — this is backend code with no neutral logic to isolate, and R16's GPU
  tests are where an upload round-trip gets asserted.

### R12b — Adopt `VK_KHR_maintenance8` and `VK_KHR_maintenance9`

- **Do:** Enable both extensions when the device has them, and fall back to R12a's behaviour
  when it does not. maintenance9 turns "does this resource need an ownership transfer" into a
  per-resource question; maintenance8 lets the transfer's barriers name real pipeline stages
  instead of being pinned to `AllCommands`. Add `--vk-disable-extension <name>` so the
  fallback path stays reachable on hardware that has the extensions.
- **Verify:** All four combinations of the two extensions render identically with zero
  validation errors, and the default configuration needs no muted messages.
- **Size:** M · **Needs:** R12a
- **As built:** done, and it earned its place by finding a bug rather than by the performance
  it does not deliver.

  **The barrier that keeps a texture on the copy queue was invalid, and nothing before this
  step could have run it.** `BarrierPresets::CopyDstToShaderResource` names `PixelStage` as its
  destination, which is illegal on a command buffer from a family without graphics
  (`VUID-vkCmdPipelineBarrier2-dstStageMask-09676`); this machine's copy family is
  compute+transfer. R12a never hit it because a separate copy queue meant *every* texture was
  transferred, and the transferred path replaces that barrier with the release. maintenance9
  makes textures stay on the copy queue for the first time, and the first run produced 10
  validation errors. The fix empties the destination scope at the call site, which is also what
  the situation truly is: nothing later in that command buffer reads the texture, and the
  consumer in a subsequent submission is reached through the fence wait the copies already rely
  on. This is precisely the failure mode §9 warns about for this area, arriving from the
  direction nobody was watching.

  **Extension and feature move together or not at all.** Every relaxation either extension
  describes is worded "if the feature is enabled", so enabling the extension alone changes
  nothing — and leaving a feature struct chained for an extension that is not enabled is
  undefined behaviour rather than a no-op. `CreateLogicalDevice` therefore keeps both structs in
  its `vk::StructureChain` and `unlink`s the ones whose extension did not make the list.

  **The rule is a pure function with unit tests, which is what makes two paths safe.**
  `RequiresOwnershipTransfer` in `src/vulkan/OwnershipTransfer.{h,cpp}` takes the device's
  promises as a value and answers per resource; `tests/unit/rhi/OwnershipTransferTests.cpp`
  covers the arrangements this GPU does not have. Two things about it are easy to get backwards
  and are locked down by tests: `optimalImageTransferToQueueFamilies` belongs to the *source*
  family and its bits are *destination* family indices, and it is 32 bits wide, so a
  destination family index of 32 or above is unrepresentable and has to fall back to
  transferring rather than to a shift with undefined behaviour.

  **maintenance8 changes what the barriers say and not what happens, exactly as predicted.**
  The release and acquire now name `Copy` where the specification would otherwise ignore the
  mask, which pins the hand-over to a stage instead of leaving it unplaced. Two things stay at
  `AllCommands` and both for reasons the extension cannot fix: the acquire's *destination*
  stage, because an upload context fills resources for a caller that has not said what will
  read them, and the semaphore wait, because that submission contains the acquire and nothing
  else, so there is no later work a narrower wait could let start sooner. Both become worth
  narrowing when the acquire moves into the command list that consumes the resource. No number
  moved, which is what R11 and R12a both also found and is why it was not promised.

  **The testing lever is `DeviceDesc::DisabledOptionalExtensions`, and the flag is a thin skin
  over it.** The field is neutral in type and backend-specific in content, and R16 sets it
  directly rather than going through a command line. `--vk-disable-extension` is repeatable, and
  a name the backend does not treat as optional is reported and ignored, so it can never turn a
  working device into a failing one. The `vk-` prefix marks it as backend-specific in a way that
  survives a second backend existing.

  Measured on the RX 580 (RADV), which has both extensions. All four runs are **pixel-identical**
  to each other and every counter but the warning count matches:

  | Configuration | Submissions | Transfers | `validationErrors` | `validationWarnings` |
  |---|---|---|---|---|
  | both enabled | 4 | none | 0 | 0 |
  | no maintenance9 | 8 | all, stage-accurate | 0 | 10 |
  | neither | 8 | all, `AllCommands` | 0 | 10 |
  | no maintenance8 | 4 | none | 0 | 0 |

  The 10 warnings in the middle two rows are the best-practices message about performing a
  transfer a maintenance9 device does not need — correct, since those runs are pretending not to
  have it. The default row is clean with nothing suppressed, which is what let R12a's mute be
  deleted. Stressed with `StagingBudget` at 1 byte: 22 submissions with maintenance9 and 44
  without, for 22 resources, both with zero errors and no leaked resources.

  **Deferred to R16:** `--vk-force-single-queue`, to reach the third path — a device with no
  separate copy family — without patching code. It is a different mechanism from disabling an
  extension, and R16 is where the GPU tests that would consume it get written.

### R13 — Growable `DescriptorAllocator`

- **Do:** `std::vector<vk::raii::DescriptorPool>`; on `eErrorOutOfPoolMemory`, allocate
  another pool at ~1.5× and retry. Use it in `MaterialFactory`, deleting
  `s_MAX_MATERIAL_SET_COUNT` (`MaterialFactory.cpp:10`). Vulkan-side by design (D7).
- **Verify:** **Temporarily set the initial pool size to 4**, load Sponza (~25 materials),
  confirm it loads with pool growth logged, then restore a sensible size. Headless report
  identical.
- **Size:** M · **Needs:** R5 · **Was:** step 31
- **As built:** done. Two things differ from the **Do** text.

  **Growth happens before a pool is overrun, not in response to the failure.** The allocator
  counts the sets it has taken from the newest pool and adds the next one when that count
  reaches the pool's capacity. Catching the failed allocation was implemented first and does
  work — Sponza loaded at an initial capacity of 4, growing 4 → 6 → 9 → 13 — but every growth
  cost three validation warnings, because the layers report each
  `VK_ERROR_OUT_OF_POOL_MEMORY`. `validationWarnings` is one of the numbers the run report
  exists to make trustworthy, so a routine, expected growth must not raise it. The counter
  brought the same run to **zero warnings and zero errors** with the growth still logged.

  The catch stays, because the counter cannot see the other reason an allocation fails.
  Fragmentation is invisible from outside the driver, and the specification's instruction is
  to treat *any* error as fragmentation and create a new pool — `VK_ERROR_FRAGMENTED_POOL`
  was added late in Vulkan 1.0, so a driver written against an earlier patch version is
  allowed to report it as something else. So the retry catches `vk::SystemError` rather than
  the two named codes, and a genuine out-of-memory surfaces from the retry instead.

  **Pool sizes are stated per set, not per pool.** The caller passes the descriptors *one*
  set needs and the allocator multiplies by the pool's set capacity. Stating it per pool
  would make every growth a place to get the arithmetic wrong: raising `maxSets` without
  raising the descriptor counts to match produces a pool that reports itself as having room
  and then refuses the allocation. `MaterialFactory` now derives its one entry from
  `TextureBinding::COUNT`, which retires the hand-maintained `s_MAX_TEXTURE_COUNT_PER_MAT`
  alongside `s_MAX_MATERIAL_SET_COUNT`.

  `kInitialMaterialSetCapacity` is 100 — the old ceiling, kept as the starting size so no
  scene shipped today pays for a second pool, and Sponza confirms it (zero growth events).
  The allocator is not thread-safe; material creation is single-threaded today, and a mutex
  would be paying now for a guarantee nothing asks for. Sponza's report is identical at
  capacity 4 and capacity 100, and `tests/baseline/`'s report is unchanged.

### R14 — Growable instance buffer

- **Do:** Replace the `throw` in `UpdateInstanceBuffer` (`main.cpp:2318`) with: wait idle,
  reallocate to `max(needed, capacity * 2)`, remap, log once. `MAX_INSTANCE_COUNT` becomes an
  initial capacity, not a ceiling.
- **Verify:** Author `content/scenes/stress.map` with > 1024 instances and confirm it renders
  instead of throwing. Existing scenes' reports identical.
- **Size:** S · **Needs:** R9 · **Was:** step 32
- **As built:** done. `MAX_INSTANCE_COUNT` is now `INITIAL_INSTANCE_CAPACITY`, and the
  capacity it names lives in `m_InstanceCapacity` because it changes.

  **Every frame's buffer is replaced at once.** Growing only the frame being filled would
  leave the other frame short by exactly the same amount, so it would grow again on the very
  next frame — two device waits and two log lines for one overflow.

  **The wait is the whole correctness argument, and it is the part the obvious test does not
  reach.** Replacing a buffer destroys the old one through `UniqueHandle::operator=`, and a
  frame still in flight has that buffer bound as a vertex buffer;
  `VUID-vkDestroyBuffer-buffer-00922` requires every submitted command referring to it to have
  completed. The current frame's fence has been waited on by the time
  `UpdateInstanceBuffer` runs, but nothing covers the others, so the growth waits on the
  device.

  A scene is fully loaded before the first frame, so the growth `stress.map` triggers happens
  on frame 0 with nothing in flight — the wait returns immediately and proves nothing. Forcing
  a growth every 30 frames instead produced five reallocations with both frames in flight and
  synchronization validation on (it is enabled unconditionally in
  `VulkanDevice::CreateInstance`), for **0 errors and 0 warnings**. That is the evidence the
  step rests on; the probe was removed afterwards.

  **No remap step was needed.** `UpdateInstanceBuffer` already calls `GetMappedData` fresh
  each frame and the vertex binding is resolved at record time, so nothing held a pointer or
  handle across the swap. The **Do** text's "remap" was already satisfied by the existing code.

  **`content/scenes/stress.map` is 80 cars in a lattice, not a flat grid.** Four depths, each
  a 5x4 arrangement sized to the frustum at that depth from camera preset 1 and offset by half
  a cell from its neighbour, so a nearer car sits between the ones behind it rather than hiding
  one. That fills the frame instead of leaving the cars in a thin band at the horizon, which
  makes the same scene usable for frustum culling later. The lateral extent is sized for a
  near-square window; a wider one only adds margin at the sides, so no car falls off the edge
  on a different window size. 80 cars x 21 meshes = **1680 instances**, one growth to 2048,
  `validationErrors: 0`. The car is authored at `scale="1"` — `test_scene.map` uses `scale="10"`,
  which is ten times life size and made the cars interpenetrate at any spacing that fits on
  screen.

  `tests/baseline/`'s report is unchanged, and `test_scene` never grows: 23 instances against
  an initial 1024.

### R15 — `PipelineCache`

- **Do:** One cache object created at startup, seeded from `<user data dir>/pipeline_cache.bin`
  via `platform/Paths.h`, passed to all five pipeline creations (all currently pass `nullptr`)
  and to `ImGui_ImplVulkan_InitInfo::PipelineCache`. Write on shutdown. Neutral interface,
  opaque blob (D8).
- **Verify:** Log pipeline-creation time; second launch measurably faster. Delete the file
  and confirm it regenerates. **Corrupt the file and confirm it is rejected gracefully**
  rather than crashing.
- **Size:** M · **Needs:** R5 · **Was:** step 33
- **As built:** done. `Rhi::IPipelineCache` has exactly one method, `Save()`, and the
  description it is created from carries the path. The file I/O sits behind the interface
  rather than in the caller because deciding a file is stale is a backend question — Vulkan
  answers it from a header the specification defines, D3D12 would answer it some other way,
  and neither answer belongs in `App`.

  **The header check is what the step is really about.** Handing a driver bytes that did not
  come from `vkGetPipelineCacheData` is invalid usage
  (`VUID-VkPipelineCacheCreateInfo-initialDataSize-00769`), and while the specification
  requires the implementation to *ignore* data it does not recognise — so a corrupt file was
  never going to crash — a knowingly invalid call is still one the validation layers may
  report, and `validationErrors` is a number this project keeps at zero. So the backend
  validates before it hands anything over: 32-byte header, `headerSize` 32,
  `headerVersion` `VK_PIPELINE_CACHE_HEADER_VERSION_ONE`, then `vendorID`, `deviceID` and
  `pipelineCacheUUID` against the physical device.

  Read a byte at a time rather than through `VkPipelineCacheHeaderVersionOne`. The
  specification writes every field least significant byte first *regardless of host byte
  order*, and explicitly declines to promise the C struct is packed to match — it says an
  application whose compiler diverges "must employ another method to set values at the
  correct offsets". Reading it by hand is that other method, and costs four lines.

  Confirmed against a real file: `20 00 00 00 | 01 00 00 00 | 02 10 00 00 | df 67 00 00`
  is 32, version one, vendor `0x1002`, device `0x67df`. Five rejection paths were exercised —
  truncated to 16 bytes, 4 KiB of `/dev/urandom`, a valid header with a foreign
  `pipelineCacheUUID`, a valid header with a foreign `deviceID`, and no file at all. Each
  starts empty, regenerates, and passes `--strict-validation`. A device mismatch logs at
  Info because a driver update legitimately causes it; a malformed header logs at Warning
  because nothing legitimate does.

  **`Save()` writes beside the file and renames over it.** A write interrupted halfway leaves
  a blob that still carries a valid 32-byte header and so passes every check above, while
  violating `VUID-VkPipelineCacheCreateInfo-initialDataSize-00768` — the size is part of what
  makes the data valid to hand back, and nothing in the header states it. The rename is
  atomic within a filesystem, so the next run sees either the whole old file or the whole new
  one.

  **Measuring this needs the driver's own cache out of the way.** Mesa keeps a shader cache
  in `~/.cache`, so the obvious before/after understates the result badly: 3.6 ms of pipeline
  creation cold against 0.6 ms warm, most of the "cold" number already being a hit somewhere
  else. With `MESA_SHADER_CACHE_DISABLE=true`, which is what a driver with no such cache
  looks like, the same five pipelines take **71.6 ms cold and 0.56 ms warm**. Anyone
  re-measuring on a different driver should disable its cache first, or conclude this does
  nothing.

  **The user data directory is new, and lives in `Paths`.** `Paths::UserDataRoot()` /
  `UserData()` resolve `HIKARI_USER_DATA`, else `SDL_GetPrefPath` — which creates the
  directory and gives the right answer per platform. Two rules differ from the content root
  deliberately: an explicit override is *created* rather than required to exist, since nothing
  ships there; and failure is not fatal. Everything written to it is regenerable, so the root
  comes back empty, `UserData()` returns an empty path, and an empty path is exactly what
  tells the cache to stay in memory for the run. The organisation argument to
  `SDL_GetPrefPath` is empty because this project has no organisation name, and passing the
  application name twice would nest it under itself.

  **Timing is logged per pipeline, inside the builders.** That covers `CloudSystem`'s two as
  well as the renderer's three without either caller doing anything, and it puts the number
  on the call the cache actually changes — `vkCreate*Pipelines`, not the shader module or the
  layout around it.

  ImGui reaches the same cache through one new accessor in `VulkanNative.h`. D9's hole grew
  by a function, for the reason it exists: `ImGui_ImplVulkan_InitInfo::PipelineCache` is a raw
  handle and ImGui builds its pipelines without going through anything this module offers.

  `tests/baseline/`'s report is unchanged, field for field.

### R16 — First GPU tests

- **Do:** Add `tests/support/RhiTestFixture.h` (one device per binary, SKIP if no ICD) and
  `ValidationGuard.h`. Tests: device creation reports the required features; buffer upload →
  readback round-trips byte-exactly; image upload → readback matches; **all six cubemap faces
  differ as expected**. Requires a `LABEL` parameter on `engine_test` in `cmake/Testing.cmake`,
  which currently hardcodes `LABELS "unit"`. Label these `gpu` and keep them out of
  `run_unit_tests.sh`.
- **Run the upload round-trips with the ownership transfer forced on and off**, via
  `DeviceDesc::DisabledOptionalExtensions` (R12b). On a device with `VK_KHR_maintenance9` the
  default configuration performs no transfer at all, so without this the release/acquire path —
  the one most hardware in the field takes — never executes in the test suite. Add
  `--vk-force-single-queue` here too, so the third arrangement (no separate copy family) is
  reachable without patching code.
- **Expect a failure:** the cubemap test should fail first time — that is companion-doc bug
  1.1, `CopyBufferToImage` and the layout transition hardcoding `layerCount = 1` (visible in
  the old `Utility.h:245`). Fix it here; the test locks it down permanently.
- **Verify:** `ctest -L gpu` passes locally, and skips with a clear message when no ICD is
  present. `ctest -L unit` unaffected.
- **Size:** L · **Needs:** R11 · **Was:** step 34
- **As built:** done. Nine cases in `rhi_gpu_tests`, all passing, all skipping cleanly on a
  machine with no ICD. Four things differ from the **Do** text.

  **The cubemap bug was already gone, and proving that took more than the passing test.** The
  step predicted companion-doc bug 1.1 would surface here, and it did not: the test passed
  first time under all four configurations. R10 removed it without noting that it had, because
  making a copy region a neutral `BufferTextureCopyRegion` meant `BaseLayer` and `LayerCount`
  became fields the caller has to fill rather than arguments with a default of one, and the
  release barrier's range is built from `desc.ArrayLayers`. A hardcoded layer count had nowhere
  left to hide. A test that passes for a reason nobody checked is worth no more than the
  assumption behind it, so the bug was put back — `.BaseLayer = 0u, .LayerCount = 1u` in the
  region `VulkanUploadContext::Flush` builds — and the test fails on face 0, all six faces
  having landed on top of each other in layer 0. Reverted immediately; the mutation is recorded
  here rather than kept, because the value was in running it once.

  **Headless device creation was the actual work of this step.** `DeviceRequirements::bPresent`
  had existed since R5 with nothing setting it false, and it did not work: a test binary never
  initialises SDL video, so `SDL_Vulkan_GetInstanceExtensions` returns nothing and the old code
  threw on the empty list. Three separate places assumed presentation — the instance extension
  list, `IsPhysicalDeviceSuitable` requiring `VK_KHR_swapchain`, and `CreateLogicalDevice`
  enabling it. All three are now conditional, which is the first time the non-present path has
  been executed rather than merely declared. Stage 6's headless mode inherits a device layer
  that already works; what it still needs is everything above the device. The macOS
  portability-enumeration extension stays unconditional, because it gates driver enumeration
  itself rather than presentation, while `VK_EXT_metal_surface` moved under the flag.

  **Every upload case runs under four device configurations, not two.** The step asked for the
  ownership transfer forced on and off. That is two of the arrangements the upload path can be
  in; the third is a device with no separate copy family, which hands nothing over because
  there is nowhere to hand it. `DeviceDesc::bForceSingleQueue` reaches it, implemented in
  `SelectQueueFamilies` as the *absence* of the dedicated-family search rather than as a second
  assignment afterwards — the existing fallback already resolves an unfilled role to the
  graphics family, so skipping the search reuses it instead of duplicating it, and compute
  correctly stays unresolved on a graphics family that cannot dispatch. The fourth
  configuration is the same transfer with `VK_KHR_maintenance8` also disabled, so the barriers
  fall back to `AllCommands`. On this machine (RX 580, RADV) three of the four are unreachable
  without the levers, and they are the three most hardware in the field is.

  **A Catch2 listener destroys the shared devices; static destruction aborts.** The first
  arrangement left them to the static that owns them, and `vkDestroyDevice` then abort()ed
  inside the validation layer — the layer's own globals are torn down by its static
  destructors, and whichever runs first wins. The crash arrives after every test has passed, so
  it reads as a failure of whatever ran last. `RhiDeviceListener` tears them down from
  `testRunEnded`, while `main()` is still running.

  Two smaller notes. `engine_test` gained `SKIP_RETURN_CODE 4` alongside the `LABEL` the step
  asked for: Catch2 returns 4 from a run in which every case skipped, and without telling CTest
  that, a machine with no ICD reports nine passes rather than nine skips — the exact failure
  mode the `gpu` label exists to prevent. And `GpuReadback.h` is Vulkan-side on purpose: the RHI
  hands out a command list but not a queue, so recording a readback is neutral and submitting it
  is not. A second backend keeps those two functions' signatures and rewrites their bodies.

  **`scripts/precommit.sh` runs them; CI still does not.** That makes precommit a superset of
  CI rather than a mirror of it, which is the one place the two are now allowed to differ: a
  developer has a GPU and CI does not. Because the cases skip rather than fail, precommit stays
  green on a machine with no ICD — so a green precommit is evidence the GPU tests ran only when
  the machine could run them, and `ctest` exits 0 either way. That is tolerable locally and is
  exactly why the same job would be worthless on a runner.

  Verified: `ctest -L gpu` 9/9 pass, and 9/9 skip under `VK_DRIVER_FILES=/nonexistent`.
  `ctest -L unit` is 145/145, up two from R15 for `bForceSingleQueue`. `scripts/precommit.sh`
  green. `tests/baseline/`'s report is unchanged, field for field. The same scene run with
  `--vk-force-single-queue` reports 13 barriers against 14 and 8 barrier calls against 9, which
  is the ownership transfer not happening, with `validationErrors` still 0.

### R17 — Seal the boundary and update the docs

- **Do:** Move anything left in `include/rhi/vulkan/` that is not `VulkanNative.h`,
  `PipelineBuilder.h`, `ComputePipelineBuilder.h` or `DescriptorAllocator.h` down into
  `src/vulkan/`. Extend `scripts/rhi_boundary_check.sh` with the `src/` rule (only the ImGui
  glue may include `rhi/vulkan/`). Walk the checklist in §8 and record the answer for each
  row. Update `CLAUDE.md`: stage table to ✅, repository layout to list `engine/rhi/`, and
  remove the Stage 5 pointer to this document.
- **Verify:** `scripts/precommit.sh` green; boundary check green; headless report identical
  to baseline with `validationErrors: 0`.
- **Size:** S · **Needs:** R16
- **As built:** Three of the four headers this step expected to move down could not, and the
  `src/` rule it expected to write could not be written.

  Only `VulkanAllocator.h` had no user outside the module; it moved to `src/vulkan/` and took
  VMA off every public path with it. `DebugNames.h`, `SwapchainUtil.h` and `CommandListUtil.h`
  are all still included from `src/`, and each for a reason Stage 5 put out of scope on
  purpose: the application names Vulkan objects it created itself, the swapchain lives in
  `App` until `IPresentTarget` (Stage 6), and the cloud bake records a dispatch, which
  `ICommandList` will not express until Stage 8. Moving them would have meant either breaking
  the build or dragging Stage 6 and Stage 8 work into this step.

  So "only the ImGui glue may include `rhi/vulkan/`" is not a rule Stage 5 can end on — and
  there is no ImGui glue file to point it at either, since ImGui is still initialised inside
  `main.cpp`. What replaced it is a **ratchet**: `cmake/RhiBoundaryCheck.cmake` now also
  freezes the transitional area to a named list of seven headers, and allowlists the sixteen
  sites outside `engine/rhi/` that include them, each entry carrying the work that will
  remove it. A new include fails. An entry that stops matching *also* fails, so the list
  cannot outlive the code it excuses — which is what makes it shrink as Stages 6–8 land,
  ending at the ImGui glue this step was aiming for. All three failure modes were tested by
  provoking them.

  The §8 audit found one row weaker than the table claimed: `FenceHandle` is declared, and
  D5's reasoning is recorded on it, but no interface takes one. That is now said at the
  declaration, along with where the API should be shaped from (Stage 6's first real wait)
  rather than guessed at now.

  Also worth knowing before this document is retired: **27 comments across the tree cite its
  decision numbers** as `(plan D2)`, `(plan D6)` and so on. Each comment states its rationale
  in full and the citation only points at the longer argument, so none of them break — but
  they all dangle unless §2 keeps its numbering wherever it is promoted to.

---

## 6. Mapping back to Part IV

| This plan | Part IV | Note |
|---|---|---|
| R1 | — | Pulled forward from §11.1 / §9 (`Core` handles) |
| R2 | — | New: neutral vocabulary + conversion tables + boundary enforcement |
| R3 | 24 | Unchanged |
| R4 | 25 | Two filesystem helpers redirected to `Platform`, not `RHI` |
| R5 | 26 | Plus `IDevice`, `DeviceCaps`, the native escape hatch |
| R6 | 27 | Plus neutral `QueueType` |
| R7 | 28 | Unchanged |
| R8 | — | New: `ICommandList` + neutral barriers (also lands the barrier-batching TODO) |
| R9 | — | New: buffers become handles |
| R10 | — | New: textures/views/samplers become handles |
| R11 | 29 | Now handle-based |
| R12a | 30 | Ownership transfer expressed as intent, not raw barriers |
| R12b | — | New: the transfer becomes conditional on `VK_KHR_maintenance9`, with the explicit path as the fallback |
| R13 | 31 | Explicitly stays Vulkan-only (D7) |
| R14 | 32 | Unchanged |
| R15 | 33 | Neutral interface over an opaque blob |
| R16 | 34 | Plus a `LABEL` parameter for `engine_test` |
| R17 | — | New: boundary audit + doc updates |

---

## 7. Out of scope

Explicitly **not** in Stage 5, to keep the boundary of this plan sharp:

- `IPresentTarget` / `SwapchainTarget` / `OffscreenTarget` — Stage 6 (steps 35–40). The
  swapchain stays in `App`.
- A second RHI implementation. Decided: the null/recording backend waits for Stage 6's
  `OffscreenTarget`. Until then D1 is held by §4's checks, not by a second compiler target.
- Draw-call recording through `ICommandList`, render passes, the frame graph — Stage 8.
- Any neutral descriptor/binding abstraction — deferred to bindless, step 69 (D7).
- Neutral pipeline creation — Stage 8, once the binding model exists (D8).
- Removing `ResourceManager` / `ModelManager` / `MaterialFactory` singletons — Stage 7.
- Moving `Texture`/`Cubemap` to an `Assets` module — Stage 7.
- Shader build changes for a DXIL target (D12).

---

## 8. D3D12 readiness checklist

> **This section is now Stage 7.5's brief.** Its §10 said the checklist "becomes the starting
> backlog for the backend", and that is what happened: the six rows still reading *Partial* or
> *Deferred* — command recording, command pool, CPU/GPU sync, descriptors, per-draw constants,
> pipelines — are, near enough, the step list in `docs/backend_readiness_plan.md`, which uses
> "no row here still reads Partial or Deferred" as half of its definition of done. One row has
> moved since R17 walked it: *present sync* is no longer Deferred, because Stage 6 landed
> `IPresentTarget` with neutral `SemaphoreHandle`s.

R17 walked this. Each row carries the verdict it was walked for — **neutral** (the public API
says nothing Vulkan-specific), **isolated** (Vulkan is still visible, but only from named
sites the boundary check holds), **deferred** (out of Stage 5 by an explicit decision), or
**partial** (the neutral form exists but does not yet cover every use).

| Concept | Vulkan today | D3D12 equivalent | Stage 5 outcome | Verdict |
|---|---|---|---|---|
| Instance / adapter | `VkInstance` + `VkPhysicalDevice` | `IDXGIFactory` + `IDXGIAdapter` | Inside `VulkanDevice` | Isolated |
| Device | `VkDevice` | `ID3D12Device` | `Rhi::IDevice`, neutral | Neutral |
| Queues | family index + `VkQueue` | `ID3D12CommandQueue` (DIRECT/COMPUTE/COPY) | `QueueType`, neutral (D6) | Neutral |
| Allocator | VMA | D3D12MA (same author, similar API) | Inside device | Neutral |
| Buffer | `VkBuffer` + `VmaAllocation` | `ID3D12Resource` | `BufferHandle` (D2) | Neutral |
| Texture | `VkImage` | `ID3D12Resource` | `TextureHandle` (D2) | Neutral |
| Texture view | `VkImageView` object | descriptor in a heap | `TextureViewHandle` — same handle, different backing | Neutral |
| Sampler | `VkSampler` object | sampler descriptor / static sampler | `SamplerHandle` | Neutral |
| Barriers | sync2 triple | Enhanced Barriers triple | Neutral triple + presets (D4) | Neutral |
| Command recording | `VkCommandBuffer` | `ID3D12GraphicsCommandList` | `ICommandList`, copies and barriers only | Partial |
| Command pool | `VkCommandPool` | `ID3D12CommandAllocator` | Inside backend for the RHI's own work | Partial |
| CPU/GPU sync | timeline semaphore | `ID3D12Fence` + value | `FenceHandle` declared; waiting is inside `IUploadContext` | Partial |
| Present sync | **binary** semaphores (VUIDs in D5) | DXGI + fence | Behind `IPresentTarget`, Stage 6 | Deferred |
| Descriptors | sets / layouts / pools | root signature + heaps | **Deferred** (D7) — isolated, not abstracted | Deferred |
| Per-draw constants | push constants | root constants | Already 1:1, but recorded through raw Vulkan | Deferred |
| Pipelines | `VkPipeline` + dynamic rendering | PSO, no render pass objects | Vulkan-side (D8); dynamic rendering is the portable choice | Deferred |
| Pipeline cache | `VkPipelineCache` | cached PSO blob / `ID3D12PipelineLibrary` | Neutral opaque blob (D8) | Neutral |
| Shaders | SPIR-V via Slang | DXIL via Slang | Already portable (D12) | Neutral |
| Clip space | Y-down | Y-up | One site, `DeviceCaps::bFlipClipSpaceY` (D10) | Neutral |
| Formats | `VkFormat` | `DXGI_FORMAT` | Curated `Rhi::Format` + tables (D11) | Neutral |
| Debug names | `VK_EXT_debug_utils` | `ID3D12Object::SetName` | `DebugName` on every `*Desc`, not a setter | Neutral |
| Validation | layers + messenger | debug layer + `ID3D12InfoQueue` | `Rhi::Diagnostics` | Neutral |

Eleven rows landed somewhere other than the plan predicted, or need a caveat a table cell
cannot hold:

- **Instance / adapter is isolated, not hidden.** `VulkanNative.h` hands out the physical
  device, device, surface and graphics queue, because ImGui's Vulkan backend takes all four
  (D9). A D3D12 backend cannot implement those functions — it does not have to, because
  `VulkanNative.h` is a Vulkan header. What it means is that the ImGui integration is
  backend-specific code, and stays so.
- **The allocator became fully private in R17.** `VulkanAllocator.h` was the last VMA type on
  a public path; it now lives in `src/vulkan/`, so VMA is invisible outside the module. This
  is the only row the step actually moved.
- **Command recording is copies and barriers.** `ICommandList` has `Begin`/`End`, `Barrier`,
  and the three copy entry points. Draws, dispatches and render passes are Stage 8, so the
  frame loop and the cloud bake still record through `vk::CommandBuffer` — the cloud bake
  wraps the same buffer in an `ICommandList` for its barriers and uses raw Vulkan for the
  dispatch, which is the shape the rest of the renderer will take as Stage 8 lands.
- **Command pools: the RHI owns its own, the application still owns nine.** The upload
  context allocates from a pool it created; `App` creates a generic pool plus one per pass
  per frame in flight. Those move with the frame loop, not with the RHI.
- **CPU/GPU sync is the weakest row.** `FenceHandle` exists as a type and D5's reasoning is
  recorded on it, but no interface takes one: `IUploadContext::Flush` waits internally on a
  `VkFence` it owns — not even the timeline semaphore D5 settles on — and the frame loop's
  fences and binary semaphores are still `App`'s raw Vulkan. The neutral shape is decided and
  unbuilt. Stage 6 is where a caller
  first needs to wait on something the RHI owns, and is where this should be built rather
  than guessed at now.
- **Per-draw constants are 1:1 but not abstracted.** `vkCmdPushConstants` and
  `SetGraphicsRoot32BitConstants` mean the same thing, so there is no design risk here; there
  is also no neutral call, because it takes a pipeline layout, which is descriptors (D7).
- **Debug names are a `Desc` field, not `SetDebugName(handle, name)`.** Every `*Desc` carries
  a `DebugName`, so the name is set at creation and there is no second call to forget. A
  setter would also have had to work on handles whose backing object does not exist yet on
  D3D12 (a view is a descriptor). Neutral either way; the checklist simply predicted the
  wrong shape.
- **Clip space holds.** `bFlipClipSpaceY` is read at exactly one site in the renderer, plus
  one assertion in the GPU tests. D10 is intact.
- **`Rhi::Diagnostics` is neutral but not virtual.** It is a concrete class with a
  backend-set callback rather than an interface. Nothing about it names Vulkan; a second
  backend feeds the same class from `ID3D12InfoQueue`.
- **Formats are curated, and the table is the enforcement.** `Rhi::Format` is a hand-picked
  enum, and the conversion functions are switches with no `default:`, so adding a format
  without mapping it fails the build — on MSVC too, which is what the `/w14062` in
  `engine/rhi/CMakeLists.txt` buys.
- **The transitional area is seven headers, used from sixteen sites.** Not the four the step
  predicted: pipeline creation (D8), descriptors (D7), the swapchain (Stage 6) and dispatch
  recording (Stage 8) are all explicitly out of Stage 5, and each is a reason a header has to
  stay reachable. `cmake/RhiBoundaryCheck.cmake` lists all sixteen with the work that removes
  each, and fails both on a new one and on an entry that stops matching.

---

## 9. Risks

- **R9 and R10 are the dangerous steps.** They touch every resource creation and use site in
  the codebase. They are deliberately split (buffers, then textures) so a baseline comparison
  runs between them. Do not merge them.
- **R12a, queue-family ownership transfer**, is the classic "compiles, renders correctly on
  one driver, fails intermittently on another" change. Read the spec's synchronization
  chapter, keep synchronization validation on, and treat a clean run as necessary but not
  sufficient.
- **The escape hatch (D9) will try to grow.** Every new `GetNative*` function is a hole in
  the abstraction. Adding one should require a line in this document saying why.
- **The neutral-header check has a known blind spot** (system include paths — see the comment
  block in `cmake/HeaderSelfContainment.cmake`). That is why the grep gate exists too. If
  both are ever bypassed, the boundary is unenforced and this plan quietly stops working.
- **Handle generation is 8 bits.** FIFO slot reuse makes aliasing require 256 full pool
  cycles; if a pool is ever used for high-churn per-frame resources, revisit the split rather
  than assuming it holds.
- **The estimate.** ~3 weeks assumes the baseline comparison stays trustworthy. If a step
  produces an unexplained report diff, stop and find it — a silent behaviour change carried
  forward is worth more than the remaining schedule.

---

## 10. Retirement

R17 is done and `CLAUDE.md`'s stage table reads Stage 5 ✅, and this file was kept anyway —
retiring it is a deliberate decision rather than a step that falls due. **Stage 7.5's B6
proposes itself as the moment**, since that is when the transitional area shrinks to its
permanent residue and §8's checklist has no unfinished rows left to track.

**Promote before deleting.** The step list is disposable; the design decisions are not. Move
into `docs/architecture_plan.md`, a small permanent `docs/rhi.md`, or
`docs/backend_readiness_plan.md` — which is the natural host, since it is retained for the
same reason and its D14–D18 already continue this numbering:

- §2 D0–D13 — the rationale future work has to respect, less D7 and D8, superseded by D14
  and D15.
- §4 — how the boundary is enforced, since the checks stay in the build.
- §8 — the D3D12 readiness checklist, which becomes the starting backlog for the backend.
- R12b's flag convention, into `CLAUDE.md`'s *Conventions*: a command-line option that only one
  backend can honour takes that backend's prefix — `--vk-disable-extension` — in the same
  `--kebab-case` as every other flag. It is the only thing that tells a reader which options
  stop meaning anything under a second backend, and it outlives this document.

**Then delete:** this file, and the Stage 5 pointer in `CLAUDE.md`'s working rules and
roadmap table.
