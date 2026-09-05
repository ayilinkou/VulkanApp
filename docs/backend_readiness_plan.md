# Stage 7.5 — Backend readiness: making a second backend possible

> **Retained document.** Unlike Stage 7's own plan, which was deleted when that stage ended,
> this one outlives its stage.
> Its decisions govern how the RHI's public seam spells recording, binding, pipelines and
> submission, and a D3D12 backend — plus everything written against the seam afterwards — has
> to respect them. See [§9 Retention](#9-retention).

**Created:** 5 September 2026 · **Supersedes:** `rhi_extraction_plan.md` D7 and D8;
`architecture_plan.md` Part IV steps 48–56 in part, and §20's bindless row ·
**Status:** not started — **not yet grilled**, see §0

---

## Table of contents

0. [Before any of this is built](#0-before-any-of-this-is-built)
1. [Purpose and authority](#1-purpose-and-authority)
2. [Design decisions](#2-design-decisions)
3. [The step sequence](#3-the-step-sequence)
4. [What this stage needs from other stages](#4-what-this-stage-needs-from-other-stages)
5. [Out of scope](#5-out-of-scope)
6. [Definition of done](#6-definition-of-done)
7. [Open questions for the grill](#7-open-questions-for-the-grill)
8. [Risks](#8-risks)
9. [Retention](#9-retention)

---

## 0. Before any of this is built

**This plan has not been pressure-tested. Run it through `/grill-me` before B1 starts.**

It was written in one conversation, from reading the code and the existing plans, and it
commits the project to a specific shape for four seams that a second backend then has to live
inside. Two of its decisions (D14, D15) reverse decisions taken deliberately in
`rhi_extraction_plan.md`, and one of them (D14) is in tension with a mitigation the
architecture plan already records. That is exactly the kind of document that reads as settled
because it is written in complete sentences, and is not.

The grill has a concrete agenda: [§7](#7-open-questions-for-the-grill) lists the questions
this document decided quickly, or did not decide at all. Start there rather than at B1.

Stage 7's plan came out of the same process — a `/grill-me` interview before the stage
started, so the work did not re-litigate its own design a step at a time. That is the
precedent this follows.

---

## 1. Purpose and authority

Stage 5 made the RHI's **resource** API backend-neutral: devices, queues, buffers, textures,
views, samplers, barriers, formats, the pipeline cache blob. That is why
`rhi_extraction_plan.md` §8's checklist has so many rows marked *Neutral*.

It did not make the **frame** neutral. `ICommandList` is `Begin`/`End`, `Barrier` and three
copy entry points, and nothing else. Every draw the engine issues is raw Vulkan recorded into
a `vk::CommandBuffer` the application owns (`src/main.cpp:1923-1960`, `:2024-2049`,
`:2091-2109`), against pipelines built by a Vulkan-side builder, bound through descriptor sets
the material layer writes by hand, submitted on a queue the RHI does not hand out.

**This stage closes that gap and nothing else.** When it ends, a D3D12 backend is a matter of
implementing interfaces rather than of designing them.

It is Stage **7.5** rather than a renumbering because a whole stage inserted at 8 would cascade
through every cross-reference in three documents. The `.5` costs nothing; the renumber would
cost a day of chasing references and would leave stale ones behind.

### Authority

**For the duration of this stage, this document is the authority on the RHI's public seam.**
Where it disagrees with `rhi_extraction_plan.md`'s D0–D13, this wins — it was written later,
with the seam built and the gaps visible. Where it disagrees with Part IV, this wins for the
same reason the RHI plan did.

Two specific reversals, both argued in §2: **D14 supersedes D7** (binding) and **D15
supersedes D8** (pipelines). Everything else in D0–D13 stands unchanged, D4's barrier triple,
D10's single clip-space site, D11's curated formats and D13's D3D12-first naming especially.

### The inclusion test

One question decides what belongs here: **does a second backend need this to exist?**

Not "is this good work", not "is this nearby", not "are we already touching the file". The
test keeps the frame graph, the data-oriented rewrite and bindless out on principle rather than
by argument, and it is also the test that says when the stage is finished. Work that fails it
and is worth doing goes to `backlog.md` or stays in the stage that already owns it.

---

## 2. Design decisions

`D` numbers continue the series `rhi_extraction_plan.md` §2 started, rather than restarting at
D1. Both documents govern the same seam, and two live decisions numbered D7 in different files
would be a trap for exactly the reader who most needs to find one.

### D14 — Bindless is deferred until after the D3D12 backend; the binding model is narrow and neutral

**Supersedes D7**, which deferred the binding model on the grounds that bindless (step 70)
would make the question "largely moot". That does not hold, for four reasons.

**The architecture plan already requires the conventional path.** §20's table, row 5, mitigates
bindless portability with "keep a non-bindless fallback path behind a device-capability flag;
the `gpu` test suite runs both". Honour that and the conventional binding model gets built
regardless — so bindless is not a way to avoid designing it, it is a second path layered on
top of it. D7 and that row have pointed in different directions since both were written.

**The groundwork is not in place.** §20's row and the plan's §5 S2-3 both say descriptor
indexing is "already enabled". What `VulkanDevice.cpp:980` enables is one bit,
`descriptorBindingPartiallyBound`, and it is used for the partially-bound material set that
lets an untextured material render (`MaterialFactory.cpp:76`). Bindless additionally needs
`runtimeDescriptorArray` and `shaderSampledImageArrayNonUniformIndexing` at minimum, and
realistically `descriptorBindingVariableDescriptorCount` and the update-after-bind bits. None
are enabled. Step 70 is rated XL for good reason.

**Bindless does not remove the binding model even where it applies.** D3D12 sampler heaps are
separate from the CBV/SRV/UAV heap and cap at 2048 entries, so samplers stay conventional in
practice. Per-frame constants stay conventional too — a root CBV or dynamic UBO beats indexing
camera and light data through a heap. Bindless removes the *material* set; the global set, the
sampler path and the pipeline layout all survive it.

**The convergence is version-gated.** D3D12's `ResourceDescriptorHeap` needs SM6.6, the
Agility SDK and a recent driver. Below that the shape is descriptor tables with volatile
ranges, which is not the same design. D7's "bindless converges the two APIs" holds at the top
of the stack and weakens underneath it.

**So:** build a neutral binding model scoped to the layouts that exist today — a global
uniform set, the material set of three combined image samplers, the composite set, the depth
set, and one fragment push-constant range. That is four layouts and one range, and it maps
1:1 onto a Vulkan descriptor set and a D3D12 descriptor table plus root constants.

**What it costs.** `TextureBinding::COUNT` stays 3 (`src/Texture.h:12`), so no emissive,
occlusion or clearcoat maps until either the cap is raised deliberately or step 70 lands.
`suggested_work.md` §2.6 argued for folding the raise into step 70 rather than doing it alone;
that argument weakens as step 70 moves further out, and the raise gets cheaper once B3 exists —
the enum, the layout, `PBRMaterial`'s writes and both surface shaders, with no descriptor-pool
consequences.

**What it buys.** Step 70 stops being on the critical path to a second backend, and lands
afterwards as a change behind a stable seam that can be verified on *both* backends instead of
guessed at on one. §20's row 5 also stops being live for the duration: with no bindless path
there is no fallback to maintain, one fewer capability flag and one fewer axis in the GPU
suite.

**The risk D7 was right about** is that a narrow model metastasises into the general one as
passes multiply. The mitigation is the ratchet that already works for `Rhi::Format`: scope it
explicitly to the layouts that exist, and grow it by amendment rather than by generalisation.
A `default:`-free switch is the enforcement mechanism there, and something equivalent should
guard this.

### D15 — Pipelines become neutral in this stage

**Supersedes D8's first half.** D8 kept `PipelineBuilder` Vulkan-side because "neutralizing
pipeline creation means neutralizing the binding model (D7), so it waits". D14 neutralises the
binding model, so the reason expires and pipelines follow in the same stage.

D8's second half stands and is reaffirmed by D17: the pipeline *cache* is already a neutral
opaque blob, and `IPipelineCache` does not change shape here.

### D16 — Submission and command-list allocation move behind `IDevice`

`rhi_extraction_plan.md` §8 calls CPU/GPU sync "the weakest row": `FenceHandle` exists as a
type in `Handles.h:30` and **no interface takes one**. `IUploadContext::Flush` waits on a
`VkFence` it owns privately — not even the timeline semaphore D5 settled on — and the frame
loop's fences, binary semaphores and nine command pools are raw Vulkan in `App`. §8 expected
Stage 6 to build this because Stage 6 was where a caller would first wait on something the RHI
owned. Stage 6 shipped without it, so the shape is decided and unbuilt, and no step in Part IV
owns it.

It lands here, as the first step. The RHI allocates and recycles command lists per queue and
takes them back at submit, with waits and signals expressed as `FenceHandle` + `uint64_t`
(D5) plus the present target's `SemaphoreHandle`s (D5's binary-semaphore carve-out for the
swapchain is unaffected).

This is deliberately first, and it is the one seam that does not depend on the other three: a
command list the RHI hands out can be recorded through the native escape hatch while the
recording API is still being built. That is what makes the ordering in §3 possible.

### D17 — Dynamic rendering is the neutral rendering-scope model

The renderer already uses `vk::RenderingInfo` rather than `VkRenderPass`/`VkFramebuffer`
objects, which D8 recorded as a favourable accident: it is much closer to D3D12's
`OMSetRenderTargets`, and `vk::PipelineRenderingCreateInfo`'s colour formats correspond to a
PSO's `RTVFormats`.

The neutral form is an attachment description — view handle, load and store op, clear value —
and `BeginRendering`/`EndRendering` on `ICommandList`. **Render pass objects are not
reintroduced**, here or later, and neither is a subpass concept: D3D12 has no equivalent and
adding one would be inventing a lowest common denominator that neither API wants.

### D18 — The seam lands before the pass conversions, not during them

Part IV's steps 50–54 convert each recorder into a `Pass` class. Those conversions and this
stage touch the same five recorders, and the order matters.

**This stage first.** B5 moves the recorders onto the neutral API in place, and Stage 8's
50–54 then move already-neutral code into `Pass` classes — a structural move with no API
change. Each recorder is touched twice, but for two clearly separated reasons, which is the
plan's own stated philosophy about not superimposing two large refactors.

The alternative — convert to `Pass` classes first — means `Pass::Execute` takes a
`vk::CommandBuffer` and every pass is then edited again when it stops doing so. That is the
same two touches with the second one spread across a class hierarchy instead of concentrated
in one step.

This is the ordering decision most worth challenging in the grill; see §7.

---

## 3. The step sequence

Six steps, `B1`–`B6`. Each ends in a compiling, running application with the baseline
unchanged, per Part IV's rule — that rule is not relaxed here, and matters more than usual
because these steps touch every draw site in the engine.

Every step's verification includes `scripts/precommit.sh` plus a baseline comparison
(`tests/scripts/baseline_test.sh`, counters and decoded pixels). **Synchronization validation
should be on for B1 and B5**, which are the two that move barriers and submits around;
`CLAUDE.md` notes it is off by default, and `backlog.md` already records that the GPU tests
assert a synchronization dependency they cannot currently detect.

### B1 — Command lists and submission become RHI-owned

- **Do:** `IDevice` allocates and recycles `ICommandList`s per `QueueType`, and gains a submit
  entry point taking recorded lists plus waits and signals as `FenceHandle` + value (D16).
  `FenceHandle` becomes a type an interface actually takes. The application's nine command
  pools and its per-frame fences move behind the RHI; the present target's semaphores are
  passed as `SemaphoreHandle` and stay behind `IPresentTarget`.
- **Retires:** three allowlist entries — `CloudSystem.cpp`'s `CommandListUtil.h`, and
  `tests/support/GpuReadback.h`'s and `tests/gpu/rhi/PresentTargetTests.cpp`'s
  `VulkanNative.h`. All three are pool allocation, submission and fence waiting rather than
  recording, which is why they go first and together.
- **Verify:** baseline unchanged, counters unchanged, zero validation errors with
  synchronization validation enabled. `backlog.md`'s "drop the wait semaphore and the suite
  still passes" experiment should now fail as it is supposed to.
- **Size:** L

### B2 — Neutral rendering scope and dynamic state

- **Do:** a neutral attachment description (view handle, load/store op, clear value),
  `BeginRendering`/`EndRendering`, `SetViewport`, `SetScissor` on `ICommandList` (D17).
  Nothing here depends on the binding model, which is why it precedes B3.
- **Verify:** baseline unchanged. The five recorders still bind pipelines and draw through the
  escape hatch at this point; only the scope and dynamic state have moved.
- **Size:** M

### B3 — Neutral bind groups

- **Do:** a bind-group layout description and a bind-group description, a handle type, and
  `SetBindGroup` on `ICommandList` (D14). Scoped to the four layouts that exist. The
  partially-bound behaviour the material set relies on must survive the move —
  `MaterialFactory.cpp:76` is what makes an untextured material render, and losing it silently
  would change what the test cubes look like rather than failing a build.
  `MaterialFactory` and `PBRMaterial` stop writing descriptors directly.
- **Retires:** `DescriptorAllocator.h` and both its allowlist entries; `MaterialFactory.cpp`'s
  and `PBRMaterial.cpp`'s `VulkanNative.h` and `DebugNames.h` entries. Six of seventeen.
- **Verify:** baseline unchanged — in particular the untextured and transparent cube cases from
  step 47's matrix, which are the ones that exercise partial binding.
- **Size:** L

### B4 — Neutral pipelines

- **Do:** neutral graphics and compute pipeline descriptions plus shader modules, consuming
  B3's layouts and feeding the existing neutral `IPipelineCache` unchanged (D15). Formats come
  from `Rhi::Format`, so `GetNativeFormat` leaves the six call sites in `main.cpp` that
  currently translate for the builders.
- **Retires:** `PipelineBuilder.h` and `ComputePipelineBuilder.h`, and their two allowlist
  entries.
- **Verify:** baseline unchanged. The pipeline cache still warms — `startupMs` on a second run
  should not regress, which is the only externally visible sign the cache is still doing its
  job.
- **Size:** L

### B5 — Draw and dispatch recording

- **Do:** `SetPipeline`, `SetBindGroup`, `PushConstants`, vertex and index buffer binding,
  `Draw`/`DrawIndexed`, `Dispatch`. Move the five frame-loop recorders and `CloudSystem`'s
  noise bake onto them. Push constants become a neutral call now that a layout is neutral
  (D14) — the Vulkan and D3D12 forms were always 1:1, only the layout blocked it.
- **Retires:** `CloudSystem.cpp`'s `VulkanNative.h` entry. **Not** `main.cpp`'s — that one is
  ImGui's (D9) and does not go away, it moves to the Vulkan UI backend at step 46.
- **Verify:** baseline unchanged, with synchronization validation on. This is the step where an
  unchanged screenshot is the load-bearing evidence rather than a formality.
- **Size:** L

### B6 — Seal the seam

- **Do:** delete `VulkanNative.h`'s RAII accessors, which exist only for code that builds
  Vulkan objects itself and by now has none. Shrink `rhi/vulkan/` to its permanent residue and
  update `cmake/RhiBoundaryCheck.cmake`'s two lists to match. Remove the remaining `DebugNames.h`
  entries as the objects they name finish moving behind the RHI.
- **Also decide, but do not assume:** B6 is the natural moment to retire
  `rhi_extraction_plan.md` by promoting its D0–D13, §4 and §8 into permanent homes.
  `CLAUDE.md` is explicit that retiring it is a deliberate decision rather than a roadmap step,
  so it is proposed here and taken then.
- **Verify:** `rhi_boundary_check` passes against the reduced lists; `precommit.sh` green.
- **Size:** M

---

## 4. What this stage needs from other stages

The stage does not stand alone. These are planned elsewhere, are load-bearing for a second
backend, and are called out here so the dependency is visible from the document that needs
them rather than only from the one that owns them.

| What | Where | Why it matters here |
|---|---|---|
| **Step 47** — headless scene tests in CI | Stage 7 | **The highest-value prerequisite in the roadmap.** It is the instrument that answers "does the D3D12 backend render the same thing" with a test rather than an eyeball. Every B-step's verification leans on it. |
| **Step 46** — `IUiBackend` + `VulkanUiBackend` | Stage 7 | Demotes D9's ImGui escape hatch from a hole in `main.cpp` to a leaf file that a D3D12 backend replaces with a sibling. Without it, the application names Vulkan permanently and B6 cannot seal anything. |
| **Step 58** — `Mesh*`/`Material*` become handles | Stage 9 | `Drawable::operator<` (`src/Drawable.h:16-22`) falls through to comparing pointers, so batch order tracks heap addresses. Cross-backend pixel comparison is the evidence this whole stage rests on, and that ordering is the known flaw in it. **Recommend pulling forward** to before or during this stage. |
| **Step 48** — `ShaderTypes.h` shared with Slang | Stage 8 | GPU struct layouts are mirrored by hand in C++ and Slang with no `static_assert` linking them. A second backend means a second shader target and doubles the ways that drifts silently. Cheap (M), and worth taking early. |
| **Steps 50–54** — recorders become `Pass` classes | Stage 8 | Follow this stage, not precede it (D18). |
| **Step 70** — bindless | Stage 10 | Explicitly after the backend (D14), not before it. |
| Device info in the run report | `backlog.md` (P2) | Its "blocked by" reads "a neutral device-info accessor on `IDevice`, which is a seam decision". This is the stage that takes seam decisions, so it unblocks here — and two backends make "which device produced this report" a question worth being able to answer. |
| Runtime-selectable validation | `backlog.md` (P2) | `main.cpp:167` gates validation on `NDEBUG`, so a release run reports zero validation errors trivially. With two backends there are two validation surfaces and more reason to assert on both in more configurations. |
| The undetected synchronization dependency | `backlog.md` (P2) | B1 moves submission and B5 moves recording. A test suite that cannot see a missing wait is worth strictly less during exactly those two steps. |

Already done, and listed so nobody re-does them: **D10** gives clip-space handedness one site
behind `DeviceCaps::bFlipClipSpaceY`; **D11**'s curated `Rhi::Format` with `default:`-free
switches already fails the build on an unmapped format; **D12**'s Slang shaders are portable as
written. Emitting DXIL alongside SPIR-V is real work, but it belongs to the backend stage
rather than to this one — there is nothing to neutralise, only a second output to add.

---

## 5. Out of scope

Everything that fails the inclusion test in §1, and specifically:

- **The D3D12 backend itself.** This stage makes it possible; it does not start it.
- **The frame graph (step 56)** and `BarrierBatcher` (55). A second backend needs a neutral
  command list, not a graph. Building the graph against one backend bakes in that backend's
  assumptions; building it after means writing it with two in front of you.
- **All of Stage 9 except step 58.** Arena, `FrameSnapshot`, radix sort, dirty flags, frustum
  culling, ECS, scene serialization — none of it touches the RHI seam, and ECS is on record as
  the largest-blast-radius change in the roadmap.
- **All of Stage 10.** Bindless is deferred by D14; the rest is independent. Note that steps 72
  (mipmaps) and 74 (reverse-Z) deliberately *change* the screenshot and need re-baselining, so
  running them while cross-backend pixel comparison is the primary evidence would be actively
  confusing.
- **Async compute, aliasing and multi-queue in the neutral API.** §20's row 4 constrains the
  frame graph this way already; the same constraint applies to the seam. Add a queue concept
  beyond D6's `QueueType` when a pass needs it.

---

## 6. Definition of done

`cmake/RhiBoundaryCheck.cmake` is the measure, because it is already enforced in CI and
already names the work that removes each entry. Today it holds **7 transitional headers used
from 17 sites**. The steps account for fourteen of those sites — B1 three, B3 six, B4 two,
B5 one, B6 the two remaining `DebugNames.h` entries — leaving **2 headers used from 3 sites**:

| Header | Site | Why it stays |
|---|---|---|
| `VulkanNative.h` | the Vulkan UI backend (after step 46) | ImGui's Vulkan backend takes raw handles. D9 is permanent by design: a D3D12 build gets a sibling file, not an edit. |
| `VulkanNative.h` | `tests/gpu/rhi/DeviceTests.cpp` | The escape hatch is what those cases assert on. |
| `SwapchainUtil.h` | `tests/unit/rhi/SwapchainUtilTests.cpp` | Deliberate, and argued in the check itself: the functions are pure and device-free so they can be unit tested, and `src/vulkan/` is on a PRIVATE include path that a test cannot reach. |

The second measure is `rhi_extraction_plan.md` §8's checklist: **no row still reads *Partial* or
*Deferred***. Six do today — command recording, command pool, CPU/GPU sync, descriptors,
per-draw constants and pipelines — and they are, near enough, this document's step list.

---

## 7. Open questions for the grill

The agenda for §0's interview. Each is something this document either decided fast or dodged.

1. **Is D18's ordering right?** Seam first, then `Pass` classes. The alternative touches each
   recorder twice in the other order. Which second touch is cheaper is not obvious, and the
   answer may differ per pass — `CloudPass` in particular also has to stop holding
   `vk::raii::Device&` and three references into `App` (step 52).
2. **Can B1 really precede B2–B5?** D16 claims a command list the RHI hands out can be recorded
   through the escape hatch while the recording API is unbuilt. That is the assumption the whole
   ordering rests on and it has not been tried.
3. **How is "narrow" enforced for bind groups?** D14 promises the `Rhi::Format` ratchet as the
   model but does not say what the mechanism is. Without one, D7's metastasis risk is real and
   the answer is "discipline", which the architecture rules explicitly do not rely on.
4. **Does step 58 get pulled forward, and into what?** §4 recommends it without saying whether
   it becomes B0, a Stage 7 addendum, or a prerequisite the stage simply waits on.
5. **Are the composite and depth sets bind groups at all?** They are per-frame renderer
   plumbing, and Stage 8's frame graph may want to own them as transient resources. Modelling
   them as bind groups now may be building something 56 replaces.
6. **What is the neutral spelling of a push constant range?** D14 says the concept is 1:1 but a
   range is declared on a layout, which raises how much of the layout's shape becomes public.
7. **Does `IPipelineCache` survive B4 unchanged?** D15 asserts it does. It was designed against
   a Vulkan-side builder, and the caller relationship changes when creation moves behind the
   RHI.
8. **Is six steps the right granularity?** B1, B3, B4 and B5 are all L, and B5 touches every
   draw site in the engine. R9/R10 in the RHI plan were split precisely so a baseline comparison
   ran between them; B5 may deserve the same treatment.
9. **What is the actual cost of the `TextureBinding::COUNT == 3` cap** over the now-longer
   window before step 70, and does anything planned need a fourth map before then?

---

## 8. Risks

- **B3 and B5 touch every draw and every material.** This is the R9/R10 hazard again, and the
  RHI plan's advice applies unchanged: do not merge them, and run a baseline comparison
  between them. Question 8 above asks whether B5 should be split further.
- **The seam is being designed against one backend.** Some of it will be wrong in ways only
  writing the D3D12 backend reveals. Mitigate by checking each neutral description against the
  D3D12 documentation as it is written rather than inferring from the Vulkan side — the same
  rule `CLAUDE.md` already sets for Vulkan semantics, applied to the API that is not in the
  tree yet. Budget for revision rather than assuming the first shape survives.
- **Synchronization mistakes in B1 and B5 will not fail locally.** Plausible-sounding
  synchronization compiles, renders correctly on one driver and fails intermittently on
  another. Synchronization validation on, and a clean run treated as necessary rather than
  sufficient.
- **The pixel comparison this stage relies on is known to be imperfect** until step 58 lands:
  `Drawable::operator<` orders by pointer value, and three or more stacked transparent layers
  can differ in the low bits because WBOIT accumulates additively and float addition is not
  associative. Known cause, not a regression — but it is noise in the primary instrument, which
  is the argument for pulling step 58 forward.
- **Scope creep through adjacency.** Four of the six steps open files that Stage 8 and Stage 9
  also want to change. The inclusion test in §1 is the defence, and it only works if it is
  actually applied when the temptation arrives.

---

## 9. Retention

**This document is kept after the stage ends.** Stage 7's plan was deleted at its stage's
close because it records how to build things that will by then be built.
`rhi_extraction_plan.md` was kept past Stage 5 because its decisions still govern a seam that
outlived it. This one is the second kind: D14–D18 say what the RHI's public API is allowed to
express about recording, binding, pipelines and submission, and a D3D12 backend — and
everything written against the seam afterwards — has to respect them.

What that means in practice:

- The step sequence in §3 becomes history once the stage completes, exactly as R1–R17 did.
  Leave it; it is short, and it explains why the seam has the shape it has.
- §2's decisions stay live and are the reason to open this file.
- §6's definition of done becomes the standing description of what the transitional area is
  *for*, and `cmake/RhiBoundaryCheck.cmake` stays its enforcement.
- If `rhi_extraction_plan.md` is retired at B6, this document is the natural place for D0–D13
  to land, which would put the whole D-series back in one file and make the numbering
  continuity in §2 pay off.
