# Stage 7.5 — Backend readiness: making a second backend possible

> **Retained document.** Unlike Stage 7's own plan, which was deleted when that stage ended,
> this one outlives its stage.
> Its decisions govern how the RHI's public seam spells recording, binding, pipelines and
> submission, and a D3D12 backend — plus everything written against the seam afterwards — has
> to respect them. See [§11 Retention](#11-retention).

**Created:** 5 September 2026 · **Rewritten:** 6 September 2026, after the `/grill-me`
interview §0 demanded · **Supersedes:** `rhi_extraction_plan.md` D7 and D8;
`architecture_plan.md` Part IV steps 48–56 in part, and §20's bindless row ·
**Status:** grilled. Step 1 may start.

---

## Table of contents

0. [What the grill changed](#0-what-the-grill-changed)
1. [Purpose and authority](#1-purpose-and-authority)
2. [Design decisions](#2-design-decisions)
3. [The step sequence](#3-the-step-sequence)
4. [Stage 7.6 — backend prerequisites](#4-stage-76--backend-prerequisites)
5. [Stage 7.7 — the D3D12 backend](#5-stage-77--the-d3d12-backend)
6. [What this stage needs from other stages](#6-what-this-stage-needs-from-other-stages)
7. [Out of scope](#7-out-of-scope)
8. [Definition of done](#8-definition-of-done)
9. [Open investigations](#9-open-investigations)
10. [Risks](#10-risks)
11. [Retention](#11-retention)

---

## 0. What the grill changed

The first draft of this document was written in one sitting, from reading the code and the
existing plans, and it committed the project to a shape for four seams that a second backend
then has to live inside. It said so, and refused to let step B1 start until it had been
pressure-tested. That interview happened on 6 September 2026 and this document is its output.

Two things are worth recording about it, because they are the argument for doing it again.

**The plan was one day old and four of its premises had already moved.** Stage 7 landed
between the writing and the grilling: `src/` ceased to exist and the renderer became
`engine/engine/src/Engine.cpp`, so every `src/main.cpp:NNNN` reference was stale. Two
prerequisites the plan listed as pending — steps 46 and 47 — had shipped. The transitional
allowlist had grown from 17 sites to 18, because the split gave the UI backend an entry of its
own. And the plan's claim that synchronization validation was off was simply wrong:
`validate_sync` is hardcoded `VK_TRUE` in `VulkanDevice::CreateInstance`, so every Debug run
and every GPU test already has it. That last one had a step built on top of it, which the
grill deleted.

**The interview found design gaps, not just stale facts.** The frame records command buffers
on several threads at once, which B1 had not accounted for. `TextureBinding::COUNT` turned out
to cap a feature that does not exist rather than one being held back. And the combined image
samplers the material set uses cannot be expressed on D3D12 at all — the single most
consequential thing the grill turned up, and one nothing in the original document was looking
for.

The rule this produced now lives in `CLAUDE.md`: **grill before every stage**, and re-grill an
already-grilled plan against four mechanical checks before starting it.

---

## 1. Purpose and authority

Stage 5 made the RHI's **resource** API backend-neutral: devices, queues, buffers, textures,
views, samplers, barriers, formats, the pipeline cache blob. That is why
`rhi_extraction_plan.md` §8's checklist has so many rows marked *Neutral*.

It did not make the **frame** neutral. `ICommandList` is `Begin`/`End`, `Barrier` and three
copy entry points, and nothing else. Every draw the engine issues is raw Vulkan recorded into
a `vk::CommandBuffer` the application owns, against pipelines built by a Vulkan-side builder,
bound through descriptor sets the material layer writes by hand, submitted on a queue the RHI
does not hand out.

**This stage closes that gap and nothing else.** When it ends, a D3D12 backend is a matter of
implementing interfaces rather than of designing them.

It is Stage **7.5** rather than a renumbering because a whole stage inserted at 8 would cascade
through every cross-reference in three documents. The `.5` costs nothing; the renumber would
cost a day of chasing references and would leave stale ones behind. Stages 7.6 and 7.7 follow
the same precedent, which is why the roadmap now reads 7, 7.5, 7.6, 7.7, 8. That looks faintly
ridiculous and is still cheaper than the alternative.

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

The test earned its keep during the grill: it is what sent the emissive texture map to the
backlog, and what kept step 58 in Stage 9.

---

## 2. Design decisions

`D` numbers continue the series `rhi_extraction_plan.md` §2 started, rather than restarting at
D1. Both documents govern the same seam, and two live decisions numbered D7 in different files
would be a trap for exactly the reader who most needs to find one.

D14–D18 come from the original draft. D19–D26 come from the grill.

### D14 — Bindless is deferred until after the D3D12 backend; the binding model is narrow and neutral

**Supersedes D7**, which deferred the binding model on the grounds that bindless (step 70)
would make the question "largely moot". That does not hold, for four reasons.

**The architecture plan already requires the conventional path.** §20's table, row 5, mitigates
bindless portability with "keep a non-bindless fallback path behind a device-capability flag;
the `gpu` test suite runs both". Honour that and the conventional binding model gets built
regardless — so bindless is not a way to avoid designing it, it is a second path layered on
top of it. D7 and that row have pointed in different directions since both were written.

**The groundwork is not in place.** §20's row and the plan's §5 S2-3 both say descriptor
indexing is "already enabled". What `VulkanDevice.cpp` enables is one bit,
`descriptorBindingPartiallyBound`, and it is used for the partially-bound material set that
lets an untextured material render. Bindless additionally needs `runtimeDescriptorArray` and
`shaderSampledImageArrayNonUniformIndexing` at minimum, and realistically
`descriptorBindingVariableDescriptorCount` and the update-after-bind bits. None are enabled.
Step 70 is rated XL for good reason.

**Bindless does not remove the binding model even where it applies.** D3D12 sampler heaps are
separate from the CBV/SRV/UAV heap and cap at 2048 entries, so samplers stay conventional in
practice. Per-frame constants stay conventional too — a root CBV or dynamic UBO beats indexing
camera and light data through a heap. Bindless removes the *material* set; the global set, the
sampler path and the pipeline layout all survive it.

**The convergence is version-gated.** D3D12's `ResourceDescriptorHeap` needs SM6.6, the
Agility SDK and a recent driver. Below that the shape is descriptor tables with volatile
ranges, which is not the same design. D7's "bindless converges the two APIs" holds at the top
of the stack and weakens underneath it.

**So:** build a neutral binding model scoped to the layouts that exist today, plus the
push-constant ranges those layouts carry. That maps 1:1 onto a Vulkan descriptor set and a
D3D12 descriptor table plus root constants.

**Correction from the grill.** The first draft said "four layouts and one range". There are
**four ranges, not one**, and they span two shader stages: fragment `MaterialData` on the
opaque and transparent layouts, compute `CloudPushConstants` on the cloud dispatch, and compute
`BakeConstants` on the noise bake. The composite layout has none. The neutral model must
therefore express a range's stage, not merely its size. See D23.

**Second correction, found while building step 4: there are six layouts, not four.** The
original count listed the global, material, composite and depth sets and stopped there.
`CloudSystem` owns two more of its own, and neither appeared in any step's scope:

| Layout | Bindings | Where |
|---|---|---|
| Global | uniform buffer | done, step 4 |
| Composite | 4 textures + 1 sampler | done, step 4 |
| Depth | 1 texture, pixel *and* compute | done, step 4 |
| Material | 3 textures, partially bound | step 5 |
| **Cloud dispatch** | **storage image + combined image sampler** | **step 11** |
| **Cloud noise bake** | **storage image** | **step 11** |

Three things follow, and the first is the one that matters.

**`UnorderedAccessTexture` is not optional.** Both cloud layouts bind storage images —
`RWTexture2D` and `RWTexture3D` in the shaders — and `BindingType` has no value for them. It
cannot be added when convenient; it is a prerequisite of those sets becoming neutral, which is
step 11. When it lands the pair reads the way `TextureLayout`'s `ShaderResource` and
`UnorderedAccess` already do.

**The cloud dispatch set holds a combined image sampler**, which D22 forbids outright. So step
11 carries a second sampler split, in `clouds.comp.slang`, exactly as step 4 carried the
composite one.

**The narrowness argument is unaffected, but its arithmetic was wrong.** Six layouts across a
whole renderer is still narrow, and the ratchet still holds — but a count used as evidence for
narrowness has to be the real count. The pinned inventory ends at six, not four, and
`BindGroupLayoutInventoryTests` grows to match at steps 5 and 11 rather than only at step 5.

**What it costs.** `TextureBinding::COUNT` stays 3, so no emissive, occlusion or clearcoat maps
until either the cap is raised deliberately or step 70 lands. The grill checked what that
actually costs and the answer is less than `suggested_work.md` §2.6 assumed: emissive and
occlusion are not parsed by the loader, are absent from `MaterialData`, and are referenced by
no shader. Nothing is being dropped on the floor, so raising the cap is *adding a feature*
rather than unblocking one — which fails the inclusion test outright. It is now its own backlog
row, not a rider on step 70.

**What it buys.** Step 70 stops being on the critical path to a second backend, and lands
afterwards as a change behind a stable seam that can be verified on *both* backends instead of
guessed at on one. §20's row 5 also stops being live for the duration: with no bindless path
there is no fallback to maintain, one fewer capability flag and one fewer axis in the GPU
suite.

**The risk D7 was right about** is that a narrow model metastasises into the general one as
passes multiply. D21 is the mitigation, and it is a stronger one than the first draft promised.

### D15 — Pipelines become neutral in this stage

**Supersedes D8's first half.** D8 kept `PipelineBuilder` Vulkan-side because "neutralizing
pipeline creation means neutralizing the binding model (D7), so it waits". D14 neutralises the
binding model, so the reason expires and pipelines follow in the same stage.

D8's second half stands and is reaffirmed by D17: the pipeline *cache* is already a neutral
opaque blob, and `IPipelineCache` does not change shape here. The grill confirmed this against
the header — `PipelineCacheDesc{Path, DebugName}` and `Save()`, whose own documentation says
the whole of what a caller does is create it, hand it to pipeline creation and save it. Only
what it is handed to changes.

One consequence does follow, from D25: a single machine can now run both backends, so the
cache's default path must be backend-distinguished or the two will overwrite each other's blob
on every run.

### D16 — Submission and command-list allocation move behind `IDevice`

`rhi_extraction_plan.md` §8 calls CPU/GPU sync "the weakest row": `FenceHandle` exists as a
type in `Handles.h` and **no interface takes one**. `IUploadContext::Flush` waits on a
`VkFence` it owns privately — not even the timeline semaphore D5 settled on — and the frame
loop's fences, binary semaphores and command pools are raw Vulkan in the engine. §8 expected
Stage 6 to build this because Stage 6 was where a caller would first wait on something the RHI
owned. Stage 6 shipped without it, so the shape is decided and unbuilt, and no step in Part IV
owns it.

It lands here, first. The RHI allocates and recycles command lists per queue and takes them
back at submit, with waits and signals expressed as `FenceHandle` + `uint64_t` (D5) plus the
present target's `SemaphoreHandle`s (D5's binary-semaphore carve-out for the swapchain is
unaffected).

This is deliberately first, and it is the one seam that does not depend on the other three: a
command list the RHI hands out can be recorded through the native escape hatch while the
recording API is still being built. The grill verified the mechanism rather than assuming it —
`WrapCommandList(IDevice&, vk::CommandBuffer)` and `GetNative(ICommandList&)` already exist and
already convert in both directions, so the intermediate state is one the codebase can express
today.

D19 settles the part this decision left open: *what* the RHI hands out.

### D17 — Dynamic rendering is the neutral rendering-scope model

The renderer already uses `vk::RenderingInfo` rather than `VkRenderPass`/`VkFramebuffer`
objects, which D8 recorded as a favourable accident: it is much closer to D3D12's
`OMSetRenderTargets`, and `vk::PipelineRenderingCreateInfo`'s colour formats correspond to a
PSO's `RTVFormats`.

The neutral form is an attachment description — view handle, load and store op, clear value —
and `BeginRendering`/`EndRendering` on `ICommandList`. **Render pass objects are not
reintroduced**, here or later, and neither is a subpass concept: D3D12 has no equivalent and
adding one would be inventing a lowest common denominator that neither API wants.

That last clause is the general principle behind D22 as well.

### D18 — The seam lands before the pass conversions, not during them

Part IV's steps 50–54 convert each recorder into a `Pass` class. Those conversions and this
stage touch the same recorders, and the order matters.

**This stage first.** Steps 8–11 move the recorders onto the neutral API in place, and Stage
8's 50–54 then move already-neutral code into `Pass` classes — a structural move with no API
change. Each recorder is touched twice, but for two clearly separated reasons, which is the
plan's own stated philosophy about not superimposing two large refactors.

The alternative — convert to `Pass` classes first — means `Pass::Execute` takes a
`vk::CommandBuffer` and every pass is then edited again when it stops doing so. That is the
same two touches with the second one spread across a class hierarchy instead of concentrated
in one step, and it means designing the `Pass` abstraction against Vulkan, which is exactly
what §7 refuses to do for the frame graph.

The grill sharpened the evidence for this. `CloudSystem`'s create info takes
`vk::raii::DescriptorSetLayout&`, `CommandPool&` and `Queue&` by reference, and `RecordDispatch`
takes a `vk::raii::CommandBuffer&` and two `vk::raii::DescriptorSet&`. Converting it to a
`CloudPass` first would mean writing a `Pass` whose *constructor signature* names Vulkan, and
rewriting that signature within the same stage.

The cost, accepted: the recorder work lands inside a 2,476-line `Engine.cpp`, which stays large
until Stage 8 dismantles it into a shape the frame graph decides. That is the right thing to
accept — dismantling it earlier means guessing at that shape. It is also why steps 8–11 are one
recorder each rather than one step (D-series aside, see §3).

### D19 — Command allocators are caller-owned, one per frame per recorder

D16 says command lists move behind `IDevice`. It does not say what a caller gets, and the
answer matters more than it looks, because **the frame records on several threads at once**:
`RecordOpaqueCommandBuffer` and `RecordTransparentCommandBuffer` are submitted to the job
system while the main thread records clouds, composite and ImGui.

`vkResetCommandPool` and `vkAllocateCommandBuffers` both require external synchronization on
the pool. `ID3D12CommandAllocator` carries the identical rule. Two threads recording
concurrently is therefore not an optimisation to be added later; it is what the frame already
does, and any neutral shape has to survive it.

**The RHI hands out a caller-owned allocator.** `ICommandAllocator` — D3D12's term, per D13 —
created per frame per recorder, reset as a unit, handing out `ICommandList`s. Queue affinity
sits on the allocator rather than the list, because both APIs put it there: a `VkCommandPool`
carries a queue family index and an `ID3D12CommandAllocator` carries a
`D3D12_COMMAND_LIST_TYPE`.

This mirrors what the engine already has. `CreateCommandPools` builds seven pools per frame —
draw-layout, opaque, cloud, transparent, composite, ImGui, final-layout — plus one generic, and
they are keyed by *recorder*, which is what makes the parallel recording safe: two threads
never touch one pool. So step 1 is a move, not a redesign.

**Rejected: device-managed thread-local pools.** The call site would be smaller
(`AcquireCommandList(QueueType)` and nothing else), and the costs are real. Pool count becomes
workers × frames rather than recorders × frames, because `SharedQueueJobSystem` does not pin a
recorder to a thread. Worse, resetting frame N's pools then means touching every worker's pool
from a thread that does not own it, under a rule that says do not — which resolves either into
a lock or into per-thread resets scheduled onto each worker, putting a job-system dependency
inside the RHI.

The deciding argument is that both APIs make the same external-synchronization promise about
the same object, so the neutral layer can express it *honestly* rather than approximate it.
Hiding a thread-affinity rule behind a convenient call site produces intermittent corruption on
someone else's driver, which is the category `CLAUDE.md` already refuses to gamble on.

**Reset is the dangerous operation**, and caller ownership is what makes it visible: reset
invalidates every list the allocator has produced, and it is the one place a use-after-reset
bug can live. A caller that names the object can see the moment.

### D20 — Bind groups are immutable

A bind group is created from a complete description and cannot be written into afterwards.
Changing what it points at means creating a new one.

Every current user already has a natural creation point. A material set is built once when the
material loads. The global set is one per frame in flight and never changes identity
afterwards — only the buffer's *contents* change, which immutability does not touch. The
composite and depth sets are rebuilt on resize, because their targets are.

**The argument is that the in-flight hazard stops being a rule and becomes inexpressible.**
Writing a descriptor the GPU is still reading is a hazard the mutable form leaves entirely to
the caller, unmentioned by any signature, caught by the validation layers only under the right
settings, and reproducing on someone else's driver rather than yours. The immutable form does
not have a call that can do it.

It also maps more directly onto D3D12, where a descriptor table is a baked range in a
shader-visible heap.

**This settles what the first draft left as an open question**: the composite and depth sets
*are* bind groups. The worry was that Stage 8's frame graph may want to own them as transient
resources and that modelling them now builds something step 56 replaces. It does not. A frame
graph owns when views are created and how their memory aliases; something still has to bind
them, and binding is what a bind group is. Step 56 changes who calls `CreateBindGroup`, not
whether the concept was worth having. Leaving them out, meanwhile, would strand the composite
recorder at step 8 and keep `GetImageView` alive in `VulkanNative.h`, so step 12 could not seal
anything.

**No deferred destruction is built here, deliberately.** Immutability plus per-frame recreation
would require a retirement queue keyed on the fence value that says a frame's work is done, and
the RHI has none: `IDevice::Destroy(handle)` is immediate. Nothing needs one yet. All four sets
are stable, and the only mutation point — resize — already stalls, so nothing is in flight when
they are replaced.

The grill checked what would force per-frame recreation and the honest answer is: nothing in
the roadmap. A frame graph forces it only if the physical resource behind a logical one changes
between frames, and in a stable graph it does not. Ping-pong effects are served by two
alternating immutable groups, exactly like frames in flight. What *would* force it is an editor
displaying arbitrary textures — an asset browser, a material inspector, a render-target debug
view — or texture streaming, where residency changes under a material. Note that bindless
removes that second pressure rather than adding to it.

**The trigger to watch for is not "bind groups changed", it is "a stall we can no longer
afford".** Deferred destruction is the mechanism that replaces stalls, and this engine's
universal answer to "the GPU might still be using it" is currently to wait — resize does it,
`GrowInstanceBuffers` does it. When a stall becomes visible in `frameMs`, build the retirement
queue; bind groups will be one more thing that uses it.

When step 56 does want per-frame bind groups, the answer is a **transient** variant allocated
from an arena reset wholesale at frame boundaries — no individual destroy, so no retirement
queue and no heap fragmentation. Immutability does not preclude it. That is a written
prerequisite of step 56 rather than a surprise found halfway through it.

### D21 — The binding vocabulary is narrow, and the inventory is pinned by a test

D14 promised "the `Rhi::Format` ratchet" as the mechanism for keeping the binding model narrow.
The grill established that the promise does not transfer, and this decision replaces it.

The `Format` ratchet works because a format is a **leaf value**: a curated enum plus a
`default:`-free switch in the backend means adding one without mapping it fails the build. That
is a *completeness* mechanism. It does not stop the enum growing; it stops it growing unmapped.
A binding model is a structure, not a leaf, so the same trick catches "you added a binding type
and did not map it on D3D12" and says nothing whatever about "the model grew into a
general-purpose descriptor abstraction" — which is the risk D7 actually named.

**Two mechanisms, therefore.**

**A curated `BindingType` with `default:`-free switches in each backend.** This is the
completeness half, and it is the `Format` ratchet applied where it genuinely fits. Whatever the
vocabulary becomes, both backends implement all of it or the build fails.

**A pinned layout inventory, as a unit test.** The test asserts that the engine creates exactly
four bind group layouts, with exactly these shapes. A fifth layout, or a fourth binding on the
material set, fails it — and `CLAUDE.md` already forbids changing an existing test's
expectation without asking first. That is what makes it work: it converts "the model grew" from
something noticed in review, or not, into something that **cannot land without a
conversation**. It is the same governance the transitional allowlist uses, expressed as a test
because layouts are built at runtime and a static check would have to parse C++ to see them.

**The caveat, recorded so it is not rediscovered as a complaint.** A test that pins an inventory
gets edited whenever the inventory legitimately changes, and if that happens often it becomes
noise people edit reflexively. Four layouts across a whole renderer, with step 70 deferred,
should not move often. If the count starts moving every other stage, that is the signal the
test has outlived its purpose — not that the rule needs relaxing.

### D22 — Samplers are separate from textures; combined image samplers are not in the vocabulary

**D3D12 cannot express a combined image sampler.** Samplers live in their own descriptor heap
type, `D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER`, separate from the CBV/SRV/UAV heap and capped at
2048 shader-visible entries. A single descriptor holding both a texture and a sampler is not a
naming difference; it does not exist.

So a neutral `CombinedTextureSampler` binding type would be a concept one backend has to
**decompose** rather than map — the D3D12 backend splitting one neutral binding into two root
signature entries across two heaps, with the layout description needing interpretation instead
of translation. That is the same trap D17 refuses when it rules out reintroducing render pass
objects.

The engine is already half-way there, which makes this smaller than it sounds: the composite
set is three `eSampledImage` plus one `eCombinedImageSampler`, the depth set is a plain sampled
image, and only the material set is fully combined. Step 5 finishes a split that is already
under way.

**It is also the better model on its own terms.** With separate samplers, a small palette bound
once — linear-wrap, point-clamp, aniso — lets a shader choose its filtering by naming one, so
changing how a material's normal map is filtered is a shader edit rather than a descriptor
rewrite in C++. That is not available at all with combined, where the sampler is baked into the
descriptor write. It survives bindless too: D14 already notes samplers stay conventional under
D3D12's heap limits, so a sampler palette is where this ends up regardless.

This is scoped into step 5 rather than deferred, because doing it later means editing the
pinned inventory of D21 — which requires a conversation — and touching the same four shaders,
`MaterialFactory` and `PBRMaterial` a second time. Same sampler state, same results, so the
baseline is expected pixel-identical, which gives the change a real check.

### D23 — Pipeline layouts are explicit

A `PipelineLayoutHandle` is created from an ordered list of bind group layouts plus
push-constant ranges. Pipeline descriptions reference one; `SetBindGroup` and `PushConstants`
take one. It is 1:1 with `VkPipelineLayout` and `ID3D12RootSignature`.

**Rejected: an implicit layout** derived from the pipeline description, with no public handle.
That is fewer public concepts, which is what D14's "narrow" asks for, and it does not actually
remove the object — both APIs have a real one — it moves it somewhere the caller cannot see and
adds a hash to find it again. The backend would then either create a layout per pipeline, which
is wasteful on D3D12 where a root signature is heavyweight and meant to be shared across many
PSOs, or deduplicate by hashing the description, which introduces a cache whose key must be
exactly right in the layer whose entire job is to not be subtly wrong on one backend.

The deciding argument is that root-signature identity is what determines whether bound
descriptor tables survive a pipeline change. That is a performance property worth being able to
reason about on both backends, and it is exactly what the opaque and transparent recorders lean
on today when they bind the global set once per recorder and the material set per batch.

It costs less against D14's narrowness goal than it appears: one new handle type and four
objects created at startup, in exchange for a pipeline description that shrinks to a layout
reference plus shader modules plus state.

**A range carries its stage.** See D14's correction: there are four ranges across two shader
stages, not one fragment range.

**Rejected, and worth recording because it looks like an obvious inclusion: dynamic offsets.**
Vulkan's dynamic uniform buffer is a descriptor *type* declared in the layout, with an offset
supplied at bind time. D3D12 has no dynamic offset for a descriptor table entry; its analogue
is a root CBV, which takes a GPU virtual address at bind time but is a different kind of root
parameter from a table. So the neutral concept is not "an offset on the bind call" — it is a
distinction in the layout between a binding that lives in a table and one that is inline in the
root signature, and only once that exists does a bind-time offset mean anything.

That is a generalisation, and D21's ratchet exists to keep generalisations out until something
needs them. Nothing does: the global set is a plain `eUniformBuffer` with one set per frame in
flight, and there is no site where a dynamic offset would save anything. If a case appears, the
cost of having waited is one enum value in the layout description plus an argument on the bind
call, added at a point where its purpose is known.

### D24 — Shader bytes come from the caller; the packaging difference is absorbed by the build

Today every caller passes a **file path** — `.Shaders(m_Paths.Shader("opaque.spv").string())` —
and the builder reads it. That is Vulkan-shaped twice over. The extension obviously is. Less
obviously, so is the packaging: `opaque.spv` is one module holding both `vertMain` and
`fragMain`, because `-fvk-use-entrypoint-name` lets Vulkan select an entry point out of a
module, whereas a D3D12 PSO takes separate bytecode per stage.

**The engine loads the bytes; the device reports what format it eats; the RHI never touches a
file.** `CreateShaderModule` takes bytes, `DeviceCaps` reports the format or extension, and
`Paths` — which the engine already owns — does the resolving.

**And the build emits one blob per stage for both targets**, so the runtime mapping is uniform:
name, stage, extension. `opaque.vert.spv` and `opaque.frag.spv` alongside `opaque.vert.dxil`
and `opaque.frag.dxil`. Without this, the packaging difference lands in the engine as a
per-backend branch — one file on Vulkan, two on D3D12 — which is precisely the `#ifdef` the
seam exists to prevent.

The cost is honest and worth stating: the Vulkan side of the shader build changes before D3D12
exists, splitting `opaque.spv` into two blobs and re-running the baseline to prove nothing
moved, for the benefit of a backend not yet written. It is worth it, because the alternative is
discovering the packaging mismatch while writing a PSO, and because it makes the second target
a copy of the first rather than a special case.

**Rejected: the RHI resolves shader names itself.** The engine would hold zero backend
knowledge, which is attractive, and the price is that the layer whose job is to talk to a GPU
acquires filesystem responsibilities and needs a shader root handed to it. That is a widening
you do not get back.

**Rejected for now: a `ShaderLibrary` layer** owning the name-to-bytes mapping, caching modules
and owning their lifetime. Roughly 120–150 lines, replacing about eight three-line sequences —
so barely shorter, and only meaningfully cleaner once it also owns variants, permutations or
hot reload, none of which exist. It fails the inclusion test. It is a backlog row whose trigger
is the second shader responsibility, and the engine-side helper that emerges from D24 is about
sixty lines short of being it, so promoting it later is a rename and a move.

### D25 — The backend is selected at run time, and Vulkan is always the default

The project is cross-platform, and a Linux build cannot contain a D3D12 backend at all. So
Windows links both backends and Linux links Vulkan only.

**Selection is at run time, not compile time.** `--backend vulkan|d3d12` feeds `RunSpec`. A
value the build does not contain is a **hard error** naming what was asked for and listing what
is available — the same policy `backlog.md` argues for `--present-mode`, for the same reason: a
run that quietly measured something else is worse than a run that refused.

The flag is `--backend` rather than `--rhi` because "backend" is the word this project already
uses everywhere for exactly this concept — "backend-neutral", "the backend lives in
`engine/rhi/src/vulkan/`", and a document titled *backend readiness*. It is also more accurate:
there is one RHI, and what is being selected is which implementation it uses. The word is
mildly overloaded against `IUiBackend`, which does not bite, because the UI backend is not
user-selectable.

**What run-time selection buys is the evidence model.** Under compile-time selection,
comparing two backends means two builds, two CI jobs and a comparison spanning build artefacts
— an orchestration problem before it is a rendering one. Under run-time selection it is one
build, one scene, two runs, diff, and a scene test can assert cross-backend counter equality in
a single CI job. An assertion that requires two builds stitched together gets written once and
then skipped; one that is a second `ctest` case runs on every push.

**The cost, accepted:** a Windows build carries both VMA and D3D12MA and ships both `.spv` and
`.dxil` beside the executable, and the RHI's creation seam gains a neutral `Backend` enum plus
an availability query — a small, permanent widening of the public API.

**Vulkan is the default on every platform, permanently.** Not "until D3D12 reaches parity" —
always. D3D12 is reached only by asking for it. The benefit is that a bug report, a baseline
capture and a run report mean the same thing whoever produced them and wherever.

**The consequence has to be designed for rather than hoped away:** if D3D12 is never a default,
**CI is the only thing that will ever run it routinely.** That promotes Stage 7.6's Windows GPU
job from useful coverage to the backend's sole regular exercise, and it is the reason that job
is a prerequisite of the backend rather than a follow-up to it.

### D26 — Cross-backend evidence: exact counters, tolerant pixels

Every comparison in the project today is exact — counters must match the committed baseline
exactly, and the pixel diff must produce an empty bounding box. That works because there is one
backend on one rasterizer, and it cannot survive a second.

lavapipe and WARP are different rasterizers with different filtering, different rounding in the
raster and blend paths, and different float contraction in their shader back ends. Two *real*
GPUs on the same API disagree in the low bits. So the instrument the first draft called "the
highest-value prerequisite in the roadmap" did not, as built, answer the question it was said
to answer.

**The two signals are split by what each can honestly promise.**

**Counters must match exactly, across backends.** `drawCalls`, `batches`, `instances`,
`barriers`, `barrierCalls`, `validationErrors`, `uploadSubmissions` are statements about what
the renderer *decided*, not about what the rasterizer produced. Two backends disagreeing about
a draw call count is a bug in one of them, always.

**Pixels are compared with tolerance**: a per-channel delta with two caps — no pixel may differ
by more than N per channel, and at most M% of pixels may differ at all. Two numbers, each
catching what the other misses. The ceiling catches one catastrophically wrong pixel; the
fraction catches an image that has drifted slightly everywhere. A failure can name the worst
pixel and its coordinates.

*Rejected: MSE or PSNR*, because averaging over a 1920×1080 frame lets a single blazing-wrong
pixel vanish into the mean, and that is precisely the backend bug being hunted. *Rejected:
SSIM or a perceptual metric*, which is more robust to differences nobody would see and gives a
single score that is hard to act on when it fails — "0.987 against a threshold of 0.99" does
not say where or what.

**Two things keep the tolerance honest.**

The thresholds are **committed constants**, so changing one is changing an expected test result
and is gated by the rule in `CLAUDE.md` that already requires asking first. That is what stops
a threshold becoming a knob nudged upward whenever CI goes red.

And the comparison **always reports the measured delta**, not just pass or fail. "Worst channel
delta 3, 0.4% of pixels differ" against limits of 8 and 2% shows drift while it is still
headroom. A bare PASS hides the approach to the cliff, and the day it fails nobody can tell
whether it fell or walked there over six months.

**One implementation, two settings.** Within a backend the tolerance is zero and the check stays
exactly as strict as it is today; across backends it is the configured limits. There is no
second tool to keep in step.

---

## 3. The step sequence

**Twelve steps.** The first draft had six, four of them L. The grill split them for two
reasons: the work is being followed step by step deliberately, and a step that ends where a
baseline comparison means something is a step whose green result is unambiguous.

Steps are numbered flat — 1 to 12, no letters.

Each ends in a compiling, running application with the baseline unchanged, per Part IV's rule.
That rule is not relaxed here, and it matters more than usual because these steps touch every
draw site in the engine.

Every step's verification includes `scripts/precommit.sh` plus a baseline comparison
(`tests/scripts/baseline_test.sh`, counters and decoded pixels). Synchronization validation is
already on — `validate_sync` is hardcoded `VK_TRUE` whenever validation is enabled — so it
applies to every step here without anything being switched on first.

### 1 — Command allocators and command lists

- **Do:** `ICommandAllocator` per queue type, caller-owned, one per frame per recorder, reset
  as a unit and handing out `ICommandList`s (D19). The engine's seven-per-frame pools move
  behind the RHI as allocators. The engine still submits its own lists on its own queue,
  recording through the escape hatch.
- **Verify:** baseline unchanged, counters unchanged.
- **Size:** M
- **Done.** Amended while building: the **generic pool stays raw**, against this step's
  original wording. `CloudSystem`'s noise bake reaches it through `CommandListUtil`, which
  begins, submits and waits on a one-shot buffer — so converting it needs submission behind the
  RHI *and* dispatch recording, which is steps 2 and 11. Forcing it here would have meant
  handing a raw pool back out of an allocator, widening the escape hatch to narrow it later.
  One thing came free in the other direction: `CloudSystem::RecordDispatch` now takes an
  `ICommandList&` rather than a `vk::raii::CommandBuffer&`, since the caller owns the allocator
  and must begin and end the list anyway.

### 2 — Submission and fences

- **Do:** `IDevice` gains a submit entry point taking recorded lists plus waits and signals as
  `FenceHandle` + value (D5, D16). `FenceHandle` becomes a type an interface actually takes.
  The per-frame fences move behind the RHI; the present target's semaphores are passed as
  `SemaphoreHandle` and stay behind `IPresentTarget`.
- **Retires:** `tests/support/GpuReadback.h`'s `VulkanNative.h` entry and
  `tests/gpu/rhi/ValidationCoverageTests.cpp`'s — both were submission and fence waiting, and
  both are now fully neutral. 19 sites down to 17.
- **Not** `tests/gpu/rhi/PresentTargetTests.cpp`'s, against this step's first estimate: its
  remaining uses are a raw render pass for the clears and a `VkImageView` for the attachment,
  which are step 3's to remove rather than this step's.
- **Verify:** baseline unchanged; zero validation errors, and those errors now mean something
  across submissions — see §9. This is the step where a clean synchronization validation run is
  load-bearing rather than decorative, because it is the step that moves every submit in the
  engine.
- **Size:** L

### 3 — Rendering scope and dynamic state

- **Do:** a neutral attachment description (view handle, load/store op, clear value),
  `BeginRendering`/`EndRendering`, `SetViewport`, `SetScissor` on `ICommandList` (D17). Nothing
  here depends on the binding model, which is why it precedes the bind groups.
- **Retires:** `tests/gpu/rhi/PresentTargetTests.cpp`'s `VulkanNative.h` entry, deferred here
  from step 2. Its clears were a raw render pass and its attachment took a `VkImageView`; both
  are neutral now, and the file reaches only for `OffscreenTarget` through the module's own
  sources. 17 sites down to 16.
- **Verify:** baseline unchanged. The recorders still bind pipelines and draw through the escape
  hatch; only the scope and dynamic state have moved.
- **Size:** M
- **Done.** `LoadOp` and `StoreOp` are named for D3D12's beginning- and ending-access types
  under D13 — `Preserve`/`Clear`/`Discard` rather than Vulkan's load/store vocabulary.
  `StoreOp` has exactly two values, and that is a correction: it briefly had a third,
  `NoAccess`, for the transparent pass reading depth it never writes. That was wrong twice
  over. D3D12's `NO_ACCESS` means the resource is **neither read nor written** and must be
  paired with a `NO_ACCESS` beginning access, so it describes the one case a depth-reading pass
  is not; and the fact it was trying to state is already stated by
  `DepthStencilTarget::bReadOnly`, so keeping both would be two fields able to disagree. The
  backend derives Vulkan's `STORE_OP_NONE` and the read-only layout from `bReadOnly`, and
  rejects a read-only target that asks to discard. The ImGui recorder came out fully neutral as
  a side effect: everything raw it did was open and close a scope.

### 4 — Bind groups: global, depth and composite

- **Do:** a bind-group layout description, a bind-group description, a handle type, and
  `SetBindGroup` on `ICommandList` (D14, D20, D23). Immutable, created from a complete
  description. The curated `BindingType` with `default:`-free switches lands here, as does the
  pinned layout inventory test (D21) covering these three layouts.
- **Why these three first:** no partial binding, no material lifetime, and their contents change
  only at the resize point that already stalls.
- **Verify:** baseline unchanged.
- **Size:** L
- **Done.** Three things worth knowing.

  **D22's sampler split happened here, not at step 5.** The composite layout already held a
  combined image sampler at binding 3, and `BindingType` has no such value, so the split could
  not wait: binding 3 became a sampled texture and binding 4 a `Sampler`, and `composite.slang`
  changed with it. Only the cloud target is sampled — the other three are fetched by texel — so
  one sampler serves the layout. **Step 5 is correspondingly smaller**: the material set alone.

  **The binding model carries stage visibility per binding**, because the depth group is read by
  the cloud dispatch as well as by pixel shaders. A graphics-only assumption would not have
  survived the first layout it met.

  **Two transitional accessors joined `VulkanNative.h`**, both expiring at step 6:
  `GetDescriptorSet` and `GetDescriptorSetLayout`. Binding a group needs a pipeline layout and
  creating a pipeline layout needs the raw set layouts, and neither is neutral until D23 makes
  `PipelineLayoutHandle` real. So the renderer creates its groups through `IDevice` and still
  binds them itself. **`SetBindGroup` therefore moves to step 6**, against this step's original
  wording — it cannot exist before the thing it takes as an argument. No allowlist entry moved:
  every site involved already had one.

  Validation earned its keep twice. It caught the pipeline layouts still naming the deleted
  raii objects, and then caught a sampled *depth* view being described as
  `SHADER_READ_ONLY_OPTIMAL` when the barrier had left it in `DEPTH_READ_ONLY_OPTIMAL`. The
  second is why `VulkanTextureView` now records its aspect: which layout a sampled view wants is
  a Vulkan rule, not something the neutral description should be made to state.

### 5 — Bind groups: the material set

- **Do:** the material set moves behind the neutral API, and combined image samplers become
  separate texture and sampler bindings (D22) — four shaders (`opaque`, `weightedBlendedOIT`,
  `composite`, `clouds.comp`), `MaterialFactory` and `PBRMaterial`. Both stop writing
  descriptors directly. The pinned inventory grows to four layouts, on its way to six — see
  D14's second correction for the two `CloudSystem` owns.
- **The partially-bound behaviour must survive the move.** It is what lets an untextured
  material render, and losing it silently would change what the test cubes look like rather
  than failing a build.
- **Retires:** `DescriptorAllocator.h` and both its allowlist entries;
  `MaterialFactory.cpp`'s and `PBRMaterial.cpp`'s `VulkanNative.h` and `DebugNames.h` entries.
  Six of eighteen.
- **Verify:** baseline unchanged — in particular the untextured and transparent cube cases from
  step 47's matrix, which are the ones that exercise partial binding. Same sampler state means
  the capture should be pixel-identical, so any movement here is a real defect.
- **Size:** L
- **Done.** The neutral layout grew one field for this step: `BindGroupLayoutBinding::bOptional`,
  which is what lets a slot be left empty. All three material textures set it, the sampler does
  not, and the bind group description simply omits the maps a material lacks. Vulkan spells the
  permission `VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT`; D3D12 reaches the same place from the
  other direction, since a descriptor a shader never accesses need not be valid. The flags
  structure is chained only when a layout actually asks for it.

  The material set carried a second sampler split, like the composite one: three combined image
  samplers became three textures plus one shared sampler, in `opaque.slang` and
  `weightedBlendedOIT.slang`.

  **`DescriptorAllocator.h` left the transitional area by moving rather than by deletion.** The
  RHI itself now allocates bind groups through it, so the header still exists — it just lives in
  `src/vulkan/` where nothing outside the module can reach it, which is the same outcome the
  ratchet was measuring. Six allowlist entries went with it: **7 headers from 16 sites down to
  6 from 10**.

  The pinned inventory is four of six layouts. `CloudSystem`'s two remain, and they need
  `UnorderedAccessTexture` before they can move — see D14's second correction.

### 6 — Graphics pipelines and pipeline layouts

- **Do:** `PipelineLayoutHandle` (D23), a neutral graphics pipeline description, and shader
  modules created from bytes the engine supplies (D24). `DeviceCaps` reports the shader format;
  `cmake/Shaders.cmake` starts emitting one blob per stage. Feeds the existing neutral
  `IPipelineCache` unchanged (D15). Formats come from `Rhi::Format`, so `GetNativeFormat`
  leaves the call sites that currently translate for the builders.
- **Retires:** `PipelineBuilder.h` and `Engine.cpp`'s entry for it.
- **Verify:** baseline unchanged. The pipeline cache still warms — `startupMs` on a second run
  should not regress, which is the only externally visible sign the cache is still working.
- **Size:** L
- **Done.** Four notes.

  **`SetPipeline`, `SetBindGroup` and `PushConstants` all landed here**, not at steps 8–11 as
  written. Each takes a pipeline layout, so none of them could exist before `PipelineLayoutHandle`
  did — and once it does, leaving those calls raw needs *more* escape-hatch accessors rather than
  fewer. Steps 8–11 keep vertex and index binding, draws and dispatches, which is a cleaner split
  by what is being recorded.

  **`Rhi::Format` grew three vertex-attribute formats** — `RG32Float`, `RGB32Float`,
  `RGBA32Float` — because vertex input needs them and both APIs carry vertex and texture formats
  in one enum. That expired a unit test's specimen: `ConversionTests` asserted that
  `eR32G32B32A32Sfloat` was outside the curated set, and its comment said the choice held only
  "until pipeline creation is neutralized". It now names `eR8G8B8Unorm` instead, and the
  mappings are spot-checked alongside the others.

  **`DeviceCaps::ShaderExtension`** is how the engine learns which blob to load. The build still
  emits one module per shader holding both entry points, so the pipeline description names the
  same module twice with different entry points — legal on Vulkan, not on D3D12. **Splitting the
  blobs per stage moved to Stage 7.6**, where DXIL emission restructures the shader build anyway;
  doing it twice would mean touching `cmake/Shaders.cmake` for the same reason in two stages.

  **`PipelineBuilder` is gone**, header and implementation: **5 transitional headers used from 9
  sites**. `ComputePipelineBuilder` stays until step 7, and the two descriptor accessors stay
  until steps 7 and 11, because `CloudSystem`'s own layouts are still raw.

### 7 — Compute pipelines

- **Do:** a neutral compute pipeline description, consuming the same layouts. `CloudSystem`'s
  dispatch pipeline and its noise-bake pipeline both move.
- **Retires:** `ComputePipelineBuilder.h` and `CloudSystem.cpp`'s entries for it and for
  `DebugNames.h`. **4 transitional headers used from 7 sites.**
- **Verify:** baseline unchanged.
- **Size:** M
- **Done.** Larger than M, because of a dependency the plan did not see.

  **`CloudSystem`'s two bind group layouts moved here, not at step 11.** A compute pipeline
  needs a pipeline layout, a pipeline layout is built from bind group layouts, and
  `CreatePipelineLayout` takes handles — so its layouts had to be neutral before its pipelines
  could be. That pulled in everything D14's second correction had assigned to step 11:
  `BindingType::UnorderedAccessTexture` for the `RWTexture2D`/`RWTexture3D` storage images, and
  a third combined-image-sampler split, in `clouds.comp.slang`. **Step 11 shrinks to the
  dispatch recording it names**, and the pinned inventory is complete at six layouts.

  `SetPipeline`, `SetComputeBindGroup` and `Dispatch` are separate compute entry points rather
  than shared ones, because both APIs keep the two apart — Vulkan by bind point, D3D12 by having
  `SetComputeRootSignature` and `SetGraphicsRootSignature` be different calls — so one call
  would have to guess which the caller meant.

  **A defect from step 4 surfaced here and is fixed.** The RHI's bind group descriptor pool was
  sized for uniform buffers, sampled images and samplers, and storage images were the first
  binding type it had never been asked for. It showed up as two validation *warnings* and a
  changed `validationWarnings` counter — not an error, because the specification lets an
  implementation fail to report the out-of-pool condition it should. The pool now carries a size
  for every `BindingType`.

### 8 — The composite recorder

- **Do:** `SetPipeline`, `SetBindGroup`, vertex and index buffer binding, `DrawIndexed`. Move
  `RecordCompositeCommandBuffer` onto them.
- **Why first of the four:** one draw, one bind group, no per-batch loop. It is the smallest
  possible proof that the recording API works before it is applied to anything harder.
- **Verify:** baseline unchanged.
- **Size:** M

### 9 — The opaque recorder

- **Do:** move `RecordOpaqueCommandBuffer`, including `PushConstants` for `MaterialData` — a
  neutral call now that a layout is neutral (D14, D23); the Vulkan and D3D12 forms were always
  1:1 and only the layout blocked it.
- **Verify:** baseline unchanged. This is the step where an unchanged screenshot is the
  load-bearing evidence rather than a formality.
- **Size:** M

### 10 — The transparent recorder

- **Do:** move `RecordTransparentCommandBuffer`.
- **Retires:** `Engine.cpp`'s `VulkanNative.h` entry — the last of its uses goes here.
- **Verify:** baseline unchanged. See §10 on why the transparent path is the noisiest place in
  the comparison.
- **Size:** M

### 11 — The clouds recorder and the noise bake

- **Do:** `Dispatch` on `ICommandList`. Move `CloudSystem::RecordDispatch` and
  `BakeNoiseTexture`, and take `vk::raii` references out of `CloudSystemCreateInfo`.
- **Retires:** `CloudSystem.cpp`'s `VulkanNative.h` and `CommandListUtil.h` entries. The latter
  needed both step 2 (submission) and this step (dispatch recording), which is why it goes last
  of the two.
- **Verify:** baseline unchanged.
- **Size:** M

### 12 — Seal the seam

- **Do:** delete `VulkanNative.h`'s RAII accessors, which exist only for code that builds Vulkan
  objects itself and by now has none. Shrink `rhi/vulkan/` to its permanent residue and update
  `cmake/RhiBoundaryCheck.cmake`'s two lists to match. Remove the remaining `DebugNames.h`
  entries as the objects they name finish moving behind the RHI.
- **Also decide, but do not assume:** this is the natural moment to retire
  `rhi_extraction_plan.md` by promoting its D0–D13, §4 and §8 into permanent homes. `CLAUDE.md`
  is explicit that retiring it is a deliberate decision rather than a roadmap step, so it is
  proposed here and taken then.
- **Verify:** `rhi_boundary_check` passes against the reduced lists; `precommit.sh` green.
- **Size:** M

---

## 4. Stage 7.6 — backend prerequisites

Three seams the backend needs that are not the RHI's, plus two things that pair with them.
They are a separate stage rather than extra steps of 7.5 for one reason: **verification
independence.** If they lived inside 7.5, that stage's own steps would be checked by machinery
being built in the same stage, and a bug in the new comparison tooling and a bug in step 10
would look identical. Keeping them apart means every step above is verified by the harness that
already works, and 7.6 is verified against a codebase that is not moving.

There is a sequencing bonus: DXIL emission can be validated before a single line of D3D12
exists.

| # | What | Why it is here | Size |
|---|---|---|---|
| 1 | **DXIL emission.** A second `slangc` target with an `sm_6_x` profile, a second output set, and a validation gate equivalent to `spirv-val`. Pairs with D24's per-stage packaging, which lands at 7.5 step 6 | `cmake/Shaders.cmake` is SPIR-V only. The first draft put this in the backend stage; that is wrong, because it is build-system and content-pipeline work with its own failure modes, and doing it there means finding out whether Slang's DXIL path handles `pbr.slangh` halfway through writing a device | M–L |
| 2 | **The comparison script**, with tolerance built in from the start (D26) | Already `backlog.md`'s P1 row — decode both PNGs, report the diff bounding box, diff the `counters`. Today `CLAUDE.md` walks a human through PIL by hand | S–M |
| 3 | **Windows GPU coverage on WARP.** `ctest -L gpu` and `-L scene` on the Windows jobs, with the adapter explicitly selected rather than enumerated — the same discipline `VK_DRIVER_FILES` gives on Linux | All three Windows jobs run `ctest -L unit` and stop. Under D25, Vulkan is always the default, so **CI is the only thing that will ever run D3D12 routinely** | M |
| 4 | **Runtime-selectable validation** | `backlog.md` P2. The engine gates validation on `NDEBUG`, so a release run reports zero validation errors trivially. Two backends mean two validation surfaces and a Windows release job worth having assert rather than silently pass | S |
| 5 | **Step 48 — `ShaderTypes.h` shared with Slang, extended with per-target layout assertions** | See below | M |

**Step 48 needs extending, and the reason is the only silent-corruption path a second backend
introduces.** Part IV's step 48 shares the declaration between C++ and Slang, which removes the
transcription error. It does not remove the layout-rule divergence: SPIR-V follows
`std140`/`std430`, HLSL constant buffers follow their own 16-byte packing rules, and the two do
not agree in every case. So a C++ struct that matches the SPIR-V layout can silently mismatch
the DXIL one and corrupt on exactly one backend. Sharing the declaration is necessary;
asserting offsets *per target* is what actually closes it.

**Definition of done for 7.6:** every shader compiles to DXIL and passes its validator; two
images that are not bit-identical can be compared and the result reported with its measured
delta; a GPU test runs on a Windows CI job; a release build can be asked for validation.

---

## 5. Stage 7.7 — the D3D12 backend

Out of scope for this document beyond three constraints that are decided:

- **It is stepped small and explained as it goes.** This is a learning exercise as much as a
  port, and the step sizing follows from that rather than from what a fluent implementer could
  manage in one sitting.
- **Neutral descriptions are checked against the D3D12 documentation as they are implemented**,
  not inferred from the Vulkan side. `CLAUDE.md`'s rule about never guessing at graphics API
  semantics applies to the API that is not in the tree yet, and applies hardest there.
- **Vulkan stays the default** (D25). The backend's routine exercise is 7.6's Windows CI job.

DXVK on Linux — running the D3D12 path over Vulkan — is a stretch goal in `backlog.md`, not
part of this.

---

## 6. What this stage needs from other stages

| What | Where | Status |
|---|---|---|
| **Step 47** — headless scene tests in CI | Stage 7 | **Done.** The instrument every step's verification leans on. D26 is what adapts it to two backends |
| **Step 46** — `IUiBackend` + `VulkanUiBackend` | Stage 7 | **Done.** D9's ImGui escape hatch is now a leaf file that a D3D12 build replaces with a sibling, rather than a hole in the renderer |
| **Step 48** — `ShaderTypes.h` shared with Slang | Stage 8 | Pulled into Stage 7.6, extended — see §4 |
| **Steps 50–54** — recorders become `Pass` classes | Stage 8 | Follow this stage, not precede it (D18) |
| **Step 58** — `Mesh*`/`Material*` become handles | Stage 9 | **Stays in Stage 9.** See below |
| **Step 70** — bindless | Stage 10 | Explicitly after the backend (D14) |
| Device info in the run report | `backlog.md` (P2) | Unblocks here: its blocker was "a neutral device-info accessor on `IDevice`, which is a seam decision", and this is where seam decisions are taken. Two backends make "which device produced this report" worth answering |
| Runtime-selectable validation | `backlog.md` (P2) | Moved into Stage 7.6 — see §4 |

**Why step 58 stays in Stage 9**, against the first draft's recommendation to pull it forward.
`Drawable::operator<` falls through to comparing `pMesh` and `pMat` pointers, so batch order
tracks heap addresses, and ASLR reshuffles them every run. The first draft called that a flaw
in the primary instrument. It is a **latent hazard, not a gate**:

- Batching groups equal keys after sorting, so the counts — `batches`, `drawCalls` — are
  order-independent whatever order the pointers fall in.
- The images are largely order-independent too: opaque is depth-tested, and weighted-blended
  OIT is order-independent by construction.
- If it were biting, the baseline would already fail intermittently across runs. It does not.
- Two backends do not make it worse. The comparison is between two runs, and pointer order
  already differs between runs today.

It remains worth doing — it is insurance against the first order-dependent pass — but that pass
arrives in Stage 8 at the earliest, and 7.5, 7.6 and 7.7 add none. See §10 for the one place
where transparency genuinely does introduce comparison noise, which is a different mechanism.

Already done, and listed so nobody re-does them: **D10** gives clip-space handedness one site
behind `DeviceCaps::bFlipClipSpaceY`; **D11**'s curated `Rhi::Format` with `default:`-free
switches already fails the build on an unmapped format; **D12**'s Slang shaders are portable as
written; and `DeviceRequirements::NativeWindowHandle` is already an opaque `void*` documented
as "a native window pointer versus an HWND", so the platform seam needs nothing.

---

## 7. Out of scope

Everything that fails the inclusion test in §1, and specifically:

- **The D3D12 backend itself.** This stage makes it possible; Stage 7.7 starts it.
- **The frame graph (step 56)** and `BarrierBatcher` (55). A second backend needs a neutral
  command list, not a graph. Building the graph against one backend bakes in that backend's
  assumptions; building it after means writing it with two in front of you.
- **Deferred destruction.** D20 explains why, and what triggers building it.
- **A fourth texture map.** D14's correction: nothing is being held back, so raising
  `TextureBinding::COUNT` is a feature. Backlog.
- **A `ShaderLibrary`.** D24. Backlog.
- **All of Stage 9 including step 58.** Arena, `FrameSnapshot`, radix sort, dirty flags, frustum
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

## 8. Definition of done

`cmake/RhiBoundaryCheck.cmake` is the measure, because it is already enforced in CI and already
names the work that removes each entry. Today it holds **7 transitional headers used from 19
sites** — 17 in the first draft, 18 once Stage 7 gave the UI backend an entry of its own, and 19
once step 1 added the validation-coverage test, which needs a queue until step 2 hands one out.

The steps account for sixteen of those sites: step 2 two, step 3 one, step 5 six, step 6 one,
step 7 one, step 10 one, step 11 two, step 12 the two remaining `DebugNames.h` entries. That leaves **2
headers used from 3 sites**:

| Header | Site | Why it stays |
|---|---|---|
| `VulkanNative.h` | `engine/editor/src/VulkanUiBackend.cpp` | ImGui's Vulkan backend takes raw handles. D9 is permanent by design: a D3D12 build gets a sibling file, not an edit |
| `VulkanNative.h` | `tests/gpu/rhi/DeviceTests.cpp` | The escape hatch is what those cases assert on |
| `SwapchainUtil.h` | `tests/unit/rhi/SwapchainUtilTests.cpp` | Deliberate, and argued in the check itself: the functions are pure and device-free so they can be unit tested, and `src/vulkan/` is on a PRIVATE include path a test cannot reach |

Note that `GetNative(ICommandList&)` survives step 12 along with the rest of the ImGui-shaped
hole — ImGui's backend takes a `VkCommandBuffer` by value, and there is no neutral shape for
that.

The second measure is `rhi_extraction_plan.md` §8's checklist: **no row still reads *Partial* or
*Deferred***. Six do today — command recording, command pool, CPU/GPU sync, descriptors,
per-draw constants and pipelines — and they are, near enough, this document's step list.

---

## 9. Open investigations

**1. ~~What does the dropped-semaphore experiment actually do today?~~ Settled 6 September
2026, and the row it came from was wrong twice over.**

`backlog.md` claimed the gpu suite asserted a synchronization dependency it could not detect,
because sync validation was off. Sync validation is not off — `validate_sync` is hardcoded
`VK_TRUE`. The obvious replacement claim, that syncval cannot see cross-submit hazards here, is
also wrong: a read-after-write on a *buffer* split across two submissions to one queue, with no
barrier, semaphore or fence between them, is reported as `vkQueueSubmit(): READ_AFTER_WRITE
hazard detected`, naming both command buffers and both submits.

**So the offscreen read has no hazard to find.** Dropping its wait semaphore leaves the suite
green because a barrier and submission order were already supplying the dependency the
semaphore was being credited with. Buffers were the discriminating case: an image carries layout
transitions, which are themselves ordered against other transitions on the same queue, so an
image test can pass for a reason unrelated to hazard detection. That is why the earlier attempt
— weakening the readback barrier's source scope — proved nothing either way.

`tests/gpu/rhi/ValidationCoverageTests.cpp` is what came out of it: the hazard above, committed
deliberately, asserting that validation *reported* it and then clearing the counters. It was
checked against its own failure — remove the hazard and the case fails — so it is a positive
control rather than another assertion that cannot fail. Every "zero validation errors" claim in
the gpu suite now rests on something.

**2. What does Slang's DXIL path require?** Profile and target flags, whether a separate
validator is needed, and what the emitted DXIL needs at load time. Determines the size of Stage
7.6's first item, and should be answered from the Slang documentation rather than by
experiment-until-it-links.

---

## 10. Risks

- **Steps 5 and 8–11 touch every draw and every material.** This is the R9/R10 hazard again,
  and the RHI plan's advice applies unchanged: do not merge them, and run a baseline comparison
  between them. The twelve-step split is that advice taken further than the first draft took
  it.
- **The seam is being designed against one backend.** Some of it will be wrong in ways only
  writing the D3D12 backend reveals. Mitigate by checking each neutral description against the
  D3D12 documentation as it is written rather than inferring from the Vulkan side. Budget for
  revision rather than assuming the first shape survives. The grill found one of these before a
  line was written — D22, combined image samplers — which is evidence both that the risk is real
  and that reading the other API's documentation early is what catches it.
- **Synchronization mistakes in steps 2 and 8–11 will not fail locally.** Plausible-sounding
  synchronization compiles, renders correctly on one driver and fails intermittently on
  another. Synchronization validation is on; a clean run is necessary rather than sufficient.
- **Transparency is the noisiest place in the comparison.** Weighted-blended OIT accumulates
  additively, float addition is not associative, and `Drawable::operator<` orders by pointer, so
  three or more stacked transparent layers can differ in the low bits between runs. Known cause,
  not a regression. It is also the strongest argument for D26's tolerance applying *within* a
  backend eventually, not only across two — the current exact check holds because the test scene
  does not stack that deep, which is a property of the content rather than of the renderer.
- **Scope creep through adjacency.** Several steps open files that Stage 8 and Stage 9 also
  want to change. The inclusion test in §1 is the defence, and it only works if it is actually
  applied when the temptation arrives.
- **Twelve steps is more ceremony than six.** Twelve baseline runs, twelve review cycles. The
  trade was made deliberately: no step touches more than one idea, so when the baseline moves,
  which idea moved it is not a question.

---

## 11. Retention

**This document is kept after the stage ends.** Stage 7's plan was deleted at its stage's close
because it records how to build things that will by then be built. `rhi_extraction_plan.md` was
kept past Stage 5 because its decisions still govern a seam that outlived it. This one is the
second kind: D14–D26 say what the RHI's public API is allowed to express about recording,
binding, pipelines and submission, and a D3D12 backend — and everything written against the seam
afterwards — has to respect them.

What that means in practice:

- The step sequence in §3 becomes history once the stage completes, exactly as R1–R17 did.
  Leave it; it is short, and it explains why the seam has the shape it has.
- §2's decisions stay live and are the reason to open this file.
- §8's definition of done becomes the standing description of what the transitional area is
  *for*, and `cmake/RhiBoundaryCheck.cmake` stays its enforcement.
- D26 is the exception that should **not** live here permanently. It is test strategy that
  applies to any two renderers, including two D3D12 driver versions, so it also lands in the
  architecture plan's Part III and outlives this document.
- If `rhi_extraction_plan.md` is retired at step 12, this document is the natural place for
  D0–D13 to land, which would put the whole D-series back in one file and make the numbering
  continuity in §2 pay off.
