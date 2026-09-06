#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <core/Extent3D.h>
#include <rhi/Barrier.h>
#include <rhi/Handles.h>
#include <rhi/Pipeline.h>
#include <rhi/Rendering.h>
#include <rhi/RhiTypes.h>

namespace Hikari::Rhi
{
/** One buffer-to-buffer copy. */
struct BufferCopyRegion
{
    uint64_t SrcOffset = 0u;
    uint64_t DstOffset = 0u;
    uint64_t Size = 0u;
};

/**
 * One copy between a buffer and a texture subresource.
 *
 * The texture side always starts at the subresource origin and covers Extent.
 * There is no offset field because nothing copies a sub-rectangle: adding one
 * means adding an Offset3D to the neutral vocabulary for a case that does not
 * exist, and the field would be dead weight every caller had to read past.
 *
 * The buffer side is tightly packed — rows are Extent.Width texels apart and
 * layers Extent.Width * Extent.Height. Vulkan spells that as bufferRowLength
 * and bufferImageHeight of zero; D3D12 requires row pitch to be aligned, which
 * is the backend's problem rather than the caller's.
 */
struct BufferTextureCopyRegion
{
    uint64_t BufferOffset = 0u;

    TextureAspect Aspect = TextureAspect::Color;
    uint32_t MipLevel = 0u;
    uint32_t BaseLayer = 0u;
    uint32_t LayerCount = 1u;

    Core::Extent3D Extent{};
};

/**
 * A command list being recorded: barriers and copies in Stage 5, draws in
 * Stage 8.
 *
 * Deliberately partial. Draw, bind and viewport recording still happens on the
 * backend's own command buffer inside the renderer, because neutralizing a draw
 * means neutralizing pipelines and the descriptor model with it (plan D7, D8).
 * What is here is what can be expressed without either: the barriers and the
 * copies, which together are every command that names a resource by handle.
 *
 * The list does not own the underlying command buffer or allocator — it records
 * into one the caller already has, and the caller still submits it.
 */
class ICommandList
{
public:
    virtual ~ICommandList() = default;

    ICommandList(const ICommandList&) = delete;
    ICommandList& operator=(const ICommandList&) = delete;
    ICommandList(ICommandList&&) = delete;
    ICommandList& operator=(ICommandList&&) = delete;

    /**
     * Opens the list for recording, and closes it. A list is recorded once per
     * frame after its allocator has been reset; re-recording without a reset is
     * a backend error rather than something this interface prevents.
     */
    virtual void Begin() = 0;
    virtual void End() = 0;

    /**
     * Records every barrier in `barriers` as one command, and returns what that
     * cost — the barrier count, and the one call it took. An empty span records
     * nothing and returns zero of both.
     *
     * Grouping matters beyond saving call overhead. Each barrier command is its
     * own execution dependency, so transitioning three textures in three
     * commands orders those three transitions against one another for no reason
     * — the driver may not begin the second until the first has completed.
     * Issued together they are independent, which is what they actually are.
     *
     * A caller whose barriers are separated by real work (a copy, a dispatch)
     * must still issue them separately; there the dependency is the point.
     */
    virtual BarrierCounts Barrier(std::span<const TextureBarrier> barriers) = 0;

    /**
     * Single-barrier shorthand, for the sites with nothing to group with.
     * Returns one of each, so a caller keeping a running total adds it the same
     * way it adds a batch rather than hard-coding numbers beside the call.
     */
    virtual BarrierCounts Barrier(const TextureBarrier& barrier) = 0;

    /**
     * Opens and closes a rendering scope. Draws are recorded between the two,
     * and every attachment must already be in the layout rendering needs -- this
     * transitions nothing, for the same reason the copies below do not.
     *
     * Scopes do not nest, and a list must close every scope it opens.
     */
    virtual void BeginRendering(const RenderingDesc& desc) = 0;
    virtual void EndRendering() = 0;

    /**
     * Binds the pipeline subsequent draws use, and the resources they read.
     *
     * SetBindGroup takes the layout as well as the group because that is what
     * both APIs bind against -- a VkPipelineLayout, a root signature -- and
     * because layout identity is what decides whether a bound group survives the
     * next SetPipeline. Passing it explicitly keeps that visible instead of
     * making it a consequence of call order.
     */
    virtual void SetPipeline(GraphicsPipelineHandle pipeline) = 0;
    virtual void SetBindGroup(PipelineLayoutHandle layout, uint32_t slot,
                              BindGroupHandle group) = 0;

    /**
     * Constants written straight into the command list.
     *
     * Takes the layout for the same reason SetBindGroup does: both APIs bind
     * against it, and the range being pushed into was declared there. `stages`
     * must name a range the layout actually declares.
     *
     * Bytes rather than a template, so the neutral interface stays free of the
     * caller's struct -- the layout of that struct is the caller's contract with
     * its shaders, not with the RHI.
     */
    virtual void PushConstants(PipelineLayoutHandle layout, ShaderStage stages, uint32_t offset,
                               std::span<const std::byte> data) = 0;

    /**
     * Viewport and scissor are always dynamic. Both APIs set them on the command
     * list rather than baking them into a pipeline, and a renderer that resizes
     * would otherwise rebuild every pipeline to change a number.
     */
    virtual void SetViewport(const Viewport& viewport) = 0;
    virtual void SetScissor(const Rect2D& rect) = 0;

    /**
     * Copies must be issued between barriers that put both resources in the
     * right layout — CopySrc/CopyDst — which is why none of these transition
     * anything themselves. A copy list that transitioned implicitly would
     * either over-synchronize or hide the transition from the barrier counts.
     */
    virtual void CopyBuffer(BufferHandle source, BufferHandle destination,
                            const BufferCopyRegion& region) = 0;
    virtual void CopyBufferToTexture(BufferHandle source, TextureHandle destination,
                                     const BufferTextureCopyRegion& region) = 0;
    virtual void CopyTextureToBuffer(TextureHandle source, BufferHandle destination,
                                     const BufferTextureCopyRegion& region) = 0;

protected:
    ICommandList() = default;
};
} // namespace Hikari::Rhi
