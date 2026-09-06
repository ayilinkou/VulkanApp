#pragma once

#include <memory>

#include "vulkan/vulkan_raii.hpp"

#include <rhi/Handles.h>
#include <rhi/ICommandAllocator.h>
#include <rhi/ICommandList.h>
#include <rhi/UniqueHandle.h>

#include "Texture.h"

/**
 * One recorder's command storage for one frame in flight.
 *
 * Per recorder rather than shared because the frame records on several threads
 * at once and a command allocator is not internally synchronized: the opaque
 * and transparent passes run as jobs while the main thread records the rest, so
 * each touching only its own allocator is what makes that safe.
 *
 * List is null until the frame has been recorded, and points into the
 * allocator, so it is valid only until that allocator's next Reset(). The
 * submit reads it back to find the buffer to hand the queue, which is why
 * recording keeps it rather than dropping it on the way out.
 */
struct FrameRecorder
{
    std::unique_ptr<Hikari::Rhi::ICommandAllocator> Allocator;
    Hikari::Rhi::ICommandList* List = nullptr;
};

struct FrameData
{
    FrameRecorder DrawLayoutCommands;
    FrameRecorder OpaqueCommands;
    FrameRecorder CloudCommands;
    FrameRecorder TransparentCommands;
    FrameRecorder CompositeCommands;
    FrameRecorder ImGuiCommands;
    FrameRecorder FinalLayoutCommands;
    /**
     * The frame fence value this slot's last submission signals. Waiting for it
     * is what makes the slot safe to reuse -- its allocators can be reset and its
     * per-frame buffers rewritten only once the GPU has passed this point.
     *
     * Zero until the slot has been submitted once, which is a value the fence
     * starts at, so the first pass through waits for nothing.
     */
    uint64_t LastSubmitValue = 0u;
    /**
     * Global never changes -- its buffer is created once and only its contents
     * are rewritten. Composite and Depth name render targets, so both are
     * replaced whenever those are recreated: a bind group is immutable, and
     * replacing one is how its contents change (RHI plan D20).
     */
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::BindGroupHandle> GlobalBindGroup;
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::BindGroupHandle> CompositeBindGroup;
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::BindGroupHandle> DepthBindGroup;
    Texture OpaqueTexture;
    Texture AccumTexture;
    Texture RevealageTexture;
    Texture DepthTexture;
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::BufferHandle> GlobalBuffer;
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::BufferHandle> InstanceBuffer;
};
