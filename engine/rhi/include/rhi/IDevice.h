#pragma once

#include <cstdint>
#include <memory>

#include <rhi/BufferDesc.h>
#include <rhi/DeviceDesc.h>
#include <rhi/Diagnostics.h>
#include <rhi/Handles.h>
#include <rhi/ICommandAllocator.h>
#include <rhi/IPresentTarget.h>
#include <rhi/PipelineCache.h>
#include <rhi/SamplerDesc.h>
#include <rhi/TextureDesc.h>
#include <rhi/TextureViewDesc.h>
#include <rhi/UploadContext.h>

namespace Hikari::Rhi
{
/**
 * The GPU device, and eventually the sole owner of every GPU resource.
 *
 * Abstract rather than a compile-time typedef to the backend type, because a
 * null/recording implementation has to be able to coexist with a real one in a
 * single test binary, and because selecting a backend at runtime should not
 * require rewriting call sites. The cost is a vtable dispatch on calls that are
 * already crossing into a driver, which is why resource creation lives here and
 * per-draw recording does not.
 */
class IDevice
{
public:
    virtual ~IDevice() = default;

    IDevice(const IDevice&) = delete;
    IDevice& operator=(const IDevice&) = delete;
    IDevice(IDevice&&) = delete;
    IDevice& operator=(IDevice&&) = delete;

    virtual const DeviceCaps& GetCaps() const = 0;

    /**
     * The device's validation counters and policy. Always valid: a device given
     * no Diagnostics creates its own rather than returning null.
     */
    virtual Diagnostics& GetDiagnostics() = 0;

    /**
     * Blocks until the device has finished everything submitted to it. A
     * shutdown and resize tool, not a synchronisation primitive — anything in
     * the frame loop wanting to wait should wait on a fence instead.
     */
    virtual void WaitIdle() = 0;

    /**
     * --- Buffers ---
     *
     * The device owns the storage and hands back a handle rather than an object
     * (plan D2). Throws on allocation failure rather than returning an invalid
     * handle: a caller that cannot have its buffer has nothing useful to do
     * with the failure, and every one of them would otherwise have to check.
     */
    virtual BufferHandle CreateBuffer(const BufferDesc& desc) = 0;

    /**
     * Frees the buffer and invalidates every outstanding copy of `handle`.
     * Destroying an already-destroyed or never-valid handle is reported through
     * Diagnostics rather than ignored — that report is the use-after-free
     * detection the handle model exists to buy, so it is worth reading.
     */
    virtual void Destroy(BufferHandle handle) = 0;

    /**
     * The CPU-visible pointer for a host-visible buffer, or nullptr for a
     * GpuOnly one or a stale handle.
     *
     * There is no matching Unmap. Host-visible allocations here are mapped for
     * as long as they live, because the buffers that need a mapping — the
     * per-frame uniform and instance buffers — are written every frame, and
     * mapping is not free. A pair of Map/Unmap calls would therefore be a
     * fiction: the pointer is valid from creation to destruction either way.
     */
    virtual void* GetMappedData(BufferHandle handle) = 0;

    /**
     * Buffers currently alive. Exists to be asserted on at shutdown, where
     * anything other than zero is a leak.
     */
    virtual uint32_t GetLiveBufferCount() const = 0;

    /**
     * --- Textures, views and samplers ---
     *
     * Three separate identities rather than one, because that is what both APIs
     * have: the texture is the memory, the view is how a shader or an
     * attachment interprets it, and the sampler is how it is filtered. D3D12
     * backs the last two with descriptors rather than objects, which changes
     * what a handle resolves to and not what it means.
     *
     * Every Create throws on failure, for the same reason CreateBuffer does.
     * Every Destroy invalidates outstanding copies of the handle and reports a
     * stale one through Diagnostics.
     *
     * Destruction order is the caller's responsibility: a view outliving its
     * texture is legal here and a use-after-free in the driver, so destroy
     * views before the texture they were made from.
     */
    virtual TextureHandle CreateTexture(const TextureDesc& desc) = 0;
    virtual void Destroy(TextureHandle handle) = 0;

    virtual TextureViewHandle CreateTextureView(const TextureViewDesc& desc) = 0;
    virtual void Destroy(TextureViewHandle handle) = 0;

    virtual SamplerHandle CreateSampler(const SamplerDesc& desc) = 0;
    virtual void Destroy(SamplerHandle handle) = 0;

    /**
     * The description `handle` was created with, or nullptr if it is stale.
     * Exists because a texture's extent and format are needed wherever it is
     * used — sizing a copy, choosing an aspect — and asking the device beats
     * every caller keeping its own copy in step with the real one.
     */
    virtual const TextureDesc* GetTextureDesc(TextureHandle handle) const = 0;

    /** Counterparts to GetLiveBufferCount, and asserted on at the same place. */
    virtual uint32_t GetLiveTextureCount() const = 0;
    virtual uint32_t GetLiveTextureViewCount() const = 0;
    virtual uint32_t GetLiveSamplerCount() const = 0;

    /**
     * --- Uploads ---
     *
     * A context owns a command allocator, a fence and the staging buffers it has
     * in flight, which is why it is an object rather than a device method taking
     * data: the batching it exists for has to live somewhere across calls.
     *
     * Contexts are independent of one another, so a second loading thread takes
     * a second context rather than sharing one (see IUploadContext).
     */
    [[nodiscard]] virtual std::unique_ptr<IUploadContext>
    CreateUploadContext(const UploadContextDesc& desc) = 0;

    /**
     * --- Command recording ---
     *
     * An allocator is the storage lists record into and is not internally
     * synchronized, so the caller creates one per thread that records: this
     * engine keeps one per recorder per frame in flight. Handing them out
     * rather than pooling them internally is what keeps that rule visible at
     * the point where it has to be obeyed (see ICommandAllocator).
     */
    [[nodiscard]] virtual std::unique_ptr<ICommandAllocator>
    CreateCommandAllocator(const CommandAllocatorDesc& desc) = 0;

    /**
     * --- Pipelines ---
     *
     * The cache is the only part of pipeline creation that is neutral in this
     * stage (plan D8). Creating a pipeline still means naming backend state and
     * so still happens against the backend's own builders; creating a cache,
     * handing it to them and saving it does not.
     *
     * One per device is the intended shape. Nothing stops a second, but two
     * caches only learn half of what one would.
     */
    [[nodiscard]] virtual std::unique_ptr<IPipelineCache>
    CreatePipelineCache(const PipelineCacheDesc& desc) = 0;

    /**
     * --- Presentation ---
     *
     * Which kind of target comes back is the device's to decide, not the
     * caller's: a device created without presentation support has no surface to
     * build a swapchain on, and the whole point of the interface is that the
     * renderer cannot tell. Throws if the device cannot present at all.
     */
    [[nodiscard]] virtual std::unique_ptr<IPresentTarget>
    CreatePresentTarget(const PresentTargetDesc& desc) = 0;

protected:
    IDevice() = default;
};

/**
 * Creates the device for whichever backend this build was compiled with.
 * Throws on failure rather than returning null: there is no useful degraded
 * mode, and every caller would otherwise have to check.
 */
[[nodiscard]] std::unique_ptr<IDevice> CreateDevice(const DeviceDesc& desc);
} // namespace Hikari::Rhi
