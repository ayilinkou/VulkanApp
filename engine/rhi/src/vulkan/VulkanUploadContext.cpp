#include "vulkan/VulkanUploadContext.h"

#include <cstring>
#include <format>
#include <limits>
#include <stdexcept>

#include <core/Log.h>

#include "vulkan/DebugNames.h"
#include <rhi/BarrierPresets.h>
#include <rhi/ICommandList.h>

#include "vulkan/VulkanCommandList.h"
#include "vulkan/VulkanConversions.h"
#include "vulkan/VulkanDevice.h"

namespace Hikari::Rhi::Vulkan
{
constexpr Core::LogCategory LogRhi("RHI");
namespace
{

/**
 * Staging is written by the CPU and read once by the copy, which is exactly what
 * CpuToGpu describes.
 */
constexpr MemoryAccess kStagingAccess = MemoryAccess::CpuToGpu;

/**
 * Every subresource starts at a 4-byte boundary within its staging buffer.
 *
 * Packing them tightly is legal on a graphics or compute queue, but a queue
 * family that supports only transfer requires every bufferOffset to be a
 * multiple of 4 (VUID-vkCmdCopyBufferToImage-commandBuffer-07737). That rule
 * has no effect on the four-byte-per-texel textures loaded today, and no
 * symptom at all until the code meets both a DMA-only queue and Format::R8Unorm
 * — which is the failure mode this whole area is prone to, so it is paid for up
 * front rather than discovered on someone else's GPU.
 *
 * Rounding up cannot break the alignment rule that applies on every queue, that
 * an offset is a multiple of the texel block size
 * (VUID-vkCmdCopyBufferToImage-dstImage-07975): every block size in Rhi::Format
 * is a power of two, so 4 is either a multiple of it or it is a multiple of 4
 * and a 4-aligned offset was already block-aligned.
 */
constexpr uint64_t kStagingCopyAlignment = 4u;

constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}
} // namespace

VulkanUploadContext::VulkanUploadContext(VulkanDevice& device, const UploadContextDesc& desc)
    : m_Device(device), m_Desc(desc)
{
    const std::string name =
        m_Desc.DebugName.empty() ? std::string("Upload Context") : m_Desc.DebugName;

    vk::raii::Device& vkDevice = m_Device.GetDevice();

    m_CopyFamily = m_Device.GetQueueFamily(QueueType::Copy);
    m_GraphicsFamily = m_Device.GetQueueFamily(QueueType::Graphics);

    // The copy family falls back to the graphics one on a device with no
    // separate transfer engine, and then there are no two families for a
    // resource to move between.
    m_bSeparateCopyQueue = m_CopyFamily != m_GraphicsFamily;

    m_TransferRules = m_Device.GetOwnershipTransferRules(m_CopyFamily);
    m_bUseAllStages = m_Device.IsMaintenance8Enabled();

    // Transient because every buffer these pools hand out is recorded, submitted
    // and reset within one flush.
    //
    // The copy pool takes the copy family because a command buffer may only be
    // submitted to a queue of the family its pool was created for, and the
    // copies are submitted to the copy queue.
    const vk::CommandPoolCreateInfo copyPoolInfo{.flags = vk::CommandPoolCreateFlagBits::eTransient,
                                                 .queueFamilyIndex = m_CopyFamily};
    m_CopyPool = vk::raii::CommandPool(vkDevice, copyPoolInfo);
    SetVkDebugName(vkDevice, *m_CopyPool, vk::ObjectType::eCommandPool,
                   (name + " Copy Command Pool").c_str());

    const vk::CommandBufferAllocateInfo copyAllocInfo{.commandPool = *m_CopyPool,
                                                      .level = vk::CommandBufferLevel::ePrimary,
                                                      .commandBufferCount = 1u};
    m_CopyCommandBuffer = std::move(vk::raii::CommandBuffers(vkDevice, copyAllocInfo).front());
    SetVkDebugName(vkDevice, *m_CopyCommandBuffer, vk::ObjectType::eCommandBuffer,
                   (name + " Copy Command Buffer").c_str());

    if (m_bSeparateCopyQueue)
    {
        const vk::CommandPoolCreateInfo acquirePoolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eTransient,
            .queueFamilyIndex = m_GraphicsFamily};
        m_AcquirePool = vk::raii::CommandPool(vkDevice, acquirePoolInfo);
        SetVkDebugName(vkDevice, *m_AcquirePool, vk::ObjectType::eCommandPool,
                       (name + " Acquire Command Pool").c_str());

        const vk::CommandBufferAllocateInfo acquireAllocInfo{.commandPool = *m_AcquirePool,
                                                             .level =
                                                                 vk::CommandBufferLevel::ePrimary,
                                                             .commandBufferCount = 1u};
        m_AcquireCommandBuffer =
            std::move(vk::raii::CommandBuffers(vkDevice, acquireAllocInfo).front());
        SetVkDebugName(vkDevice, *m_AcquireCommandBuffer, vk::ObjectType::eCommandBuffer,
                       (name + " Acquire Command Buffer").c_str());

        // Binary rather than a timeline semaphore, even though the RHI models
        // waits as fence + value (plan D5): this one never leaves the backend,
        // is signalled and waited exactly once per flush, and a timeline would
        // mean enabling VkPhysicalDeviceVulkan12Features::timelineSemaphore for
        // a use that gains nothing from a counter.
        //
        // Reuse across flushes is safe because a binary semaphore returns to
        // unsignalled once the wait it satisfied is executed, and Flush() blocks
        // on the fence that submission signals before it can be called again.
        m_OwnershipSemaphore = vk::raii::Semaphore(vkDevice, vk::SemaphoreCreateInfo{});
        SetVkDebugName(vkDevice, *m_OwnershipSemaphore, vk::ObjectType::eSemaphore,
                       (name + " Ownership Semaphore").c_str());
    }

    // Created unsignaled: the first thing done with it is a submit, and the wait
    // that follows must not return before that submit completes.
    m_Fence = vk::raii::Fence(vkDevice, vk::FenceCreateInfo{});
    SetVkDebugName(vkDevice, *m_Fence, vk::ObjectType::eFence, (name + " Fence").c_str());

    if (!m_bSeparateCopyQueue)
    {
        Core::LogMsg(
            Core::LogSeverity::Info, LogRhi,
            "Upload context '{}' uploads on the graphics queue family {}; the device has no "
            "separate copy family, so nothing is ever handed over.",
            name, m_CopyFamily);
        return;
    }

    Core::LogMsg(
        Core::LogSeverity::Info, LogRhi,
        "Upload context '{}' uploads on queue family {} for the graphics family {}. Buffers {} "
        "an ownership transfer, images are decided per resource, and ownership barriers name {}.",
        name, m_CopyFamily, m_GraphicsFamily,
        BufferRequiresOwnershipTransfer(m_TransferRules, m_CopyFamily, m_GraphicsFamily)
            ? "need"
            : "do not need",
        m_bUseAllStages ? "real pipeline stages" : "AllCommands");
}

VulkanUploadContext::~VulkanUploadContext()
{
    // Anything still pending was recorded and never flushed, which means the
    // resources it was meant to fill are holding uninitialised memory. Doing the
    // upload now would be too late to help — whoever owns those resources has
    // already been handed them — so this reports rather than papers over it, and
    // releases the staging so the device does not also report leaked buffers.
    if (!m_BufferCopies.empty() || !m_TextureCopies.empty())
    {
        Core::LogMsg(
            Core::LogSeverity::Error, LogRhi,
            "Upload context destroyed with {} buffer and {} texture upload(s) never flushed — "
            "those resources were never filled.",
            m_BufferCopies.size(), m_TextureCopies.size());
    }

    for (const BufferHandle staging : m_Staging)
        m_Device.Destroy(staging);

    // The one line that says what the batching actually bought, which is
    // otherwise only visible by counting flush lines in the log.
    Core::LogMsg(Core::LogSeverity::Info, LogRhi,
                 "Upload context destroyed after {} submission(s) for {} resource(s), {:.1f} MiB.",
                 m_Stats.Submits, m_Stats.Uploads,
                 static_cast<double>(m_Stats.Bytes) / (1024.0 * 1024.0));
}

BufferHandle VulkanUploadContext::CreateStaging(uint64_t size, const char* what)
{
    const BufferHandle staging = m_Device.CreateBuffer(
        BufferDesc{.Size = size,
                   .Usage = BufferUsage::CopySrc,
                   .Access = kStagingAccess,
                   .DebugName = std::format(
                       "{} Staging ({})",
                       m_Desc.DebugName.empty() ? std::string("Upload") : m_Desc.DebugName, what)});
    m_Staging.push_back(staging);
    return staging;
}

void VulkanUploadContext::FlushIfOverBudget(uint64_t bytes)
{
    const bool bPending = !m_BufferCopies.empty() || !m_TextureCopies.empty();
    if (bPending && m_PendingBytes + bytes > m_Desc.StagingBudget)
        Flush();
}

void VulkanUploadContext::UploadBuffer(BufferHandle destination, uint64_t destinationOffset,
                                       std::span<const std::byte> data)
{
    if (data.empty())
        return;

    FlushIfOverBudget(data.size_bytes());

    const BufferHandle staging = CreateStaging(data.size_bytes(), "buffer");

    void* pMapped = m_Device.GetMappedData(staging);
    if (pMapped == nullptr)
        throw std::runtime_error("Rhi::IUploadContext::UploadBuffer: staging is not host-visible.");

    std::memcpy(pMapped, data.data(), data.size_bytes());

    m_BufferCopies.push_back(PendingBufferCopy{.Staging = staging,
                                               .Destination = destination,
                                               .DestinationOffset = destinationOffset,
                                               .Size = data.size_bytes()});
    m_PendingBytes += data.size_bytes();
    ++m_Stats.Uploads;
    m_Stats.Bytes += data.size_bytes();
}

void VulkanUploadContext::UploadTexture(TextureHandle destination,
                                        std::span<const TextureUpload> subresources)
{
    if (subresources.empty())
        return;

    uint64_t total = 0u;
    for (const TextureUpload& subresource : subresources)
        total = AlignUp(total, kStagingCopyAlignment) + subresource.Data.size_bytes();

    if (total == 0u)
        return;

    // Whole texture or nothing: see IUploadContext::UploadTexture for why
    // splitting one across two batches would discard the first batch's pixels.
    FlushIfOverBudget(total);

    const BufferHandle staging = CreateStaging(total, "texture");

    auto* pMapped = static_cast<std::byte*>(m_Device.GetMappedData(staging));
    if (pMapped == nullptr)
    {
        throw std::runtime_error(
            "Rhi::IUploadContext::UploadTexture: staging is not host-visible.");
    }

    PendingTextureCopy pending{.Staging = staging, .Destination = destination, .Subresources = {}};
    pending.Subresources.reserve(subresources.size());

    uint64_t offset = 0u;
    for (const TextureUpload& subresource : subresources)
    {
        offset = AlignUp(offset, kStagingCopyAlignment);
        std::memcpy(pMapped + offset, subresource.Data.data(), subresource.Data.size_bytes());

        pending.Subresources.push_back(
            PendingTextureCopy::Subresource{.StagingOffset = offset,
                                            .Aspect = subresource.Aspect,
                                            .MipLevel = subresource.MipLevel,
                                            .BaseLayer = subresource.BaseLayer,
                                            .LayerCount = subresource.LayerCount,
                                            .Extent = subresource.Extent});

        offset += subresource.Data.size_bytes();
    }

    m_TextureCopies.push_back(std::move(pending));
    m_PendingBytes += total;
    ++m_Stats.Uploads;
    m_Stats.Bytes += total;
}

vk::ImageMemoryBarrier2 VulkanUploadContext::MakeReleaseBarrier(vk::Image image,
                                                                const TextureDesc& desc) const
{
    // The destination half is empty on purpose. A release performs no
    // visibility operation, so its destination access mask has no effect and
    // the specification asks for it to be zero (Vulkan 1.4, *Queue Family
    // Ownership Transfer*).
    //
    // The destination *stage* is ignored too, unless maintenance8's dependency
    // flag is set — and then it stops being ignored in both directions: it
    // becomes the stage the release is ordered at, and it must be one the copy
    // family supports (VUID-vkCmdPipelineBarrier2-dstStageMask-09676). Copy is
    // both, which is what makes naming it an improvement on leaving the
    // operation unpinned.
    return vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = m_bUseAllStages ? vk::PipelineStageFlags2{vk::PipelineStageFlagBits2::eCopy}
                                        : vk::PipelineStageFlags2{},
        .dstAccessMask = vk::AccessFlagBits2::eNone,
        .oldLayout = ToVk(TextureLayout::CopyDst),
        .newLayout = ToVk(TextureLayout::ShaderResource),
        .srcQueueFamilyIndex = m_CopyFamily,
        .dstQueueFamilyIndex = m_GraphicsFamily,
        .image = image,
        .subresourceRange = {.aspectMask = ToVk(DefaultAspect(desc.Format)),
                             .baseMipLevel = 0u,
                             .levelCount = desc.MipLevels,
                             .baseArrayLayer = 0u,
                             .layerCount = desc.ArrayLayers}};
}

void VulkanUploadContext::MakeAcquireBarriers(OwnershipBarriers& barriers) const
{
    // An acquire performs no availability operation, so its source access mask
    // says nothing, and its source stage is ignored for the same reason the
    // release's destination stage is — unless maintenance8's flag is set, when
    // it becomes the stage the acquire is ordered at and must be valid on the
    // graphics family. Copy is valid there, and matching the release's stage is
    // what makes the pair express "this hand-over happens at the copy stage"
    // rather than "somewhere between the two submissions".
    const vk::PipelineStageFlags2 srcStage =
        m_bUseAllStages ? vk::PipelineStageFlags2{vk::PipelineStageFlagBits2::eCopy}
                        : vk::PipelineStageFlags2{};

    // The destination stage stays AllCommands even with maintenance8, and that
    // is a limit of what an upload context can know rather than caution: it
    // fills resources for a caller that has not said what will read them, so
    // naming a narrower stage would be a guess. It becomes worth narrowing when
    // the acquire moves into the command list that actually consumes the
    // resource.
    //
    // The two destination access masks differ because an image's has to agree
    // with its layout. ShaderReadOnlyOptimal permits only the shader and
    // attachment read flags, and the blanket MemoryRead is not among them — a
    // mismatch the validation layer reports even though MemoryRead is defined
    // as every read access. A buffer has no layout to disagree with, so the
    // blanket flag is the honest answer there.
    for (vk::ImageMemoryBarrier2& barrier : barriers.Images)
    {
        barrier.srcStageMask = srcStage;
        barrier.srcAccessMask = vk::AccessFlagBits2::eNone;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    }

    for (vk::BufferMemoryBarrier2& barrier : barriers.Buffers)
    {
        barrier.srcStageMask = srcStage;
        barrier.srcAccessMask = vk::AccessFlagBits2::eNone;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead;
    }
}

vk::DependencyFlags VulkanUploadContext::OwnershipDependencyFlags() const
{
    return m_bUseAllStages ? vk::DependencyFlags{vk::DependencyFlagBits::
                                                     eQueueFamilyOwnershipTransferUseAllStagesKHR}
                           : vk::DependencyFlags{};
}

void VulkanUploadContext::Flush()
{
    if (m_BufferCopies.empty() && m_TextureCopies.empty())
        return;

    // Safe because every buffer from these pools was submitted and waited on by
    // the previous flush.
    m_CopyPool.reset();

    VulkanCommandList list(m_Device, *m_CopyCommandBuffer, QueueType::Copy);
    list.Begin();

    // Every texture barrier here covers the whole of each texture rather than
    // only the subresources being written. A layout is a property of a
    // subresource, and leaving the untouched ones in Undefined would make them
    // illegal to sample through a view that spans the whole resource — which is
    // the only kind of view anything here creates.
    std::vector<TextureBarrier> toCopyDst;
    toCopyDst.reserve(m_TextureCopies.size());

    // A texture that stays on this queue takes the ordinary CopyDst ->
    // ShaderResource transition. One being handed over takes the same layout
    // move as part of its release, because a transition inside an ownership
    // transfer must be named identically in both halves and is then executed
    // exactly once.
    std::vector<TextureBarrier> toShaderResource;
    OwnershipBarriers transfer;

    for (const PendingTextureCopy& copy : m_TextureCopies)
    {
        const TextureDesc* pDesc = m_Device.GetTextureDesc(copy.Destination);
        const vk::Image image = m_Device.GetImage(copy.Destination);
        if (pDesc == nullptr || !image)
            continue; // Reported by the command list when the copy is recorded.

        const uint32_t mips = pDesc->MipLevels;
        const uint32_t layers = pDesc->ArrayLayers;
        const TextureAspect aspect = DefaultAspect(pDesc->Format);

        toCopyDst.push_back(
            BarrierPresets::UndefinedToCopyDst(layers, mips, aspect).On(copy.Destination));

        if (m_Device.RequiresOwnershipTransfer(copy.Destination, m_CopyFamily, m_GraphicsFamily))
        {
            transfer.Images.push_back(MakeReleaseBarrier(image, *pDesc));
            continue;
        }

        // The preset's destination scope is emptied, and that is a requirement
        // rather than an economy. This barrier is recorded on the copy queue,
        // whose family need not support graphics — and naming a stage the
        // recording family does not have is invalid
        // (VUID-vkCmdPipelineBarrier2-dstStageMask-09676), which on this
        // machine's compute+transfer copy family means the pixel shader stage
        // the preset names.
        //
        // Emptying it is also what the situation actually is: nothing later in
        // this command buffer reads the texture. The consumer is a subsequent
        // submission, reached through the fence wait below, which is the same
        // guarantee the copies themselves rely on and is cited at that wait.
        TextureBarrier readable =
            BarrierPresets::CopyDstToShaderResource(layers, mips, aspect).On(copy.Destination);
        readable.DstStage = PipelineStage::None;
        readable.DstAccess = AccessFlags::None;
        toShaderResource.push_back(readable);
    }

    // No ownership is acquired for this one: the textures are in Undefined, so
    // there are no contents for the copy family to inherit, and the copies are
    // the first thing to touch them.
    list.Barrier(toCopyDst);

    for (const PendingBufferCopy& copy : m_BufferCopies)
    {
        list.CopyBuffer(copy.Staging, copy.Destination,
                        BufferCopyRegion{.SrcOffset = 0u,
                                         .DstOffset = copy.DestinationOffset,
                                         .Size = copy.Size});
    }

    for (const PendingTextureCopy& copy : m_TextureCopies)
    {
        for (const PendingTextureCopy::Subresource& subresource : copy.Subresources)
        {
            list.CopyBufferToTexture(
                copy.Staging, copy.Destination,
                BufferTextureCopyRegion{.BufferOffset = subresource.StagingOffset,
                                        .Aspect = subresource.Aspect,
                                        .MipLevel = subresource.MipLevel,
                                        .BaseLayer = subresource.BaseLayer,
                                        .LayerCount = subresource.LayerCount,
                                        .Extent = subresource.Extent});
        }
    }

    // Buffers are all-or-nothing: nothing about an individual buffer changes
    // the answer, so the question is asked once for the batch.
    if (BufferRequiresOwnershipTransfer(m_TransferRules, m_CopyFamily, m_GraphicsFamily))
    {
        transfer.Buffers.reserve(m_BufferCopies.size());
        for (const PendingBufferCopy& copy : m_BufferCopies)
        {
            const vk::Buffer buffer = m_Device.GetBuffer(copy.Destination);
            if (!buffer)
                continue;

            transfer.Buffers.push_back(vk::BufferMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = m_bUseAllStages
                                    ? vk::PipelineStageFlags2{vk::PipelineStageFlagBits2::eCopy}
                                    : vk::PipelineStageFlags2{},
                .dstAccessMask = vk::AccessFlagBits2::eNone,
                .srcQueueFamilyIndex = m_CopyFamily,
                .dstQueueFamilyIndex = m_GraphicsFamily,
                .buffer = buffer,
                .offset = copy.DestinationOffset,
                .size = copy.Size});
        }
    }

    list.Barrier(toShaderResource);

    const bool bHandOver = !transfer.Empty();
    if (bHandOver)
    {
        const vk::DependencyInfo releaseInfo{
            .dependencyFlags = OwnershipDependencyFlags(),
            .bufferMemoryBarrierCount = static_cast<uint32_t>(transfer.Buffers.size()),
            .pBufferMemoryBarriers = transfer.Buffers.data(),
            .imageMemoryBarrierCount = static_cast<uint32_t>(transfer.Images.size()),
            .pImageMemoryBarriers = transfer.Images.data()};
        m_CopyCommandBuffer.pipelineBarrier2(releaseInfo);
    }

    list.End();

    // The copy queue, whichever family that resolved to — a command buffer may
    // only be submitted to a queue of the family its pool was created for, and
    // this pool is the copy family's.
    const vk::CommandBuffer copyCommandBuffer = *m_CopyCommandBuffer;
    vk::raii::Queue& copyQueue = m_Device.GetQueue(QueueType::Copy);

    if (bHandOver)
    {
        MakeAcquireBarriers(transfer);

        m_AcquirePool.reset();
        m_AcquireCommandBuffer.begin(vk::CommandBufferBeginInfo{});

        const vk::DependencyInfo acquireInfo{
            .dependencyFlags = OwnershipDependencyFlags(),
            .bufferMemoryBarrierCount = static_cast<uint32_t>(transfer.Buffers.size()),
            .pBufferMemoryBarriers = transfer.Buffers.data(),
            .imageMemoryBarrierCount = static_cast<uint32_t>(transfer.Images.size()),
            .pImageMemoryBarriers = transfer.Images.data()};
        m_AcquireCommandBuffer.pipelineBarrier2(acquireInfo);
        m_AcquireCommandBuffer.end();

        const vk::Semaphore released = *m_OwnershipSemaphore;
        const vk::SubmitInfo copySubmit{.commandBufferCount = 1u,
                                        .pCommandBuffers = &copyCommandBuffer,
                                        .signalSemaphoreCount = 1u,
                                        .pSignalSemaphores = &released};
        copyQueue.submit(copySubmit);

        // The semaphore is what orders the acquire after the release, which the
        // specification requires of every ownership transfer — an execution
        // dependency between two queues cannot be expressed with a barrier.
        //
        // The wait stage stays AllCommands even under maintenance8. Narrowing it
        // is the remaining win the extension offers, and there is nothing here
        // to win: this submission contains the acquire and nothing else, so
        // there is no later work for a narrower wait to let start sooner.
        const vk::CommandBuffer acquireCommandBuffer = *m_AcquireCommandBuffer;
        const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eAllCommands;
        const vk::SubmitInfo acquireSubmit{.waitSemaphoreCount = 1u,
                                           .pWaitSemaphores = &released,
                                           .pWaitDstStageMask = &waitStage,
                                           .commandBufferCount = 1u,
                                           .pCommandBuffers = &acquireCommandBuffer};
        m_Device.GetGraphicsQueue().submit(acquireSubmit, *m_Fence);

        m_Stats.Submits += 2u;
    }
    else
    {
        const vk::SubmitInfo submitInfo{.commandBufferCount = 1u,
                                        .pCommandBuffers = &copyCommandBuffer};
        copyQueue.submit(submitInfo, *m_Fence);

        ++m_Stats.Submits;
    }

    // No barrier is needed between these copies and whatever reads the results
    // in a later submission, and that is a specification guarantee rather than
    // an assumption. A fence signal's first access scope is "all memory access
    // performed by the device", so waiting on it makes these writes available;
    // the next queue submission's second access scope is likewise all device
    // access, which makes them visible to everything it contains (Vulkan 1.4,
    // *Fences* and *Host Write Ordering Guarantees*). The queue drain this
    // replaced was defined as exactly this fence wait, so nothing is given up.
    //
    // That covers visibility, and only visibility. It says nothing about which
    // queue family owns the memory, which is why the hand-over above exists as
    // well: without it the contents would be undefined to the graphics family
    // no matter how well synchronized the two queues were.
    const vk::Result result = m_Device.GetDevice().waitForFences(
        *m_Fence, vk::True, std::numeric_limits<uint64_t>::max());
    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error(
            std::format("Rhi::IUploadContext::Flush: waiting on the upload fence failed: {}.",
                        vk::to_string(result)));
    }
    m_Device.GetDevice().resetFences(*m_Fence);

    const uint64_t flushedBytes = m_PendingBytes;
    const size_t flushedUploads = m_BufferCopies.size() + m_TextureCopies.size();

    for (const BufferHandle staging : m_Staging)
        m_Device.Destroy(staging);

    m_Staging.clear();
    m_BufferCopies.clear();
    m_TextureCopies.clear();
    m_PendingBytes = 0u;

    Core::LogMsg(Core::LogSeverity::Info, LogRhi,
                 "Upload flush: {} resource(s), {:.1f} MiB, in {}.", flushedUploads,
                 static_cast<double>(flushedBytes) / (1024.0 * 1024.0),
                 bHandOver ? "2 submissions (copy, then ownership acquire)" : "1 submission");
}
} // namespace Hikari::Rhi::Vulkan
