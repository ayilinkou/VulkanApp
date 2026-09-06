#include "vulkan/VulkanCommandList.h"

#include <array>
#include <format>
#include <vector>

#include "vulkan/VulkanConversions.h"
#include "vulkan/VulkanDevice.h"

namespace Hikari::Rhi::Vulkan
{
namespace
{
/**
 * Not an overload of ToVk: an unqualified call inside this anonymous namespace
 * would find only the name declared here and never reach the conversion table's
 * overloads in the enclosing namespace.
 */
vk::ImageMemoryBarrier2 MakeVkBarrier(vk::Image image, const TextureBarrier& desc)
{
    const vk::ImageSubresourceRange range{.aspectMask = ToVk(desc.Aspect),
                                          .baseMipLevel = desc.BaseMip,
                                          .levelCount = desc.MipCount,
                                          .baseArrayLayer = desc.BaseLayer,
                                          .layerCount = desc.LayerCount};

    // Both queue-family fields are IGNORED, which is what a barrier that stays
    // on one queue must say. A neutral TextureBarrier cannot describe a queue
    // family ownership transfer at all — the concept has no D3D12 counterpart
    // (plan D6) — so the component that submits to a second queue builds those
    // barriers itself; VulkanUploadContext is the one that does.
    return vk::ImageMemoryBarrier2{.srcStageMask = ToVk(desc.SrcStage),
                                   .srcAccessMask = ToVk(desc.SrcAccess),
                                   .dstStageMask = ToVk(desc.DstStage),
                                   .dstAccessMask = ToVk(desc.DstAccess),
                                   .oldLayout = ToVk(desc.OldLayout),
                                   .newLayout = ToVk(desc.NewLayout),
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange = range};
}

vk::BufferImageCopy MakeVkCopy(const BufferTextureCopyRegion& region)
{
    // bufferRowLength and bufferImageHeight of 0 mean "tightly packed to the
    // image extent", which is the only layout ICommandList's region describes.
    return vk::BufferImageCopy{.bufferOffset = region.BufferOffset,
                               .bufferRowLength = 0u,
                               .bufferImageHeight = 0u,
                               .imageSubresource = {.aspectMask = ToVk(region.Aspect),
                                                    .mipLevel = region.MipLevel,
                                                    .baseArrayLayer = region.BaseLayer,
                                                    .layerCount = region.LayerCount},
                               .imageOffset = {0, 0, 0},
                               .imageExtent = vk::Extent3D{
                                   region.Extent.Width, region.Extent.Height, region.Extent.Depth}};
}
} // namespace

VulkanCommandList::VulkanCommandList(VulkanDevice& device, vk::CommandBuffer cmd)
    : m_Device(device), m_Cmd(cmd)
{
}

void VulkanCommandList::SetPipeline(GraphicsPipelineHandle pipeline)
{
    m_Cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_Device.GetPipeline(pipeline));
}

void VulkanCommandList::SetPipeline(ComputePipelineHandle pipeline)
{
    m_Cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_Device.GetPipeline(pipeline));
}

void VulkanCommandList::SetComputeBindGroup(PipelineLayoutHandle layout, uint32_t slot,
                                            BindGroupHandle group)
{
    m_Cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_Device.GetPipelineLayout(layout),
                             slot, m_Device.GetDescriptorSet(group), nullptr);
}

void VulkanCommandList::Dispatch(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
{
    m_Cmd.dispatch(groupsX, groupsY, groupsZ);
}

void VulkanCommandList::SetBindGroup(PipelineLayoutHandle layout, uint32_t slot,
                                     BindGroupHandle group)
{
    m_Cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_Device.GetPipelineLayout(layout),
                             slot, m_Device.GetDescriptorSet(group), nullptr);
}

void VulkanCommandList::PushConstants(PipelineLayoutHandle layout, ShaderStage stages,
                                      uint32_t offset, std::span<const std::byte> data)
{
    m_Cmd.pushConstants(m_Device.GetPipelineLayout(layout), ToVk(stages), offset,
                        static_cast<uint32_t>(data.size()), data.data());
}

void VulkanCommandList::BeginRendering(const RenderingDesc& desc)
{
    // Fixed capacity rather than an allocation per pass: this runs several times
    // a frame, and a renderer wanting more than this many colour targets at once
    // has outgrown the assumption rather than hit a limit worth growing.
    constexpr size_t kMaxRenderTargets = 8u;
    if (desc.RenderTargets.size() > kMaxRenderTargets)
        throw std::runtime_error(
            "Rhi::VulkanCommandList::BeginRendering: too many render targets.");

    std::array<vk::RenderingAttachmentInfo, kMaxRenderTargets> colors{};
    for (size_t i = 0; i < desc.RenderTargets.size(); i++)
    {
        const RenderTarget& target = desc.RenderTargets[i];
        colors[i] = vk::RenderingAttachmentInfo{
            .imageView = m_Device.GetImageView(target.View),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = ToVkLoadOp(target.Load),
            .storeOp = ToVkStoreOp(target.Store),
            .clearValue = vk::ClearColorValue(target.ClearColor[0], target.ClearColor[1],
                                              target.ClearColor[2], target.ClearColor[3])};
    }

    vk::RenderingAttachmentInfo depth{};
    if (desc.pDepthStencil != nullptr)
    {
        if (desc.pDepthStencil->bReadOnly && desc.pDepthStencil->Store != StoreOp::Preserve)
        {
            throw std::runtime_error("Rhi::VulkanCommandList::BeginRendering: a read-only "
                                     "depth target cannot discard contents it never wrote.");
        }

        depth = vk::RenderingAttachmentInfo{
            .imageView = m_Device.GetImageView(desc.pDepthStencil->View),
            .imageLayout = desc.pDepthStencil->bReadOnly ? vk::ImageLayout::eDepthReadOnlyOptimal
                                                         : vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = ToVkLoadOp(desc.pDepthStencil->Load),
            // A read-only pass writes nothing, so the store is NONE rather than
            // STORE: the layout forbids the write that STORE would claim.
            .storeOp = desc.pDepthStencil->bReadOnly ? vk::AttachmentStoreOp::eNone
                                                     : ToVkStoreOp(desc.pDepthStencil->Store),
            .clearValue = vk::ClearDepthStencilValue(desc.pDepthStencil->ClearDepth,
                                                     desc.pDepthStencil->ClearStencil)};
    }

    const vk::RenderingInfo renderingInfo{
        .renderArea =
            vk::Rect2D{vk::Offset2D{desc.RenderArea.Offset.X, desc.RenderArea.Offset.Y},
                       vk::Extent2D{desc.RenderArea.Extent.Width, desc.RenderArea.Extent.Height}},
        .layerCount = 1u,
        .colorAttachmentCount = static_cast<uint32_t>(desc.RenderTargets.size()),
        .pColorAttachments = colors.data(),
        .pDepthAttachment = desc.pDepthStencil != nullptr ? &depth : nullptr};

    m_Cmd.beginRendering(renderingInfo);
}

void VulkanCommandList::EndRendering()
{
    m_Cmd.endRendering();
}

void VulkanCommandList::SetViewport(const Viewport& viewport)
{
    m_Cmd.setViewport(0u, vk::Viewport{viewport.X, viewport.Y, viewport.Width, viewport.Height,
                                       viewport.MinDepth, viewport.MaxDepth});
}

void VulkanCommandList::SetScissor(const Rect2D& rect)
{
    m_Cmd.setScissor(0u, vk::Rect2D{vk::Offset2D{rect.Offset.X, rect.Offset.Y},
                                    vk::Extent2D{rect.Extent.Width, rect.Extent.Height}});
}

void VulkanCommandList::Begin()
{
    m_Cmd.begin(vk::CommandBufferBeginInfo{});
}

void VulkanCommandList::End()
{
    m_Cmd.end();
}

BarrierCounts VulkanCommandList::Barrier(std::span<const TextureBarrier> barriers)
{
    if (barriers.empty())
        return {};

    std::vector<vk::ImageMemoryBarrier2> converted;
    converted.reserve(barriers.size());

    for (const TextureBarrier& barrier : barriers)
    {
        const vk::Image image = m_Device.GetImage(barrier.Texture);
        if (!image)
        {
            // Dropped rather than recorded against a null image, which the
            // driver would reject. The barrier counts in the run report go down
            // by one when this happens, which is the point of reporting them.
            m_Device.ReportStaleHandle(
                std::format("Rhi::ICommandList::Barrier: texture handle {:#010x} is stale or was "
                            "never valid; the barrier was not recorded.",
                            barrier.Texture.Value));
            continue;
        }

        converted.push_back(MakeVkBarrier(image, barrier));
    }

    if (converted.empty())
        return {};

    const vk::DependencyInfo dependencyInfo{.imageMemoryBarrierCount =
                                                static_cast<uint32_t>(converted.size()),
                                            .pImageMemoryBarriers = converted.data()};
    m_Cmd.pipelineBarrier2(dependencyInfo);

    return BarrierCounts{.Barriers = static_cast<uint32_t>(converted.size()), .Calls = 1u};
}

BarrierCounts VulkanCommandList::Barrier(const TextureBarrier& barrier)
{
    const std::array one{barrier};
    return Barrier(one);
}

void VulkanCommandList::CopyBuffer(BufferHandle source, BufferHandle destination,
                                   const BufferCopyRegion& region)
{
    const vk::Buffer src = m_Device.GetBuffer(source);
    const vk::Buffer dst = m_Device.GetBuffer(destination);
    if (!src || !dst)
    {
        m_Device.ReportStaleHandle(
            std::format("Rhi::ICommandList::CopyBuffer: buffer handle {:#010x} or {:#010x} is "
                        "stale; the copy was not recorded.",
                        source.Value, destination.Value));
        return;
    }

    m_Cmd.copyBuffer(src, dst,
                     vk::BufferCopy{.srcOffset = region.SrcOffset,
                                    .dstOffset = region.DstOffset,
                                    .size = region.Size});
}

void VulkanCommandList::CopyBufferToTexture(BufferHandle source, TextureHandle destination,
                                            const BufferTextureCopyRegion& region)
{
    const vk::Buffer src = m_Device.GetBuffer(source);
    const vk::Image dst = m_Device.GetImage(destination);
    if (!src || !dst)
    {
        m_Device.ReportStaleHandle(
            std::format("Rhi::ICommandList::CopyBufferToTexture: buffer handle {:#010x} or "
                        "texture handle {:#010x} is stale; the copy was not recorded.",
                        source.Value, destination.Value));
        return;
    }

    // The destination has to already be in CopyDst; see ICommandList's comment
    // on why a copy transitions nothing itself.
    m_Cmd.copyBufferToImage(src, dst, vk::ImageLayout::eTransferDstOptimal, MakeVkCopy(region));
}

void VulkanCommandList::CopyTextureToBuffer(TextureHandle source, BufferHandle destination,
                                            const BufferTextureCopyRegion& region)
{
    const vk::Image src = m_Device.GetImage(source);
    const vk::Buffer dst = m_Device.GetBuffer(destination);
    if (!src || !dst)
    {
        m_Device.ReportStaleHandle(
            std::format("Rhi::ICommandList::CopyTextureToBuffer: texture handle {:#010x} or "
                        "buffer handle {:#010x} is stale; the copy was not recorded.",
                        source.Value, destination.Value));
        return;
    }

    m_Cmd.copyImageToBuffer(src, vk::ImageLayout::eTransferSrcOptimal, dst, MakeVkCopy(region));
}
} // namespace Hikari::Rhi::Vulkan
