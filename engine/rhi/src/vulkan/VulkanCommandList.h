#pragma once

#include <span>

#include "vulkan/vulkan.hpp"

#include <rhi/Barrier.h>
#include <rhi/Handles.h>
#include <rhi/ICommandList.h>

namespace Hikari::Rhi::Vulkan
{
class VulkanDevice;

/**
 * ICommandList over a VkCommandBuffer the caller owns.
 *
 * Non-owning on purpose: the renderer still allocates its command buffers from
 * its own pools and submits them itself, because draw recording stays Vulkan-
 * side until Stage 8. This wraps one so the commands that *can* be expressed
 * neutrally — barriers and copies — go through the neutral interface, while the
 * rest keeps using the raw buffer.
 *
 * Construct one wherever a command buffer is being recorded; it holds nothing
 * but two references and is cheaper than the first command recorded into it.
 * Code outside the module gets one from Rhi::Vulkan::WrapCommandList().
 */
class VulkanCommandList final : public ICommandList
{
public:
    VulkanCommandList(VulkanDevice& device, vk::CommandBuffer cmd);

    void Begin() override;
    void End() override;

    void SetPipeline(GraphicsPipelineHandle pipeline) override;
    void SetBindGroup(PipelineLayoutHandle layout, uint32_t slot, BindGroupHandle group) override;

    void PushConstants(PipelineLayoutHandle layout, ShaderStage stages, uint32_t offset,
                       std::span<const std::byte> data) override;

    void BeginRendering(const RenderingDesc& desc) override;
    void EndRendering() override;
    void SetViewport(const Viewport& viewport) override;
    void SetScissor(const Rect2D& rect) override;

    BarrierCounts Barrier(std::span<const TextureBarrier> barriers) override;
    BarrierCounts Barrier(const TextureBarrier& barrier) override;

    void CopyBuffer(BufferHandle source, BufferHandle destination,
                    const BufferCopyRegion& region) override;
    void CopyBufferToTexture(BufferHandle source, TextureHandle destination,
                             const BufferTextureCopyRegion& region) override;
    void CopyTextureToBuffer(TextureHandle source, BufferHandle destination,
                             const BufferTextureCopyRegion& region) override;

private:
    VulkanDevice& m_Device;
    vk::CommandBuffer m_Cmd;

public:
    /** The buffer this records into, for the native escape hatch. */
    vk::CommandBuffer Native() const { return m_Cmd; }
};
} // namespace Hikari::Rhi::Vulkan
