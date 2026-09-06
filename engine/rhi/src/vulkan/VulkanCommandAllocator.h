#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vulkan/vulkan_raii.hpp"

#include <rhi/ICommandAllocator.h>

#include "VulkanCommandList.h"

namespace Hikari::Rhi::Vulkan
{
class VulkanDevice;

/**
 * ICommandAllocator over a VkCommandPool.
 *
 * The pool is created without eResetCommandBuffer: Reset() recycles the whole
 * pool in one call, which is both cheaper than resetting buffers individually
 * and the only shape D3D12 has, since an allocator there is reset as a unit and
 * has no per-list equivalent.
 */
class VulkanCommandAllocator final : public ICommandAllocator
{
public:
    VulkanCommandAllocator(VulkanDevice& device, const CommandAllocatorDesc& desc);

    ICommandList& Acquire() override;
    void Reset() override;

private:
    VulkanDevice& m_Device;
    std::string m_DebugName;

    /** Fixed at creation, and stamped on every list this hands out. */
    QueueType m_Queue;

    vk::raii::CommandPool m_Pool = nullptr;

    /**
     * Buffers and their wrappers are kept across resets rather than freed and
     * reallocated. Resetting a pool returns its buffers to the initial state
     * and keeps their memory, so allocating again each frame would hand back
     * the same storage by a longer route -- and would churn the wrappers, which
     * callers hold references to for the length of a frame.
     *
     * The two vectors are index-aligned, and m_Acquired is how far into them
     * this recycle has reached.
     */
    std::vector<vk::raii::CommandBuffer> m_Buffers;
    std::vector<std::unique_ptr<VulkanCommandList>> m_Lists;
    size_t m_Acquired = 0u;
};
} // namespace Hikari::Rhi::Vulkan
