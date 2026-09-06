#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "vulkan/vulkan_raii.hpp"

namespace Hikari::Rhi::Vulkan
{

/**
 * Hands out descriptor sets from a chain of pools that grows on demand, so a
 * scene with more materials than expected loads instead of aborting.
 *
 * Vulkan-shaped on purpose. Descriptor sets and D3D12's root signatures plus
 * descriptor heaps have no cheap common denominator, and bindless converges the
 * two later anyway — so this is kept isolated rather than abstracted, and
 * replacing it stays a contained change.
 */
class DescriptorAllocator
{
public:
    /**
     * descriptorsPerSet counts descriptors in *one* set, not in a pool: the
     * allocator scales it by the pool's set capacity. Stating it per set is
     * what keeps growth correct — a caller cannot raise the set count and
     * forget to raise the descriptor counts with it.
     */
    DescriptorAllocator(vk::raii::Device& device,
                        std::span<const vk::DescriptorPoolSize> descriptorsPerSet,
                        uint32_t initialSetCapacity, std::string debugName);

    /**
     * Not copyable: it owns pools whose handles the sets it handed out still
     * name.
     */
    DescriptorAllocator(const DescriptorAllocator&) = delete;
    DescriptorAllocator& operator=(const DescriptorAllocator&) = delete;

    /**
     * The returned set frees itself back to its pool, so it must not outlive
     * this allocator.
     */
    [[nodiscard]] vk::raii::DescriptorSet Allocate(vk::DescriptorSetLayout layout);

private:
    [[nodiscard]] vk::raii::DescriptorSet AllocateFromNewestPool(vk::DescriptorSetLayout layout);
    void AddPool(uint32_t setCapacity);
    void Grow();

private:
    vk::raii::Device& m_Device;

    /**
     * std::vector is safe despite handing out self-freeing sets: a
     * vk::raii::DescriptorSet stores its VkDescriptorPool by value, not as a
     * pointer to the owning object, so growing this vector moves the pool
     * objects without dangling anything already allocated.
     */
    std::vector<vk::raii::DescriptorPool> m_Pools;

    std::vector<vk::DescriptorPoolSize> m_DescriptorsPerSet;

    /**
     * Capacity of the newest pool, and how much of it is spoken for. Only the
     * newest pool is ever allocated from, so one counter covers it. Sets freed
     * back into it are not counted back, which can grow one pool early — cheap
     * next to tracking every free.
     */
    uint32_t m_SetCapacity = 0u;
    uint32_t m_SetsFromNewestPool = 0u;
    const uint32_t m_InitialSetCapacity;

    const std::string m_DebugName;
};
} // namespace Hikari::Rhi::Vulkan
