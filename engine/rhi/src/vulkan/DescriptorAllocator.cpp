#include "vulkan/DescriptorAllocator.h"

#include <algorithm>
#include <format>
#include <utility>

#include <core/Log.h>

#include "vulkan/DebugNames.h"

namespace Hikari::Rhi::Vulkan
{
constexpr Core::LogCategory LogRhi("RHI");

DescriptorAllocator::DescriptorAllocator(vk::raii::Device& device,
                                         std::span<const vk::DescriptorPoolSize> descriptorsPerSet,
                                         uint32_t initialSetCapacity, std::string debugName)
    : m_Device(device), m_DescriptorsPerSet(descriptorsPerSet.begin(), descriptorsPerSet.end()),
      m_InitialSetCapacity(std::max(initialSetCapacity, 1u)), m_DebugName(std::move(debugName))
{
    AddPool(m_InitialSetCapacity);
}

void DescriptorAllocator::AddPool(uint32_t setCapacity)
{
    std::vector<vk::DescriptorPoolSize> poolSizes(m_DescriptorsPerSet);
    for (vk::DescriptorPoolSize& size : poolSizes)
        size.descriptorCount *= setCapacity;

    vk::DescriptorPoolCreateInfo createInfo{
        // Mandatory rather than a preference: every set handed out is a
        // vk::raii::DescriptorSet, which frees itself on destruction, and
        // vkFreeDescriptorSets requires the pool it came from to carry this
        // flag (VUID-vkFreeDescriptorSets-descriptorPool-00312).
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = setCapacity,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()};

    m_Pools.emplace_back(m_Device, createInfo);
    m_SetCapacity = setCapacity;
    m_SetsFromNewestPool = 0u;

    SetVkDebugName(m_Device, *m_Pools.back(), vk::ObjectType::eDescriptorPool,
                   std::format("{} Descriptor Pool {}", m_DebugName, m_Pools.size() - 1).c_str());
}

void DescriptorAllocator::Grow()
{
    // Sets freed back into the older pools are never reclaimed — allocation
    // only ever touches the newest one. Growing geometrically is what keeps
    // that affordable; a per-pool free list would cost more bookkeeping than
    // the handful of sets it would recover. The +1 floor keeps a capacity of
    // one making progress.
    AddPool(m_SetCapacity + std::max(m_SetCapacity / 2u, 1u));

    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "{} descriptor pool grew to {} sets ({} pools)",
                 m_DebugName, m_SetCapacity, m_Pools.size());
}

vk::raii::DescriptorSet DescriptorAllocator::AllocateFromNewestPool(vk::DescriptorSetLayout layout)
{
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *m_Pools.back(), .descriptorSetCount = 1u, .pSetLayouts = &layout};

    vk::raii::DescriptorSet set = std::move(m_Device.allocateDescriptorSets(allocInfo).front());
    ++m_SetsFromNewestPool;
    return set;
}

vk::raii::DescriptorSet DescriptorAllocator::Allocate(vk::DescriptorSetLayout layout)
{
    // Grow before asking a pool for one set more than it holds. The retry below
    // would cover it, but the validation layers report every
    // VK_ERROR_OUT_OF_POOL_MEMORY as a warning, and the run report's warning
    // count is a signal worth keeping meaningful — routine growth should not
    // raise it.
    if (m_SetsFromNewestPool >= m_SetCapacity)
        Grow();

    try
    {
        return AllocateFromNewestPool(layout);
    }
    catch (const vk::SystemError&)
    {
        // Any error means "this pool is spent", not just eErrorOutOfPoolMemory
        // and eErrorFragmentedPool. The specification asks for exactly that:
        // VK_ERROR_FRAGMENTED_POOL was added late in Vulkan 1.0, so drivers
        // written against earlier patch versions may report fragmentation as
        // something else, and applications "should assume that the allocation
        // failed due to fragmentation, and create a new descriptor pool". A
        // genuine out-of-memory surfaces from the retry below instead.
    }

    Grow();
    return AllocateFromNewestPool(layout);
}
} // namespace Hikari::Rhi::Vulkan
