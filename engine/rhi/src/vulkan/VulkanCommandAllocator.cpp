#include "VulkanCommandAllocator.h"

#include <format>

#include <rhi/vulkan/DebugNames.h>

#include "VulkanDevice.h"

namespace Hikari::Rhi::Vulkan
{
VulkanCommandAllocator::VulkanCommandAllocator(VulkanDevice& device,
                                               const CommandAllocatorDesc& desc)
    : m_Device(device), m_DebugName(desc.DebugName)
{
    const vk::CommandPoolCreateInfo createInfo{.queueFamilyIndex =
                                                   device.GetQueueFamily(desc.Queue)};

    m_Pool = vk::raii::CommandPool(device.GetDevice(), createInfo);
    SetVkDebugName(device.GetDevice(), *m_Pool, vk::ObjectType::eCommandPool, m_DebugName.c_str());
}

ICommandList& VulkanCommandAllocator::Acquire()
{
    if (m_Acquired == m_Lists.size())
    {
        const vk::CommandBufferAllocateInfo allocInfo{.commandPool = *m_Pool,
                                                      .level = vk::CommandBufferLevel::ePrimary,
                                                      .commandBufferCount = 1u};

        m_Buffers.push_back(
            std::move(vk::raii::CommandBuffers(m_Device.GetDevice(), allocInfo).front()));
        SetVkDebugName(m_Device.GetDevice(), *m_Buffers.back(), vk::ObjectType::eCommandBuffer,
                       std::format("{} [{}]", m_DebugName, m_Lists.size()).c_str());

        m_Lists.push_back(std::make_unique<VulkanCommandList>(m_Device, *m_Buffers.back()));
    }

    return *m_Lists[m_Acquired++];
}

void VulkanCommandAllocator::Reset()
{
    m_Pool.reset();
    m_Acquired = 0u;
}
} // namespace Hikari::Rhi::Vulkan
