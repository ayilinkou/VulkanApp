#pragma once

#include "vulkan/vulkan_raii.hpp"

namespace Hikari::Rhi::Vulkan
{
/**
 * What a FenceHandle resolves to: a timeline semaphore, not a VkFence.
 *
 * The neutral fence is a monotonic counter because that is what D3D12 offers
 * (plan D5), and Vulkan's counter is the timeline semaphore. A VkFence cannot
 * stand in: it is a binary flag that has to be reset before reuse, so it can
 * neither be waited on for a value it has already passed nor be signalled twice
 * without host intervention -- both of which the frame loop does.
 *
 * A wrapper struct for the same reason VulkanSemaphore is one: Core::HandlePool
 * needs a default-constructible payload, and vk::raii::Semaphore has no default
 * constructor.
 */
struct VulkanFence
{
    vk::raii::Semaphore Timeline = nullptr;
};
} // namespace Hikari::Rhi::Vulkan
