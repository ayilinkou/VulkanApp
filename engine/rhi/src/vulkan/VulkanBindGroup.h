#pragma once

#include "vulkan/vulkan_raii.hpp"

namespace Hikari::Rhi::Vulkan
{
/** What a BindGroupLayoutHandle resolves to. */
struct VulkanBindGroupLayout
{
    vk::raii::DescriptorSetLayout Layout = nullptr;
};

/**
 * What a BindGroupHandle resolves to.
 *
 * The set frees itself back to the pool it came from, so the device's allocator
 * must outlive every group -- which it does, both being members destroyed in
 * declaration order after the pools they name.
 */
struct VulkanBindGroup
{
    vk::raii::DescriptorSet Set = nullptr;
};
} // namespace Hikari::Rhi::Vulkan
