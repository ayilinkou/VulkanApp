#pragma once

#include "vulkan/vulkan_raii.hpp"

namespace Hikari::Rhi::Vulkan
{
/** What a PipelineLayoutHandle resolves to. */
struct VulkanPipelineLayout
{
    vk::raii::PipelineLayout Layout = nullptr;
};

/** What a ShaderModuleHandle resolves to. */
struct VulkanShaderModule
{
    vk::raii::ShaderModule Module = nullptr;
};

/** What a GraphicsPipelineHandle resolves to. */
struct VulkanGraphicsPipeline
{
    vk::raii::Pipeline Pipeline = nullptr;
};

/** What a ComputePipelineHandle resolves to. Separate pool, same vk::Pipeline. */
struct VulkanComputePipeline
{
    vk::raii::Pipeline Pipeline = nullptr;
};
} // namespace Hikari::Rhi::Vulkan
