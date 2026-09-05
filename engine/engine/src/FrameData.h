#pragma once

#include "vulkan/vulkan_raii.hpp"

#include <rhi/Handles.h>
#include <rhi/UniqueHandle.h>

#include "Texture.h"

struct FrameData
{
    vk::raii::CommandPool DrawLayoutCommandPool = nullptr;
    vk::raii::CommandPool OpaqueCommandPool = nullptr;
    vk::raii::CommandPool CloudCommandPool = nullptr;
    vk::raii::CommandPool TransparentCommandPool = nullptr;
    vk::raii::CommandPool CompositeCommandPool = nullptr;
    vk::raii::CommandPool ImGuiCommandPool = nullptr;
    vk::raii::CommandPool FinalLayoutCommandPool = nullptr;
    vk::raii::CommandBuffer DrawLayoutCommandBuffer = nullptr;
    vk::raii::CommandBuffer OpaqueCommandBuffer = nullptr;
    vk::raii::CommandBuffer CloudCommandBuffer = nullptr;
    vk::raii::CommandBuffer TransparentCommandBuffer = nullptr;
    vk::raii::CommandBuffer CompositeCommandBuffer = nullptr;
    vk::raii::CommandBuffer ImGuiCommandBuffer = nullptr;
    vk::raii::CommandBuffer FinalLayoutCommandBuffer = nullptr;
    vk::raii::Fence DrawFence = nullptr;
    vk::raii::DescriptorSet GlobalBufferDescriptorSet = nullptr;
    vk::raii::DescriptorSet CompositeDescriptorSet = nullptr;
    vk::raii::DescriptorSet DepthBufferDescriptorSet = nullptr;
    Texture OpaqueTexture;
    Texture AccumTexture;
    Texture RevealageTexture;
    Texture DepthTexture;
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::BufferHandle> GlobalBuffer;
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::BufferHandle> InstanceBuffer;
};
