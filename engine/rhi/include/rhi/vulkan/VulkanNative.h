#pragma once

#include <cstdint>
#include <memory>

#include "vulkan/vulkan_raii.hpp"

#include <rhi/Handles.h>
#include <rhi/ICommandList.h>
#include <rhi/IDevice.h>
#include <rhi/PipelineCache.h>
#include <rhi/RhiTypes.h>

/**
 * The one sanctioned way to get a Vulkan handle out of an IDevice.
 *
 * It exists because ImGui's Vulkan backend needs raw instance, physical device,
 * device, queue family index and queue handles, and wrapping ImGui to avoid that
 * is not worth doing. Anything that reaches in here is by definition
 * backend-specific and will not compile against a second backend — which is the
 * point: the leak is *listed*, in one file, rather than spread through the
 * renderer where nobody can count it.
 *
 * What is left is permanent by design. ImGui's Vulkan backend takes raw
 * instance, device and queue handles, a VkCommandBuffer by value, a VkFormat and
 * a VkPipelineCache; a D3D12 build answers that with a sibling file rather than
 * an edit. The physical device is here for one query the neutral API cannot yet
 * answer — which depth formats this hardware supports — and goes when it can.
 *
 * The transitional half is gone: the RAII accessors, the buffer, view, sampler
 * and semaphore resolvers, and WrapCommandList all existed because the renderer
 * built Vulkan objects itself, and it no longer does.
 */
namespace Hikari::Rhi::Vulkan
{
/** Raw handles, for C APIs such as ImGui that take them by value. */
struct NativeDevice
{
    VkInstance Instance = VK_NULL_HANDLE;
    VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
    VkDevice Device = VK_NULL_HANDLE;
    VkQueue GraphicsQueue = VK_NULL_HANDLE;
    uint32_t GraphicsQueueFamily = ~0u;
    uint32_t ApiVersion = 0;
};

NativeDevice GetNative(IDevice& device);

/**
 * The physical device, for the one question the neutral API cannot yet answer:
 * which depth formats this hardware actually supports. The renderer picks its
 * depth format by querying VkFormatProperties, and there is no neutral
 * "is this format usable for this" on IDevice. This goes when there is.
 */
vk::raii::PhysicalDevice& GetPhysicalDevice(IDevice& device);

/**
 * The buffer a command list records into.
 *
 * The inverse of WrapCommandList, and it exists for the same kind of caller:
 * ImGui's Vulkan backend takes a VkCommandBuffer by value, and there is no
 * neutral shape for that until the RHI records draws itself.
 */
vk::CommandBuffer GetNative(ICommandList& commandList);

/**
 * vk::Format for a neutral one and back.
 *
 * Needed because format selection and pipeline creation are still Vulkan-shaped
 * in the renderer: the depth format is chosen by querying
 * VkFormatProperties, and PipelineBuilder takes vk::Format for the dynamic-
 * rendering attachment formats (D8). Both move behind the neutral API when
 * pipeline creation does. Throws on a vk::Format with no neutral equivalent —
 * Rhi::Format is curated, not a mirror (D11).
 */
vk::Format GetNativeFormat(Format format);
Format FromNativeFormat(vk::Format format);

/**
 * The VkPipelineCache behind a neutral cache.
 *
 * The same ImGui-shaped hole as NativeDevice, and there for the same reason:
 * ImGui_ImplVulkan_InitInfo::PipelineCache is a raw handle, and ImGui builds its
 * own pipelines without going through anything this module offers. Every other
 * caller hands the neutral cache to a builder instead.
 */
VkPipelineCache GetNativePipelineCache(IPipelineCache& cache);
} // namespace Hikari::Rhi::Vulkan
