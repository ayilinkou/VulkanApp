#include <rhi/vulkan/VulkanNative.h>

#include <stdexcept>

#include "vulkan/VulkanCommandList.h"
#include "vulkan/VulkanConversions.h"
#include "vulkan/VulkanDevice.h"
#include "vulkan/VulkanPipelineCache.h"

namespace Hikari::Rhi::Vulkan
{
namespace
{
/**
 * A dynamic_cast rather than a static_cast because getting this wrong is
 * undefined behaviour rather than a diagnosable error, and the cost is
 * irrelevant: these are called a handful of times during startup, never per
 * frame. If a second backend ever exists, this is where passing the wrong
 * device type surfaces as an exception instead of as memory corruption.
 */
VulkanDevice& AsVulkan(IDevice& device)
{
    auto* pVulkanDevice = dynamic_cast<VulkanDevice*>(&device);
    if (!pVulkanDevice)
        throw std::runtime_error("Rhi::Vulkan native accessor used on a non-Vulkan device!");
    return *pVulkanDevice;
}
} // namespace

NativeDevice GetNative(IDevice& device)
{
    VulkanDevice& vulkanDevice = AsVulkan(device);
    return NativeDevice{.Instance = *vulkanDevice.GetInstance(),
                        .PhysicalDevice = *vulkanDevice.GetPhysicalDevice(),
                        .Device = *vulkanDevice.GetDevice(),
                        .GraphicsQueue = *vulkanDevice.GetGraphicsQueue(),
                        .GraphicsQueueFamily = vulkanDevice.GetQueueFamily(QueueType::Graphics),
                        .ApiVersion = vulkanDevice.GetApiVersion()};
}

vk::raii::PhysicalDevice& GetPhysicalDevice(IDevice& device)
{
    return AsVulkan(device).GetPhysicalDevice();
}

vk::CommandBuffer GetNative(ICommandList& commandList)
{
    return static_cast<VulkanCommandList&>(commandList).Native();
}

vk::Format GetNativeFormat(Format format)
{
    return ToVk(format);
}

Format FromNativeFormat(vk::Format format)
{
    return FromVk(format);
}

VkPipelineCache GetNativePipelineCache(IPipelineCache& cache)
{
    return *ToVulkan(cache).Get();
}
} // namespace Hikari::Rhi::Vulkan
