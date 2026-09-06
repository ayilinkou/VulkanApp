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

vk::raii::Device& GetDevice(IDevice& device)
{
    return AsVulkan(device).GetDevice();
}

vk::raii::Queue& GetGraphicsQueue(IDevice& device)
{
    return AsVulkan(device).GetGraphicsQueue();
}

uint32_t GetGraphicsQueueFamily(IDevice& device)
{
    return AsVulkan(device).GetQueueFamily(QueueType::Graphics);
}

vk::Buffer GetBuffer(IDevice& device, BufferHandle handle)
{
    return AsVulkan(device).GetBuffer(handle);
}

vk::ImageView GetImageView(IDevice& device, TextureViewHandle handle)
{
    return AsVulkan(device).GetImageView(handle);
}

vk::Sampler GetSampler(IDevice& device, SamplerHandle handle)
{
    return AsVulkan(device).GetSampler(handle);
}

vk::DescriptorSetLayout GetDescriptorSetLayout(IDevice& device, BindGroupLayoutHandle handle)
{
    return static_cast<VulkanDevice&>(device).GetDescriptorSetLayout(handle);
}

vk::DescriptorSet GetDescriptorSet(IDevice& device, BindGroupHandle handle)
{
    return static_cast<VulkanDevice&>(device).GetDescriptorSet(handle);
}

vk::Semaphore GetSemaphore(IDevice& device, SemaphoreHandle handle)
{
    return AsVulkan(device).GetSemaphore(handle);
}

std::unique_ptr<ICommandList> WrapCommandList(IDevice& device, vk::CommandBuffer cmd)
{
    return std::make_unique<VulkanCommandList>(AsVulkan(device), cmd);
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
