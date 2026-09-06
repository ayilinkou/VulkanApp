#include "vulkan/VulkanDevice.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <format>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <core/Log.h>

#include <rhi/vulkan/DebugNames.h>

#include "vulkan/OffscreenTarget.h"
#include "vulkan/SwapchainTarget.h"
#include "vulkan/VulkanCommandAllocator.h"
#include "vulkan/VulkanCommandList.h"
#include "vulkan/VulkanConversions.h"
#include "vulkan/VulkanPipelineCache.h"
#include "vulkan/VulkanUploadContext.h"

namespace Hikari::Rhi::Vulkan
{
constexpr Core::LogCategory LogRhi("RHI");
namespace
{

/**
 * The tiling every texture this device allocates is created with. Named rather
 * than written inline because whether an image needs a queue family ownership
 * transfer depends on it, and the two answers have to come from one place.
 */
constexpr vk::ImageTiling kTextureTiling = vk::ImageTiling::eOptimal;

constexpr const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";

/** "family 1 (dedicated)", "family 0", or "none", for the startup log. */
std::string DescribeFamily(const QueueFamilies& families, QueueType role)
{
    const uint32_t index = families.Get(role);
    if (index == QueueFamilies::kInvalid)
        return "none";

    return std::format("family {}{}", index, families.IsDedicated(role) ? " (dedicated)" : "");
}

/**
 * Rejects the descriptions Vulkan would reject anyway, but with a message that
 * names the caller's field rather than a VUID. Every one of these is a
 * programming error rather than a runtime condition, so they throw.
 */
void ValidateTextureDesc(const TextureDesc& desc)
{
    const auto fail = [&desc](std::string_view why)
    {
        throw std::runtime_error(
            std::format("Rhi::IDevice::CreateTexture('{}'): {}", desc.DebugName, why));
    };

    if (desc.Format == Rhi::Format::Undefined)
        fail("no format.");

    if (desc.Extent.Width == 0u || desc.Extent.Height == 0u || desc.Extent.Depth == 0u)
        fail("every extent must be at least 1.");

    if (desc.MipLevels == 0u || desc.ArrayLayers == 0u)
        fail("MipLevels and ArrayLayers must be at least 1.");

    // Depth is the third dimension of a 3D texture and the array is the layers;
    // Vulkan has no 3D array images, and mixing the two is the classic way to
    // describe a cubemap as six slices deep instead of six layers wide.
    if (desc.Dimension == TextureDimension::Texture3D && desc.ArrayLayers != 1u)
        fail("a 3D texture cannot have array layers.");

    if (desc.Dimension == TextureDimension::Texture2D && desc.Extent.Depth != 1u)
        fail("a 2D texture must have a depth of 1; use ArrayLayers for slices.");

    if (desc.bCubeCompatible &&
        (desc.Dimension != TextureDimension::Texture2D || desc.ArrayLayers % 6u != 0u))
        fail("a cube-compatible texture must be 2D with a multiple of 6 array layers.");
}
/**
 * Adds the validation layer vcpkg installed to the loader's layer search path.
 *
 * Without this the layer has to come from a system-wide Vulkan SDK, which is the
 * only thing a Linux or Windows build would still need one for. VK_ADD_LAYER_PATH
 * adds to the standard search paths rather than replacing them, and the loader
 * reads it when layers are enumerated rather than at its own startup — so setting
 * it here, before the first enumeration in CreateInstance, is early enough. That
 * is not true of every loader variable: VK_LOADER_* settings are read during
 * loader initialisation, which has already happened by the time any of this runs.
 *
 * Whatever the environment already asked for is kept and comes first, so a
 * developer pointing at their own layers still gets them.
 */
void AddBundledLayerPath()
{
#if defined(_WIN32)
    constexpr char kSeparator = ';';
#else
    constexpr char kSeparator = ':';
#endif

    std::string value = HIKARI_VULKAN_LAYER_PATH;

#if defined(_WIN32)
    // getenv is flagged unsafe by MSVC (C4996); _dupenv_s is its thread-safe replacement.
    char* existing = nullptr;
    std::size_t existingLen = 0;
    _dupenv_s(&existing, &existingLen, "VK_ADD_LAYER_PATH");
    if (existing != nullptr && existing[0] != '\0')
        value = std::string(existing) + kSeparator + value;
    std::free(existing);

    _putenv_s("VK_ADD_LAYER_PATH", value.c_str());
#else
    const char* existing = std::getenv("VK_ADD_LAYER_PATH");
    if (existing != nullptr && existing[0] != '\0')
        value = std::string(existing) + kSeparator + value;

    setenv("VK_ADD_LAYER_PATH", value.c_str(), 1);
#endif
}

} // namespace

VulkanDevice::VulkanDevice(const DeviceDesc& desc)
    : m_OwnedDiagnostics(desc.pDiagnostics ? nullptr : std::make_unique<Diagnostics>()),
      m_pDiagnostics(desc.pDiagnostics ? desc.pDiagnostics : m_OwnedDiagnostics.get())
{
    CreateInstance(desc);
    SetupDebugMessenger(desc);
    CreateSurface(desc.Requirements);
    PickPhysicalDevice(desc.Requirements);
    SelectOptionalExtensions(desc);
    FindQueueFamilies(desc);
    CreateLogicalDevice(desc.Requirements);

    m_Allocator = VulkanAllocator(m_Instance, m_PhysicalDevice, m_Device, kApiVersion);

    // Vulkan's clip space has Y pointing down relative to what GLM produces.
    m_Caps.bFlipClipSpaceY = true;
    m_Caps.bPresentSupported = desc.Requirements.bPresent;
    m_Caps.bHasDedicatedComputeQueue = m_QueueFamilies.IsDedicated(QueueType::Compute);
    m_Caps.bHasDedicatedCopyQueue = m_QueueFamilies.IsDedicated(QueueType::Copy);
}

VulkanDevice::~VulkanDevice()
{
    // A resource still alive here was never destroyed. It is not a crash — the
    // pools free their payloads on the way out, and they are declared after the
    // allocator and the device so that happens in the right order — but it is a
    // leak for as long as the device ran, and the whole point of routing
    // resources through handles is that the count is knowable. Reported rather
    // than asserted so that a shutdown already unwinding from an error is not
    // made worse.
    const std::array<std::pair<const char*, uint32_t>, 7> live{
        std::pair{"buffer", m_Buffers.Size()},
        std::pair{"texture", m_Textures.Size()},
        std::pair{"texture view", m_TextureViews.Size()},
        std::pair{"sampler", m_Samplers.Size()},
        std::pair{"fence", m_Fences.Size()},
        std::pair{"bind group layout", m_BindGroupLayouts.Size()},
        std::pair{"bind group", m_BindGroups.Size()},
    };

    bool bAnyLive = false;
    for (const auto& [kind, count] : live)
    {
        if (count == 0u)
            continue;

        bAnyLive = true;
        Core::LogMsg(Core::LogSeverity::Warning, LogRhi,
                     "Device destroyed with {} {}(s) still alive — each is a resource whose owner "
                     "never released it.",
                     count, kind);
    }

    if (!bAnyLive)
    {
        Core::LogMsg(Core::LogSeverity::Info, LogRhi,
                     "Device destroyed with 0 live buffers, textures, texture views, samplers, "
                     "fences and bind groups.");
    }
}

void VulkanDevice::WaitIdle()
{
    m_Device.waitIdle();
}

vk::raii::Queue& VulkanDevice::GetQueue(QueueType role)
{
    if (role == QueueType::Copy && *m_CopyQueue)
        return m_CopyQueue;

    // Compute lands here too. The graphics queue is the only other one created,
    // and it can serve any role its family advertises — which SelectQueueFamilies
    // already checked before falling a role back to it.
    return m_GraphicsQueue;
}

BufferHandle VulkanDevice::CreateBuffer(const BufferDesc& desc)
{
    if (desc.Size == 0u)
        throw std::runtime_error("Rhi::IDevice::CreateBuffer: a buffer must have a non-zero size.");

    const VmaMemoryParams memoryParams = ToVk(desc.Access);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = static_cast<VkDeviceSize>(desc.Size);
    bufferInfo.usage = static_cast<VkBufferUsageFlags>(ToVk(desc.Usage));
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memoryParams.Usage;
    allocInfo.flags = memoryParams.Flags;

    VkBuffer rawBuffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo allocationInfo{};

    const vk::Result result = static_cast<vk::Result>(vmaCreateBuffer(
        m_Allocator, &bufferInfo, &allocInfo, &rawBuffer, &allocation, &allocationInfo));

    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error(std::format("Rhi::IDevice::CreateBuffer: VMA failed to allocate "
                                             "'{}' ({} bytes): {}.",
                                             desc.DebugName, desc.Size, vk::to_string(result)));
    }

    if (!desc.DebugName.empty())
    {
        SetVkDebugName(m_Device, vk::Buffer(rawBuffer), vk::ObjectType::eBuffer,
                       desc.DebugName.c_str());
        // Names the allocation as well as the buffer, because VMA's own leak
        // and budget dumps report allocations, not Vulkan objects.
        vmaSetAllocationName(m_Allocator, allocation, desc.DebugName.c_str());
    }

    return m_Buffers.Create(m_Allocator, vk::Buffer(rawBuffer), allocation, allocationInfo);
}

void VulkanDevice::Destroy(BufferHandle handle)
{
    if (m_Buffers.Release(handle))
        return;

    // Either a double destroy or a handle outliving what it named. Both are the
    // bug the generation counter exists to catch, so neither is silently
    // ignored — but neither is fatal either, since the slot is already free.
    ReportDiagnostic(DiagnosticSeverity::Error,
                     std::format("Rhi::IDevice::Destroy(BufferHandle): handle {:#010x} is stale or "
                                 "was never valid; it may have been destroyed already.",
                                 handle.Value));
}

void* VulkanDevice::GetMappedData(BufferHandle handle)
{
    const VulkanBuffer* pBuffer = m_Buffers.Get(handle);
    return pBuffer ? pBuffer->AllocationInfo.pMappedData : nullptr;
}

vk::Buffer VulkanDevice::GetBuffer(BufferHandle handle) const
{
    const VulkanBuffer* pBuffer = m_Buffers.Get(handle);
    return pBuffer ? pBuffer->Buffer : vk::Buffer{};
}

TextureHandle VulkanDevice::CreateTexture(const TextureDesc& desc)
{
    ValidateTextureDesc(desc);

    const vk::ImageCreateInfo imageInfo{
        .flags = desc.bCubeCompatible
                     ? vk::ImageCreateFlags{vk::ImageCreateFlagBits::eCubeCompatible}
                     : vk::ImageCreateFlags{},
        .imageType = ToVk(desc.Dimension),
        .format = ToVk(desc.Format),
        .extent = vk::Extent3D{desc.Extent.Width, desc.Extent.Height, desc.Extent.Depth},
        .mipLevels = desc.MipLevels,
        .arrayLayers = desc.ArrayLayers,
        .samples = ToVk(desc.Samples),
        .tiling = kTextureTiling,
        .usage = ToVk(desc.Usage),
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined};

    // Textures are always device-local: nothing here uploads by writing an
    // image's memory directly, it stages through a buffer and copies. That is
    // also the only portable path — D3D12 has no equivalent of a linear-tiled
    // host-visible image that a shader can sample.
    const VmaMemoryParams memoryParams = ToVk(MemoryAccess::GpuOnly);

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memoryParams.Usage;
    allocInfo.flags = memoryParams.Flags;

    const VkImageCreateInfo cImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
    VkImage rawImage = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;

    const vk::Result result = static_cast<vk::Result>(
        vmaCreateImage(m_Allocator, &cImageInfo, &allocInfo, &rawImage, &allocation, nullptr));

    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error(std::format("Rhi::IDevice::CreateTexture: VMA failed to allocate "
                                             "'{}' ({}x{}x{}): {}.",
                                             desc.DebugName, desc.Extent.Width, desc.Extent.Height,
                                             desc.Extent.Depth, vk::to_string(result)));
    }

    if (!desc.DebugName.empty())
    {
        SetVkDebugName(m_Device, vk::Image(rawImage), vk::ObjectType::eImage,
                       desc.DebugName.c_str());
        // Names the allocation as well as the image, because VMA's own leak and
        // budget dumps report allocations, not Vulkan objects.
        vmaSetAllocationName(m_Allocator, allocation, desc.DebugName.c_str());
    }

    return m_Textures.Create(m_Allocator, vk::Image(rawImage), allocation, desc);
}

TextureHandle VulkanDevice::RegisterExternalTexture(vk::Image image, const TextureDesc& desc)
{
    if (!image)
        throw std::runtime_error("Rhi::Vulkan::RegisterExternalTexture: null image.");

    ValidateTextureDesc(desc);

    if (!desc.DebugName.empty())
        SetVkDebugName(m_Device, image, vk::ObjectType::eImage, desc.DebugName.c_str());

    // No allocator and no allocation: VulkanTexture reads that as "not ours"
    // and frees nothing when the slot is released.
    return m_Textures.Create(VmaAllocator{}, image, VmaAllocation{}, desc);
}

void VulkanDevice::Destroy(TextureHandle handle)
{
    if (m_Textures.Release(handle))
        return;

    ReportDiagnostic(DiagnosticSeverity::Error,
                     std::format("Rhi::IDevice::Destroy(TextureHandle): handle {:#010x} is stale "
                                 "or was never valid; it may have been destroyed already.",
                                 handle.Value));
}

TextureViewHandle VulkanDevice::CreateTextureView(const TextureViewDesc& desc)
{
    const VulkanTexture* pTexture = m_Textures.Get(desc.Texture);
    if (!pTexture)
    {
        throw std::runtime_error(
            std::format("Rhi::IDevice::CreateTextureView: '{}' names texture handle {:#010x}, "
                        "which is stale or was never valid.",
                        desc.DebugName, desc.Texture.Value));
    }

    const Rhi::Format format =
        desc.Format == Rhi::Format::Undefined ? pTexture->Desc.Format : desc.Format;
    const TextureAspect aspect = Any(desc.Aspect) ? desc.Aspect : DefaultAspect(format);

    const vk::ImageViewCreateInfo createInfo{.image = pTexture->Image,
                                             .viewType = ToVk(desc.Dimension),
                                             .format = ToVk(format),
                                             .subresourceRange = {.aspectMask = ToVk(aspect),
                                                                  .baseMipLevel = desc.BaseMip,
                                                                  .levelCount = desc.MipCount,
                                                                  .baseArrayLayer = desc.BaseLayer,
                                                                  .layerCount = desc.LayerCount}};

    VulkanTextureView view{vk::raii::ImageView(m_Device, createInfo), aspect};

    if (!desc.DebugName.empty())
    {
        SetVkDebugName(m_Device, *view.View, vk::ObjectType::eImageView, desc.DebugName.c_str());
    }

    return m_TextureViews.Create(std::move(view));
}

void VulkanDevice::Destroy(TextureViewHandle handle)
{
    if (m_TextureViews.Release(handle))
        return;

    ReportDiagnostic(
        DiagnosticSeverity::Error,
        std::format("Rhi::IDevice::Destroy(TextureViewHandle): handle {:#010x} is stale or "
                    "was never valid; it may have been destroyed already.",
                    handle.Value));
}

SamplerHandle VulkanDevice::CreateSampler(const SamplerDesc& desc)
{
    // Only meaningful when anisotropy is enabled, and then it must lie within
    // the device's limit (VUID-VkSamplerCreateInfo-anisotropyEnable-01071). A
    // desc asking for 0 is asking for the best the device offers, which is what
    // spares every caller from plumbing the limit through to get it.
    float maxAnisotropy = desc.MaxAnisotropy;
    if (desc.bAnisotropyEnable)
    {
        const float limit = m_PhysicalDevice.getProperties().limits.maxSamplerAnisotropy;
        maxAnisotropy = desc.MaxAnisotropy <= 0.f ? limit : std::min(desc.MaxAnisotropy, limit);
    }

    const vk::SamplerCreateInfo createInfo{
        .magFilter = ToVk(desc.MagFilter),
        .minFilter = ToVk(desc.MinFilter),
        .mipmapMode = ToVk(desc.MipmapFilter),
        .addressModeU = ToVk(desc.AddressU),
        .addressModeV = ToVk(desc.AddressV),
        .addressModeW = ToVk(desc.AddressW),
        .mipLodBias = desc.MipLodBias,
        .anisotropyEnable = static_cast<vk::Bool32>(desc.bAnisotropyEnable),
        .maxAnisotropy = maxAnisotropy,
        .compareEnable = static_cast<vk::Bool32>(desc.bCompareEnable),
        .compareOp = ToVk(desc.Compare),
        .minLod = desc.MinLod,
        .maxLod = desc.MaxLod,
        .borderColor = ToVk(desc.Border),
        .unnormalizedCoordinates = vk::False};

    VulkanSampler sampler{vk::raii::Sampler(m_Device, createInfo)};

    if (!desc.DebugName.empty())
    {
        SetVkDebugName(m_Device, *sampler.Sampler, vk::ObjectType::eSampler,
                       desc.DebugName.c_str());
    }

    return m_Samplers.Create(std::move(sampler));
}

void VulkanDevice::Destroy(SamplerHandle handle)
{
    if (m_Samplers.Release(handle))
        return;

    ReportDiagnostic(DiagnosticSeverity::Error,
                     std::format("Rhi::IDevice::Destroy(SamplerHandle): handle {:#010x} is stale "
                                 "or was never valid; it may have been destroyed already.",
                                 handle.Value));
}

const TextureDesc* VulkanDevice::GetTextureDesc(TextureHandle handle) const
{
    const VulkanTexture* pTexture = m_Textures.Get(handle);
    return pTexture ? &pTexture->Desc : nullptr;
}

vk::Image VulkanDevice::GetImage(TextureHandle handle) const
{
    const VulkanTexture* pTexture = m_Textures.Get(handle);
    return pTexture ? pTexture->Image : vk::Image{};
}

vk::ImageView VulkanDevice::GetImageView(TextureViewHandle handle) const
{
    const VulkanTextureView* pView = m_TextureViews.Get(handle);
    return pView ? *pView->View : vk::ImageView{};
}

vk::Sampler VulkanDevice::GetSampler(SamplerHandle handle) const
{
    const VulkanSampler* pSampler = m_Samplers.Get(handle);
    return pSampler ? *pSampler->Sampler : vk::Sampler{};
}

std::unique_ptr<IUploadContext> VulkanDevice::CreateUploadContext(const UploadContextDesc& desc)
{
    return std::make_unique<VulkanUploadContext>(*this, desc);
}

std::unique_ptr<ICommandAllocator>
VulkanDevice::CreateCommandAllocator(const CommandAllocatorDesc& desc)
{
    return std::make_unique<VulkanCommandAllocator>(*this, desc);
}

std::unique_ptr<IPipelineCache> VulkanDevice::CreatePipelineCache(const PipelineCacheDesc& desc)
{
    return std::make_unique<VulkanPipelineCache>(m_Device, m_PhysicalDevice.getProperties(), desc);
}

std::unique_ptr<IPresentTarget> VulkanDevice::CreatePresentTarget(const PresentTargetDesc& desc)
{
    // The caller does not choose, and cannot tell: a device with a surface
    // presents through a swapchain, one without renders into images of its own.
    // That is the whole seam — everything above this line is written once and
    // runs both ways.
    if (*m_Surface == nullptr)
        return std::make_unique<OffscreenTarget>(*this, desc);

    return std::make_unique<SwapchainTarget>(*this, desc);
}

SemaphoreHandle VulkanDevice::CreateSemaphore(std::string_view debugName)
{
    VulkanSemaphore semaphore{vk::raii::Semaphore(m_Device, vk::SemaphoreCreateInfo{})};

    if (!debugName.empty())
    {
        SetVkDebugName(m_Device, *semaphore.Semaphore, vk::ObjectType::eSemaphore,
                       std::string(debugName).c_str());
    }

    return m_Semaphores.Create(std::move(semaphore));
}

void VulkanDevice::Destroy(SemaphoreHandle handle)
{
    if (m_Semaphores.Release(handle))
        return;

    ReportDiagnostic(DiagnosticSeverity::Error,
                     std::format("Rhi::VulkanDevice::Destroy(SemaphoreHandle): handle {:#010x} is "
                                 "stale or was never valid; it may have been destroyed already.",
                                 handle.Value));
}

vk::Semaphore VulkanDevice::GetSemaphore(SemaphoreHandle handle) const
{
    const VulkanSemaphore* pSemaphore = m_Semaphores.Get(handle);
    return pSemaphore ? *pSemaphore->Semaphore : vk::Semaphore{};
}

namespace
{
vk::DescriptorType ToVkDescriptorType(BindingType type)
{
    switch (type)
    {
        case BindingType::UniformBuffer:
            return vk::DescriptorType::eUniformBuffer;
        case BindingType::Texture:
            return vk::DescriptorType::eSampledImage;
        case BindingType::Sampler:
            return vk::DescriptorType::eSampler;
    }

    throw std::runtime_error("Rhi::VulkanDevice: unmapped BindingType.");
}

vk::ShaderStageFlags ToVkShaderStages(ShaderStage stages)
{
    vk::ShaderStageFlags result{};
    if ((stages & ShaderStage::Vertex) != ShaderStage::None)
        result |= vk::ShaderStageFlagBits::eVertex;
    if ((stages & ShaderStage::Pixel) != ShaderStage::None)
        result |= vk::ShaderStageFlagBits::eFragment;
    if ((stages & ShaderStage::Compute) != ShaderStage::None)
        result |= vk::ShaderStageFlagBits::eCompute;

    if (!result)
        throw std::runtime_error("Rhi::VulkanDevice: a binding visible to no shader stage.");

    return result;
}
} // namespace

BindGroupLayoutHandle VulkanDevice::CreateBindGroupLayout(const BindGroupLayoutDesc& desc)
{
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(desc.Bindings.size());
    for (const BindGroupLayoutBinding& binding : desc.Bindings)
    {
        bindings.push_back(
            vk::DescriptorSetLayoutBinding{.binding = binding.Slot,
                                           .descriptorType = ToVkDescriptorType(binding.Type),
                                           .descriptorCount = 1u,
                                           .stageFlags = ToVkShaderStages(binding.Visibility)});
    }

    const vk::DescriptorSetLayoutCreateInfo createInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};

    VulkanBindGroupLayout layout{vk::raii::DescriptorSetLayout(m_Device, createInfo)};

    if (!desc.DebugName.empty())
    {
        SetVkDebugName(m_Device, *layout.Layout, vk::ObjectType::eDescriptorSetLayout,
                       desc.DebugName.c_str());
    }

    return m_BindGroupLayouts.Create(std::move(layout));
}

void VulkanDevice::Destroy(BindGroupLayoutHandle handle)
{
    if (m_BindGroupLayouts.Release(handle))
        return;

    ReportDiagnostic(DiagnosticSeverity::Error,
                     std::format("Rhi::VulkanDevice::Destroy(BindGroupLayoutHandle): handle "
                                 "{:#010x} is stale or was never valid.",
                                 handle.Value));
}

BindGroupHandle VulkanDevice::CreateBindGroup(const BindGroupDesc& desc)
{
    const VulkanBindGroupLayout* pLayout = m_BindGroupLayouts.Get(desc.Layout);
    if (pLayout == nullptr)
    {
        ReportStaleHandle(std::format("Rhi::VulkanDevice::CreateBindGroup: layout {:#010x} is "
                                      "stale or was never valid.",
                                      desc.Layout.Value));
        return BindGroupHandle{};
    }

    VulkanBindGroup group{m_BindGroupAllocator->Allocate(*pLayout->Layout)};

    // Written once, here, and never again: a group is immutable (plan D20), so
    // this is the only point at which its descriptors are filled in.
    std::vector<vk::DescriptorBufferInfo> bufferInfos(desc.Bindings.size());
    std::vector<vk::DescriptorImageInfo> imageInfos(desc.Bindings.size());
    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(desc.Bindings.size());

    for (size_t i = 0; i < desc.Bindings.size(); i++)
    {
        const BindGroupBinding& binding = desc.Bindings[i];
        vk::WriteDescriptorSet write{.dstSet = *group.Set,
                                     .dstBinding = binding.Slot,
                                     .dstArrayElement = 0u,
                                     .descriptorCount = 1u,
                                     .descriptorType = ToVkDescriptorType(binding.Type)};

        switch (binding.Type)
        {
            case BindingType::UniformBuffer:
                bufferInfos[i] = vk::DescriptorBufferInfo{
                    .buffer = GetBuffer(binding.Buffer), .offset = 0u, .range = VK_WHOLE_SIZE};
                write.pBufferInfo = &bufferInfos[i];
                break;
            case BindingType::Texture:
            {
                // A sampled depth view reads from DEPTH_READ_ONLY_OPTIMAL, not
                // from SHADER_READ_ONLY_OPTIMAL. Which one is a Vulkan layout
                // rule rather than anything the caller said, so it comes from
                // the view's aspect.
                const VulkanTextureView* pView = m_TextureViews.Get(binding.View);
                const bool bDepth = pView != nullptr && Any(pView->Aspect & TextureAspect::Depth);
                imageInfos[i] = vk::DescriptorImageInfo{
                    .imageView = GetImageView(binding.View),
                    .imageLayout = bDepth ? vk::ImageLayout::eDepthReadOnlyOptimal
                                          : vk::ImageLayout::eShaderReadOnlyOptimal};
                write.pImageInfo = &imageInfos[i];
                break;
            }
            case BindingType::Sampler:
                imageInfos[i] = vk::DescriptorImageInfo{.sampler = GetSampler(binding.Sampler)};
                write.pImageInfo = &imageInfos[i];
                break;
        }

        writes.push_back(write);
    }

    m_Device.updateDescriptorSets(writes, nullptr);

    if (!desc.DebugName.empty())
    {
        SetVkDebugName(m_Device, *group.Set, vk::ObjectType::eDescriptorSet,
                       desc.DebugName.c_str());
    }

    return m_BindGroups.Create(std::move(group));
}

void VulkanDevice::Destroy(BindGroupHandle handle)
{
    if (m_BindGroups.Release(handle))
        return;

    ReportDiagnostic(DiagnosticSeverity::Error,
                     std::format("Rhi::VulkanDevice::Destroy(BindGroupHandle): handle {:#010x} is "
                                 "stale or was never valid.",
                                 handle.Value));
}

vk::DescriptorSetLayout VulkanDevice::GetDescriptorSetLayout(BindGroupLayoutHandle handle) const
{
    const VulkanBindGroupLayout* pLayout = m_BindGroupLayouts.Get(handle);
    return pLayout ? *pLayout->Layout : vk::DescriptorSetLayout{};
}

vk::DescriptorSet VulkanDevice::GetDescriptorSet(BindGroupHandle handle) const
{
    const VulkanBindGroup* pGroup = m_BindGroups.Get(handle);
    return pGroup ? *pGroup->Set : vk::DescriptorSet{};
}

FenceHandle VulkanDevice::CreateFence(const FenceDesc& desc)
{
    const vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> createInfo{
        {}, {.semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = desc.InitialValue}};

    VulkanFence fence{vk::raii::Semaphore(m_Device, createInfo.get<vk::SemaphoreCreateInfo>())};

    if (!desc.DebugName.empty())
    {
        SetVkDebugName(m_Device, *fence.Timeline, vk::ObjectType::eSemaphore,
                       desc.DebugName.c_str());
    }

    return m_Fences.Create(std::move(fence));
}

void VulkanDevice::Destroy(FenceHandle handle)
{
    if (m_Fences.Release(handle))
        return;

    ReportDiagnostic(DiagnosticSeverity::Error,
                     std::format("Rhi::VulkanDevice::Destroy(FenceHandle): handle {:#010x} is "
                                 "stale or was never valid; it may have been destroyed already.",
                                 handle.Value));
}

void VulkanDevice::WaitForFence(FenceHandle handle, uint64_t value)
{
    const VulkanFence* pFence = m_Fences.Get(handle);
    if (pFence == nullptr)
    {
        ReportStaleHandle(std::format(
            "Rhi::VulkanDevice::WaitForFence: handle {:#010x} is stale or was never valid.",
            handle.Value));
        return;
    }

    const vk::Semaphore timeline = *pFence->Timeline;
    const vk::SemaphoreWaitInfo waitInfo{
        .semaphoreCount = 1u, .pSemaphores = &timeline, .pValues = &value};

    const vk::Result result =
        m_Device.waitSemaphores(waitInfo, std::numeric_limits<uint64_t>::max());
    if (result != vk::Result::eSuccess)
        throw std::runtime_error("Rhi::VulkanDevice::WaitForFence: the wait did not succeed.");
}

void VulkanDevice::Submit(const SubmitDesc& desc)
{
    // Fixed capacities rather than per-submit allocations: this runs once per
    // frame per queue, and the counts are bounded by what the frame actually
    // has. Overflowing throws rather than silently dropping a wait, which is the
    // failure mode that would corrupt a frame invisibly.
    constexpr size_t kMaxLists = 16u;
    constexpr size_t kMaxSyncs = 8u;

    if (desc.CommandLists.size() > kMaxLists ||
        desc.WaitFences.size() + desc.WaitSemaphores.size() > kMaxSyncs ||
        desc.SignalFences.size() + desc.SignalSemaphores.size() > kMaxSyncs)
    {
        throw std::runtime_error("Rhi::VulkanDevice::Submit: more lists or synchronization "
                                 "operations than a submission is sized for.");
    }

    std::array<vk::CommandBufferSubmitInfo, kMaxLists> lists{};
    for (size_t i = 0; i < desc.CommandLists.size(); i++)
    {
        const VulkanCommandList* pList =
            static_cast<const VulkanCommandList*>(desc.CommandLists[i]);
        lists[i] = vk::CommandBufferSubmitInfo{.commandBuffer = pList->Native()};
    }

    std::array<vk::SemaphoreSubmitInfo, kMaxSyncs> waits{};
    size_t waitCount = 0u;
    for (const FenceOperation& wait : desc.WaitFences)
    {
        const VulkanFence* pFence = m_Fences.Get(wait.Fence);
        if (pFence == nullptr)
        {
            ReportStaleHandle(std::format(
                "Rhi::VulkanDevice::Submit: wait fence {:#010x} is stale or was never valid.",
                wait.Fence.Value));
            continue;
        }

        // AllCommands because a fence wait orders whole submissions against one
        // another: D3D12's queue wait has no stage at all, so narrowing this
        // would be expressing something the neutral API cannot say.
        waits[waitCount++] =
            vk::SemaphoreSubmitInfo{.semaphore = *pFence->Timeline,
                                    .value = wait.Value,
                                    .stageMask = vk::PipelineStageFlagBits2::eAllCommands};
    }
    for (const SemaphoreHandle handle : desc.WaitSemaphores)
    {
        // ColorAttachmentOutput: the only semaphores that reach here guard writes
        // to an image a present target just handed out, and that is the first
        // stage which can write one. See SubmitDesc on why the caller does not
        // name a stage.
        waits[waitCount++] = vk::SemaphoreSubmitInfo{
            .semaphore = GetSemaphore(handle),
            .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput};
    }

    std::array<vk::SemaphoreSubmitInfo, kMaxSyncs> signals{};
    size_t signalCount = 0u;
    for (const FenceOperation& signal : desc.SignalFences)
    {
        const VulkanFence* pFence = m_Fences.Get(signal.Fence);
        if (pFence == nullptr)
        {
            ReportStaleHandle(std::format(
                "Rhi::VulkanDevice::Submit: signal fence {:#010x} is stale or was never valid.",
                signal.Fence.Value));
            continue;
        }

        signals[signalCount++] =
            vk::SemaphoreSubmitInfo{.semaphore = *pFence->Timeline,
                                    .value = signal.Value,
                                    .stageMask = vk::PipelineStageFlagBits2::eAllCommands};
    }
    for (const SemaphoreHandle handle : desc.SignalSemaphores)
    {
        signals[signalCount++] =
            vk::SemaphoreSubmitInfo{.semaphore = GetSemaphore(handle),
                                    .stageMask = vk::PipelineStageFlagBits2::eAllCommands};
    }

    const vk::SubmitInfo2 submitInfo{.waitSemaphoreInfoCount = static_cast<uint32_t>(waitCount),
                                     .pWaitSemaphoreInfos = waits.data(),
                                     .commandBufferInfoCount =
                                         static_cast<uint32_t>(desc.CommandLists.size()),
                                     .pCommandBufferInfos = lists.data(),
                                     .signalSemaphoreInfoCount = static_cast<uint32_t>(signalCount),
                                     .pSignalSemaphoreInfos = signals.data()};

    GetQueue(desc.Queue).submit2(submitInfo);
}

void VulkanDevice::ReportStaleHandle(std::string_view what) const
{
    ReportDiagnostic(DiagnosticSeverity::Error, what);
}

void VulkanDevice::ReportDiagnostic(DiagnosticSeverity severity, std::string_view message) const
{
    m_pDiagnostics->Report(severity, message);
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL VulkanDevice::DebugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
    const auto* device = static_cast<const VulkanDevice*>(pUserData);
    if (!device)
        return vk::False;

    // Filtered here as well as by the messenger's severity flags, so that a
    // message destined to be dropped does not pay for the std::format below.
    // The comparison is on the raw bit values, which the specification orders
    // verbose < info < warning < error, so this reads as a threshold.
    if (severity < ToVk(device->m_pDiagnostics->MinSeverity()))
        return vk::False;

    device->ReportDiagnostic(FromVk(severity), std::format("Type: {}. Msg: {}", vk::to_string(type),
                                                           pCallbackData->pMessage));

    // False tells the driver to carry on. Returning true is reserved for layer
    // development and aborts the call that triggered the message.
    return vk::False;
}

void VulkanDevice::CreateInstance(const DeviceDesc& desc)
{
    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "CreateInstance()");

    // Before the first enumeration below, which is what the loader answers from
    // its manifest scan.
    if (desc.bEnableValidation)
        AddBundledLayerPath();

    const vk::ApplicationInfo appInfo{.pApplicationName = desc.ApplicationName.c_str(),
                                      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                      .pEngineName = "No Engine",
                                      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                      .apiVersion = kApiVersion};

    // extensions
    std::vector<const char*> requiredExtensions;
    requiredExtensions.push_back(vk::EXTDebugUtilsExtensionName);

    // The surface extensions come from SDL because only it knows which
    // windowing system this build is talking to. A device that will not present
    // must not ask: SDL only loads the Vulkan library when video has been
    // initialised, so this returns nothing in a process that never opened a
    // window — and none of what it would return is needed there anyway.
    if (desc.Requirements.bPresent)
    {
        uint32_t countInstanceExtensions = 0;
        const char* const* instanceExtensions =
            SDL_Vulkan_GetInstanceExtensions(&countInstanceExtensions);

        if (countInstanceExtensions == 0)
            throw std::runtime_error("No available instance extensions found!");

        requiredExtensions.insert(requiredExtensions.end(), instanceExtensions,
                                  instanceExtensions + countInstanceExtensions);
    }

    auto extensionProperties = m_Context.enumerateInstanceExtensionProperties();

    // VK_EXT_layer_settings is implemented by layers (e.g.
    // VK_LAYER_KHRONOS_validation), not by the loader, so it never appears in
    // vkEnumerateInstanceExtensionProperties(nullptr). It takes effect purely
    // through the VkLayerSettingsCreateInfoEXT pNext chain below; the gate for
    // attaching it is whether the validation layer is enabled, not whether the
    // extension happens to be enumerated.
    auto unsupportedExtensionIt = std::ranges::find_if(
        requiredExtensions,
        [&extensionProperties](auto const& requiredExtension)
        {
            return std::ranges::none_of(
                extensionProperties, [requiredExtension](auto const& extensionProperty)
                { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; });
        });

    if (unsupportedExtensionIt != requiredExtensions.end())
        throw std::runtime_error("Required extension not supported: " +
                                 std::string(*unsupportedExtensionIt));

    // layers
    std::vector<const char*> requiredLayers;
    if (desc.bEnableValidation)
        requiredLayers.push_back(kValidationLayerName);

    auto layerProperties = m_Context.enumerateInstanceLayerProperties();

    auto unsupportedLayerIt = std::ranges::find_if(
        requiredLayers,
        [&layerProperties](auto const& requiredLayer)
        {
            return std::ranges::none_of(
                layerProperties, [requiredLayer](auto const& layerProperty)
                { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
        });

    if (unsupportedLayerIt != requiredLayers.end())
        throw std::runtime_error("Required layer not supported: " +
                                 std::string(*unsupportedLayerIt));

    const vk::Bool32 bSyncValEnabled = VK_TRUE;

    // Best-practices validation is off because the layer crashes on it, not because
    // we stopped wanting it. vulkan-validationlayers 1.4.357.0 — the newest version
    // vcpkg offers — reads an image's last-used queue family in a maintenance9-gated
    // branch of BestPractices::ValidateImageInQueue without first checking it against
    // VK_QUEUE_FAMILY_IGNORED, so the first use of any image in a submit indexes at
    // 0xFFFFFFFF and segfaults inside the layer. We enable maintenance9, so this hit
    // every debug run and two GPU tests. Fixed upstream by Vulkan-ValidationLayers
    // PR #12922 (issue #12449), merged after the 1.4.357.0 tag was cut, so the first
    // SDK release carrying it is later than anything vcpkg has today. Restore this
    // line and the setting below once vcpkg offers such a version.
    // const vk::Bool32 bBestPracticesValEnabled = VK_TRUE;

    // VUIDs the validation layer must not emit. The best-practices layer raises a
    // performance warning for every VK_SUBOPTIMAL_KHR returned by
    // vkQueuePresentKHR, but a suboptimal swapchain is precisely the signal the
    // frame loop acts on — it recreates the swapchain on the next iteration. A
    // live resize makes that fire once per frame, which is noise rather than a
    // defect. The id names vkCreateSharedSwapchainsKHR for historical reasons but
    // is the one the present-time check actually uses (Vulkan-ValidationLayers
    // bp_wsi.cpp, PostCallRecordQueuePresentKHR). Muted at the layer via
    // message_id_filter so it is never generated, keeping DebugCallback generic
    // and the run report's validation counts clean. The filter matches nothing while
    // best-practices validation is disabled above, and is kept so that re-enabling it
    // brings the mute back with it.
    static constexpr const char* kMutedMessageIds[] = {
        "BestPractices-vkCreateSharedSwapchainsKHR-SuboptimalSwapchain",
    };

    std::array<vk::LayerSettingEXT, 2> settings = {
        vk::LayerSettingEXT{.pLayerName = kValidationLayerName,
                            .pSettingName = "validate_sync",
                            .type = vk::LayerSettingTypeEXT::eBool32,
                            .valueCount = 1,
                            .pValues = &bSyncValEnabled},
        // vk::LayerSettingEXT{.pLayerName = kValidationLayerName,
        //                     .pSettingName = "validate_best_practices",
        //                     .type = vk::LayerSettingTypeEXT::eBool32,
        //                     .valueCount = 1,
        //                     .pValues = &bBestPracticesValEnabled},
        vk::LayerSettingEXT{.pLayerName = kValidationLayerName,
                            .pSettingName = "message_id_filter",
                            .type = vk::LayerSettingTypeEXT::eString,
                            .valueCount = static_cast<uint32_t>(std::size(kMutedMessageIds)),
                            .pValues = kMutedMessageIds},
    };

    vk::LayerSettingsCreateInfoEXT layerSettingsInfo{
        .settingCount = static_cast<uint32_t>(settings.size()), .pSettings = settings.data()};

    vk::InstanceCreateInfo createInfo{.pNext =
                                          desc.bEnableValidation ? &layerSettingsInfo : nullptr,
                                      .pApplicationInfo = &appInfo,
                                      .enabledLayerCount = (uint32_t)requiredLayers.size(),
                                      .ppEnabledLayerNames = requiredLayers.data(),
                                      .enabledExtensionCount = (uint32_t)requiredExtensions.size(),
                                      .ppEnabledExtensionNames = requiredExtensions.data()};

    m_Instance = vk::raii::Instance(m_Context, createInfo);
}

void VulkanDevice::SetupDebugMessenger(const DeviceDesc& desc)
{
    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "SetupDebugMessenger()");

    if (!desc.bEnableValidation)
        return;

    // Ask the driver only for what the threshold admits, so the filtering
    // happens before the callback rather than inside it. Verbose is never
    // requested: it collapses to Info on the neutral scale, and asking for it
    // would multiply the message volume for nothing a caller can distinguish.
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    if (m_pDiagnostics->MinSeverity() <= DiagnosticSeverity::Warning)
        severityFlags |= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning;
    if (m_pDiagnostics->MinSeverity() <= DiagnosticSeverity::Info)
        severityFlags |= vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo;

    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);

    vk::DebugUtilsMessengerCreateInfoEXT createInfo{.messageSeverity = severityFlags,
                                                    .messageType = messageTypeFlags,
                                                    .pfnUserCallback = &VulkanDevice::DebugCallback,
                                                    .pUserData = this};

    m_DebugMessenger = m_Instance.createDebugUtilsMessengerEXT(createInfo);
}

void VulkanDevice::CreateSurface(const DeviceRequirements& requirements)
{
    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "CreateSurface()");

    if (!requirements.bPresent)
        return;

    if (!requirements.NativeWindowHandle)
        throw std::runtime_error("A present-capable device needs a native window handle!");

    auto* pWindow = static_cast<SDL_Window*>(requirements.NativeWindowHandle);

    VkSurfaceKHR rawSurface;
    if (!SDL_Vulkan_CreateSurface(pWindow, *m_Instance, nullptr, &rawSurface))
        throw std::runtime_error("Failed to create Vulkan surface!");

    m_Surface = vk::raii::SurfaceKHR(m_Instance, rawSurface);
}

bool VulkanDevice::IsPhysicalDeviceSuitable(const vk::raii::PhysicalDevice& device,
                                            const DeviceRequirements& requirements) const
{
    auto properties = device.getProperties();

    bool bSupportsVulkan13 = properties.apiVersion >= vk::ApiVersion13;

    auto queueFamilies = device.getQueueFamilyProperties();
    bool bSupportsGraphicsQ =
        std::ranges::any_of(queueFamilies, [](const auto& qfp)
                            { return FamilySupports(qfp.queueFlags, QueueType::Graphics); });

    std::vector<const char*> requiredExtensions = {vk::EXTDescriptorIndexingExtensionName};
    if (requirements.bPresent)
        requiredExtensions.push_back(vk::KHRSwapchainExtensionName);
    auto availableExtensions = device.enumerateDeviceExtensionProperties();
    bool bSupportsAllExtensions = std::ranges::all_of(
        requiredExtensions,
        [&availableExtensions](const auto& requiredExtension)
        {
            return std::ranges::any_of(
                availableExtensions, [requiredExtension](const auto& availableExtension)
                { return strcmp(availableExtension.extensionName, requiredExtension) == 0; });
        });

    auto features =
        device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
                            vk::PhysicalDeviceVulkan12Features,
                            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    bool bSupportsAllFeatures =
        features.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
        features.get<vk::PhysicalDeviceFeatures2>().features.independentBlend &&
        features.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore &&
        features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
        features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
        features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

    if (bSupportsVulkan13 && bSupportsGraphicsQ && bSupportsAllExtensions && bSupportsAllFeatures)
        return true;
    return false;
}

void VulkanDevice::PickPhysicalDevice(const DeviceRequirements& requirements)
{
    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "PickPhysicalDevice()");

    auto devices = m_Instance.enumeratePhysicalDevices();
    const auto deviceIt =
        std::ranges::find_if(devices, [&](const auto& device)
                             { return IsPhysicalDeviceSuitable(device, requirements); });

    if (deviceIt == devices.end())
        throw std::runtime_error("Failed to find a suitable GPU!");

    m_PhysicalDevice = *deviceIt;
}

void VulkanDevice::SelectOptionalExtensions(const DeviceDesc& desc)
{
    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "SelectOptionalExtensions()");

    const std::vector<vk::ExtensionProperties> available =
        m_PhysicalDevice.enumerateDeviceExtensionProperties();

    // Every optional extension this backend knows what to do with. A name
    // outside this list cannot be disabled, because nothing would read the
    // answer.
    const std::array<const char*, 2> optional = {vk::KHRMaintenance8ExtensionName,
                                                 vk::KHRMaintenance9ExtensionName};

    for (const std::string& name : desc.DisabledOptionalExtensions)
    {
        if (std::ranges::none_of(optional, [&name](const char* entry) { return name == entry; }))
        {
            Core::LogMsg(
                Core::LogSeverity::Warning, LogRhi,
                "DisabledOptionalExtensions names '{}', which is not an optional extension this "
                "backend uses. It has no effect.",
                name);
        }
    }

    auto resolve = [&](const char* name)
    {
        const bool bSupported =
            std::ranges::any_of(available, [name](const vk::ExtensionProperties& properties)
                                { return strcmp(properties.extensionName, name) == 0; });

        const bool bDisabled =
            std::ranges::any_of(desc.DisabledOptionalExtensions,
                                [name](const std::string& entry) { return entry == name; });

        Core::LogMsg(Core::LogSeverity::Info, LogRhi, "{}: {}", name,
                     !bSupported ? "not supported"
                     : bDisabled ? "supported, disabled by request"
                                 : "enabled");

        return bSupported && !bDisabled;
    };

    m_bMaintenance8Enabled = resolve(vk::KHRMaintenance8ExtensionName);
    m_bMaintenance9Enabled = resolve(vk::KHRMaintenance9ExtensionName);

    if (!m_bMaintenance9Enabled)
        return;

    // Fetched for every family rather than only the copy one, because the
    // property describes the family *releasing* a resource and this type has no
    // opinion yet about which family that will be.
    using QueueFamilyChain = vk::StructureChain<vk::QueueFamilyProperties2,
                                                vk::QueueFamilyOwnershipTransferPropertiesKHR>;

    const std::vector<QueueFamilyChain> families =
        m_PhysicalDevice.getQueueFamilyProperties2<QueueFamilyChain>();

    m_OptimalImageTransferToQueueFamilies.reserve(families.size());
    for (const QueueFamilyChain& family : families)
    {
        m_OptimalImageTransferToQueueFamilies.push_back(
            family.get<vk::QueueFamilyOwnershipTransferPropertiesKHR>()
                .optimalImageTransferToQueueFamilies);
    }
}

bool VulkanDevice::RequiresOwnershipTransfer(TextureHandle handle, uint32_t srcFamily,
                                             uint32_t dstFamily) const
{
    // Images are asked about one at a time, and buffers are not, because the
    // specification treats them differently: maintenance9 preserves every
    // buffer unconditionally, while an image's answer depends on how it was
    // created. Only the device knows that, which is why this lives here rather
    // than at the call site.
    const TextureDesc* pDesc = GetTextureDesc(handle);
    if (pDesc == nullptr)
        return true;

    return ImageRequiresOwnershipTransfer(GetOwnershipTransferRules(srcFamily), kTextureTiling,
                                          ToVk(pDesc->Usage), srcFamily, dstFamily);
}

OwnershipTransferRules VulkanDevice::GetOwnershipTransferRules(uint32_t srcFamily) const
{
    OwnershipTransferRules rules{.bMaintenance9 = m_bMaintenance9Enabled};

    if (srcFamily < m_OptimalImageTransferToQueueFamilies.size())
        rules.OptimalImageTransferToQueueFamilies =
            m_OptimalImageTransferToQueueFamilies[srcFamily];

    return rules;
}

void VulkanDevice::FindQueueFamilies(const DeviceDesc& desc)
{
    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "FindQueueFamilies()");

    const std::vector<vk::QueueFamilyProperties> families =
        m_PhysicalDevice.getQueueFamilyProperties();

    // Presentation is a property of a (family, surface) pair rather than a queue
    // capability, so it has to be queried — and only where there is a surface to
    // query against, since a null one is not a valid argument.
    const PresentSupportFn presentSupported = [this](uint32_t index)
    { return static_cast<bool>(m_PhysicalDevice.getSurfaceSupportKHR(index, m_Surface)); };

    for (uint32_t index = 0; index < families.size(); index++)
    {
        const char* presentNote = "";
        if (*m_Surface)
            presentNote = presentSupported(index) ? ", present" : ", no present";

        Core::LogMsg(Core::LogSeverity::Info, LogRhi, "Family {}: {} queue(s), {}{}", index,
                     families[index].queueCount, vk::to_string(families[index].queueFlags),
                     presentNote);
    }

    if (desc.bForceSingleQueue)
        Core::LogMsg(Core::LogSeverity::Info, LogRhi,
                     "DeviceDesc::bForceSingleQueue is set: every role resolves to the graphics "
                     "family.");

    m_QueueFamilies = SelectQueueFamilies(families, presentSupported, desc.Requirements.bPresent,
                                          desc.bForceSingleQueue);

    if (m_QueueFamilies.Graphics == QueueFamilies::kInvalid)
        throw std::runtime_error(desc.Requirements.bPresent
                                     ? "Could not find a queue for graphics and presenting!"
                                     : "Could not find a queue for graphics!");

    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "Graphics: {}. Compute: {}. Copy: {}.",
                 DescribeFamily(m_QueueFamilies, QueueType::Graphics),
                 DescribeFamily(m_QueueFamilies, QueueType::Compute),
                 DescribeFamily(m_QueueFamilies, QueueType::Copy));
}

void VulkanDevice::CreateLogicalDevice(const DeviceRequirements& requirements)
{
    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "CreateLogicalDevice()");

    // A queue per family that is actually submitted to, and no more: an unused
    // queue is one the driver schedules for nothing. Uploads run on the copy
    // family, so it gets one; nothing dispatches on the compute family, so it
    // does not.
    //
    // The indices have to be distinct — vkCreateDevice rejects a repeated
    // queueFamilyIndex — which IsDedicated() is exactly the test for, since it
    // is false precisely when the copy role resolved to the graphics family.
    float queuePriority = 0.5f;
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    queueCreateInfos.push_back(
        vk::DeviceQueueCreateInfo{.queueFamilyIndex = m_QueueFamilies.Graphics,
                                  .queueCount = 1,
                                  .pQueuePriorities = &queuePriority});

    if (m_QueueFamilies.IsDedicated(QueueType::Copy))
    {
        queueCreateInfos.push_back(
            vk::DeviceQueueCreateInfo{.queueFamilyIndex = m_QueueFamilies.Copy,
                                      .queueCount = 1,
                                      .pQueuePriorities = &queuePriority});
    }

    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                       vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                       vk::PhysicalDeviceMaintenance8FeaturesKHR,
                       vk::PhysicalDeviceMaintenance9FeaturesKHR>
        featureChain = {{.features = {.independentBlend = true, .samplerAnisotropy = true}},
                        {.shaderDrawParameters = true},
                        // descriptorBindingPartiallyBound belongs here rather than
                        // in a VkPhysicalDeviceDescriptorIndexingFeatures of its
                        // own: descriptor indexing was promoted into Vulkan 1.2,
                        // and chaining both structures is forbidden outright by
                        // VUID-VkDeviceCreateInfo-pNext-02830, so that one feature
                        // cannot be enabled in one and disabled in the other.
                        {.descriptorBindingPartiallyBound = true, .timelineSemaphore = true},
                        {.synchronization2 = true, .dynamicRendering = true},
                        {.extendedDynamicState = true},
                        {.maintenance8 = true},
                        {.maintenance9 = true}};

    // Both halves of an optional extension move together. A feature struct left
    // chained for an extension that is not in the enabled list is not merely
    // ignored — chaining it is undefined behaviour — and enabling the extension
    // without setting its feature bit does nothing at all, because every
    // relaxation either describes is worded "if the feature is enabled".
    if (!m_bMaintenance8Enabled)
        featureChain.unlink<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
    if (!m_bMaintenance9Enabled)
        featureChain.unlink<vk::PhysicalDeviceMaintenance9FeaturesKHR>();

    std::vector<const char*> deviceExtensions;
    if (requirements.bPresent)
        deviceExtensions.push_back(vk::KHRSwapchainExtensionName);

    if (m_bMaintenance8Enabled)
        deviceExtensions.push_back(vk::KHRMaintenance8ExtensionName);
    if (m_bMaintenance9Enabled)
        deviceExtensions.push_back(vk::KHRMaintenance9ExtensionName);

    vk::DeviceCreateInfo deviceCreateInfo{
        .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledExtensionCount = (uint32_t)deviceExtensions.size(),
        .ppEnabledExtensionNames = deviceExtensions.data()};

    m_Device = vk::raii::Device(m_PhysicalDevice, deviceCreateInfo);
    SetVkDebugName(m_Device, *m_Device, vk::ObjectType::eDevice, "Device");
    m_GraphicsQueue = vk::raii::Queue(m_Device, m_QueueFamilies.Graphics, 0);
    SetVkDebugName(m_Device, *m_GraphicsQueue, vk::ObjectType::eQueue, "Graphics Queue");

    if (m_QueueFamilies.IsDedicated(QueueType::Copy))
    {
        m_CopyQueue = vk::raii::Queue(m_Device, m_QueueFamilies.Copy, 0);
        SetVkDebugName(m_Device, *m_CopyQueue, vk::ObjectType::eQueue, "Copy Queue");
    }

    // Sized per set rather than per pool, and it grows: bind groups are created
    // at startup and rebuilt on resize, so the count is small but not a number
    // this layer should be asserting a ceiling on.
    static constexpr std::array kBindGroupSizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1u},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 4u},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1u}};
    m_BindGroupAllocator =
        std::make_unique<DescriptorAllocator>(m_Device, kBindGroupSizes, 16u, "Bind Groups");

    // Setting debug names for objects which were created before the device was
    // created.
    SetVkDebugName(m_Device, *m_Instance, vk::ObjectType::eInstance, "Instance");
    SetVkDebugName(m_Device, *m_PhysicalDevice, vk::ObjectType::ePhysicalDevice, "Physical Device");
    // The surface is deliberately not named. It is created by the loader rather than by the
    // driver, so an ICD has no object of its own to attach a name to: Mesa falls back to a side
    // hash table, creates it on the first such name and never frees it, which every windowed
    // sanitizer run then reports as a 600-byte leak. Naming it is legal —
    // VUID-vkSetDebugUtilsObjectNameEXT-pNameInfo-07872 allows an instance-level object to be
    // named through a device descended from the same instance — but the name is worth less than
    // the report it costs, and a surface is not something a capture is read through anyway.
}

} // namespace Hikari::Rhi::Vulkan

namespace Hikari::Rhi
{
std::unique_ptr<IDevice> CreateDevice(const DeviceDesc& desc)
{
    return std::make_unique<Vulkan::VulkanDevice>(desc);
}
} // namespace Hikari::Rhi
