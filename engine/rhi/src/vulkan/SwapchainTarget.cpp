#include "vulkan/SwapchainTarget.h"

#include <format>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <core/Log.h>

#include "vulkan/DebugNames.h"
#include <core/Extent2D.h>
#include <rhi/TextureDesc.h>
#include <rhi/TextureViewDesc.h>
#include <rhi/vulkan/SwapchainUtil.h>
#include <rhi/vulkan/VulkanNative.h>

#include "vulkan/VulkanDevice.h"

namespace Hikari::Rhi::Vulkan
{
constexpr Core::LogCategory LogRhi("RHI");
SwapchainTarget::SwapchainTarget(VulkanDevice& device, const PresentTargetDesc& desc)
    : m_Device(device), m_FramesInFlight(desc.FramesInFlight)
{
    if (m_FramesInFlight == 0u)
        throw std::runtime_error("PresentTargetDesc::FramesInFlight must be at least 1.");

    Create(desc.Extent);
}

SwapchainTarget::~SwapchainTarget()
{
    Destroy();
}

Core::Extent2D SwapchainTarget::GetExtent() const
{
    return Core::Extent2D{m_Extent.width, m_Extent.height};
}

void SwapchainTarget::Create(Core::Extent2D extent)
{
    vk::raii::PhysicalDevice& physicalDevice = m_Device.GetPhysicalDevice();
    vk::raii::SurfaceKHR& surface = m_Device.GetSurface();

    const vk::SurfaceCapabilitiesKHR capabilities =
        physicalDevice.getSurfaceCapabilitiesKHR(surface);
    m_SurfaceFormat = ChooseSwapchainFormat(physicalDevice.getSurfaceFormatsKHR(surface));
    m_Extent = ChooseSwapchainExtent(capabilities, vk::Extent2D{extent.Width, extent.Height});

    // Kept rather than passed straight into the create info: the default is a
    // preference, so which mode the surface actually offered is a fact about
    // this target, and the run report carries it.
    m_PresentMode = ChoosePresentMode(physicalDevice.getSurfacePresentModesKHR(surface));

    const vk::SwapchainCreateInfoKHR createInfo{
        .surface = *surface,
        .minImageCount = ChooseSwapMinImageCount(capabilities),
        .imageFormat = m_SurfaceFormat.format,
        .imageColorSpace = m_SurfaceFormat.colorSpace,
        .imageExtent = m_Extent,
        .imageArrayLayers = 1,
        .imageUsage =
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = m_PresentMode,
        .clipped = true,
        .oldSwapchain = nullptr};

    vk::raii::Device& device = m_Device.GetDevice();
    m_Swapchain = vk::raii::SwapchainKHR(device, createInfo);
    SetVkDebugName(device, *m_Swapchain, vk::ObjectType::eSwapchainKHR, "Swapchain");

    // Every swapchain texture and view is described in neutral terms, so a
    // surface offering nothing Rhi::Format can name is an unrecoverable init
    // failure rather than something to fall back from — that is the deal the
    // curated format list makes. ChooseSwapchainFormat asks for BGRA8Unorm
    // first, which every desktop surface offers.
    m_Format = FromNativeFormat(m_SurfaceFormat.format);

    const std::vector<vk::Image> images = m_Swapchain.getImages();
    m_Images.reserve(images.size());
    for (size_t i = 0; i < images.size(); i++)
    {
        // Registered rather than created: the images belong to the presentation
        // engine, and a handle is only how the rest of the RHI names one.
        const TextureDesc textureDesc{.Format = m_Format,
                                      .Extent = {m_Extent.width, m_Extent.height, 1u},
                                      .Usage =
                                          TextureUsage::ColorAttachment | TextureUsage::CopySrc,
                                      .DebugName = std::format("Swapchain Image_{}", i)};

        Image image{};
        image.Texture = m_Device.RegisterExternalTexture(images[i], textureDesc);
        image.View = m_Device.CreateTextureView(TextureViewDesc{
            .Texture = image.Texture, .DebugName = std::format("Swapchain Image View_{}", i)});
        image.RenderComplete =
            m_Device.CreateSemaphore(std::format("Render Complete Semaphore_{}", i));

        m_Images.push_back(image);
    }

    m_AcquireSemaphores.reserve(m_FramesInFlight);
    for (uint32_t i = 0u; i < m_FramesInFlight; i++)
        m_AcquireSemaphores.push_back(
            m_Device.CreateSemaphore(std::format("Acquire Semaphore_{}", i)));

    m_AcquireIndex = 0u;

    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "Swapchain: {}x{}, {} images", m_Extent.width,
                 m_Extent.height, m_Images.size());
}

void SwapchainTarget::Destroy()
{
    // Views before the images they were made from — a VkImageView outliving its
    // VkImage is undefined behaviour rather than something the driver
    // diagnoses — and both before the swapchain that owns the images.
    for (const Image& image : m_Images)
    {
        m_Device.Destroy(image.View);
        m_Device.Destroy(image.Texture);
        m_Device.Destroy(image.RenderComplete);
    }
    m_Images.clear();

    for (SemaphoreHandle semaphore : m_AcquireSemaphores)
        m_Device.Destroy(semaphore);
    m_AcquireSemaphores.clear();

    m_Swapchain = nullptr;
}

AcquiredImage SwapchainTarget::Acquire()
{
    const size_t acquireIndex = m_AcquireIndex;
    const SemaphoreHandle available = m_AcquireSemaphores[acquireIndex];

    // vk::raii throws on the error results, so the out-of-date case — which is
    // ordinary rather than exceptional, since a resize races every frame —
    // arrives here as an exception and is turned back into a value.
    vk::Result result = vk::Result::eSuccess;
    uint32_t imageIndex = 0u;
    try
    {
        std::tie(result, imageIndex) =
            m_Swapchain.acquireNextImage(UINT64_MAX, m_Device.GetSemaphore(available), nullptr);
    }
    catch (const vk::OutOfDateKHRError&)
    {
        AcquiredImage needsRecreate{};
        needsRecreate.bNeedsRecreate = true;
        return needsRecreate;
    }

    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        AcquiredImage needsRecreate{};
        needsRecreate.bNeedsRecreate = true;
        return needsRecreate;
    }

    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
        throw std::runtime_error("Failed to acquire next swapchain image!");

    // Only on success: a failed acquire signals nothing, so the semaphore is
    // still unsignalled and has to be the one the next attempt uses. Advancing
    // regardless would walk the ring on every frame of a resize and hand out a
    // semaphore whose acquire is still outstanding.
    m_AcquireIndex = (m_AcquireIndex + 1u) % m_FramesInFlight;

    const Image& image = m_Images[imageIndex];

    // The span points at the ring slot rather than at a copy, so it stays valid
    // exactly as long as AcquiredImage documents: until the next Acquire or
    // Recreate, neither of which resizes the ring without rebuilding it.
    return AcquiredImage{.Texture = image.Texture,
                         .View = image.View,
                         .Index = imageIndex,
                         .WaitSemaphores = std::span(&m_AcquireSemaphores[acquireIndex], 1u),
                         .bNeedsRecreate = false};
}

SemaphoreHandle SwapchainTarget::GetRenderCompleteSemaphore(uint32_t index) const
{
    if (index >= m_Images.size())
        throw std::runtime_error("IPresentTarget::GetRenderCompleteSemaphore: index out of range.");

    return m_Images[index].RenderComplete;
}

bool SwapchainTarget::Present(uint32_t index)
{
    if (index >= m_Images.size())
        throw std::runtime_error("IPresentTarget::Present: index out of range.");

    const vk::Semaphore waitOn = m_Device.GetSemaphore(m_Images[index].RenderComplete);
    const vk::PresentInfoKHR presentInfo{.waitSemaphoreCount = 1u,
                                         .pWaitSemaphores = &waitOn,
                                         .swapchainCount = 1u,
                                         .pSwapchains = &*m_Swapchain,
                                         .pImageIndices = &index};

    vk::Result result = vk::Result::eSuccess;
    try
    {
        result = m_Device.GetGraphicsQueue().presentKHR(presentInfo);
    }
    catch (const vk::OutOfDateKHRError&)
    {
        return false;
    }

    if (result == vk::Result::eSuboptimalKHR || result == vk::Result::eErrorOutOfDateKHR)
        return false;

    if (result != vk::Result::eSuccess)
        throw std::runtime_error("Failed to present image!");

    return true;
}

bool SwapchainTarget::Recreate(Core::Extent2D newExtent)
{
    // Asked before anything is destroyed, so that a surface which cannot back a
    // swapchain leaves the existing one intact and presentable. A minimised
    // window is how that state is reached, and it is not rare: SDL minimises
    // the window itself when focus is lost from exclusive fullscreen. Waiting
    // idle and tearing down first would give up a working swapchain to build
    // nothing, once per frame, for as long as the window stayed minimised.
    vk::raii::PhysicalDevice& physicalDevice = m_Device.GetPhysicalDevice();
    const vk::SurfaceCapabilitiesKHR capabilities =
        physicalDevice.getSurfaceCapabilitiesKHR(m_Device.GetSurface());
    if (!CanCreateSwapchain(capabilities, vk::Extent2D{newExtent.Width, newExtent.Height}))
        return false;

    // Everything below is either in use by work still in flight or about to be
    // destroyed while it is, and there is no finer-grained wait available: the
    // semaphores being rebuilt are the ones that would have ordered it.
    m_Device.WaitIdle();

    Destroy();
    Create(newExtent);
    return true;
}
} // namespace Hikari::Rhi::Vulkan
