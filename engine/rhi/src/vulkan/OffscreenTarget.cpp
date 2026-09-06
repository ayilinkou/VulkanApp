#include "vulkan/OffscreenTarget.h"

#include <array>
#include <cstring>
#include <format>
#include <limits>
#include <span>
#include <stdexcept>

#include "vulkan/vulkan_raii.hpp"

#include <core/Log.h>

#include "vulkan/DebugNames.h"
#include <core/Extent2D.h>
#include <rhi/BufferDesc.h>
#include <rhi/ICommandList.h>
#include <rhi/TextureDesc.h>
#include <rhi/TextureViewDesc.h>
#include <rhi/UniqueHandle.h>

#include "vulkan/VulkanCommandList.h"

#include "vulkan/VulkanConversions.h"
#include "vulkan/VulkanDevice.h"

namespace Hikari::Rhi::Vulkan
{
constexpr Core::LogCategory LogRhi("RHI");
namespace
{
/**
 * Everything an offscreen image is used for: rendered into, copied out of for a
 * screenshot or a readback, and sampled by a pass that composites over it.
 */
constexpr TextureUsage kOffscreenUsage =
    TextureUsage::ColorAttachment | TextureUsage::CopySrc | TextureUsage::Sampled;

/**
 * The format the images are created with.
 *
 * BGRA8Unorm first, for a reason that has nothing to do with the format itself:
 * ChooseSwapchainFormat asks for it first too, so a headless capture and a
 * windowed one come out in the same byte order and can be compared directly.
 * A headless run whose screenshots needed a different comparison than a
 * windowed run's would not be much of a regression harness.
 *
 * Queried rather than assumed. The Vulkan required-format tables do guarantee
 * colour-attachment and sampled support for both of these, but a target that
 * silently produces an unusable image on the one device that disagrees is worse
 * than one that says so at creation — and the query is two lines.
 */
Format ChooseFormat(vk::raii::PhysicalDevice& physicalDevice)
{
    constexpr std::array kPreferred{Format::BGRA8Unorm, Format::RGBA8Unorm};

    constexpr vk::FormatFeatureFlags kRequired = vk::FormatFeatureFlagBits::eColorAttachment |
                                                 vk::FormatFeatureFlagBits::eSampledImage |
                                                 vk::FormatFeatureFlagBits::eTransferSrc;

    for (const Format format : kPreferred)
    {
        const vk::FormatProperties properties = physicalDevice.getFormatProperties(ToVk(format));

        if ((properties.optimalTilingFeatures & kRequired) == kRequired)
            return format;
    }

    throw std::runtime_error("Rhi::IDevice::CreatePresentTarget: this device supports neither "
                             "BGRA8Unorm nor RGBA8Unorm as an offscreen render target.");
}
} // namespace

OffscreenTarget::OffscreenTarget(VulkanDevice& device, const PresentTargetDesc& desc)
    : m_Device(device), m_ImageCount(desc.FramesInFlight)
{
    if (desc.FramesInFlight == 0u)
        throw std::runtime_error("PresentTargetDesc::FramesInFlight must be at least 1.");

    m_Format = ChooseFormat(m_Device.GetPhysicalDevice());

    Create(desc.Extent);
}

OffscreenTarget::~OffscreenTarget()
{
    Destroy();
}

void OffscreenTarget::Create(Core::Extent2D extent)
{
    m_Extent = extent;

    m_Images.reserve(m_ImageCount);
    for (uint32_t i = 0u; i < m_ImageCount; i++)
    {
        // Created rather than registered, which is the whole difference from
        // SwapchainTarget: these images are ours, so they are ordinary textures
        // that the device allocated and will free.
        //
        // Appended one at a time rather than written into a pre-sized vector so
        // that m_Images only ever holds entries that exist. A CreateTexture that
        // throws part-way through then leaves Destroy() something it can tear
        // down, instead of a tail of invalid handles it would report as
        // double-frees on the way out.
        Image image{};
        image.Texture =
            m_Device.CreateTexture(TextureDesc{.Format = m_Format,
                                               .Extent = {m_Extent.Width, m_Extent.Height, 1u},
                                               .Usage = kOffscreenUsage,
                                               .DebugName = std::format("Offscreen Image_{}", i)});
        image.View = m_Device.CreateTextureView(TextureViewDesc{
            .Texture = image.Texture, .DebugName = std::format("Offscreen Image View_{}", i)});
        image.RenderComplete =
            m_Device.CreateSemaphore(std::format("Render Complete Semaphore_{}", i));

        m_Images.push_back(image);
    }

    // Reset with the semaphores: the first pass over a fresh set of images has
    // nothing to wait on, and the counter is what says so.
    m_AcquireCount = 0u;
    m_CurrentWait = SemaphoreHandle{};

    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "Offscreen target: {}x{}, {} images",
                 m_Extent.Width, m_Extent.Height, m_Images.size());
}

void OffscreenTarget::Destroy()
{
    // Views before the images they were made from, for the reason
    // SwapchainTarget::Destroy gives: a VkImageView outliving its VkImage is
    // undefined behaviour rather than something the driver diagnoses.
    for (const Image& image : m_Images)
    {
        m_Device.Destroy(image.View);
        m_Device.Destroy(image.Texture);
        m_Device.Destroy(image.RenderComplete);
    }
    m_Images.clear();
}

AcquiredImage OffscreenTarget::Acquire()
{
    const uint32_t index = static_cast<uint32_t>(m_AcquireCount % m_Images.size());
    m_AcquireCount++;

    Image& image = m_Images[index];

    AcquiredImage acquired{};
    acquired.Texture = image.Texture;
    acquired.View = image.View;
    acquired.Index = index;

    // The previous frame that wrote this image signalled its render-complete
    // semaphore and nothing has waited on it, because there is no presentation
    // engine to do the waiting. Handing it back here does both jobs at once:
    //
    //   * It is the real dependency. The image is about to be written again,
    //     and the previous write has to have finished first. Expressing it here
    //     is what keeps the image count independent of anything the caller does
    //     with fences — a caller that waits on a per-frame fence covers this
    //     hazard already, but nothing in IPresentTarget says it must.
    //   * It leaves the semaphore unsignalled, which is what a binary semaphore
    //     needs before it may be signalled again. The wait executes before the
    //     batch's commands and the signal after them, so the caller's own
    //     submit may legally wait on and signal the same semaphore.
    //
    // Empty on the first pass, when nothing has written the image yet.
    if (image.bSignalPending)
    {
        m_CurrentWait = image.RenderComplete;
        image.bSignalPending = false;
        acquired.WaitSemaphores = std::span(&m_CurrentWait, 1u);
    }

    // Never true: an offscreen target owns its images outright, so nothing can
    // invalidate them behind the caller's back the way a surface can.
    acquired.bNeedsRecreate = false;
    return acquired;
}

SemaphoreHandle OffscreenTarget::GetRenderCompleteSemaphore(uint32_t index) const
{
    if (index >= m_Images.size())
        throw std::runtime_error("IPresentTarget::GetRenderCompleteSemaphore: index out of range.");

    return m_Images[index].RenderComplete;
}

bool OffscreenTarget::Present(uint32_t index)
{
    if (index >= m_Images.size())
        throw std::runtime_error("IPresentTarget::Present: index out of range.");

    // Nothing to present to, so this records only that the caller has signalled
    // the image's render-complete semaphore — which the interface requires
    // before Present may be called, and which the next Acquire of this image
    // consumes.
    m_Images[index].bSignalPending = true;
    return true;
}

std::optional<SemaphoreHandle> OffscreenTarget::TakePendingSignal(uint32_t index)
{
    if (index >= m_Images.size())
        throw std::runtime_error("OffscreenTarget::TakePendingSignal: index out of range.");

    Image& image = m_Images[index];
    if (!image.bSignalPending)
        return std::nullopt;

    // Cleared for the same reason Acquire clears it: the caller's wait consumes
    // the signal, and leaving the flag set would have the next Acquire hand back
    // a wait for a signal that has already been taken.
    image.bSignalPending = false;
    return image.RenderComplete;
}

bool OffscreenTarget::Recreate(Core::Extent2D newExtent)
{
    // A zero extent is the one request that cannot be met — Vulkan images are
    // at least one texel in every dimension — so it takes the same answer a
    // minimised window gets from a swapchain: nothing was touched, ask again.
    // Reporting it as an error instead would make a caller that resizes through
    // zero have to special-case a target it is not supposed to be able to
    // identify.
    if (newExtent.Width == 0u || newExtent.Height == 0u)
        return false;

    // Every image and semaphore below is either in use by work still in flight
    // or about to be destroyed while it is, and the semaphores being rebuilt
    // are the ones that would have ordered it — the same bind SwapchainTarget
    // is in, and the same answer.
    m_Device.WaitIdle();

    Destroy();
    Create(newExtent);
    return true;
}
} // namespace Hikari::Rhi::Vulkan
