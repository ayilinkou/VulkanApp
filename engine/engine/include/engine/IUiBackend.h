#pragma once

#include <cstdint>

#include <rhi/ICommandList.h>
#include <rhi/IDevice.h>
#include <rhi/PipelineCache.h>
#include <rhi/RhiTypes.h>

namespace Hikari::Engine
{

/** What a UI backend needs to bring itself up against the current target. */
struct UiBackendDesc
{
    Rhi::IDevice* pDevice = nullptr;
    Rhi::IPipelineCache* pPipelineCache = nullptr;

    /**
     * Null for a run with no window. The platform half of a UI backend is the
     * only half that needs one; the rendering half draws into whatever image
     * the present target hands back, window or not.
     */
    void* pNativeWindowHandle = nullptr;

    /** The format the UI is composited onto, which its pipeline is built for. */
    Rhi::Format TargetFormat = Rhi::Format::Undefined;

    /**
     * How many slots the backend's own per-frame ring needs. At least the
     * engine's frames in flight: a shorter ring would be overwritten while an
     * earlier frame was still reading it.
     */
    uint32_t RingSize = 2u;
};

/**
 * The UI layer's rendering and platform integration, behind an interface so
 * that the engine's frame loop never names one.
 *
 * There is one implementation today, over Vulkan. A second backend is a sibling
 * file rather than an edit here, which is the point: the UI library's
 * integration is inherently backend-specific, so it is demoted to a leaf rather
 * than abstracted away.
 */
class IUiBackend
{
public:
    virtual ~IUiBackend();

    virtual void Init(const UiBackendDesc& desc) = 0;
    virtual void Shutdown() = 0;

    /** Starts a UI frame. Panels are built between this and Render. */
    virtual void NewFrame() = 0;

    /** Records the frame's UI into an already-open rendering scope. */
    virtual void Render(Rhi::ICommandList& commandList) = 0;

    /** The present target was rebuilt, possibly with a different image count. */
    virtual void OnTargetRecreated(uint32_t imageCount, Rhi::Format targetFormat) = 0;

    /** One platform event, in whatever form the platform delivers it. */
    virtual void ProcessPlatformEvent(const void* pEvent) = 0;
};

} // namespace Hikari::Engine
