#pragma once

#include <span>

#include <core/Extent2D.h>
#include <platform/IPlatform.h>
#include <platform/InputScript.h>
#include <platform/ScriptedInputSource.h>

namespace Hikari::Platform
{

/**
 * IPlatform with no window system behind it at all — the implementation that
 * lets the engine run in CI, where there is no display to open a window on.
 *
 * It needs no SDL. Nothing consumes SDL's Vulkan loader: the RHI reaches Vulkan
 * through vulkan.hpp's own dispatcher, which is why the gpu tests create real
 * devices with no SDL anywhere. So this is a plain class, and constructing one
 * touches no window-system subsystem, initialised or not.
 *
 * Takes a WindowDesc, the same as SdlPlatform, so that the caller builds one
 * description and hands it to whichever implementation it picked — being able
 * to swap the two behind IPlatform is the entire point of the seam. Title,
 * bResizable and bBorderless are ignored; a run with no window has no use for
 * any of them, and a window mode cannot reach here anyway because --headless
 * with --borderless or --fullscreen is rejected at parse time.
 */
class HeadlessPlatform final : public IPlatform
{
public:
    explicit HeadlessPlatform(const WindowDesc& desc);
    ~HeadlessPlatform() override;

    HeadlessPlatform(const HeadlessPlatform&) = delete;
    HeadlessPlatform& operator=(const HeadlessPlatform&) = delete;

    bool IsHeadless() const override { return true; }

    /**
     * Never {0, 0}, unlike SdlPlatform's: there is no window to be minimised.
     * It moves only where a script asks it to, which is the point of a scripted
     * resize — target recreation is exercised without a window system.
     */
    Core::Extent2D GetFramebufferExtent() const override { return m_Extent; }

    /**
     * All no-ops, and silent. Each asks the window system for something, and
     * there is no window system; none of them is reachable in a headless run
     * anyway — the window-mode flags are rejected at parse time, and the two
     * cursor calls are driven by key events that never arrive.
     */
    /**
     * The events this frame's script entries call for, and nothing else. Each
     * call is one frame: there is no window system to ask, so the frame number
     * is the count of calls made so far.
     */
    std::span<const PlatformEvent> PumpEvents() override;

    /** Held keys, tracked from the script's own key.down and key.up entries. */
    bool IsKeyDown(Key key) const override;

    /**
     * Replays `script` as the run proceeds. Without one a headless run receives
     * no events at all, which is what every run before scripted input did.
     */
    void SetInputScript(InputScript script) { m_Input.SetScript(std::move(script)); }

    void Show() override {}
    void SetWindowMode(WindowMode mode) override;
    void SetRelativeMouseMode(bool bEnabled) override;
    void WarpMouse(float x, float y) override;

    /**
     * Null, which is what the RHI reads as "no surface": DeviceDesc's
     * Requirements.bPresent is what actually decides, and App derives it from
     * IsHeadless().
     */
    void* GetNativeWindowHandle() const override { return nullptr; }

private:
    ScriptedInputSource m_Input;

    Core::Extent2D m_Extent{};
};
} // namespace Hikari::Platform
