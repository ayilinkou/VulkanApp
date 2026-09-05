#pragma once

#include <cstdint>
#include <span>
#include <string>

#include <core/Extent2D.h>
#include <platform/PlatformEvent.h>

namespace Hikari::Platform
{

struct WindowDesc
{
    /**
     * Zero asks the platform to pick a size from the display it opens on. It
     * cannot be a default here: how big the display is, and which one is the
     * main one, are unknown until the window system has been initialised.
     */
    uint32_t Width = 0u;
    uint32_t Height = 0u;
    std::string Title = "Hikari Engine";
    bool bResizable = true;
    bool bBorderless = false;
};

/**
 * How the window covers the display. One value rather than a bool plus a mode,
 * because "windowed, but exclusive" is not a state that exists.
 */
enum class WindowMode
{
    Windowed,

    /**
     * The window is resized to fill the display and loses its decorations, but
     * stays an ordinary composited window and the display's video mode is not
     * touched. Alt-tab is immediate and nothing else on the desktop is
     * disturbed, which is why this is the one to reach for by default.
     */
    BorderlessFullscreen,

    /**
     * A real display mode is selected for the fullscreen window. What that
     * buys differs by platform: on Windows it is a video mode change, while on
     * Wayland — where a client cannot mode-set at all — the compositor scales
     * a surface sized to the mode instead. Neither grants exclusive ownership
     * of the display in the Vulkan sense; that is VK_EXT_full_screen_exclusive,
     * which is Windows-only and not something the RHI asks for.
     */
    ExclusiveFullscreen,
};

/**
 * The windowing/OS seam, with two implementations: SdlPlatform opens a real
 * window, HeadlessPlatform has none at all. Having both is what lets the engine
 * run in CI with no display attached — and what keeps the renderer above this
 * line unable to tell which it got.
 */
class IPlatform
{
public:
    virtual ~IPlatform() = default;

    virtual bool IsHeadless() const = 0;

    /**
     * Size of the drawable surface in *pixels*, which differs from the window
     * size in screen coordinates on high-DPI displays — so this, not the
     * window size, is what the swapchain must be sized against. Reports
     * {0, 0} while the window is minimised.
     */
    virtual Core::Extent2D GetFramebufferExtent() const = 0;

    /**
     * Reveals the window, which is created hidden so that initialisation is
     * not visible as a blank frame.
     */
    virtual void Show() = 0;

    /**
     * Asynchronous on some window systems, and a request the window system is
     * allowed to refuse. The resulting size change arrives as a normal resize
     * event, so callers rebuild nothing here.
     */
    virtual void SetWindowMode(WindowMode mode) = 0;

    /**
     * Everything that happened since the last call, in order.
     *
     * The span is valid until the next call: a caller walks it within the frame
     * that asked for it. This is the only way input reaches the engine — polling
     * a window system directly is what this seam exists to prevent, because it
     * is also what a headless run cannot do.
     */
    virtual std::span<const PlatformEvent> PumpEvents() = 0;

    /**
     * Whether `key` is held right now.
     *
     * Separate from the event stream because held-key movement asks a different
     * question from key transitions: a frame wants to know what is down, not
     * what changed. Implementations track this from the same source their
     * events come from.
     */
    virtual bool IsKeyDown(Key key) const = 0;

    virtual void SetRelativeMouseMode(bool bEnabled) = 0;
    virtual void WarpMouse(float x, float y) = 0;

    /**
     * TEMPORARY. Vulkan surface creation (moves into rhi at step 35) and the
     * ImGui SDL3 backend (moves into engine/editor at step 53) both still run
     * inside App and need the concrete SDL_Window*. Those two callers are the
     * only permitted users; this disappears once they move.
     */
    virtual void* GetNativeWindowHandle() const = 0;
};
} // namespace Hikari::Platform
