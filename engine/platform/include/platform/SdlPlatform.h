#pragma once

#include <stdexcept>
#include <string>

#include <memory>
#include <span>

#include <platform/IPlatform.h>
#include <platform/InputScript.h>
#include <platform/ScriptedInputSource.h>

struct SDL_Window;

namespace Hikari::Platform
{

/** Thrown when an SDL call fails; appends SDL_GetError() to the message. */
class SDLException : public std::runtime_error
{
public:
    explicit SDLException(const std::string& message);
};

/**
 * Owns SDL init and the window — brought up in that order by the constructor,
 * torn down in reverse by the destructor.
 *
 * An SdlPlatform must outlive every object holding a Vulkan handle, because its
 * destructor destroys the window and that invalidates the surface created from
 * it. Destroying a VkSurfaceKHR after its window is gone, or presenting to it,
 * is use-after-free — the swapchain built on that surface has to go first.
 *
 * The Vulkan library itself is not the reason. SDL loads it as part of creating
 * an SDL_WINDOW_VULKAN window and unloads it in SDL_DestroyWindow, and the app's
 * own dispatcher holds its own reference regardless of what SDL does.
 */
class SdlPlatform final : public IPlatform
{
public:
    explicit SdlPlatform(const WindowDesc& desc);
    ~SdlPlatform() override;

    SdlPlatform(const SdlPlatform&) = delete;
    SdlPlatform& operator=(const SdlPlatform&) = delete;

    bool IsHeadless() const override { return false; }
    Core::Extent2D GetFramebufferExtent() const override;
    void Show() override;
    void SetWindowMode(WindowMode mode) override;
    std::span<const PlatformEvent> PumpEvents() override;
    bool IsKeyDown(Key key) const override;

    /**
     * Replays `script` alongside whatever the window system reports, so a
     * scripted run can be watched rather than only asserted on. A scripted
     * resize asks the window system to resize the window, and the resize event
     * that follows is the window system's own — synthesising one here would
     * report a size the window did not have.
     */
    void SetInputScript(InputScript script) { m_Input.SetScript(std::move(script)); }
    void SetRelativeMouseMode(bool bEnabled) override;
    void WarpMouse(float x, float y) override;
    void* GetNativeWindowHandle() const override;

    /**
     * Static because it is called from the handler for "constructing an
     * SdlPlatform failed", where there is no instance to call it on.
     */
    static void ShowErrorMessageBox(const char* title, const char* message);

private:
    SDL_Window* m_pWindow = nullptr;

    /**
     * A frame's events, translated and native, refilled by each PumpEvents call
     * and handed out as a span.
     *
     * Behind a pointer because SDL_Event is a union and cannot be forward
     * declared, and this header is included by code that has no business
     * seeing SDL. The natives are kept because PlatformEvent::pNative points
     * into them: a pointer to a poll-loop local would dangle before the caller
     * read it.
     */
    struct EventBuffer;
    std::unique_ptr<EventBuffer> m_pEvents;

    ScriptedInputSource m_Input;
};
} // namespace Hikari::Platform
