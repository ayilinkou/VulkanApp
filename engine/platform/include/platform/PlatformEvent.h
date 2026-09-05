#pragma once

#include <cstdint>

#include <core/Extent2D.h>

namespace Hikari::Platform
{

/** The keys anything above this seam asks about, named rather than scan-coded. */
enum class Key : uint8_t
{
    Unknown,
    W,
    A,
    S,
    D,
    Q,
    E,
    Escape,
    F9,
    F10,
    F11,
};

/** What happened. One event carries exactly the fields its type names. */
enum class EventType : uint8_t
{
    /**
     * Something the window system reported that this seam does not name. Kept
     * rather than dropped because the native event still reaches the UI
     * backend, whose business text input and mouse buttons are.
     */
    Unknown,

    /** The window system, the user or a script asked the run to end. */
    Quit,

    /** The drawable surface changed size; the new extent is in `Size`. */
    Resized,

    FocusGained,
    FocusLost,

    /** `key` went down or came up. Repeats arrive as further KeyDown events. */
    KeyDown,
    KeyUp,

    /** Relative motion since the last event, in `MouseDelta`. */
    MouseMotion,

    /**
     * Capture this frame. Not something a window system produces — a scripted
     * run raises it so that a capture can be asked for at a chosen frame rather
     * than only at the last one.
     */
    CaptureRequested,
};

/**
 * One thing that happened, in terms the engine can act on without naming a
 * window system.
 */
struct PlatformEvent
{
    EventType Type = EventType::Quit;

    Key key = Key::Unknown;
    Core::Extent2D Size{};
    float MouseDeltaX = 0.f;
    float MouseDeltaY = 0.f;

    /**
     * The backing native event, or null where there is none — a scripted event
     * has no window system behind it.
     *
     * TEMPORARY, and the same escape hatch as GetNativeWindowHandle: the UI
     * library's platform backend takes the window system's own event type, and
     * there is no neutral shape for that short of driving the UI from this
     * enum instead. It disappears when the UI's input path does.
     */
    const void* pNative = nullptr;
};

} // namespace Hikari::Platform
