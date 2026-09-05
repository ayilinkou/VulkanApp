#include <platform/HeadlessPlatform.h>

#include <core/Log.h>

namespace Hikari::Platform
{

namespace
{
constexpr Core::LogCategory LogHeadless("Headless");

/**
 * The size a headless run renders at when --resolution said nothing. There is
 * no display to size against, so it can only be a documented constant; this one
 * matches SdlPlatform's kFallbackWindowSize, which is what that platform falls
 * back to when it cannot query a display either.
 *
 * Resolved here rather than by the caller so that "zero means the platform
 * decides" keeps one meaning across both implementations, instead of being
 * SdlPlatform's rule for one path and main()'s for the other.
 */
constexpr Core::Extent2D kDefaultHeadlessSize{1280u, 720u};
} // namespace

HeadlessPlatform::HeadlessPlatform(const WindowDesc& desc)
    : m_Extent(desc.Width == 0u || desc.Height == 0u ? kDefaultHeadlessSize
                                                     : Core::Extent2D{desc.Width, desc.Height})
{
    Core::LogMsg(Core::LogSeverity::Info, LogHeadless, "Running headless at {}x{}", m_Extent.Width,
                 m_Extent.Height);
}

void HeadlessPlatform::SetWindowMode(WindowMode) {}

void HeadlessPlatform::SetRelativeMouseMode(bool) {}

void HeadlessPlatform::WarpMouse(float, float) {}
HeadlessPlatform::~HeadlessPlatform()
{
    ReportUndeliveredScriptEvents(m_Input, LogHeadless);
}

std::span<const PlatformEvent> HeadlessPlatform::PumpEvents()
{
    const std::span<const PlatformEvent> events = m_Input.Advance();

    // Applied here rather than left to the caller: a resize event from a window
    // system arrives *after* the surface changed, and a replay that reported one
    // without changing the extent would describe a state this platform was never
    // in.
    for (const PlatformEvent& event : events)
    {
        if (event.Type == EventType::Resized)
            m_Extent = event.Size;
    }

    return events;
}

bool HeadlessPlatform::IsKeyDown(Key key) const
{
    return m_Input.IsKeyDown(key);
}

} // namespace Hikari::Platform
