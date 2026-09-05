#include <platform/ScriptedInputSource.h>

namespace Hikari::Platform
{

std::span<const PlatformEvent> ScriptedInputSource::Advance()
{
    m_Events.clear();

    for (const ScriptedEvent& scripted : m_Script.Events())
    {
        if (scripted.Frame != m_Frame)
            continue;

        if (scripted.Event.Type == EventType::KeyDown)
            m_KeysDown.insert(static_cast<uint8_t>(scripted.Event.key));
        else if (scripted.Event.Type == EventType::KeyUp)
            m_KeysDown.erase(static_cast<uint8_t>(scripted.Event.key));

        m_Events.push_back(scripted.Event);
    }

    ++m_Frame;
    return m_Events;
}

size_t ScriptedInputSource::UndeliveredCount() const
{
    size_t undelivered = 0u;
    for (const ScriptedEvent& scripted : m_Script.Events())
        undelivered += scripted.Frame >= m_Frame ? 1u : 0u;

    return undelivered;
}

void ReportUndeliveredScriptEvents(const ScriptedInputSource& source,
                                   const Core::LogCategory& category)
{
    if (!source.HasScript())
        return;

    const size_t undelivered = source.UndeliveredCount();
    if (undelivered == 0u)
        return;

    Core::LogMsg(Core::LogSeverity::Warning, category,
                 "{} input script event(s) never fired: the run ended before the frames they "
                 "name. A script that outruns its own run does nothing and looks like input "
                 "that had no effect.",
                 undelivered);
}

} // namespace Hikari::Platform
