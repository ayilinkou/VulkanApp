#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

#include <core/Log.h>

#include <platform/InputScript.h>
#include <platform/PlatformEvent.h>

namespace Hikari::Platform
{

/**
 * A script being replayed, and the held-key state it implies.
 *
 * Shared by both platforms rather than living in the headless one: a scripted
 * run is worth watching in a window, and a replay that only works where nobody
 * can see it is a replay nobody trusts. What differs between the two is what
 * happens around it — SdlPlatform merges these events with the window system's,
 * HeadlessPlatform has none to merge with.
 */
class ScriptedInputSource
{
public:
    void SetScript(InputScript script) { m_Script = std::move(script); }

    bool HasScript() const { return !m_Script.Events().empty(); }

    /**
     * The events this frame calls for, and the frame counter moves on.
     *
     * A frame is one call: neither platform has anything better to count by,
     * and keying a script to frames rather than to time is what makes a replay
     * land in the same place whatever the machine managed.
     */
    std::span<const PlatformEvent> Advance();

    /**
     * Events on frames the run never reached.
     *
     * Worth reporting rather than ignoring: a script whose frame numbers outrun
     * the run does nothing and says nothing, which reads exactly like input that
     * had no effect.
     */
    size_t UndeliveredCount() const;

    /** Held from a scripted key.down until the matching key.up. */
    bool IsKeyDown(Key key) const { return m_KeysDown.contains(static_cast<uint8_t>(key)); }

private:
    InputScript m_Script;
    uint64_t m_Frame = 0u;
    std::vector<PlatformEvent> m_Events;
    std::unordered_set<uint8_t> m_KeysDown;
};

/**
 * Logs a warning for a script whose events the run never reached.
 *
 * Called by each platform as it goes, which is after the run has ended. Silent
 * when everything fired, so a well-behaved script says nothing.
 */
void ReportUndeliveredScriptEvents(const ScriptedInputSource& source,
                                   const Core::LogCategory& category);

} // namespace Hikari::Platform
