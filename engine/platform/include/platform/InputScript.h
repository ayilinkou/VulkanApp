#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <platform/PlatformEvent.h>

namespace Hikari::Platform
{

/** Thrown for a line the format does not describe. Names the line number. */
class InputScriptError : public std::runtime_error
{
public:
    explicit InputScriptError(const std::string& message) : std::runtime_error(message) {}
};

/** One event and the frame it is delivered on. */
struct ScriptedEvent
{
    uint64_t Frame = 0u;
    PlatformEvent Event;
};

/**
 * A recorded sequence of input, replayed through the same event path a window
 * system's events take.
 *
 * The point is that a headless run exercises input handling, resize handling and
 * target recreation — which is where a large share of real crashes live, and
 * none of which `--frames N` alone reaches.
 *
 * The format is one command per line, blank lines and `#` comments ignored:
 *
 *     frame 5   key.down W
 *     frame 15  key.up W
 *     frame 20  window.resize 320x240
 *     frame 30  screenshot
 *     frame 40  quit
 *
 * Unknown commands are an error rather than a silent skip: a typo in a script
 * that quietly does nothing is a test that quietly stops testing. `camera.set`
 * from the plan's sketch is deliberately not implemented — a camera is engine
 * state rather than platform input, and `--camera-preset` already places it
 * deterministically.
 */
class InputScript
{
public:
    /** Parses `text`, throwing InputScriptError on the first line it cannot read. */
    static InputScript Parse(std::string_view text);

    /** The same, reading the file at `path` first. */
    static InputScript Load(const std::string& path);

    /** In the order they were written; two events on one frame keep their order. */
    std::span<const ScriptedEvent> Events() const { return m_Events; }

    /** The frame the last event lands on, or 0 for an empty script. */
    uint64_t LastFrame() const;

    /**
     * Whether the script ends the run itself, with a quit.
     *
     * What lets a headless run go without --frames: something has to be able to
     * end a run with no window, and a script that quits is as good an answer as
     * a frame count.
     */
    bool EndsRun() const;

private:
    std::vector<ScriptedEvent> m_Events;
};

} // namespace Hikari::Platform
