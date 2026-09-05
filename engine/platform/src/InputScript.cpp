#include <platform/InputScript.h>

#include <cctype>
#include <charconv>
#include <format>
#include <fstream>
#include <sstream>

namespace Hikari::Platform
{

namespace
{

/** Every key the seam names, matched case-insensitively. */
Key ParseKey(std::string_view name, size_t lineNumber)
{
    std::string lowered(name);
    for (char& character : lowered)
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));

    if (lowered == "w")
        return Key::W;
    if (lowered == "a")
        return Key::A;
    if (lowered == "s")
        return Key::S;
    if (lowered == "d")
        return Key::D;
    if (lowered == "q")
        return Key::Q;
    if (lowered == "e")
        return Key::E;
    if (lowered == "escape")
        return Key::Escape;
    if (lowered == "f9")
        return Key::F9;
    if (lowered == "f10")
        return Key::F10;
    if (lowered == "f11")
        return Key::F11;

    throw InputScriptError(
        std::format("line {}: not a key this format names: {}", lineNumber, name));
}

/** `<width>x<height>`, both required. */
Core::Extent2D ParseExtent(std::string_view text, size_t lineNumber)
{
    const size_t separator = text.find('x');
    if (separator == std::string_view::npos)
        throw InputScriptError(
            std::format("line {}: a size is <width>x<height>, got: {}", lineNumber, text));

    const auto number = [&](std::string_view part)
    {
        uint32_t value = 0u;
        const auto [end, error] = std::from_chars(part.data(), part.data() + part.size(), value);
        if (error != std::errc{} || end != part.data() + part.size())
            throw InputScriptError(std::format("line {}: not a number: {}", lineNumber, part));

        return value;
    };

    return Core::Extent2D{number(text.substr(0, separator)), number(text.substr(separator + 1))};
}

} // namespace

InputScript InputScript::Parse(std::string_view text)
{
    InputScript script;

    std::istringstream stream{std::string(text)};
    std::string line;
    for (size_t lineNumber = 1u; std::getline(stream, line); ++lineNumber)
    {
        if (const size_t comment = line.find('#'); comment != std::string::npos)
            line.erase(comment);

        std::istringstream fields(line);
        std::string keyword;
        if (!(fields >> keyword))
            continue;

        if (keyword != "frame")
            throw InputScriptError(std::format(
                "line {}: every line begins with 'frame <N>', got: {}", lineNumber, keyword));

        uint64_t frame = 0u;
        if (!(fields >> frame))
            throw InputScriptError(
                std::format("line {}: 'frame' needs a frame number", lineNumber));

        std::string command;
        if (!(fields >> command))
            throw InputScriptError(
                std::format("line {}: frame {} has no command", lineNumber, frame));

        PlatformEvent event;
        if (command == "key.down" || command == "key.up")
        {
            std::string keyName;
            if (!(fields >> keyName))
                throw InputScriptError(std::format("line {}: {} needs a key", lineNumber, command));

            event.Type = command == "key.down" ? EventType::KeyDown : EventType::KeyUp;
            event.key = ParseKey(keyName, lineNumber);
        }
        else if (command == "window.resize")
        {
            std::string size;
            if (!(fields >> size))
                throw InputScriptError(
                    std::format("line {}: window.resize needs a size", lineNumber));

            event.Type = EventType::Resized;
            event.Size = ParseExtent(size, lineNumber);
        }
        else if (command == "screenshot")
        {
            event.Type = EventType::CaptureRequested;
        }
        else if (command == "quit")
        {
            event.Type = EventType::Quit;
        }
        else
        {
            throw InputScriptError(
                std::format("line {}: not a command this format names: {}", lineNumber, command));
        }

        script.m_Events.push_back(ScriptedEvent{.Frame = frame, .Event = event});
    }

    return script;
}

InputScript InputScript::Load(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw InputScriptError(std::format("Failed to open input script: {}", path));

    const std::string contents((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
    return Parse(contents);
}

bool InputScript::EndsRun() const
{
    for (const ScriptedEvent& event : m_Events)
    {
        if (event.Event.Type == EventType::Quit)
            return true;
    }

    return false;
}

uint64_t InputScript::LastFrame() const
{
    uint64_t last = 0u;
    for (const ScriptedEvent& event : m_Events)
        last = std::max(last, event.Frame);

    return last;
}

} // namespace Hikari::Platform
