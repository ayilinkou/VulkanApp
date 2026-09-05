#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#include <core/Log.h>

#include <platform/CommandLine.h>
#include <platform/IPlatform.h>
#include <platform/InputScript.h>
#include <platform/SdlPlatform.h>

#include <editor/VulkanUiBackend.h>

#include <engine/ParseEngineOptions.h>
#include <engine/RunApp.h>

using namespace Hikari;
using namespace Hikari::Core;
using namespace Hikari::Platform;

namespace
{

constexpr LogCategory LogEditor("Editor");

/** The engine's inputs, plus the flags only a binary with a window can answer for. */
struct EditorOptions
{
    Engine::AppRunSpec Run;

    /**
     * --resolution. Zero in either half leaves the choice to the platform,
     * which sizes the window from the display it opens on.
     */
    Extent2D WindowSize{};

    /**
     * --borderless / --fullscreen. One field rather than two flags, so that
     * "borderless and exclusive at once" cannot be represented past parsing.
     */
    WindowMode StartWindowMode = WindowMode::Windowed;

    /**
     * --input: a script replayed alongside real input. The same scripts the
     * headless binary takes, so a run that fails in CI can be watched here.
     */
    std::string InputScriptPath;
};

void PrintUsage()
{
    std::cout << "HikariEditor\n"
                 "\n"
                 "Usage: HikariEditor [options]\n"
                 "\n"
                 "Options:\n";

    // Printed by the engine rather than copied here, so that this binary and the
    // headless one cannot come to describe the same flag differently.
    Engine::PrintEngineUsage();

    std::cout << "  --screenshot <path>     Write a PNG of the final frame "
                 "before exiting\n"
                 "  --report <path>         Write a JSON run report before "
                 "exiting\n"
                 "  --resolution <W>x<H>    Start with a window of this size (default: "
                 "three quarters of\n"
                 "                          the display)\n"
                 "  --borderless            Start covering the display as a borderless "
                 "window\n"
                 "  --fullscreen            Start in exclusive fullscreen, selecting a "
                 "display mode\n"
                 "  --input <path>          Replay an input script alongside real input: key "
                 "presses, resizes,\n"
                 "                          captures and quit, delivered on the frames it "
                 "names\n"
                 "  --help                  Print this message and exit\n";
}

[[noreturn]] void ExitWithUsage(int code)
{
    PrintUsage();
    std::exit(code);
}

/**
 * --borderless and --fullscreen name two different ways of covering the
 * display, so a command line carrying both asks for nothing coherent. Rejected
 * rather than settled by precedence or by order: a launcher passing both is
 * misconfigured, and honouring one of them quietly hides that. Repeating the
 * same flag is not a conflict.
 */
void RejectConflictingWindowMode(WindowMode current, WindowMode requested, const std::string& flag)
{
    if (current == WindowMode::Windowed || current == requested)
        return;

    LogMsg(LogSeverity::Error, LogEditor, "{} cannot be combined with {}", flag,
           current == WindowMode::BorderlessFullscreen ? "--borderless" : "--fullscreen");
    ExitWithUsage(EXIT_FAILURE);
}

EditorOptions ParseArgs(int argc, char** argv)
{
    EditorOptions options;

    try
    {
        // Named rather than a temporary in the range-init: Options() hands out a
        // reference into the CommandLine, which C++20 would not keep alive for
        // the duration of the loop.
        const CommandLine commandLine(argc, argv);

        for (const CommandLineOption& option : commandLine.Options())
        {
            const std::string& flag = option.Flag;

            // Offered to the engine first: a flag it claims is one this app must
            // not also answer for, and the order is what guarantees that.
            if (Engine::ParseEngineOption(option, options.Run.Spec, options.Run.Config))
                continue;

            if (flag == "--help" || flag == "-h")
                ExitWithUsage(EXIT_SUCCESS);
            else if (flag == "--screenshot")
            {
                if (option.Value)
                    options.Run.ScreenshotPath = *option.Value;
                else
                    options.Run.bScreenshotAutoPath = true;

                // The engine is asked for pixels; where they land is this app's
                // business and none of the engine's.
                options.Run.Spec.bCaptureFinalFrame = true;
            }
            else if (flag == "--report")
            {
                if (option.Value)
                    options.Run.ReportPath = *option.Value;
                else
                    options.Run.bReportAutoPath = true;

                // Same split as --screenshot: the engine is asked to measure,
                // and where the numbers land is this app's business.
                options.Run.Spec.bRecordTimings = true;
            }
            else if (flag == "--resolution")
                options.WindowSize = option.RequireExtent2D();
            else if (flag == "--input")
                options.InputScriptPath = option.RequireValue();
            else if (flag == "--borderless")
            {
                option.RequireNoValue();
                RejectConflictingWindowMode(options.StartWindowMode,
                                            WindowMode::BorderlessFullscreen, flag);
                options.StartWindowMode = WindowMode::BorderlessFullscreen;
            }
            else if (flag == "--fullscreen")
            {
                option.RequireNoValue();
                RejectConflictingWindowMode(options.StartWindowMode,
                                            WindowMode::ExclusiveFullscreen, flag);
                options.StartWindowMode = WindowMode::ExclusiveFullscreen;
            }
            else
            {
                LogMsg(LogSeverity::Error, LogEditor, "Unknown option: {}", flag);
                ExitWithUsage(EXIT_FAILURE);
            }
        }
    }
    catch (const CommandLineError& e)
    {
        LogMsg(LogSeverity::Error, LogEditor, "{}", e.what());
        ExitWithUsage(EXIT_FAILURE);
    }

    // Ignore stops errors ever being counted, so --strict-validation would pass
    // a run that had them. Rejected rather than silently preferred either way:
    // in CI that combination reads as "validation is enforced" and is not.
    if (options.Run.Spec.bStrictValidation &&
        options.Run.Spec.ValidationPolicy == Rhi::ValidationPolicy::Ignore)
    {
        LogMsg(LogSeverity::Error, LogEditor,
               "--strict-validation cannot be combined with --validation-policy ignore: "
               "no errors would be counted for it to act on");
        ExitWithUsage(EXIT_FAILURE);
    }

    return options;
}

} // namespace

int main(int argc, char** argv)
{
    // First statement in the process, so that startupMs in the run report covers
    // argument parsing and window creation as well as device setup.
    const auto processStart = std::chrono::steady_clock::now();

    Engine::InstallProcessDefaults();

    const EditorOptions options = ParseArgs(argc, argv);

    try
    {
        SdlPlatform platform(
            WindowDesc{.Width = options.WindowSize.Width, .Height = options.WindowSize.Height});

        // Before the device, so that the first swapchain is built at the size the
        // window ends up rather than at the windowed one. Where the transition is
        // asynchronous the resize still arrives late, as a resize event, which
        // costs one recreation and nothing else.
        if (options.StartWindowMode != WindowMode::Windowed)
            platform.SetWindowMode(options.StartWindowMode);

        if (!options.InputScriptPath.empty())
            platform.SetInputScript(InputScript::Load(options.InputScriptPath));

        Editor::VulkanUiBackend uiBackend;
        return Engine::RunApp(platform, uiBackend, options.Run, processStart);
    }
    catch (const InputScriptError& e)
    {
        LogMsg(LogSeverity::Error, LogEditor, "{}", e.what());
        return EXIT_FAILURE;
    }
    catch (const SDLException& e)
    {
        SdlPlatform::ShowErrorMessageBox("SDL Error", e.what());
        LogMsg(LogSeverity::Error, LogEditor, "{}", e.what());
        return EXIT_FAILURE;
    }
}
