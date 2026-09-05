#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#include <core/Log.h>

#include <platform/CommandLine.h>
#include <platform/HeadlessPlatform.h>
#include <platform/IPlatform.h>

#include <editor/VulkanUiBackend.h>

#include <engine/ParseEngineOptions.h>
#include <engine/RunApp.h>

using namespace Hikari;
using namespace Hikari::Core;
using namespace Hikari::Platform;

namespace
{

constexpr LogCategory LogHeadless("Headless");

void PrintUsage()
{
    std::cout << "HikariHeadless — renders with no window, into an offscreen target\n"
                 "\n"
                 "Usage: HikariHeadless --frames <N> [options]\n"
                 "\n"
                 "Options:\n";

    // Printed by the engine rather than copied here, so that this binary and the
    // editor cannot come to describe the same flag differently.
    Engine::PrintEngineUsage();

    std::cout << "  --screenshot <path>     Write a PNG of the final frame "
                 "before exiting\n"
                 "  --report <path>         Write a JSON run report before "
                 "exiting\n"
                 "  --resolution <W>x<H>    Render into an offscreen target of this size "
                 "(default: 1280x720)\n"
                 "  --help                  Print this message and exit\n";
}

[[noreturn]] void ExitWithUsage(int code)
{
    PrintUsage();
    std::exit(code);
}

/** The engine's inputs, plus the size of the offscreen target to render into. */
struct HeadlessOptions
{
    Engine::AppRunSpec Run;

    /**
     * --resolution. Not a window size here — there is no window — but the
     * extent of the offscreen target. Zero leaves it to the platform, which
     * with no display to ask uses its own documented constant.
     */
    Extent2D TargetSize{};
};

HeadlessOptions ParseArgs(int argc, char** argv)
{
    HeadlessOptions options;
    Engine::AppRunSpec& run = options.Run;

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
            if (Engine::ParseEngineOption(option, run.Spec, run.Config))
                continue;

            if (flag == "--help" || flag == "-h")
                ExitWithUsage(EXIT_SUCCESS);
            else if (flag == "--screenshot")
            {
                if (option.Value)
                    run.ScreenshotPath = *option.Value;
                else
                    run.bScreenshotAutoPath = true;

                // The engine is asked for pixels; where they land is this app's
                // business and none of the engine's.
                run.Spec.bCaptureFinalFrame = true;
            }
            else if (flag == "--report")
            {
                if (option.Value)
                    run.ReportPath = *option.Value;
                else
                    run.bReportAutoPath = true;

                // Same split as --screenshot: the engine is asked to measure,
                // and where the numbers land is this app's business.
                run.Spec.bRecordTimings = true;
            }
            else if (flag == "--resolution")
                options.TargetSize = option.RequireExtent2D();
            else
            {
                LogMsg(LogSeverity::Error, LogHeadless, "Unknown option: {}", flag);
                ExitWithUsage(EXIT_FAILURE);
            }
        }
    }
    catch (const CommandLineError& e)
    {
        LogMsg(LogSeverity::Error, LogHeadless, "{}", e.what());
        ExitWithUsage(EXIT_FAILURE);
    }

    // Of the frame loop's exits, the ones that need a window cannot fire here,
    // leaving the frame counter — which only fires when Frames != 0. So a run
    // without --frames ends on a signal, and this binary exists for CI, where
    // nobody is there to send one: the job burns its whole timeout and dies to
    // SIGTERM, writing neither screenshot nor report.
    if (run.Spec.Frames == 0)
    {
        LogMsg(LogSeverity::Error, LogHeadless,
               "--frames is required: with no window there is nothing that can end the run, so it "
               "would render forever and write nothing");
        ExitWithUsage(EXIT_FAILURE);
    }

    // Ignore stops errors ever being counted, so --strict-validation would pass
    // a run that had them. Rejected rather than silently preferred either way:
    // in CI that combination reads as "validation is enforced" and is not.
    if (run.Spec.bStrictValidation && run.Spec.ValidationPolicy == Rhi::ValidationPolicy::Ignore)
    {
        LogMsg(LogSeverity::Error, LogHeadless,
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
    // argument parsing as well as device setup.
    const auto processStart = std::chrono::steady_clock::now();

    Engine::InstallProcessDefaults();

    const HeadlessOptions options = ParseArgs(argc, argv);

    // A zero size means "you decide", which for a platform with no display to
    // ask is its own documented constant.
    HeadlessPlatform platform(
        WindowDesc{.Width = options.TargetSize.Width, .Height = options.TargetSize.Height});

    // The UI is attached here too, deliberately: headless means no window, not a
    // feature-reduced build. A headless run is the only place the UI's bring-up
    // and drawing are exercised automatically, so dropping it would cost exactly
    // the coverage this binary exists to provide.
    Editor::VulkanUiBackend uiBackend;
    return Engine::RunApp(platform, uiBackend, options.Run, processStart);
}
