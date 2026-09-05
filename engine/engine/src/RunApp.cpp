#include <engine/RunApp.h>

#include <csignal>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <format>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <asset/ImageWriter.h>

#include <core/IJobSystem.h>
#include <core/Log.h>
#include <core/SerialJobSystem.h>
#include <core/SharedQueueJobSystem.h>

#include <platform/FileSystem.h>
#include <platform/SdlPlatform.h>

#include <rhi/Diagnostics.h>
#include <rhi/RhiTypes.h>

#include <engine/IEngine.h>
#include <engine/RunResult.h>

using namespace Hikari::Core;
using namespace Hikari::Platform;

namespace Hikari::Engine
{

namespace
{

constexpr LogCategory LogValidationLayer("Validation Layer");
constexpr LogCategory LogDiagnostics("Diagnostics");
constexpr LogCategory LogSDL("SDL");
constexpr LogCategory LogApp("App");

/**
 * Routes the RHI's validation messages into the log. Rhi::Diagnostics has
 * already counted and captured the message by the time this runs; this decides
 * only how it is presented. Called from the driver's debug callback, so it may
 * run on any thread.
 */
void HandleRhiDiagnostic(Rhi::DiagnosticSeverity severity, std::string_view message)
{
    LogSeverity logSeverity = LogSeverity::Info;
    switch (severity)
    {
        case Rhi::DiagnosticSeverity::Info:
            logSeverity = LogSeverity::Info;
            break;
        case Rhi::DiagnosticSeverity::Warning:
            logSeverity = LogSeverity::Warning;
            break;
        case Rhi::DiagnosticSeverity::Error:
            logSeverity = LogSeverity::Error;
            break;
    }

    LogMsg(logSeverity, LogValidationLayer, "{}", message);
}

/**
 * What the app parses out of the command line: the engine's two inputs, plus
 * the flags only a binary with a window and somewhere to write can answer for.
 */
std::string GenerateTimestamp()
{
    using namespace std::chrono;
    std::time_t now = system_clock::to_time_t(system_clock::now());
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d_%m_%Y_%H_%M_%S"); // DD_MM_YYYY_HH_mm_SS
    return oss.str();
}

/** Where an auto-pathed capture and report land, relative to the working directory. */
constexpr const char* kDefaultScreenshotPath = "tests/screenshots/screenshot_";
constexpr const char* kDefaultReportPath = "tests/reports/report_";

/**
 * The present mode as a JSON value: a quoted name, or null where the target
 * does not present at all, which is what an offscreen run reports.
 */
std::string PresentModeJson(std::optional<Rhi::PresentMode> mode)
{
    if (!mode)
        return "null";

    switch (*mode)
    {
        case Rhi::PresentMode::Immediate:
            return "\"immediate\"";
        case Rhi::PresentMode::Mailbox:
            return "\"mailbox\"";
        case Rhi::PresentMode::Fifo:
            return "\"fifo\"";
        case Rhi::PresentMode::FifoRelaxed:
            return "\"fifo-relaxed\"";
    }

    return "null";
}

/**
 * Serialises a run report to JSON.
 *
 * Three objects rather than one flat list, because they are read differently:
 * everything under "counters" is an expectation that must match exactly,
 * everything under "timings" is a measurement that varies with the machine. A
 * reader cannot tell those apart in a flat object, and a number that looks
 * authoritative and is not is worse than no number. "run" is what makes two
 * reports comparable at all — the same scene at a different resolution, present
 * mode or build configuration is not the same measurement.
 */
void WriteRunReport(const RunReport& report, const std::string& path)
{
    EnsureParentDirectoryExists(kDefaultReportPath);

    const std::string finalPath = EnsureExtension(path, ".json");
    std::ofstream file(finalPath);
    if (!file.is_open())
    {
        LogMsg(LogSeverity::Error, LogApp, "Failed to open report file for writing: {}", finalPath);
        return;
    }

    const auto stats = [](const TimingStats& s)
    {
        return std::format("{{ \"mean\": {:.4f}, \"p99\": {:.4f}, \"min\": {:.4f}, "
                           "\"max\": {:.4f} }}",
                           s.Mean, s.P99, s.Min, s.Max);
    };

    file << "{\n"
         << "  \"frames\": " << report.Frames << ",\n"
         << "  \"counters\": {\n"
         << "    \"frame\": {\n"
         << "      \"drawCalls\": " << report.Counters.Frame.DrawCalls << ",\n"
         << "      \"batches\": " << report.Counters.Frame.Batches << ",\n"
         << "      \"instances\": " << report.Counters.Frame.Instances << ",\n"
         << "      \"barriers\": " << report.Counters.Frame.Barriers << ",\n"
         << "      \"barrierCalls\": " << report.Counters.Frame.BarrierCalls << "\n"
         << "    },\n"
         << "    \"run\": {\n"
         << "      \"validationErrors\": " << report.Counters.Run.ValidationErrors << ",\n"
         << "      \"validationWarnings\": " << report.Counters.Run.ValidationWarnings << ",\n"
         << "      \"uploadSubmissions\": " << report.Counters.Run.UploadSubmissions << "\n"
         << "    }\n"
         << "  },\n"
         << "  \"timings\": {\n"
         << std::format("    \"startupMs\": {:.4f},\n", report.Timings.StartupMs)
         << std::format("    \"firstFrame\": {{ \"frameMs\": {:.4f}, \"cpuMs\": {:.4f} }},\n",
                        report.Timings.FirstFrame.FrameMs, report.Timings.FirstFrame.CpuMs)
         << "    \"frameMs\": " << stats(report.Timings.FrameMs) << ",\n"
         << "    \"cpuMs\": " << stats(report.Timings.CpuMs) << "\n"
         << "  },\n"
         << "  \"run\": {\n"
         << "    \"fixedDt\": " << (report.Run.bFixedDt ? "true" : "false") << ",\n"
         << "    \"headless\": " << (report.Run.bHeadless ? "true" : "false") << ",\n"
         << "    \"noUi\": " << (report.Run.bNoUi ? "true" : "false") << ",\n"
         << "    \"width\": " << report.Run.Width << ",\n"
         << "    \"height\": " << report.Run.Height << ",\n"
         << "    \"jobCount\": " << report.Run.JobCount << ",\n"
         << "    \"presentMode\": " << PresentModeJson(report.Run.PresentMode) << ",\n"
         << "    \"buildConfig\": \"" << report.Run.BuildConfig << "\"\n"
         << "  }\n"
         << "}\n";
    file.close();

    LogMsg(LogSeverity::Info, LogApp, "Wrote report to {}", finalPath);
}

/** Writes a captured frame out as a PNG, through the module that owns the encoder. */
void WriteCapturePng(const CapturedFrame& capture, const std::string& path)
{
    EnsureParentDirectoryExists(kDefaultScreenshotPath);

    if (capture.IsEmpty())
    {
        LogMsg(LogSeverity::Error, LogApp, "No frame was captured, so no screenshot was written.");
        return;
    }

    const std::string finalPath = EnsureExtension(path, ".png");
    if (Asset::WritePng(capture.Pixels, capture.Extent, finalPath))
        LogMsg(LogSeverity::Info, LogApp, "Wrote screenshot to {}", finalPath);
}

/** The job system the run asked for, and a line saying which it got. */
std::unique_ptr<IJobSystem> CreateJobSystem(int jobCount)
{
    if (jobCount == 0)
    {
        LogMsg(LogSeverity::Info, LogApp,
               "JobSystem selected: SerialJobSystem (no worker threads)");
        return std::make_unique<SerialJobSystem>();
    }

    std::unique_ptr<IJobSystem> jobSystem =
        jobCount > 0 ? std::make_unique<SharedQueueJobSystem>(static_cast<uint32_t>(jobCount))
                     : std::make_unique<SharedQueueJobSystem>();

    LogMsg(LogSeverity::Info, LogApp,
           "JobSystem selected: SharedQueueJobSystem ({} worker threads)",
           jobSystem->WorkerCount());
    return jobSystem;
}

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <io.h>
#include <windows.h>

inline void EnableAnsiColors()
{
    for (DWORD stdHandle : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE})
    {
        HANDLE handle = GetStdHandle(stdHandle);
        if (handle == INVALID_HANDLE_VALUE)
            continue;

        DWORD mode = 0;
        if (!GetConsoleMode(handle, &mode))
            continue;

        SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
}
#else
// For write(), which is what the signal handler prints its newline with.
#include <unistd.h>
#endif

/**
 * Asks the loop to stop, for either signal that means "shut down": SIGINT from
 * Ctrl-C at a terminal, SIGTERM from a CI runner's timeout or a process
 * manager. Both leave the loop the ordinary way, so the screenshot and the run
 * report are still written — which is the whole point of handling SIGTERM,
 * since a killed run produces no artefacts to diagnose the timeout with.
 *
 * A second signal deliberately does nothing new. Escalating to _Exit would drop
 * exactly those artefacts for a user who was merely impatient; a run whose
 * shutdown is genuinely wedged is what SIGKILL is for.
 *
 * RequestStop() stores to a lock-free std::atomic<bool>, which a handler may touch.
 * The newline goes through write() rather than std::cout because only
 * async-signal-safe functions may be called from a handler, and formatted
 * output is not one of them: interrupting a stream mid-write and re-entering it
 * is undefined, and the failure is a corrupted or deadlocked stdout rather than
 * anything that announces itself.
 */
void HandleTerminationSignal(int)
{
    RequestStop();

#ifdef _WIN32
    const int written = _write(1, "\n", 1);
#else
    const ssize_t written = write(STDOUT_FILENO, "\n", 1);
#endif
    // Nothing useful to do if the write fails, and a handler cannot report it.
    // Consumed because write() is declared warn_unused_result.
    (void)written;
}

} // namespace

void InstallProcessDefaults()
{
#ifdef _WIN32
    EnableAnsiColors();
#endif

    // Both signals that mean "stop": Ctrl-C, and the SIGTERM a CI runner sends
    // when a job outlives its timeout. Without the second, a timed-out run is
    // killed with its screenshot and report unwritten — the two things anyone
    // diagnosing the timeout would want.
    std::signal(SIGINT, HandleTerminationSignal);
    std::signal(SIGTERM, HandleTerminationSignal);

    Log::g_MinSeverity = LogSeverity::Info;
}

int RunApp(IPlatform& platform, IUiBackend& uiBackend, const AppRunSpec& app,
           std::chrono::steady_clock::time_point processStart)
{
    // Declared before the engine so that it outlives the device reporting into
    // it, and so its counters are still readable for the strict-validation
    // check below, which runs after everything has been torn down.
    Rhi::Diagnostics diagnostics(
        Rhi::Diagnostics::Desc{.Policy = app.Spec.ValidationPolicy,
                               .MinSeverity = Rhi::DiagnosticSeverity::Info,
                               .OnMessage = &HandleRhiDiagnostic});

    std::unique_ptr<Paths> pPaths = nullptr;
    std::unique_ptr<IJobSystem> pJobSystem = nullptr;
    std::unique_ptr<IEngine> pEngine = nullptr;

    try
    {
        pJobSystem = CreateJobSystem(app.Spec.JobCount);
        pPaths = std::make_unique<Paths>(app.Spec.ContentRoot);

        pEngine = CreateEngine(EngineDesc{.pPlatform = &platform,
                                          .pPaths = pPaths.get(),
                                          .pJobSystem = pJobSystem.get(),
                                          .pDiagnostics = &diagnostics,
                                          .pUiBackend = &uiBackend,
                                          .Spec = app.Spec,
                                          .Config = app.Config,
                                          .ProcessStart = processStart});
        const RunResult result = pEngine->Run();

        // One stamp for both files, taken when the first is written and reused
        // by the second, so that a screenshot and the report describing the same
        // moment share a name — asking the clock twice puts them a second apart
        // whenever the two writes straddle a second boundary.
        std::string captureStamp;
        const auto stamp = [&captureStamp]() -> const std::string&
        {
            if (captureStamp.empty())
                captureStamp = GenerateTimestamp();

            return captureStamp;
        };

        if (app.Spec.bCaptureFinalFrame)
        {
            WriteCapturePng(result.Capture, app.bScreenshotAutoPath
                                                ? kDefaultScreenshotPath + stamp()
                                                : app.ScreenshotPath);
        }

        if (!app.ReportPath.empty() || app.bReportAutoPath)
        {
            WriteRunReport(result.Report,
                           app.bReportAutoPath ? kDefaultReportPath + stamp() : app.ReportPath);
        }
    }
    catch (const SDLException& e)
    {
        SdlPlatform::ShowErrorMessageBox("SDL Error", e.what());
        LogMsg(LogSeverity::Error, LogSDL, "{}", e.what());
        return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        LogMsg(LogSeverity::Error, LogApp, "Error: {}", e.what());
        return EXIT_FAILURE;
    }

    pEngine.reset();
    pJobSystem.reset();
    pPaths.reset();

    // Everything above is destroyed by this point, so this covers teardown
    // messages too — which is where the interesting ones tend to be, since a
    // resource freed while still in use is only detectable at destruction.
    const uint64_t validationErrors = diagnostics.ErrorCount();
    const uint64_t validationWarnings = diagnostics.WarningCount();

    if (validationErrors > 0 || validationWarnings > 0)
    {
        LogMsg(LogSeverity::Warning, LogDiagnostics, "{} error(s), {} warning(s)", validationErrors,
               validationWarnings);

        const std::vector<std::string> recent = diagnostics.RecentMessages();
        const uint64_t dropped = diagnostics.DroppedMessageCount();
        if (dropped > 0)
            LogMsg(LogSeverity::Warning, LogDiagnostics,
                   "  last {} message(s), {} earlier one(s) dropped:", recent.size(), dropped);
        else
            LogMsg(LogSeverity::Warning, LogDiagnostics, "  {} message(s):", recent.size());

        for (const std::string& message : recent)
            LogMsg(LogSeverity::Warning, LogDiagnostics, "    {}", message);
    }

    if (app.Spec.bStrictValidation && validationErrors > 0)
    {
        LogMsg(LogSeverity::Error, LogDiagnostics,
               "Strict validation failed: {} validation error(s) occurred", validationErrors);
        return EXIT_FAILURE;
    }

    LogMsg(LogSeverity::Info, LogApp, "Exiting gracefully...");
    return EXIT_SUCCESS;
}

} // namespace Hikari::Engine
