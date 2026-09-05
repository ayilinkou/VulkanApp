#pragma once

#include <chrono>
#include <string>

#include <platform/IPlatform.h>

#include <engine/EngineConfig.h>
#include <engine/IUiBackend.h>
#include <engine/RunSpec.h>

namespace Hikari::Engine
{

/** What an app decided about a run: the engine's inputs, plus where output goes. */
struct AppRunSpec
{
    RunSpec Spec;
    EngineConfig Config;

    /** --screenshot: a path, or the flag alone for an automatically named one. */
    std::string ScreenshotPath;
    bool bScreenshotAutoPath = false;

    /** --report, the same way. */
    std::string ReportPath;
    bool bReportAutoPath = false;
};

/**
 * Everything both binaries do once they know their platform: the diagnostics
 * sink, the job system, the content root, the engine, and the files a run
 * leaves behind. Returns the process exit code.
 *
 * Shared rather than copied because the two apps differ only in how they get a
 * platform and which flags they accept — and a second copy of this is how two
 * binaries come to disagree about what a run means.
 */
int RunApp(Platform::IPlatform& platform, IUiBackend& uiBackend, const AppRunSpec& app,
           std::chrono::steady_clock::time_point processStart);

/**
 * Colour support on the console, the two signals that mean "stop", and the
 * default log level. Called first by every binary, because a run killed by a
 * SIGTERM before this has run leaves its report and capture unwritten.
 */
void InstallProcessDefaults();

} // namespace Hikari::Engine
