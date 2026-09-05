#pragma once

#include <chrono>
#include <memory>

#include <core/IJobSystem.h>

#include <platform/IPlatform.h>
#include <platform/Paths.h>

#include <rhi/Diagnostics.h>

#include <engine/EngineConfig.h>
#include <engine/IUiBackend.h>
#include <engine/RunResult.h>
#include <engine/RunSpec.h>

namespace Hikari::Engine
{

/** Everything the engine borrows for the length of a run, and does not own. */
struct EngineDesc
{
    Platform::IPlatform* pPlatform = nullptr;
    const Platform::Paths* pPaths = nullptr;
    Core::IJobSystem* pJobSystem = nullptr;

    /** Outlives the engine: its counts are read after everything is torn down. */
    Rhi::Diagnostics* pDiagnostics = nullptr;

    /**
     * The UI stack. Built by the app and required: a run with no window still
     * draws the panel, which is what keeps the UI path exercised in CI.
     */
    IUiBackend* pUiBackend = nullptr;

    RunSpec Spec;
    EngineConfig Config;

    /**
     * Taken as the app's first statement, so that the report's startupMs covers
     * argument parsing and window creation rather than starting at the device.
     */
    std::chrono::steady_clock::time_point ProcessStart;
};

/**
 * One run of the engine, behind an interface because an app cannot see what the
 * engine is made of: the renderer, the scene and the asset types are private to
 * this module until Stage 8 promotes them into modules of their own.
 */
class IEngine
{
public:
    virtual ~IEngine();

    /** Runs to completion and hands back what the run measured and captured. */
    virtual RunResult Run() = 0;
};

[[nodiscard]] std::unique_ptr<IEngine> CreateEngine(const EngineDesc& desc);

/**
 * Asks the run to stop at the end of the current frame.
 *
 * Safe from a signal handler: it stores to an atomic and does nothing else,
 * which is what lets a Ctrl-C or a CI runner's SIGTERM still leave a report and
 * a capture behind.
 */
void RequestStop();

/** Whether a stop has been asked for. */
bool StopRequested();

} // namespace Hikari::Engine
