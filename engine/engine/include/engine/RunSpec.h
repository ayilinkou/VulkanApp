#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <rhi/Diagnostics.h>

namespace Hikari::Engine
{

/**
 * What one run of the engine does: which scene, for how long, and what it
 * should hand back when it ends.
 *
 * Engine-scoped, so nothing here describes a window. An app parses its own
 * platform flags into a WindowDesc and keeps them to itself — a binary that
 * cannot open a window has no business being handed a window mode.
 *
 * A few fields are read by the app rather than by the engine, because the app
 * builds what it injects: the job system is constructed from JobCount and the
 * diagnostics sink from ValidationPolicy, both before the engine exists. They
 * belong here anyway — they describe the run, not the platform, and putting
 * them in each app's own parsing is how two binaries drift on a shared flag.
 */
struct RunSpec
{
    /** Content-relative unless absolute. Empty loads no scene. */
    std::string ScenePath;

    /** --content; empty resolves the content root the usual way. Read by the app. */
    std::string ContentRoot;

    /** 0 runs until something asks the run to stop. */
    uint64_t Frames = 0;

    /** Use a fixed 1/60s timestep rather than wall-clock time. */
    bool bFixedDt = false;

    /** Index into kCameraPresets; -1 leaves the camera free. */
    int CameraPreset = -1;

    /**
     * Hand the final frame's pixels back in RunResult. A path is the app's
     * business: a test that wants the pixels should not have to name a file and
     * read them back off disk.
     */
    bool bCaptureFinalFrame = false;

    /**
     * Collect the per-frame timings the report's `timings` block is computed
     * from. Off by default because the samples are one pair of floats per frame
     * with no bound: a run that never ends would accumulate them forever, and
     * an interactive session has nobody to read them.
     */
    bool bRecordTimings = false;

    /**
     * Suppress the editor panel without touching anything else: the UI still
     * initialises and its pass still records, so a run with this differs from
     * one without it only in what is drawn.
     */
    bool bNoUi = false;

    /** -1 default, 0 serial with no threads, N worker threads. Read by the app. */
    int JobCount = -1;

    /** Exit non-zero if any validation error occurred. Read by the app. */
    bool bStrictValidation = false;

    /** How the diagnostics sink treats validation messages. Read by the app. */
    Rhi::ValidationPolicy ValidationPolicy = Rhi::ValidationPolicy::Count;

    /**
     * Optional extensions to behave as though the device did not support, so a
     * run can exercise the fallback path. Backend-specific by nature, which is
     * what the flag's --vk- prefix says.
     */
    std::vector<std::string> DisabledVulkanExtensions;

    /** Behave as though the device exposed one queue family. */
    bool bForceSingleQueue = false;
};

} // namespace Hikari::Engine
