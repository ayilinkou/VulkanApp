#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <rhi/RhiTypes.h>

namespace Hikari::Engine
{

/** The four numbers the run report carries for a series of frame timings. */
struct TimingStats
{
    float Mean = 0.f;
    float P99 = 0.f;
    float Min = 0.f;
    float Max = 0.f;
};

/**
 * What one run measured, as data rather than as a file. The engine fills it and
 * returns it; an app decides whether it becomes JSON, an assertion, or nothing.
 *
 * The three groups are separate because they are read differently. Counters are
 * expectations that must match a committed baseline exactly. Timings are
 * measurements that vary with the machine, so they are read for drift rather
 * than diffed. Run is what makes two reports comparable at all — the same scene
 * at a different resolution, present mode or build configuration is not the
 * same measurement.
 */
struct RunReport
{
    /** Counts from the last frame drawn, which is what a capture describes. */
    struct RunCounters
    {
        uint64_t ValidationErrors = 0;
        uint64_t ValidationWarnings = 0;
        uint32_t DrawCalls = 0;
        uint32_t Batches = 0;
        uint32_t Instances = 0;
        uint32_t Barriers = 0;
        uint32_t BarrierCalls = 0;
    };

    /**
     * Frame 0 is held apart from the series rather than mixed into it: it pays
     * for first use of every pipeline and the first acquire, so averaging it
     * with the rest describes neither.
     */
    struct FirstFrameTimings
    {
        float FrameMs = 0.f;
        float CpuMs = 0.f;
    };

    /** Wall clock per frame, and the same minus what the frame spent blocked. */
    struct RunTimings
    {
        float StartupMs = 0.f;
        FirstFrameTimings FirstFrame;
        TimingStats FrameMs;
        TimingStats CpuMs;
    };

    /** The conditions the numbers above were measured under. */
    struct RunInfo
    {
        bool bFixedDt = false;
        bool bHeadless = false;
        bool bNoUi = false;
        uint32_t Width = 0;
        uint32_t Height = 0;
        uint32_t JobCount = 0;

        /** Absent where the target does not present at all, as offscreen ones do not. */
        std::optional<Rhi::PresentMode> PresentMode;
        std::string BuildConfig;
    };

    uint64_t Frames = 0;
    RunCounters Counters;
    RunTimings Timings;
    RunInfo Run;
};

} // namespace Hikari::Engine
