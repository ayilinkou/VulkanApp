#pragma once

#include <engine/CapturedFrame.h>
#include <engine/RunReport.h>

namespace Hikari::Engine
{

/**
 * Everything a run hands back. The engine writes no files: an app decides what
 * to do with these, which is what lets a test read the pixels without naming a
 * path and reading them back off disk.
 */
struct RunResult
{
    RunReport Report;

    /** Empty unless RunSpec::bCaptureFinalFrame asked for it. */
    CapturedFrame Capture;
};

} // namespace Hikari::Engine
