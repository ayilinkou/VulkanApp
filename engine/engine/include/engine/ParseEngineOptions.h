#pragma once

#include <platform/CommandLine.h>

#include <engine/EngineConfig.h>
#include <engine/RunSpec.h>

namespace Hikari::Engine
{

/**
 * Applies one command-line option to the engine's inputs, and says whether it
 * was the engine's to apply.
 *
 * One option at a time rather than the whole command line, because an app has
 * flags of its own to interleave: it walks the options once, offers each here
 * first, handles what this declines, and rejects what neither claims. The
 * alternative — a parser per binary — is how two apps come to disagree about a
 * flag they share.
 *
 * Fills both inputs because a command line does not distinguish them: all but
 * one flag describe the run, and --frames-in-flight sizes the engine itself.
 *
 * Returns false and touches nothing for a flag the engine does not know, which
 * includes the ones about files: an app owns --screenshot and --report, because
 * it is the app that writes them, and asks for the pixels by setting
 * RunSpec::bCaptureFinalFrame itself.
 *
 * Throws Platform::CommandLineError where an engine flag's value is missing or
 * malformed, so a caller reports it wherever it reports its own.
 */
bool ParseEngineOption(const Platform::CommandLineOption& option, RunSpec& spec,
                       EngineConfig& config);

/**
 * The engine's section of --help, with no heading of its own so that an app can
 * print its own flags into the same list. Two spaces of indent, descriptions
 * starting at column 26, which is what an app's lines should match.
 */
void PrintEngineUsage();

} // namespace Hikari::Engine
