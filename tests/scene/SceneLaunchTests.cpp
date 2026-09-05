#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

#include <core/SerialJobSystem.h>

#include <platform/HeadlessPlatform.h>
#include <platform/InputScript.h>
#include <platform/Paths.h>

#include <rhi/Diagnostics.h>
#include <rhi/IDevice.h>

#include <editor/VulkanUiBackend.h>

#include <engine/IEngine.h>

#include "TestEnvironment.h"
#include "TestPaths.h"

using namespace Hikari;
using namespace Hikari::Core;
using namespace Hikari::Platform;

namespace
{

/**
 * Small enough to keep twelve runs quick, large enough that a divergence has
 * somewhere to show up. Nothing here compares against a committed image — the
 * comparisons are between two runs on the same machine.
 */
constexpr uint32_t kWidth = 640u;
constexpr uint32_t kHeight = 360u;

/** Enough to get past the first frame's one-off costs and wrap the frames in flight. */
constexpr uint64_t kFrames = 3u;

/**
 * Whether this machine can create a device at all, probed once.
 *
 * Without it a machine with no Vulkan ICD reports every case as a failure, which
 * is indistinguishable from a real one. The probe device is destroyed
 * immediately; each case builds its own through the engine.
 */
/** The reason no device could be created, empty while one could. */
std::string& DeviceFailureReason()
{
    static std::string reason;
    return reason;
}

bool HasUsableDevice()
{
    static const bool bUsable = []
    {
        try
        {
            Rhi::Diagnostics::Desc diagnosticsDesc;
            diagnosticsDesc.Policy = Rhi::ValidationPolicy::Ignore;
            Rhi::Diagnostics diagnostics(diagnosticsDesc);

            Rhi::DeviceDesc desc;
            desc.ApplicationName = "HikariEngine scene tests";
            desc.bEnableValidation = false;
            desc.pDiagnostics = &diagnostics;
            desc.Requirements.bPresent = false;

            return Rhi::CreateDevice(desc) != nullptr;
        }
        catch (const std::exception& error)
        {
            DeviceFailureReason() = error.what();
            return false;
        }
    }();

    return bUsable;
}

/**
 * Skips the calling case where there is no device — or fails it where the
 * environment says one was supposed to be there.
 *
 * CI supplies an ICD on purpose, so a skipped case teaches it nothing: CTest
 * counts a skip as not-failed, and a run of nothing then reports 100% passed.
 * That is what happened the first time these ran in CI, and the reason went
 * unprinted because a skip prints nothing.
 */
void RequireDevice()
{
    if (HasUsableDevice())
        return;

    const std::string reason =
        DeviceFailureReason().empty() ? "No usable Vulkan device"
                                      : "No usable Vulkan device: " + DeviceFailureReason();

    if (TestEnvironment::DeviceRequired())
        FAIL(reason);

    SKIP(reason);
}

/**
 * One headless run of one scene, start to finish, in this process.
 *
 * In-process rather than by launching the binary: every counter is assertable
 * without a file or a JSON parser, and a failure points at a line. The UI is
 * attached for the same reason the headless binary attaches it — the barrier
 * counts then describe the frame that actually ships.
 */
Engine::RunResult RunScene(const std::string& contentRoot, const std::string& scenePath)
{
    HeadlessPlatform platform(WindowDesc{.Width = kWidth, .Height = kHeight});
    Editor::VulkanUiBackend uiBackend;

    // Counted rather than fail-fast: a failing assertion that names the count is
    // more useful here than an abort inside the driver.
    Rhi::Diagnostics::Desc diagnosticsDesc;
    diagnosticsDesc.Policy = Rhi::ValidationPolicy::Count;
    diagnosticsDesc.MinSeverity = Rhi::DiagnosticSeverity::Info;
    Rhi::Diagnostics diagnostics(diagnosticsDesc);

    // Serial on purpose: a scene that renders differently under one job system
    // than another is a race in whatever submitted the jobs, and this suite
    // should not be where that surfaces as a flake.
    SerialJobSystem jobSystem;
    Paths paths(contentRoot);

    Engine::RunSpec spec;
    spec.ScenePath = scenePath;
    spec.Frames = kFrames;
    spec.bFixedDt = true;
    spec.CameraPreset = 0;
    spec.bCaptureFinalFrame = true;
    spec.bRecordTimings = true;

    // The UI is attached but its panel is suppressed. The pass still records —
    // its barrier and its rendering scope cost the same either way — so the
    // counters describe the frame that ships, while the pixels stay comparable:
    // the panel prints the frame time and the FPS, which no two runs agree on.
    spec.bNoUi = true;

    const std::unique_ptr<Engine::IEngine> engine = Engine::CreateEngine(
        Engine::EngineDesc{.pPlatform = &platform,
                           .pPaths = &paths,
                           .pJobSystem = &jobSystem,
                           .pDiagnostics = &diagnostics,
                           .pUiBackend = &uiBackend,
                           .Spec = spec,
                           .Config = Engine::EngineConfig{},
                           .ProcessStart = std::chrono::steady_clock::now()});

    return engine->Run();
}

/** What a scene's geometry must come out as, derived from the scene itself. */
struct SceneExpectation
{
    const char* Scene;
    uint32_t DrawCalls;
    uint32_t Batches;
    uint32_t Instances;
};

void CheckScene(const SceneExpectation& expected)
{
    RequireDevice();

    INFO("scene: " << expected.Scene);

    const Engine::RunResult first = RunScene(TestDataDir(), expected.Scene);

    CHECK(first.Report.Counters.Run.ValidationErrors == 0u);
    CHECK(first.Report.Frames == kFrames);
    CHECK(first.Report.Counters.Frame.DrawCalls == expected.DrawCalls);
    CHECK(first.Report.Counters.Frame.Batches == expected.Batches);
    CHECK(first.Report.Counters.Frame.Instances == expected.Instances);

    // Every frame records its passes whether or not anything is drawn, so a
    // scene with no geometry still moves images between layouts.
    CHECK(first.Report.Counters.Frame.Barriers > 0u);
    REQUIRE_FALSE(first.Capture.IsEmpty());

    // The same run again, in the same process on the same device: the counters
    // and the pixels have to agree with themselves before they are worth
    // comparing against anything else.
    //
    // Batch order still tracks heap addresses — Drawable::operator< falls
    // through to comparing mesh and material pointers — but opaque geometry is
    // order-independent (depth tested, blending off) and weighted-blended
    // transparency only diverges in the low bits with three or more layers over
    // one pixel, which no scene here has. Step 58 removes the caveat.
    const Engine::RunResult second = RunScene(TestDataDir(), expected.Scene);

    CHECK(second.Report.Counters.Frame.DrawCalls == first.Report.Counters.Frame.DrawCalls);
    CHECK(second.Report.Counters.Frame.Batches == first.Report.Counters.Frame.Batches);
    CHECK(second.Report.Counters.Frame.Instances == first.Report.Counters.Frame.Instances);
    CHECK(second.Report.Counters.Frame.Barriers == first.Report.Counters.Frame.Barriers);
    CHECK(second.Report.Counters.Frame.BarrierCalls == first.Report.Counters.Frame.BarrierCalls);
    CHECK(second.Report.Counters.Run.ValidationErrors == 0u);
    // Compared as a bool: Catch2 stringifies the operands of a failed
    // comparison, and these are a megabyte of bytes each.
    const bool bPixelsMatch = second.Capture.Pixels == first.Capture.Pixels;
    CHECK(bPixelsMatch);

    if (!bPixelsMatch && second.Capture.Pixels.size() == first.Capture.Pixels.size())
    {
        size_t differing = 0u;
        for (size_t i = 0u; i < first.Capture.Pixels.size(); ++i)
            differing += first.Capture.Pixels[i] != second.Capture.Pixels[i] ? 1u : 0u;

        FAIL_CHECK("pixel bytes differing between the two runs: "
                   << differing << " of " << first.Capture.Pixels.size());
    }
}

} // namespace

TEST_CASE("A scene with nothing in it renders and reports nothing drawn", "[scene]")
{
    CheckScene({.Scene = "scenes/empty.map", .DrawCalls = 0u, .Batches = 0u, .Instances = 0u});
}

TEST_CASE("A scene of lights alone draws no geometry", "[scene]")
{
    CheckScene(
        {.Scene = "scenes/lights_only.map", .DrawCalls = 0u, .Batches = 0u, .Instances = 0u});
}

TEST_CASE("One opaque cube is one batch of one instance", "[scene]")
{
    CheckScene(
        {.Scene = "scenes/single_cube.map", .DrawCalls = 1u, .Batches = 1u, .Instances = 1u});
}

TEST_CASE("One transparent cube draws through the blended path", "[scene]")
{
    CheckScene(
        {.Scene = "scenes/transparent_only.map", .DrawCalls = 1u, .Batches = 1u, .Instances = 1u});
}

TEST_CASE("Two materials cannot merge into one batch", "[scene]")
{
    CheckScene(
        {.Scene = "scenes/two_materials.map", .DrawCalls = 2u, .Batches = 2u, .Instances = 2u});
}

TEST_CASE("Two entities of one model merge into a single instanced batch", "[scene]")
{
    CheckScene(
        {.Scene = "scenes/instanced_cubes.map", .DrawCalls = 1u, .Batches = 1u, .Instances = 2u});
}

TEST_CASE("The shipped scene loads, renders and reports no validation errors", "[scene]")
{
    RequireDevice();

    // The hand-authored cubes exist so the expected counters are derivable; this
    // case exists so the suite is not exclusively testing geometry nobody ships.
    // Its counters are not pinned here — the committed baseline is what does
    // that — but a load failure, a validation error or an empty frame is caught.
    const Engine::RunResult result = RunScene(HIKARI_CONTENT_DIR, "scenes/test_scene.map");

    CHECK(result.Report.Counters.Run.ValidationErrors == 0u);
    CHECK(result.Report.Frames == kFrames);
    CHECK(result.Report.Counters.Frame.DrawCalls > 0u);
    CHECK(result.Report.Counters.Frame.Batches > 0u);
    CHECK(result.Report.Counters.Frame.Instances >= result.Report.Counters.Frame.Batches);
    REQUIRE_FALSE(result.Capture.IsEmpty());
}

TEST_CASE("The headless binary runs a scene and writes what it was asked for", "[scene]")
{
    RequireDevice();

    // The one case that launches the real binary. Everything above tests the
    // engine; this tests the program around it — argument parsing, the platform
    // and UI it builds, the files it writes, and the exit code CI reads.
    const std::filesystem::path outputDir =
        std::filesystem::temp_directory_path() / "hikari_scene_tests";
    std::filesystem::create_directories(outputDir);

    const std::filesystem::path screenshot = outputDir / "smoke.png";
    const std::filesystem::path report = outputDir / "smoke.json";
    std::filesystem::remove(screenshot);
    std::filesystem::remove(report);

    // --strict-validation is what makes the exit code carry the validation
    // result: without it a run with errors still exits 0.
    const std::string command = std::string("\"") + HIKARI_HEADLESS_BINARY + "\"" +
                                " --content \"" + TestDataDir() + "\"" +
                                " --scene scenes/single_cube.map --frames 3 --fixed-dt" +
                                " --resolution 320x180 --no-ui --strict-validation" +
                                " --screenshot \"" + screenshot.string() + "\"" + " --report \"" +
                                report.string() + "\"";

    INFO("command: " << command);
    REQUIRE(std::system(command.c_str()) == 0);

    REQUIRE(std::filesystem::exists(screenshot));
    REQUIRE(std::filesystem::file_size(screenshot) > 0u);
    REQUIRE(std::filesystem::exists(report));

    std::ifstream reportFile(report);
    const std::string contents((std::istreambuf_iterator<char>(reportFile)),
                               std::istreambuf_iterator<char>());
    CHECK(contents.find("\"counters\"") != std::string::npos);
    CHECK(contents.find("\"validationErrors\": 0") != std::string::npos);
}

TEST_CASE("A scripted run replays input, resizes and captures where it was told to", "[scene]")
{
    RequireDevice();

    // The coverage --frames alone cannot reach: held-key movement, a resize and
    // the target recreation it forces, a capture at a chosen frame, and a quit
    // that ends the run from inside.
    HeadlessPlatform platform(WindowDesc{.Width = kWidth, .Height = kHeight});
    platform.SetInputScript(
        InputScript::Load(std::string(TestDataDir()) + "input/scripted_replay.txt"));

    Editor::VulkanUiBackend uiBackend;

    Rhi::Diagnostics::Desc diagnosticsDesc;
    diagnosticsDesc.Policy = Rhi::ValidationPolicy::Count;
    diagnosticsDesc.MinSeverity = Rhi::DiagnosticSeverity::Info;
    Rhi::Diagnostics diagnostics(diagnosticsDesc);

    SerialJobSystem jobSystem;
    Paths paths(TestDataDir());

    Engine::RunSpec spec;
    spec.ScenePath = "scenes/single_cube.map";
    spec.bFixedDt = true;
    spec.bNoUi = true;
    spec.bRecordTimings = true;

    // No --frames equivalent and no bCaptureFinalFrame: the script's quit is
    // what ends this run and its screenshot is what captures it, which is the
    // whole point of the case.
    spec.Frames = 0u;

    const std::unique_ptr<Engine::IEngine> engine = Engine::CreateEngine(
        Engine::EngineDesc{.pPlatform = &platform,
                           .pPaths = &paths,
                           .pJobSystem = &jobSystem,
                           .pDiagnostics = &diagnostics,
                           .pUiBackend = &uiBackend,
                           .Spec = spec,
                           .Config = Engine::EngineConfig{},
                           .ProcessStart = std::chrono::steady_clock::now()});

    const Engine::RunResult result = engine->Run();

    CHECK(result.Report.Counters.Run.ValidationErrors == 0u);

    // The script quits on frame 14, so the run ends there rather than going on
    // forever — which is what an unbounded headless run does without input.
    CHECK(result.Report.Frames == 15u);

    // ...and resizes to 320x240 on frame 8, which the report and the capture
    // both have to reflect: a resize that did not recreate the target would
    // leave one of them at the size the run started with.
    CHECK(result.Report.Run.Width == 320u);
    CHECK(result.Report.Run.Height == 240u);

    REQUIRE_FALSE(result.Capture.IsEmpty());
    CHECK(result.Capture.Extent.Width == 320u);
    CHECK(result.Capture.Extent.Height == 240u);
}

TEST_CASE("The headless binary replays a script and needs no frame count", "[scene]")
{
    RequireDevice();

    // The other subprocess case covers --frames; this one covers the script as a
    // command-line surface: the flag, the file being read through it, and the
    // rule that a script which quits stands in for a frame count. In-process
    // cases reach none of that — they hand the platform a script object.
    const std::filesystem::path outputDir =
        std::filesystem::temp_directory_path() / "hikari_scene_tests";
    std::filesystem::create_directories(outputDir);

    const std::filesystem::path report = outputDir / "scripted.json";
    std::filesystem::remove(report);

    const std::string command = std::string("\"") + HIKARI_HEADLESS_BINARY + "\"" +
                                " --content \"" + TestDataDir() + "\"" +
                                " --scene scenes/single_cube.map --fixed-dt --no-ui" +
                                " --strict-validation --input \"" + TestDataDir() +
                                "input/scripted_replay.txt\"" + " --report \"" + report.string() +
                                "\"";

    INFO("command: " << command);
    REQUIRE(std::system(command.c_str()) == 0);
    REQUIRE(std::filesystem::exists(report));

    std::ifstream reportFile(report);
    const std::string contents((std::istreambuf_iterator<char>(reportFile)),
                               std::istreambuf_iterator<char>());

    // The script quits on frame 14 and resizes on frame 8, so the report is
    // evidence that it drove the run rather than being loaded and ignored.
    CHECK(contents.find("\"frames\": 15") != std::string::npos);
    CHECK(contents.find("\"width\": 320") != std::string::npos);
    CHECK(contents.find("\"height\": 240") != std::string::npos);
    CHECK(contents.find("\"validationErrors\": 0") != std::string::npos);
}

TEST_CASE("The headless binary refuses a script it cannot run", "[scene]")
{
    // No device needed: both refusals happen while parsing, before anything asks
    // for one, which is also why they are worth having — a run that cannot end
    // should fail at the command line rather than after loading a scene.
    const std::filesystem::path outputDir =
        std::filesystem::temp_directory_path() / "hikari_scene_tests";
    std::filesystem::create_directories(outputDir);

    const std::string binary = std::string("\"") + HIKARI_HEADLESS_BINARY + "\"";

    SECTION("a script that is not there")
    {
        const std::string command = binary + " --input \"" + (outputDir / "missing.txt").string() +
                                    "\" > /dev/null 2>&1";
        CHECK(std::system(command.c_str()) != 0);
    }

    SECTION("a script that never quits, with no frame count to fall back on")
    {
        const std::filesystem::path script = outputDir / "never_ends.txt";
        std::ofstream(script) << "frame 1 key.down W\n";

        const std::string command =
            binary + " --input \"" + script.string() + "\" > /dev/null 2>&1";
        CHECK(std::system(command.c_str()) != 0);
    }
}
