// The engine itself, whole. Private to this module — it names the renderer, the
// scene and the asset types that live beside it in src/ — and reached from
// outside only through <engine/IEngine.h>, which CreateEngine below implements.
//
// One translation unit rather than a header and a source: nothing else in the
// module needs the class, and a header carrying it would have to qualify every
// name in 2,000 lines that the using-directives below cover. Stage 8 splits it
// by promoting whole passes out, not by cutting it in half here.

#include <atomic>
#include <span>

#include "AssetRegistry.h"
#include "BindGroupLayouts.h"
#include "Camera.h"
#include "CloudSystem.h"
#include "Common.h"
#include "Cubemap.h"
#include "Entity.h"
#include "FrameData.h"
#include "InstanceData.h"
#include "Lights.h"
#include "MaterialFactory.h"
#include "Model.h"
#include "ModelManager.h"
#include "PBRMaterial.h"
#include "Texture.h"
#include "Vertex.h"
#include "XmlParser.h"

#include <core/Clock.h>
#include <core/IJobSystem.h>
#include <core/Log.h>
#include <core/Timer.h>

#include <engine/CameraPresets.h>
#include <engine/EngineConfig.h>
#include <engine/IEngine.h>
#include <engine/IUiBackend.h>
#include <engine/RunResult.h>
#include <engine/RunSpec.h>

#include <platform/FileSystem.h>
#include <platform/IPlatform.h>
#include <platform/Paths.h>

#include <rhi/BarrierPresets.h>
#include <rhi/BufferDesc.h>
#include <rhi/DeviceDesc.h>
#include <rhi/Diagnostics.h>
#include <rhi/Handles.h>
#include <rhi/ICommandList.h>
#include <rhi/IDevice.h>
#include <rhi/IPresentTarget.h>
#include <rhi/RhiTypes.h>
#include <rhi/SamplerDesc.h>
#include <rhi/TextureDesc.h>
#include <rhi/TextureViewDesc.h>
#include <rhi/UniqueHandle.h>
#include <rhi/UploadContext.h>
#include <rhi/vulkan/VulkanNative.h>

#include <SDL3/SDL.h>

#include "ImGuiFileDialog.h"
#include "imgui.h"

using namespace Hikari;
using namespace Hikari::Core;
using namespace Hikari::Platform;
using namespace Hikari::Rhi::Vulkan;

constexpr LogCategory LogWindow("Window");
constexpr LogCategory LogEngine("Engine");
constexpr LogCategory LogRenderer("Renderer");

struct LightData
{
    uint32_t PointLightCount;
    uint32_t DirLightCount;
    glm::vec2 Padding;
    PointLight::Data PointLights[MAX_POINT_LIGHTS];
    DirectionalLight::Data DirLights[MAX_DIR_LIGHTS];
};

struct CameraData
{
    glm::mat4 View;
    glm::mat4 Proj;
    glm::mat4 InvViewProj;
    glm::vec3 Pos;
    float NearPlane;
    glm::vec3 Padding;
    float FarPlane;
};

/**
 * Each member must start at an offset that is a multiple of its base alignment.
 * Eg. a float can start on offset 0, 4, 8 or 12.
 * glm::vec3 is 12 bytes wide by default but is 16 byte aligned.
 */
struct GlobalBuffer
{
    LightData Lights;
    CameraData CamData;
    glm::vec3 SkyColor;
    float Time;
};

#ifdef NDEBUG
constexpr bool bEnableValidationLayers = false;
#else
constexpr bool bEnableValidationLayers = true;
#endif

/**
 * Whether a frame can be captured from a present target in `format`.
 *
 * A capture is 8-bit RGBA and nothing else, so a 16-bit float target — which
 * nothing asks for today but the neutral format list can name — would need tone
 * mapping rather than a channel swap. Rejected with a message instead of
 * silently reinterpreting the bytes, which is what a bare BytesPerTexel check
 * would allow.
 */
constexpr bool IsCapturableFormat(Rhi::Format format)
{
    return format == Rhi::Format::BGRA8Unorm || format == Rhi::Format::RGBA8Unorm ||
           format == Rhi::Format::RGBA8Srgb;
}

namespace Hikari::Engine
{

/** Set by RequestStop(), read by the frame loop. */
std::atomic<bool> g_bShouldClose = false;

/**
 * Elapsed milliseconds since `start`.
 *
 * steady_clock, not high_resolution_clock: libstdc++ makes the latter an alias
 * for system_clock, which the standard does not require to move only forwards,
 * so an NTP correction mid-run can produce a nonsense interval or a negative
 * one. core/Timer.h makes the same choice for the same reason.
 */
inline float MillisecondsSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

/** Takes the samples by value because it sorts them to find the percentile. */
inline TimingStats ComputeTimingStats(std::vector<float> samples)
{
    if (samples.empty())
        return {};

    double sum = 0.0;
    for (const float sample : samples)
        sum += sample;

    std::sort(samples.begin(), samples.end());

    // Nearest-rank: the smallest sample at or above the 99th percentile, which
    // for fewer than 100 samples is the largest one.
    size_t index = static_cast<size_t>(std::ceil(0.99 * static_cast<double>(samples.size())));
    index = std::min(index, samples.size()) - 1;

    return TimingStats{.Mean = static_cast<float>(sum / static_cast<double>(samples.size())),
                       .P99 = samples[index],
                       .Min = samples.front(),
                       .Max = samples.back()};
}

class Engine final : public IEngine
{
public:
    Engine(IPlatform& platform, const Paths& paths, IUiBackend* pUiBackend, RunSpec spec,
           EngineConfig config, IJobSystem& jobSystem, Rhi::Diagnostics& diagnostics,
           std::chrono::steady_clock::time_point processStart)
        : m_Platform(platform), m_Paths(paths), m_pUiBackend(pUiBackend), m_Spec(std::move(spec)),
          m_Config(config), m_JobSystem(jobSystem), m_Diagnostics(diagnostics),
          m_RhiDevice(Rhi::CreateDevice(MakeDeviceDesc())),
          m_PhysicalDevice(Rhi::Vulkan::GetPhysicalDevice(*m_RhiDevice)),
          m_ProcessStart(processStart)
    {
        // Sized here rather than at first use: every per-frame resource below is
        // built by index into this, and a run with one frame in flight has to
        // find one slot rather than the two a fixed array would always hold.
        m_Frames.resize(m_Config.FramesInFlight);
    }
    ~Engine()
    {
        if (!m_bShutdown)
        {
            m_RhiDevice->WaitIdle();
            Shutdown();
        }
    }

    RunResult Run() override
    {
        // A process may run the engine more than once — the scene tests run it
        // once per case — and the stop flag is a global because a signal handler
        // has to be able to set it. Left over from a previous run it would end
        // this one before its first frame.
        g_bShouldClose = false;

        Init();

        m_Platform.Show();

        // Everything before the first frame: argument parsing, the window, the
        // device, every pipeline, and the scene's uploads.
        m_StartupMs = MillisecondsSince(m_ProcessStart);

        while (!g_bShouldClose)
        {
            const auto frameStart = std::chrono::steady_clock::now();
            m_FrameWaitMs = 0.f;

            if (!m_bIsFocused)
            {
                // Counted as waiting rather than as work: an unfocused frame is
                // idling on purpose, and charging that to the CPU cost would
                // make a backgrounded run look like a slow one.
                const auto idleStart = std::chrono::steady_clock::now();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                m_FrameWaitMs += MillisecondsSince(idleStart);
            }

            // Simulation time, which is not the frame's measured duration: the
            // timings below read the steady clock themselves, so a fixed step
            // moves the world without touching what the report says the frame
            // cost.
            m_DeltaTime = m_Clock->Tick();
            m_RunTime = m_Clock->Elapsed();

            // Through the platform seam rather than by polling a window system:
            // that is what lets a scripted run replay the same events a real one
            // produces, so input, resizing and target recreation are exercised
            // with no display attached.
            for (const PlatformEvent& event : m_Platform.PumpEvents())
            {
                // Every event, translated or not — text input and mouse buttons
                // are the UI's business and this switch does not name them.
                if (event.pNative != nullptr)
                    m_pUiBackend->ProcessPlatformEvent(event.pNative);

                switch (event.Type)
                {
                    case EventType::MouseMotion:
                        HandleMouse(event.MouseDeltaX, event.MouseDeltaY);
                        break;
                    case EventType::Quit:
                        g_bShouldClose = true;
                        break;
                    case EventType::Resized:
                        RecreateSwapchainAndRenderImages();
                        break;
                    case EventType::FocusGained:
                        m_bIsFocused = true;
                        LogMsg(LogSeverity::Info, LogWindow, "Focus gained");
                        break;
                    case EventType::FocusLost:
                        m_bIsFocused = false;
                        LogMsg(LogSeverity::Info, LogWindow, "Focus lost");
                        break;
                    case EventType::KeyDown:
                        if (m_bIsFocused)
                            HandleKey(event.key);
                        break;
                    case EventType::CaptureRequested:
                        // Staged on this frame rather than the last one, which is
                        // the only way a run captures a chosen moment.
                        m_bCaptureThisFrame = true;
                        break;
                    case EventType::KeyUp:
                    case EventType::Unknown:
                        break;
                }
            }

            m_Camera->Tick();
            HandleMovement();

            if (m_bCursorVisible && !m_Spec.bNoUi)
                DrawImGuiFrame();

            m_ModelManager.GenerateBatches(*m_SceneGraph);

            const bool bIsLastFrame =
                g_bShouldClose || (m_Spec.Frames != 0 && (m_FrameCounter + 1) >= m_Spec.Frames);
            // Either the run's last frame, when a capture was asked for at all,
            // or whichever frame a script pointed at.
            const bool captureScreenshot =
                m_bCaptureThisFrame || (bIsLastFrame && m_Spec.bCaptureFinalFrame);
            m_bCaptureThisFrame = false;

            DrawFrame(captureScreenshot);

            ++m_FrameCounter;
            RecordFrameTiming(frameStart);

            if (m_Spec.Frames != 0 && m_FrameCounter >= m_Spec.Frames)
            {
                g_bShouldClose = true;
            }
        }

        const bool bWantsScreenshot = m_Spec.bCaptureFinalFrame;

        // The in-frame decision at the top of the loop asks whether this is the
        // last frame, and a signal arriving after that line makes the answer
        // wrong: the loop exits at the top of the next iteration with nothing
        // staged, and the run ends with the "called without a captured frame"
        // error instead of a PNG. Draw one more frame and capture that instead.
        //
        // Only reachable on the interrupted path — a --frames N run stages its
        // capture on frame N-1 and never gets here. The extra frame counts as an
        // ordinary one, in the frame total and in the timing series: it is a
        // frame the app drew, and an interrupted run is not a measured one.
        //
        // The obvious alternative — stage a copy every frame while --screenshot
        // is set, so one is always ready — is what this avoids. It would add a
        // copy and a barrier to every frame of the mode used for measurement,
        // moving barriers and barrierCalls in every captured run and inflating
        // the very timings the report exists to make honest.
        if (bWantsScreenshot && !m_bScreenshotBufferReady)
        {
            const auto frameStart = std::chrono::steady_clock::now();
            m_FrameWaitMs = 0.f;

            DrawFrame(true);

            ++m_FrameCounter;
            RecordFrameTiming(frameStart);
        }

        RunResult result{.Report = BuildRunReport(), .Capture = {}};

        // Whatever was staged, however it came to be asked for: the run
        // description's "capture the final frame", or a script pointing at a
        // frame of its own.
        if (m_bScreenshotBufferReady)
            result.Capture = CaptureFinalFrame();

        // After both: the report reads the present target's extent and format,
        // and the capture reads the staging buffer the device still owns.
        m_RhiDevice->WaitIdle();
        Shutdown();

        return result;
    }

private:
    /**
     * Called from the constructor's initialiser list, so it may only touch
     * members declared above m_RhiDevice.
     */
    [[nodiscard]] Rhi::DeviceDesc MakeDeviceDesc() const
    {
        Rhi::DeviceDesc desc;
        desc.ApplicationName = "HikariEngine";
        desc.bEnableValidation = bEnableValidationLayers;
        desc.pDiagnostics = &m_Diagnostics;
        // The line the whole headless path turns on: no present requirement
        // means the device creates no surface, and CreatePresentTarget hands
        // back an OffscreenTarget instead of a SwapchainTarget.
        desc.Requirements.bPresent = !m_Platform.IsHeadless();
        desc.Requirements.NativeWindowHandle = m_Platform.GetNativeWindowHandle();
        desc.DisabledOptionalExtensions = m_Spec.DisabledVulkanExtensions;
        desc.bForceSingleQueue = m_Spec.bForceSingleQueue;
        return desc;
    }

    void Init()
    {
        LogMsg(LogSeverity::Info, LogEngine, "Init()");

        // Started here rather than at construction, so that the world's clock
        // begins when the frame loop is about to, not while the device is still
        // being built.
        if (m_Spec.bFixedDt)
            m_Clock = std::make_unique<Core::FixedStepClock>();
        else
            m_Clock = std::make_unique<Core::RealClock>();

        InitVulkan();
        InitImGui();

        if (!m_Spec.ScenePath.empty())
        {
            m_SceneGraph =
                XmlParser::LoadScene(m_Paths.Content(m_Spec.ScenePath).string(), *m_Assets);
            if (!m_SceneGraph)
            {
                throw std::runtime_error("Failed to load scene: " + m_Spec.ScenePath);
            }
            m_Assets->PurgeCaches();
        }
        else
        {
            m_SceneGraph = std::make_unique<SceneGraph>();
        }

        // TODO: read from scene
        CubemapCreateInfo createInfo{};
        createInfo.Name = "Skybox";
        createInfo.Format = Rhi::Format::RGBA8Srgb;

        const auto skyboxFace = [this](std::string_view face)
        { return m_Paths.Content("textures/skybox/" + std::string(face)).string(); };

        createInfo.RightPath = skyboxFace("right.jpg");
        createInfo.LeftPath = skyboxFace("left.jpg");
        createInfo.TopPath = skyboxFace("top.jpg");
        createInfo.BottomPath = skyboxFace("bottom.jpg");
        createInfo.FrontPath = skyboxFace("front.jpg");
        createInfo.BackPath = skyboxFace("back.jpg");

        // Logged and skipped rather than fatal, which is what the error convention
        // asks of asset loading: nothing renders the skybox yet, so a content
        // root without one is a scene that looks unfinished rather than a run
        // that cannot start.
        try
        {
            m_Skybox = m_Assets->LoadCubemap(createInfo);
        }
        catch (const std::exception& error)
        {
            LogMsg(LogSeverity::Warning, LogEngine, "Skybox not loaded: {}", error.what());
        }

        m_Camera = std::make_unique<Camera>();

        if (m_Spec.CameraPreset >= 0)
        {
            if (m_Spec.CameraPreset >= kNumCameraPresets)
            {
                throw std::runtime_error(
                    "Invalid --camera-preset index: " + std::to_string(m_Spec.CameraPreset) +
                    " (valid range: 0-" + std::to_string(kNumCameraPresets - 1) + ")");
            }

            const CameraPresetData& preset = kCameraPresets[m_Spec.CameraPreset];
            m_Camera->GetTransform().Position = preset.Position;
            m_Camera->GetTransform().Rotation = preset.Rotation;
        }
        else
        {
            m_Camera->GetTransform().Position += glm::vec3(0.f, 0.f, 10.f);
        }

        LogMsg(LogSeverity::Info, LogEngine, "Init() succeeded");
    }

    void InitImGui()
    {
        // The ring has to be at least as long as the frames in flight; the
        // backend clamps the rest. An offscreen target makes one image per
        // frame in flight, so its count alone would be too short.
        m_pUiBackend->Init(UiBackendDesc{
            .pDevice = m_RhiDevice.get(),
            .pPipelineCache = m_PipelineCache.get(),
            .pNativeWindowHandle =
                m_Platform.IsHeadless() ? nullptr : m_Platform.GetNativeWindowHandle(),
            .TargetFormat = m_PresentTarget->GetFormat(),
            .RingSize = std::max(m_PresentTarget->GetImageCount(), m_Config.FramesInFlight)});
    }

    /**
     * The present target speaks neutral types; these are the two places the
     * renderer still needs the Vulkan spelling, and it needs it often enough
     * that converting at each use site would drown the call sites.
     */
    vk::Extent2D SwapchainExtent() const
    {
        const Core::Extent2D extent = m_PresentTarget->GetExtent();
        return vk::Extent2D{extent.Width, extent.Height};
    }

    void InitVulkan()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "InitVulkan()");

        // The device itself was created in the constructor, so that every member
        // below can assume it exists.
        //
        // First, because almost everything after it is sized to the target's
        // extent or built against its format.
        const Extent2D framebufferExtent = m_Platform.GetFramebufferExtent();
        m_PresentTarget = m_RhiDevice->CreatePresentTarget(
            Rhi::PresentTargetDesc{.Extent = {framebufferExtent.Width, framebufferExtent.Height},
                                   .FramesInFlight = m_Config.FramesInFlight});

        CreateDepthResources();
        CreateBindGroupLayouts();
        CreateTextureSampler();

        m_UploadContext = m_RhiDevice->CreateUploadContext(
            Rhi::UploadContextDesc{.DebugName = "Asset Upload Context"});

        // Before any pipeline is built, and before ImGui, which is handed the
        // same one. Paths::UserData is empty when the platform gave us nowhere
        // to write, and an empty path is how the cache is told to stay in
        // memory for the run.
        m_PipelineCache = m_RhiDevice->CreatePipelineCache(Rhi::PipelineCacheDesc{
            .Path = m_Paths.UserData("pipeline_cache.bin"), .DebugName = "Pipeline Cache"});

        // Before the registry, which hands it to the loader that builds
        // materials, and before the pipelines, which are laid out against its
        // descriptor set layout.
        m_MaterialFactory = std::make_unique<MaterialFactory>(*m_RhiDevice, m_TextureSampler.Get());

        // After the upload context it loads through, and before anything that
        // loads: the registry is what every asset in the run comes from.
        m_Assets = std::make_unique<AssetRegistry>(*m_RhiDevice, *m_UploadContext, m_Paths,
                                                   *m_MaterialFactory);

        CreatePipelines();
        CreateCommandAllocators();
        CreateGlobalBuffers();
        CreateInstanceBuffers(m_Config.InitialInstanceCapacity);
        CreateRenderTargets();

        // TODO: read from scene
        CloudSystemCreateInfo cloudCreateInfo{.RhiDevice = *m_RhiDevice,
                                              .PipelineCache = *m_PipelineCache,
                                              .ContentPaths = m_Paths,
                                              .GlobalSetLayout = m_GlobalLayout.Get(),
                                              .DepthSetLayout = m_DepthLayout.Get(),
                                              .SwapchainWidth = SwapchainExtent().width,
                                              .SwapchainHeight = SwapchainExtent().height,
                                              .FramesInFlight = m_Config.FramesInFlight};
        m_CloudSystem = std::make_unique<CloudSystem>(cloudCreateInfo);

        CreateGlobalBindGroups();
        RecreateFrameBindGroups();
        CreateSyncObjects();
        CreateQuadBuffers();
    }

    /**
     * What this run measured, as data. Nothing here decides whether it becomes
     * a file: the app is handed the struct and makes that call.
     */
    RunReport BuildRunReport() const
    {
        // The last frame's counts, as the draw call and batch numbers below also
        // are. Worth knowing when comparing two reports: a captured run copies
        // the final frame, and that copy costs one extra barrier and one extra
        // call, so it legitimately reads one higher than an uncaptured one.
        const Rhi::BarrierCounts barrierCounts = FrameBarrierCounts();

        RunReport report;
        report.Frames = m_FrameCounter;

        report.Counters.Frame = {.DrawCalls = m_OpaqueDrawCallCount + m_TransparentDrawCallCount,
                                 .Batches = m_OpaqueBatchCount + m_TransparentBatchCount,
                                 .Instances = m_OpaqueInstanceCount + m_TransparentInstanceCount,
                                 .Barriers = barrierCounts.Barriers,
                                 .BarrierCalls = barrierCounts.Calls};

        report.Counters.Run = {.ValidationErrors = m_Diagnostics.ErrorCount(),
                               .ValidationWarnings = m_Diagnostics.WarningCount(),
                               .UploadSubmissions = m_UploadContext->GetStats().Submits};

        report.Timings = {.StartupMs = m_StartupMs,
                          .FirstFrame = {.FrameMs = m_FirstFrameMs, .CpuMs = m_FirstFrameCpuMs},
                          .FrameMs = ComputeTimingStats(m_FrameMs),
                          .CpuMs = ComputeTimingStats(m_CpuMs)};

        report.Run = {.bFixedDt = m_Spec.bFixedDt,
                      .bHeadless = m_Platform.IsHeadless(),
                      .bNoUi = m_Spec.bNoUi,
                      .Width = SwapchainExtent().width,
                      .Height = SwapchainExtent().height,
                      .JobCount = static_cast<uint32_t>(m_JobSystem.WorkerCount()),
                      .PresentMode = m_PresentTarget->GetPresentMode(),
                      .BuildConfig = HIKARI_BUILD_CONFIG};

        return report;
    }

    /**
     * Reads back the frame staged during the final DrawFrame() call, before it
     * was presented, as tightly packed 8-bit RGBA.
     *
     * The swizzle happens here because this is where the target's format is
     * known: a present target picks its own — an offscreen one need not agree
     * with a surface — so a caller that assumed a channel order would be wrong
     * on the first target that chose differently.
     */
    CapturedFrame CaptureFinalFrame()
    {
        m_RhiDevice->WaitIdle();

        if (!m_bScreenshotBufferReady)
        {
            LogMsg(LogSeverity::Error, LogEngine,
                   "A capture was asked for without a captured frame. No frame was drawn?");
            return {};
        }

        const Rhi::Format format = m_PresentTarget->GetFormat();
        if (!IsCapturableFormat(format))
        {
            LogMsg(LogSeverity::Error, LogEngine,
                   "Cannot capture the final frame: the present target's format is not an 8-bit "
                   "four-channel one, which is all a capture can describe.");
            return {};
        }

        const uint32_t bytesPerPixel = Rhi::BytesPerTexel(format);
        const uint32_t width = SwapchainExtent().width;
        const uint32_t height = SwapchainExtent().height;

        // A capture is RGBA, so a BGRA target needs its first and third channels
        // swapped and an RGBA one needs nothing. The shader is indifferent
        // either way: it writes SV_Target component 0 and the hardware maps that
        // to whatever the format's first component is.
        const bool bSwapRedAndBlue = format == Rhi::Format::BGRA8Unorm;

        const auto* src = static_cast<const uint8_t*>(
            m_RhiDevice->GetMappedData(m_ScreenshotStagingBuffer.Get()));

        CapturedFrame capture;
        capture.Extent = {width, height};
        capture.Pixels.resize(static_cast<size_t>(width) * height * bytesPerPixel);
        for (size_t i = 0; i < static_cast<size_t>(width) * height; i++)
        {
            capture.Pixels[i * 4 + 0] = bSwapRedAndBlue ? src[i * 4 + 2] : src[i * 4 + 0];
            capture.Pixels[i * 4 + 1] = src[i * 4 + 1];
            capture.Pixels[i * 4 + 2] = bSwapRedAndBlue ? src[i * 4 + 0] : src[i * 4 + 2];
            capture.Pixels[i * 4 + 3] = src[i * 4 + 3];
        }

        return capture;
    }

    void Shutdown()
    {
        LogMsg(LogSeverity::Info, LogEngine, "Shutdown()");

        // Before ImGui, which built pipelines into the same cache, and before
        // the device that owns it goes away.
        m_PipelineCache->Save();

        m_Skybox.reset();
        m_SceneGraph.reset();
        ShutdownImGui();
        // The registry's caches assert they are empty, which only holds once
        // everything above has dropped what it borrowed; and the factory owns
        // the descriptor sets those materials were allocated from, so it goes
        // after the models that held them.
        m_Assets.reset();
        m_MaterialFactory.reset();

        m_bShutdown = true;
    }

    void ShutdownImGui()
    {
        // The backend owns the context it created, so destroying it is its job.
        m_pUiBackend->Shutdown();
    }

    void HandleMouse(float x, float y)
    {
        // Same reasoning as HandleMovement: the cursor decides, not the preset.
        if (!m_bCursorVisible)
            m_Camera->Rotate(x, y);
    }

    void ShowCursor()
    {
        m_Platform.WarpMouse(static_cast<float>(SwapchainExtent().width / 2.f),
                             static_cast<float>(SwapchainExtent().height / 2.f));
        m_Platform.SetRelativeMouseMode(false);
        m_bCursorVisible = true;
    }

    void HideCursor()
    {
        m_Platform.SetRelativeMouseMode(true);
        m_bCursorVisible = false;
    }

    /** This includes OS key repeat delay. */
    void HandleKey(Key key)
    {
        switch (key)
        {
            case Key::Escape:
                if (m_bCursorVisible)
                    HideCursor();
                else
                    ShowCursor();
                break;
            case Key::F9:
                m_Platform.SetWindowMode(WindowMode::Windowed);
                break;
            case Key::F10:
                m_Platform.SetWindowMode(WindowMode::BorderlessFullscreen);
                break;
            case Key::F11:
                m_Platform.SetWindowMode(WindowMode::ExclusiveFullscreen);
                break;
            default:
                break;
        }
    }

    /**
     * Checking the state of the keys every frame, bypassing OS key repeat
     * delay.
     */
    void HandleMovement()
    {
        // A visible cursor means the UI is being driven rather than the camera.
        //
        // A camera preset used to block movement too, from a time when the only
        // input was a hand on a keyboard and a preset run had to be pinned down
        // to be comparable. A script is as deterministic as the preset is, so a
        // preset now means where the camera starts and nothing more; a run that
        // wants it to stay there simply sends no input, which is what every
        // capture run does.
        if (m_bCursorVisible)
            return;

        glm::vec3 camOffset = {0.f, 0.f, 0.f};
        if (m_Platform.IsKeyDown(Key::A))
        {
            camOffset += -m_Camera->GetRightVector() * m_Camera->GetMoveSpeed() * m_DeltaTime;
        }
        if (m_Platform.IsKeyDown(Key::D))
        {
            camOffset += m_Camera->GetRightVector() * m_Camera->GetMoveSpeed() * m_DeltaTime;
        }
        if (m_Platform.IsKeyDown(Key::W))
        {
            camOffset += m_Camera->GetForwardVector() * m_Camera->GetMoveSpeed() * m_DeltaTime;
        }
        if (m_Platform.IsKeyDown(Key::S))
        {
            camOffset += -m_Camera->GetForwardVector() * m_Camera->GetMoveSpeed() * m_DeltaTime;
        }
        if (m_Platform.IsKeyDown(Key::Q))
        {
            camOffset += glm::vec3(0.f, -1.f, 0.f) * m_Camera->GetMoveSpeed() * m_DeltaTime;
        }
        if (m_Platform.IsKeyDown(Key::E))
        {
            camOffset += glm::vec3(0.f, 1.f, 0.f) * m_Camera->GetMoveSpeed() * m_DeltaTime;
        }

        if ((std::fabs(camOffset.x) + std::fabs(camOffset.y) + std::fabs(camOffset.z)) > 0.f)
            m_Camera->GetTransform().Position += camOffset;
    }

    /**
     * Closes out the frame that started at `frameStart`.
     *
     * Frame 0 goes to its own pair of fields rather than into the series: it
     * pays for the first use of every pipeline, the first descriptor writes and
     * the first acquire, so it is an outlier by construction and mixing it in
     * describes neither it nor a steady-state frame. The panel's readout is fed
     * from the measurement too, so that it shows what a frame cost rather than
     * what --fixed-dt told the simulation to pretend.
     */
    void RecordFrameTiming(std::chrono::steady_clock::time_point frameStart)
    {
        const float frameMs = MillisecondsSince(frameStart);

        // Clamped because the two are measured by different calls to the same
        // clock: a wait can be charged a few microseconds the frame total has
        // not yet accrued, and a negative CPU cost is worse than a zero one.
        const float cpuMs = std::max(0.f, frameMs - m_FrameWaitMs);

        const float smoothing = 0.9f;
        m_DisplayFrameTime = (m_DisplayFrameTime * smoothing) + (frameMs * (1.f - smoothing));
        if (frameMs > 0.f)
            m_DisplayFPS = (m_DisplayFPS * smoothing) + ((1000.f / frameMs) * (1.f - smoothing));

        if (!m_Spec.bRecordTimings)
            return;

        if (m_FrameCounter == 1)
        {
            m_FirstFrameMs = frameMs;
            m_FirstFrameCpuMs = cpuMs;
            return;
        }

        m_FrameMs.push_back(frameMs);
        m_CpuMs.push_back(cpuMs);
    }

    void DrawFrame(bool captureScreenshot = false)
    {
        // Semaphores coordinate GPU to GPU synchronisation, for example
        // ordering work between queues. They get reset automatically after the
        // waiting operation begins.
        //
        // Fences coordinate CPU to GPU synchronisation, for times when
        // the CPU needs to know that the GPU has finished a task. Must be
        // explicitely reset by the host.

        // A recreation that was deferred means the surface had no area when it
        // was last asked. Retry it here, and skip the frame while the answer
        // has not changed: there is nothing to draw into.
        if (m_bSwapchainOutOfDate)
        {
            RecreateSwapchainAndRenderImages();
            if (m_bSwapchainOutOfDate)
                return;
        }

        FrameData& frameData = m_Frames[m_FrameIndex];

        // The frame's blocking calls are measured rather than assumed away: on a
        // FIFO surface the wait is most of the wall clock, and a CPU cost that
        // included it would read as a regression whenever the display paced us.
        const auto fenceWaitStart = std::chrono::steady_clock::now();
        m_RhiDevice->WaitForFence(m_FrameFence.Get(), frameData.LastSubmitValue);
        m_FrameWaitMs += MillisecondsSince(fenceWaitStart);

        const auto acquireStart = std::chrono::steady_clock::now();
        const Rhi::AcquiredImage image = m_PresentTarget->Acquire();
        m_FrameWaitMs += MillisecondsSince(acquireStart);
        if (image.bNeedsRecreate)
        {
            RecreateSwapchainAndRenderImages();
            return;
        }

        UpdateGlobalBuffer(m_FrameIndex);
        UpdateInstanceBuffer(m_FrameIndex);

        if (captureScreenshot && !m_bScreenshotBufferReady)
        {
            const vk::DeviceSize bufferSize = static_cast<vk::DeviceSize>(SwapchainExtent().width) *
                                              SwapchainExtent().height *
                                              Rhi::BytesPerTexel(m_PresentTarget->GetFormat());
            m_ScreenshotStagingBuffer = Rhi::UniqueHandle<Rhi::BufferHandle>(
                *m_RhiDevice,
                m_RhiDevice->CreateBuffer(Rhi::BufferDesc{.Size = bufferSize,
                                                          .Usage = Rhi::BufferUsage::CopyDst,
                                                          .Access = Rhi::MemoryAccess::GpuToCpu,
                                                          .DebugName = "Screenshot Staging"}));
            m_bScreenshotBufferReady = true;
        }

        {
            // Timer recordTimer("Command buffer recording");
            m_MainThreadBarrierCounts = {};

            m_JobSystem.Submit([&] { RecordOpaqueCommandBuffer(); });
            m_JobSystem.Submit([&] { RecordTransparentCommandBuffer(); });

            // these command buffers are very small and so likely faster to
            // record on main thread
            RecordSwapImageToDrawLayout(image);
            RecordCloudsCommandBuffer();
            RecordCompositeCommandBuffer(image);
            RecordImGui(image);
            RecordSwapImageToFinalLayout(image, captureScreenshot);

            m_JobSystem.Wait();
            LogBarrierCounts();
        }

        // TODO: even when ImGui is not showing, it's being submitted
        const std::array<Rhi::ICommandList*, 7> commandLists = {
            frameData.DrawLayoutCommands.List,  frameData.OpaqueCommands.List,
            frameData.TransparentCommands.List, frameData.CloudCommands.List,
            frameData.CompositeCommands.List,   frameData.ImGuiCommands.List,
            frameData.FinalLayoutCommands.List};
        // The waits arrive as a span because how many there are is the target's
        // business: a swapchain hands back the one its acquire signalled, and a
        // headless target hands back the previous write of the same image, or
        // nothing at all on the first pass.
        const Rhi::SemaphoreHandle renderComplete =
            m_PresentTarget->GetRenderCompleteSemaphore(image.Index);

        // The value this slot will wait for next time round.
        frameData.LastSubmitValue = ++m_FrameSubmitCount;
        const Rhi::FenceOperation signalFrame{.Fence = m_FrameFence.Get(),
                                              .Value = frameData.LastSubmitValue};

        m_RhiDevice->Submit(Rhi::SubmitDesc{.Queue = Rhi::QueueType::Graphics,
                                            .CommandLists = commandLists,
                                            .SignalFences = {&signalFrame, 1u},
                                            .WaitSemaphores = image.WaitSemaphores,
                                            .SignalSemaphores = {&renderComplete, 1u}});

        const auto presentStart = std::chrono::steady_clock::now();
        const bool bPresented = m_PresentTarget->Present(image.Index);
        m_FrameWaitMs += MillisecondsSince(presentStart);
        if (!bPresented)
            RecreateSwapchainAndRenderImages();

        m_FrameIndex = (m_FrameIndex + 1) % m_Config.FramesInFlight;
    }

    void DrawImGuiFrame()
    {
        m_pUiBackend->NewFrame();
        if (m_Platform.IsHeadless())
        {
            // Standing in for the platform backend. DeltaTime is the
            // load-bearing one — NewFrame asserts it is positive, where
            // DisplaySize need only be non-negative.
            ImGuiIO& io = ImGui::GetIO();
            io.DisplaySize = ImVec2(static_cast<float>(SwapchainExtent().width),
                                    static_cast<float>(SwapchainExtent().height));
            io.DeltaTime = m_DeltaTime;
        }

        ImGui::NewFrame();

        if (ImGui::Begin("Menu"))
        {
            for (size_t i = 0; i < m_SceneGraph->PointLights.size(); i++)
            {
                PointLight* pPointLight = m_SceneGraph->PointLights[i];
                ImGui::PushID(static_cast<int>(i));

                ImGui::Text("Point Light");
                ImGui::DragFloat3("Position", &pPointLight->GetPosition().x, 0.5f);
                ImGui::ColorEdit3("Color##PointLight", &pPointLight->GetColor().r);
                ImGui::SliderFloat("Intensity##PointLight", &pPointLight->GetIntensity(), 0.f,
                                   1000.f);
                ImGui::PopID();

                ImGui::Dummy(ImVec2(0.f, 5.f));
            }

            for (size_t i = 0; i < m_SceneGraph->DirLights.size(); i++)
            {
                DirectionalLight* pDirLight = m_SceneGraph->DirLights[i];
                ImGui::PushID(static_cast<int>(i));

                ImGui::Text("Directional Light");
                glm::vec3 dir = pDirLight->GetDirection();
                ImGui::DragFloat3("Direction", &dir.x, 0.5f);
                if (dir != pDirLight->GetDirection())
                    pDirLight->SetDirection(dir);

                ImGui::ColorEdit3("Color##DirLight", &pDirLight->GetColor().r);
                ImGui::SliderFloat("Intensity##DirLight", &pDirLight->GetIntensity(), 0.f, 10.f);

                ImGui::PopID();
            }

            ImGui::Dummy(ImVec2(0.f, 20.f));

            ImVec2 minFileDialogSize = ImVec2(600, 400);
            if (ImGui::Button("Load Scene"))
            {
                IGFD::FileDialogConfig config;
                config.path = m_Paths.Content("scenes").string();
                ImGuiFileDialog::Instance()->OpenDialog("LoadSceneDlg", "Choose Scene to Load",
                                                        ".map", config);
            }

            if (ImGuiFileDialog::Instance()->Display("LoadSceneDlg", ImGuiWindowFlags_NoCollapse,
                                                     minFileDialogSize))
            {
                if (ImGuiFileDialog::Instance()->IsOk())
                {
                    std::string path = ImGuiFileDialog::Instance()->GetFilePathName();

                    m_RhiDevice->WaitIdle();

                    // Loading the new scene before unloading current scene.
                    // Speeds up load times by not unloading resources which are
                    // used in both scenes. This does mean that you have to
                    // temporarily store both scenes in memory until the load if
                    // finished though. Can look into this in the future if it
                    // becomes a problem.
                    std::unique_ptr<SceneGraph> tempSceneGraph =
                        XmlParser::LoadScene(path, *m_Assets);
                    if (tempSceneGraph.get())
                    {
                        m_SceneGraph.reset();
                        m_SceneGraph = std::move(tempSceneGraph);
                        m_Assets->PurgeCaches();
                    }
                }
                ImGuiFileDialog::Instance()->Close();
            }

            if (ImGui::Button("Save Scene"))
            {
                IGFD::FileDialogConfig config;
                config.path = m_Paths.Content("scenes").string();
                config.fileName = "new_scene.map";
                ImGuiFileDialog::Instance()->OpenDialog("SaveSceneDlg", "Save Scene As", ".map",
                                                        config);
            }

            if (ImGuiFileDialog::Instance()->Display("SaveSceneDlg", ImGuiWindowFlags_NoCollapse,
                                                     minFileDialogSize))
            {
                if (ImGuiFileDialog::Instance()->IsOk())
                {
                    std::string path = ImGuiFileDialog::Instance()->GetFilePathName();

                    XmlParser::SaveScene(m_SceneGraph, path);
                }
                ImGuiFileDialog::Instance()->Close();
            }

            if (ImGui::Button("Quit"))
            {
                g_bShouldClose = true;
            }

            ImGui::Text("Frame time: %.4fms", m_DisplayFrameTime);
            ImGui::Text("FPS: %.1f", m_DisplayFPS);
        }
        ImGui::End();

        ImGui::Render();
    }

    /**
     * The compiled shader named `name`, as a module the device owns.
     *
     * The engine resolves the file and the device says which kind it can read
     * (plan D24): resolving a name to a path is a content question, and the RHI
     * has no business owning a filesystem. Modules are kept alive for the run
     * because a pipeline may be rebuilt on resize.
     */
    Rhi::ShaderModuleHandle LoadShader(const std::string& name)
    {
        const std::string file = std::format("{}.{}", name, m_RhiDevice->GetCaps().ShaderExtension);
        const std::vector<char> code = Platform::ReadFile(m_Paths.Shader(file).string());

        m_ShaderModules.push_back(Rhi::UniqueHandle<Rhi::ShaderModuleHandle>(
            *m_RhiDevice, m_RhiDevice->CreateShaderModule(Rhi::ShaderModuleDesc{
                              .Bytes = std::as_bytes(std::span(code)), .DebugName = file})));

        return m_ShaderModules.back().Get();
    }

    /** Vertex plus instance streams, which every surface pipeline uses. */
    static constexpr std::array kSurfaceVertexBuffers{Vertex::GetBindingDescription(),
                                                      InstanceData::GetBindingDescription()};

    static constexpr auto SurfaceVertexAttributes()
    {
        std::array<Rhi::VertexAttribute, Vertex::AttributeCount + InstanceData::AttributeCount>
            attributes{};
        const auto vertexAttributes = Vertex::GetAttributeDescriptions();
        const auto instanceAttributes = InstanceData::GetAttributeDescriptions();
        std::ranges::copy(vertexAttributes, attributes.begin());
        std::ranges::copy(instanceAttributes, attributes.begin() + Vertex::AttributeCount);
        return attributes;
    }

    void CreateOpaquePipeline()
    {
        static constexpr auto kAttributes = SurfaceVertexAttributes();

        const std::array bindGroupLayouts{m_GlobalLayout.Get(), m_MaterialFactory->GetLayout()};
        const std::array pushRanges{Rhi::PushConstantRange{
            .Stages = Rhi::ShaderStage::Pixel, .Size = sizeof(PBRMaterial::MaterialData)}};

        m_OpaquePipelineLayout = Rhi::UniqueHandle<Rhi::PipelineLayoutHandle>(
            *m_RhiDevice, m_RhiDevice->CreatePipelineLayout(
                              Rhi::PipelineLayoutDesc{.BindGroupLayouts = bindGroupLayouts,
                                                      .PushConstantRanges = pushRanges,
                                                      .DebugName = "Opaque Layout"}));

        const std::array formats{m_OpaqueImageFormat};
        const std::array blends{Rhi::RenderTargetBlend{}};

        m_OpaquePipeline = Rhi::UniqueHandle<Rhi::GraphicsPipelineHandle>(
            *m_RhiDevice,
            m_RhiDevice->CreateGraphicsPipeline(
                Rhi::GraphicsPipelineDesc{
                    .Layout = m_OpaquePipelineLayout.Get(),
                    .VertexShader = {LoadShader("opaque"), "vertMain"},
                    .PixelShader = {m_ShaderModules.back().Get(), "fragMain"},
                    .VertexBuffers = kSurfaceVertexBuffers,
                    .VertexAttributes = kAttributes,
                    .RenderTargetFormats = formats,
                    .RenderTargetBlends = blends,
                    .DepthFormat = m_DepthFormat,
                    .Depth = {.bTest = true, .bWrite = true, .Compare = Rhi::CompareOp::Less},
                    // Two-sided materials are a per-batch property, so the mode is
                    // set per draw rather than baked in.
                    .bDynamicCull = true,
                    .DebugName = "Opaque"},
                *m_PipelineCache));
    }

    void CreateTransparentPipeline()
    {
        static constexpr auto kAttributes = SurfaceVertexAttributes();

        const std::array bindGroupLayouts{m_GlobalLayout.Get(), m_MaterialFactory->GetLayout()};
        const std::array pushRanges{Rhi::PushConstantRange{
            .Stages = Rhi::ShaderStage::Pixel, .Size = sizeof(PBRMaterial::MaterialData)}};

        m_TransparentPipelineLayout = Rhi::UniqueHandle<Rhi::PipelineLayoutHandle>(
            *m_RhiDevice, m_RhiDevice->CreatePipelineLayout(
                              Rhi::PipelineLayoutDesc{.BindGroupLayouts = bindGroupLayouts,
                                                      .PushConstantRanges = pushRanges,
                                                      .DebugName = "Transparent Layout"}));

        const std::array formats{m_AccumImageFormat, m_RevealageImageFormat};

        // Weighted-blended OIT: the accumulation target sums contributions, and
        // the revealage target multiplies what is left of the background through.
        const std::array blends{
            Rhi::RenderTargetBlend{.bEnable = true,
                                   .SrcColor = Rhi::BlendFactor::One,
                                   .DstColor = Rhi::BlendFactor::One,
                                   .SrcAlpha = Rhi::BlendFactor::One,
                                   .DstAlpha = Rhi::BlendFactor::One},
            Rhi::RenderTargetBlend{.bEnable = true,
                                   .SrcColor = Rhi::BlendFactor::Zero,
                                   .DstColor = Rhi::BlendFactor::OneMinusSrcColor,
                                   .SrcAlpha = Rhi::BlendFactor::Zero,
                                   .DstAlpha = Rhi::BlendFactor::OneMinusSrcColor}};

        m_TransparentPipeline = Rhi::UniqueHandle<Rhi::GraphicsPipelineHandle>(
            *m_RhiDevice,
            m_RhiDevice->CreateGraphicsPipeline(
                Rhi::GraphicsPipelineDesc{
                    .Layout = m_TransparentPipelineLayout.Get(),
                    .VertexShader = {LoadShader("weightedBlendedOIT"), "vertMain"},
                    .PixelShader = {m_ShaderModules.back().Get(), "fragMain"},
                    .VertexBuffers = kSurfaceVertexBuffers,
                    .VertexAttributes = kAttributes,
                    .RenderTargetFormats = formats,
                    .RenderTargetBlends = blends,
                    .DepthFormat = m_DepthFormat,
                    // Tested against the opaque depth, never written to it.
                    .Depth = {.bTest = true, .bWrite = false, .Compare = Rhi::CompareOp::Less},
                    // Not dynamic, unlike the opaque pass: transparent surfaces
                    // are drawn from both sides regardless of what the material
                    // says, so there is nothing per batch to vary.
                    .DebugName = "Transparent"},
                *m_PipelineCache));
    }

    void CreateCompositePipeline()
    {
        static constexpr std::array kQuadBuffers{QuadVertex::GetBindingDescription()};
        static constexpr auto kQuadAttributes = QuadVertex::GetAttributeDescription();

        const std::array bindGroupLayouts{m_GlobalLayout.Get(), m_CompositeLayout.Get()};

        m_CompositePipelineLayout = Rhi::UniqueHandle<Rhi::PipelineLayoutHandle>(
            *m_RhiDevice,
            m_RhiDevice->CreatePipelineLayout(Rhi::PipelineLayoutDesc{
                .BindGroupLayouts = bindGroupLayouts, .DebugName = "Composite Layout"}));

        const std::array formats{m_PresentTarget->GetFormat()};
        const std::array blends{Rhi::RenderTargetBlend{}};

        m_CompositePipeline = Rhi::UniqueHandle<Rhi::GraphicsPipelineHandle>(
            *m_RhiDevice,
            m_RhiDevice->CreateGraphicsPipeline(
                Rhi::GraphicsPipelineDesc{.Layout = m_CompositePipelineLayout.Get(),
                                          .VertexShader = {LoadShader("composite"), "vertMain"},
                                          .PixelShader = {m_ShaderModules.back().Get(), "fragMain"},
                                          .VertexBuffers = kQuadBuffers,
                                          .VertexAttributes = kQuadAttributes,
                                          .RenderTargetFormats = formats,
                                          .RenderTargetBlends = blends,
                                          .DebugName = "Composite"},
                *m_PipelineCache));
    }

    void CreatePipelines()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "CreatePipelines()");

        CreateOpaquePipeline();
        CreateTransparentPipeline();
        CreateCompositePipeline();
    }

    void CreateCommandAllocators()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "CreateCommandAllocators()");

        // Graphics for all seven, the cloud dispatch included: the frame is one
        // submit to the graphics queue, so every list in it has to come from an
        // allocator that queue accepts.
        for (size_t i = 0; i < m_Config.FramesInFlight; i++)
        {
            FrameData& frame = m_Frames[i];

            const auto make = [&](const char* name)
            {
                return m_RhiDevice->CreateCommandAllocator(
                    Rhi::CommandAllocatorDesc{.Queue = Rhi::QueueType::Graphics,
                                              .DebugName = std::format("{} Frame {}", name, i)});
            };

            frame.DrawLayoutCommands.Allocator = make("Draw Layout Commands");
            frame.OpaqueCommands.Allocator = make("Opaque Commands");
            frame.CloudCommands.Allocator = make("Cloud Commands");
            frame.TransparentCommands.Allocator = make("Transparent Commands");
            frame.CompositeCommands.Allocator = make("Composite Commands");
            frame.ImGuiCommands.Allocator = make("ImGui Commands");
            frame.FinalLayoutCommands.Allocator = make("Final Layout Commands");
        }
    }

    /**
     * Recycles a recorder's allocator and opens a list on it for this frame.
     *
     * Safe at this point because the frame's fence has already been waited on, so
     * nothing this allocator produced last time round is still executing -- which
     * is the one condition Reset() cannot check for itself.
     */
    Rhi::ICommandList& BeginRecording(FrameRecorder& recorder)
    {
        recorder.Allocator->Reset();
        recorder.List = &recorder.Allocator->Acquire();
        recorder.List->Begin();
        return *recorder.List;
    }

    /** The whole render target, which is the only area any pass draws to. */
    Rhi::Rect2D WholeTarget() const
    {
        return Rhi::Rect2D{.Extent = {SwapchainExtent().width, SwapchainExtent().height}};
    }

    Rhi::Viewport FullViewport() const
    {
        return Rhi::Viewport{.Width = static_cast<float>(SwapchainExtent().width),
                             .Height = static_cast<float>(SwapchainExtent().height)};
    }

    /**
     * What the frame's three recording threads produced between them. Summed on
     * demand rather than accumulated into one member, because two of the three
     * are written from job threads.
     */
    Rhi::BarrierCounts FrameBarrierCounts() const
    {
        Rhi::BarrierCounts total = m_OpaqueBarrierCounts;
        total += m_TransparentBarrierCounts;
        total += m_MainThreadBarrierCounts;
        return total;
    }

    /**
     * Reports the frame's barrier counts the first time they are seen and
     * whenever they change afterwards. Logging them every frame would drown the
     * log, and never logging them would make a change in the barriers — the
     * easiest thing to get wrong here and the hardest to see — invisible
     * between runs of the report.
     */
    void LogBarrierCounts()
    {
        const Rhi::BarrierCounts counts = FrameBarrierCounts();
        if (counts == m_LoggedBarrierCounts)
            return;

        LogMsg(LogSeverity::Info, LogRenderer, "Barriers recorded this frame: {} over {} calls",
               counts.Barriers, counts.Calls);
        m_LoggedBarrierCounts = counts;
    }

    void RecordOpaqueCommandBuffer()
    {
        FrameData& frame = m_Frames[m_FrameIndex];
        Rhi::ICommandList* list = &BeginRecording(frame.OpaqueCommands);

        const std::array openingBarriers{
            Rhi::BarrierPresets::UndefinedToDepthStencilWrite().On(frame.DepthTexture.GetHandle()),
            Rhi::BarrierPresets::UndefinedToRenderTarget().On(frame.OpaqueTexture.GetHandle())};
        m_OpaqueBarrierCounts = list->Barrier(openingBarriers);

        const std::array renderTargets{Rhi::RenderTarget{
            .View = frame.OpaqueTexture.GetView(),
            .Load = Rhi::LoadOp::Clear,
            .ClearColor = {m_Config.SkyColor.r, m_Config.SkyColor.g, m_Config.SkyColor.b, 1.f}}};
        const Rhi::DepthStencilTarget depthTarget{.View = frame.DepthTexture.GetView(),
                                                  .Load = Rhi::LoadOp::Clear};

        list->BeginRendering(Rhi::RenderingDesc{.RenderArea = WholeTarget(),
                                                .RenderTargets = renderTargets,
                                                .pDepthStencil = &depthTarget});
        list->SetPipeline(m_OpaquePipeline.Get());

        list->SetViewport(FullViewport());
        list->SetScissor(WholeTarget());
        list->SetBindGroup(m_OpaquePipelineLayout.Get(), 0, frame.GlobalBindGroup.Get());

        list->SetVertexBuffer(1u, frame.InstanceBuffer.Get());

        // per mesh batch
        const std::vector<MeshBatch>& batches = m_ModelManager.GetOpaqueBatches();
        uint32_t instanceCount = 0;
        for (const MeshBatch& batch : batches)
        {
            // Set every batch rather than tracked and skipped when unchanged. A
            // command buffer starts with no dynamic cull mode at all, so anything
            // that skips the first set leaves the draw invalid
            // (VUID-vkCmdDrawIndexed-None-07840) — which is what the previous
            // version did for a single-sided material, and what every material
            // shipped today being two-sided hid. Recording one more state token
            // per batch is not worth a rule about when it may be skipped.
            list->SetCullMode(batch.pMaterial->IsTwoSided() ? Rhi::CullMode::None
                                                            : Rhi::CullMode::Back);

            list->SetVertexBuffer(0u, batch.VertexBuffer);
            list->SetIndexBuffer(batch.IndexBuffer, Rhi::IndexFormat::Uint32);
            list->SetBindGroup(m_OpaquePipelineLayout.Get(), 1, batch.pMaterial->GetBindGroup());
            const auto& materialData = *static_cast<const PBRMaterial::MaterialData*>(
                batch.pMaterial->GetPushConstantData());
            list->PushConstants(m_OpaquePipelineLayout.Get(), Rhi::ShaderStage::Pixel, 0u,
                                std::as_bytes(std::span(&materialData, 1)));
            list->DrawIndexed(batch.IndexCount, batch.InstanceCount, batch.FirstIndex, 0,
                              batch.FirstInstance);
            instanceCount += batch.InstanceCount;
        }
        m_OpaqueDrawCallCount = static_cast<uint32_t>(batches.size());
        m_OpaqueBatchCount = static_cast<uint32_t>(batches.size());
        m_OpaqueInstanceCount = instanceCount;

        list->EndRendering();

        list->End();
    }

    void RecordCloudsCommandBuffer()
    {
        FrameData& frame = m_Frames[m_FrameIndex];
        Rhi::ICommandList& list = BeginRecording(frame.CloudCommands);

        m_MainThreadBarrierCounts += m_CloudSystem->RecordDispatch(
            list, m_FrameIndex, frame.GlobalBindGroup.Get(), frame.DepthBindGroup.Get());

        list.End();
    }

    void RecordTransparentCommandBuffer()
    {
        FrameData& frame = m_Frames[m_FrameIndex];
        Rhi::ICommandList* list = &BeginRecording(frame.TransparentCommands);

        const std::array openingBarriers{
            Rhi::BarrierPresets::UndefinedToRenderTarget().On(frame.AccumTexture.GetHandle()),
            Rhi::BarrierPresets::UndefinedToRenderTarget().On(frame.RevealageTexture.GetHandle()),
            Rhi::BarrierPresets::DepthStencilWriteToShaderResource().On(
                frame.DepthTexture.GetHandle())};
        m_TransparentBarrierCounts = list->Barrier(openingBarriers);

        const std::array renderTargets{Rhi::RenderTarget{.View = frame.AccumTexture.GetView(),
                                                         .Load = Rhi::LoadOp::Clear,
                                                         .ClearColor = {0.f, 0.f, 0.f, 0.f}},
                                       Rhi::RenderTarget{.View = frame.RevealageTexture.GetView(),
                                                         .Load = Rhi::LoadOp::Clear,
                                                         .ClearColor = {1.f, 0.f, 0.f, 0.f}}};

        // Read, not written: the pass tests against the opaque depth and samples
        // it, so neither backend may bind it writable while it is sampled.
        const Rhi::DepthStencilTarget depthTarget{
            .View = frame.DepthTexture.GetView(), .Load = Rhi::LoadOp::Preserve, .bReadOnly = true};

        list->BeginRendering(Rhi::RenderingDesc{.RenderArea = WholeTarget(),
                                                .RenderTargets = renderTargets,
                                                .pDepthStencil = &depthTarget});
        list->SetPipeline(m_TransparentPipeline.Get());

        list->SetViewport(FullViewport());
        list->SetScissor(WholeTarget());
        list->SetBindGroup(m_TransparentPipelineLayout.Get(), 0, frame.GlobalBindGroup.Get());

        list->SetVertexBuffer(1u, frame.InstanceBuffer.Get());

        // per mesh batch
        const std::vector<MeshBatch>& batches = m_ModelManager.GetTransparentBatches();
        uint32_t instanceCount = 0;
        for (const MeshBatch& batch : batches)
        {
            list->SetVertexBuffer(0u, batch.VertexBuffer);
            list->SetIndexBuffer(batch.IndexBuffer, Rhi::IndexFormat::Uint32);
            list->SetBindGroup(m_TransparentPipelineLayout.Get(), 1,
                               batch.pMaterial->GetBindGroup());
            const auto& materialData = *static_cast<const PBRMaterial::MaterialData*>(
                batch.pMaterial->GetPushConstantData());
            list->PushConstants(m_TransparentPipelineLayout.Get(), Rhi::ShaderStage::Pixel, 0u,
                                std::as_bytes(std::span(&materialData, 1)));
            list->DrawIndexed(batch.IndexCount, batch.InstanceCount, batch.FirstIndex, 0,
                              batch.FirstInstance);
            instanceCount += batch.InstanceCount;
        }
        m_TransparentDrawCallCount = static_cast<uint32_t>(batches.size());
        m_TransparentBatchCount = static_cast<uint32_t>(batches.size());
        m_TransparentInstanceCount = instanceCount;

        list->EndRendering();

        list->End();
    }

    void RecordCompositeCommandBuffer(const Rhi::AcquiredImage& image)
    {
        FrameData& frame = m_Frames[m_FrameIndex];
        Rhi::ICommandList* list = &BeginRecording(frame.CompositeCommands);

        const std::array openingBarriers{
            Rhi::BarrierPresets::RenderTargetToShaderResource().On(frame.OpaqueTexture.GetHandle()),
            Rhi::BarrierPresets::RenderTargetToShaderResource().On(frame.AccumTexture.GetHandle()),
            Rhi::BarrierPresets::RenderTargetToShaderResource().On(
                frame.RevealageTexture.GetHandle())};
        m_MainThreadBarrierCounts += list->Barrier(openingBarriers);

        const std::array renderTargets{Rhi::RenderTarget{
            .View = image.View, .Load = Rhi::LoadOp::Clear, .ClearColor = {0.f, 0.f, 0.f, 1.f}}};

        list->BeginRendering(
            Rhi::RenderingDesc{.RenderArea = WholeTarget(), .RenderTargets = renderTargets});
        list->SetPipeline(m_CompositePipeline.Get());

        list->SetViewport(FullViewport());
        list->SetScissor(WholeTarget());

        list->SetBindGroup(m_CompositePipelineLayout.Get(), 0u, frame.GlobalBindGroup.Get());
        list->SetBindGroup(m_CompositePipelineLayout.Get(), 1u, frame.CompositeBindGroup.Get());

        list->SetVertexBuffer(0u, m_QuadVertexBuffer.Get());
        list->SetIndexBuffer(m_QuadIndexBuffer.Get(), Rhi::IndexFormat::Uint32);

        constexpr uint32_t QUAD_INDEX_COUNT = 6u;
        list->DrawIndexed(QUAD_INDEX_COUNT, 1u, 0u, 0, 0u);

        list->EndRendering();

        list->End();
    }

    void RecordImGui(const Rhi::AcquiredImage& image)
    {
        Rhi::ICommandList* list = &BeginRecording(m_Frames[m_FrameIndex].ImGuiCommands);

        // ImGui draws over the composited frame with loadOp eLoad, so the
        // composite pass's writes have to be visible to this pass's load.
        m_MainThreadBarrierCounts +=
            list->Barrier(Rhi::BarrierPresets::PreserveRenderTarget().On(image.Texture));

        const std::array renderTargets{Rhi::RenderTarget{.View = image.View}};

        list->BeginRendering(
            Rhi::RenderingDesc{.RenderArea = WholeTarget(), .RenderTargets = renderTargets});

        // The pass itself still records: its barrier and its render pass are what
        // a frame costs whether or not the panel is drawn, so suppressing the
        // panel alone leaves every counter in the run report untouched.
        if (m_bCursorVisible && !m_Spec.bNoUi)
            m_pUiBackend->Render(*list);

        list->EndRendering();
        list->End();
    }

    void RecordSwapImageToDrawLayout(const Rhi::AcquiredImage& image)
    {
        Rhi::ICommandList* list = &BeginRecording(m_Frames[m_FrameIndex].DrawLayoutCommands);
        m_MainThreadBarrierCounts +=
            list->Barrier(Rhi::BarrierPresets::AcquiredImageToRenderTarget().On(image.Texture));
        list->End();
    }

    void RecordSwapImageToFinalLayout(const Rhi::AcquiredImage& image, bool captureScreenshot)
    {
        Rhi::ICommandList* list = &BeginRecording(m_Frames[m_FrameIndex].FinalLayoutCommands);

        const Rhi::TextureHandle swapTexture = image.Texture;

        // Undefined means the target requires no particular layout, and then the
        // frame ends with no closing barrier at all. Only a target with a
        // presentation engine answers otherwise; the command buffer still goes
        // out, empty, because the submit takes a fixed array of seven and the
        // ImGui one is already empty whenever the panel is hidden.
        const Rhi::TextureLayout finalLayout = m_PresentTarget->GetRequiredFinalLayout();
        const bool bNeedsFinalTransition = finalLayout != Rhi::TextureLayout::Undefined;

        if (captureScreenshot)
        {
            // Copy out the composited frame while it is still safely between
            // acquire and present (see RenderTargetToCopySrc()).
            m_MainThreadBarrierCounts +=
                list->Barrier(Rhi::BarrierPresets::RenderTargetToCopySrc().On(swapTexture));

            list->CopyTextureToBuffer(
                swapTexture, m_ScreenshotStagingBuffer.Get(),
                Rhi::BufferTextureCopyRegion{
                    .Extent = {SwapchainExtent().width, SwapchainExtent().height, 1u}});

            if (bNeedsFinalTransition)
            {
                m_MainThreadBarrierCounts +=
                    list->Barrier(Rhi::BarrierPresets::CopySrcToFinal(finalLayout).On(swapTexture));
            }
        }
        else if (bNeedsFinalTransition)
        {
            m_MainThreadBarrierCounts += list->Barrier(
                Rhi::BarrierPresets::RenderTargetToFinal(finalLayout).On(swapTexture));
        }

        list->End();
    }

    void CreateSyncObjects()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "CreateSyncObjects()");

        // The semaphores ordering acquire and present belong to the present
        // target: they have to be rebuilt in lockstep with the images they order
        // access to, and only the object owning the images knows when that is.
        //
        // One fence rather than one per frame in flight. A neutral fence is a
        // monotonic counter, so a single counter and a value per slot says
        // everything a fence each would: the slot records the value its last
        // submission signals and waits for exactly that.
        m_FrameFence = Rhi::UniqueHandle<Rhi::FenceHandle>(
            *m_RhiDevice, m_RhiDevice->CreateFence(Rhi::FenceDesc{.DebugName = "Frame Fence"}));
    }

    void RecreateSwapchainAndRenderImages()
    {
        // Read here rather than beside its use at the end of the function: the
        // rebuild below is what changes it.
        const uint32_t previousImageCount = m_PresentTarget->GetImageCount();

        // A target that refuses to rebuild is the state a minimised window
        // leaves the surface in. Alt-tabbing out of exclusive fullscreen is the
        // common way to reach it: SDL minimises the window itself on focus loss
        // there, so that the desktop video mode comes back.
        //
        // Deferring rather than blocking keeps the event loop running, so the
        // window can be restored — or the application closed — while it lasts.
        // The old target is left intact, and DrawFrame skips frames until this
        // succeeds.
        const Extent2D framebufferExtent = m_Platform.GetFramebufferExtent();
        if (!m_PresentTarget->Recreate({framebufferExtent.Width, framebufferExtent.Height}))
        {
            if (!m_bSwapchainOutOfDate)
                LogMsg(LogSeverity::Info, LogRenderer,
                       "Present target cannot be rebuilt yet; deferring recreation.");

            m_bSwapchainOutOfDate = true;
            return;
        }

        // Logged after the fact because the attempt above is what can fail, and
        // a minimised window would otherwise announce a recreation every frame.
        LogMsg(LogSeverity::Info, LogRenderer, "Recreating swapchain and render images...");

        m_bSwapchainOutOfDate = false;

        // Recreate() waited for work in flight before invalidating the images,
        // so the resources rebuilt below are no longer in use either.
        m_DepthFormat = Rhi::Format::Undefined;

        CreateDepthResources();
        CreateRenderTargets();

        m_CloudSystem->Resize(SwapchainExtent().width, SwapchainExtent().height);
        RecreateFrameBindGroups();

        m_Camera->SetProjection(m_Camera->GetFOV(),
                                static_cast<float>(SwapchainExtent().width) /
                                    static_cast<float>(SwapchainExtent().height),
                                m_Camera->GetNearPlane(), m_Camera->GetFarPlane());

        // A recreate may hand back a different format or a different number of
        // images than the last one — entering or leaving fullscreen is the usual
        // cause, because the driver switches between composited and direct
        // presentation. Nothing this class owns is sized to that count (every
        // per-frame array is as long as the frames in flight, and the per-image
        // semaphores belong to the target), but a UI backend caches both at
        // init, so it is the one thing that has to be told.
        const uint32_t imageCount = m_PresentTarget->GetImageCount();
        if (imageCount != previousImageCount)
        {
            LogMsg(LogSeverity::Info, LogRenderer, "Swapchain image count changed: {} -> {}",
                   previousImageCount, imageCount);
        }

        m_pUiBackend->OnTargetRecreated(imageCount, m_PresentTarget->GetFormat());
    }

    void CreateBindGroupLayouts()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "CreateBindGroupLayouts()");

        const auto make =
            [&](std::span<const Rhi::BindGroupLayoutBinding> bindings, const char* name)
        {
            return Rhi::UniqueHandle<Rhi::BindGroupLayoutHandle>(
                *m_RhiDevice, m_RhiDevice->CreateBindGroupLayout(Rhi::BindGroupLayoutDesc{
                                  .Bindings = bindings, .DebugName = name}));
        };

        m_GlobalLayout = make(EngineBindGroups::kGlobal, "Global Layout");
        m_CompositeLayout = make(EngineBindGroups::kComposite, "Composite Layout");
        m_DepthLayout = make(EngineBindGroups::kDepth, "Depth Layout");
    }

    /**
     * The global group never changes: the buffer it names is created once per
     * frame in flight and only its contents are rewritten. So it is built here
     * and never rebuilt, unlike the two below.
     */
    void CreateGlobalBindGroups()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "CreateGlobalBindGroups()");

        for (size_t i = 0; i < m_Config.FramesInFlight; i++)
        {
            FrameData& frame = m_Frames[i];
            const std::array bindings{Rhi::BindGroupBinding{.Slot = 0u,
                                                            .Type = Rhi::BindingType::UniformBuffer,
                                                            .Buffer = frame.GlobalBuffer.Get()}};

            frame.GlobalBindGroup = Rhi::UniqueHandle<Rhi::BindGroupHandle>(
                *m_RhiDevice, m_RhiDevice->CreateBindGroup(Rhi::BindGroupDesc{
                                  .Layout = m_GlobalLayout.Get(),
                                  .Bindings = bindings,
                                  .DebugName = std::format("Global Bind Group Frame {}", i)}));
        }
    }

    /**
     * Rebuilds the two groups whose contents are render targets, which change
     * whenever those targets are recreated.
     *
     * Recreated rather than rewritten, because a bind group is immutable (RHI
     * plan D20). That is safe here for the reason it is safe anywhere in this
     * renderer: the only caller besides startup is the resize path, which has
     * already waited for the device to go idle, so nothing in flight still names
     * the groups being replaced.
     */
    void RecreateFrameBindGroups()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "RecreateFrameBindGroups()");

        for (size_t i = 0; i < m_Config.FramesInFlight; i++)
        {
            FrameData& frame = m_Frames[i];

            const std::array compositeBindings{
                Rhi::BindGroupBinding{.Slot = 0u,
                                      .Type = Rhi::BindingType::Texture,
                                      .View = frame.OpaqueTexture.GetView()},
                Rhi::BindGroupBinding{.Slot = 1u,
                                      .Type = Rhi::BindingType::Texture,
                                      .View = frame.AccumTexture.GetView()},
                Rhi::BindGroupBinding{.Slot = 2u,
                                      .Type = Rhi::BindingType::Texture,
                                      .View = frame.RevealageTexture.GetView()},
                Rhi::BindGroupBinding{.Slot = 3u,
                                      .Type = Rhi::BindingType::Texture,
                                      .View =
                                          m_CloudSystem->GetOutputView(static_cast<uint8_t>(i))},
                Rhi::BindGroupBinding{.Slot = 4u,
                                      .Type = Rhi::BindingType::Sampler,
                                      .Sampler = m_TextureSampler.Get()}};

            frame.CompositeBindGroup = Rhi::UniqueHandle<Rhi::BindGroupHandle>(
                *m_RhiDevice, m_RhiDevice->CreateBindGroup(Rhi::BindGroupDesc{
                                  .Layout = m_CompositeLayout.Get(),
                                  .Bindings = compositeBindings,
                                  .DebugName = std::format("Composite Bind Group Frame {}", i)}));

            const std::array depthBindings{
                Rhi::BindGroupBinding{.Slot = 0u,
                                      .Type = Rhi::BindingType::Texture,
                                      .View = frame.DepthTexture.GetView()}};

            frame.DepthBindGroup = Rhi::UniqueHandle<Rhi::BindGroupHandle>(
                *m_RhiDevice, m_RhiDevice->CreateBindGroup(Rhi::BindGroupDesc{
                                  .Layout = m_DepthLayout.Get(),
                                  .Bindings = depthBindings,
                                  .DebugName = std::format("Depth Bind Group Frame {}", i)}));
        }
    }

    void CreateGlobalBuffers()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "CreateGlobalBuffers()");

        vk::DeviceSize size = sizeof(GlobalBuffer);
        if (size % 16 != 0)
            throw std::runtime_error(
                std::format("Buffer must be 16 byte aligned! Size is {}", size));

        for (size_t i = 0; i < m_Config.FramesInFlight; i++)
        {
            m_Frames[i].GlobalBuffer = Rhi::UniqueHandle<Rhi::BufferHandle>(
                *m_RhiDevice, m_RhiDevice->CreateBuffer(Rhi::BufferDesc{
                                  .Size = size,
                                  .Usage = Rhi::BufferUsage::Uniform,
                                  .Access = Rhi::MemoryAccess::CpuToGpu,
                                  .DebugName = std::format("Global Buffer Frame {}", i)}));
        }

        m_GlobalBuffer.SkyColor = m_Config.SkyColor;
    }

    void UpdateGlobalBuffer(uint32_t frameIndex)
    {
        m_GlobalBuffer.Time = m_RunTime;
        m_GlobalBuffer.CamData.Pos = m_Camera->GetPosition();
        glm::mat4 view = m_Camera->GetViewMatrix();
        m_GlobalBuffer.CamData.View = glm::transpose(view);
        glm::mat4 proj = m_Camera->GetProjMatrix();
        // GLM was designed for OpenGL, which has its Y coordinate in clip
        // space inverted. Compensate for this by scaling here.
        //
        // Driven by the device capability rather than by the build, because
        // whether this is needed is a property of the graphics API: Vulkan wants
        // it, D3D12 does not. This is the only site permitted to apply it.
        if (m_RhiDevice->GetCaps().bFlipClipSpaceY)
            proj[1][1] *= -1.f;
        m_GlobalBuffer.CamData.Proj = glm::transpose(proj);
        m_GlobalBuffer.CamData.NearPlane = m_Camera->GetNearPlane();
        m_GlobalBuffer.CamData.FarPlane = m_Camera->GetFarPlane();

        m_GlobalBuffer.CamData.InvViewProj =
            glm::inverse(glm::transpose(m_GlobalBuffer.CamData.Proj) * view);

        uint32_t& pointLightCount = m_GlobalBuffer.Lights.PointLightCount;
        for (pointLightCount = 0u;
             pointLightCount < std::min(static_cast<uint32_t>(m_SceneGraph->PointLights.size()),
                                        static_cast<uint32_t>(MAX_POINT_LIGHTS));
             pointLightCount++)
        {
            m_GlobalBuffer.Lights.PointLights[pointLightCount] =
                m_SceneGraph->PointLights[pointLightCount]->GetData();
        }

        uint32_t& dirLightCount = m_GlobalBuffer.Lights.DirLightCount;
        for (dirLightCount = 0u;
             dirLightCount < std::min(static_cast<uint32_t>(m_SceneGraph->DirLights.size()),
                                      static_cast<uint32_t>(MAX_DIR_LIGHTS));
             dirLightCount++)
        {
            m_GlobalBuffer.Lights.DirLights[dirLightCount] =
                m_SceneGraph->DirLights[dirLightCount]->GetData();
        }

        if (m_SceneGraph->PointLights.size() > MAX_POINT_LIGHTS ||
            m_SceneGraph->DirLights.size() > MAX_DIR_LIGHTS)
        {
            const uint32_t pointLightTotal =
                static_cast<uint32_t>(m_SceneGraph->PointLights.size());
            const uint32_t dirLightTotal = static_cast<uint32_t>(m_SceneGraph->DirLights.size());
            LogMsg(LogSeverity::Warning, LogRenderer,
                   "Scene defines more lights than the shader supports; excess lights are "
                   "dropped. PointLights: {} (max {}), DirLights: {} (max {})",
                   pointLightTotal, MAX_POINT_LIGHTS, dirLightTotal, MAX_DIR_LIGHTS);
        }

        memcpy(m_RhiDevice->GetMappedData(m_Frames[frameIndex].GlobalBuffer.Get()), &m_GlobalBuffer,
               sizeof(m_GlobalBuffer));
    }

    void CreateTextureSampler()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "CreateTextureSampler()");

        // MaxAnisotropy left at 0 asks for the device maximum, which is what this
        // used to read off the physical device's limits itself.
        m_TextureSampler = Rhi::UniqueHandle<Rhi::SamplerHandle>(
            *m_RhiDevice, m_RhiDevice->CreateSampler(Rhi::SamplerDesc{
                              .bAnisotropyEnable = true, .DebugName = "Texture Sampler"}));
    }

    void CreateDepthResources()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "CreateDepthResources()");

        m_DepthFormat = FindDepthFormat();
        for (size_t i = 0; i < m_Config.FramesInFlight; i++)
        {
            m_Frames[i].DepthTexture = Texture(
                *m_RhiDevice,
                Rhi::TextureDesc{.Format = m_DepthFormat,
                                 .Extent = {SwapchainExtent().width, SwapchainExtent().height, 1u},
                                 .Usage = Rhi::TextureUsage::DepthStencilAttachment |
                                          Rhi::TextureUsage::Sampled,
                                 .DebugName = std::format("Frame_{} Depth Image", i)},
                Rhi::TextureViewDimension::Texture2D);
        }
    }

    Rhi::Format FindSupportedFormat(std::span<const Rhi::Format> candidates, vk::ImageTiling tiling,
                                    vk::FormatFeatureFlags features)
    {
        for (const Rhi::Format format : candidates)
        {
            const vk::FormatProperties properties =
                m_PhysicalDevice.getFormatProperties(Rhi::Vulkan::GetNativeFormat(format));

            if (tiling == vk::ImageTiling::eLinear &&
                (properties.linearTilingFeatures & features) == features)
                return format;
            if (tiling == vk::ImageTiling::eOptimal &&
                (properties.optimalTilingFeatures & features) == features)
                return format;
        }
        throw std::runtime_error("Failed to find a supported format!");
    }

    Rhi::Format FindDepthFormat()
    {
        // D16UnormS8Uint used to be a fourth candidate here and is deliberately
        // gone: it has no DXGI equivalent, so Rhi::Format cannot carry it. That
        // costs nothing, because it was unreachable. The specification's
        // mandatory format table requires VK_FORMAT_FEATURE_DEPTH_STENCIL_
        // ATTACHMENT_BIT to be "supported for at least one of
        // VK_FORMAT_D24_UNORM_S8_UINT and VK_FORMAT_D32_SFLOAT_S8_UINT"
        // (Vulkan 1.4, "Mandatory Format Support: Depth/Stencil"), and both are
        // above it in this list — so no conformant device could ever fall
        // through to a fourth candidate.
        static constexpr std::array candidates{Rhi::Format::D32Float, Rhi::Format::D32FloatS8Uint,
                                               Rhi::Format::D24UnormS8Uint};
        return FindSupportedFormat(candidates, vk::ImageTiling::eOptimal,
                                   vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    }

    void CreateInstanceBuffers(uint32_t instanceCapacity)
    {
        LogMsg(LogSeverity::Info, LogRenderer, "CreateInstanceBuffers()");

        // TODO: allocating memory 3 times, can probably allocate once and
        // store offsets Can do the same with uniform buffer.
        vk::DeviceSize size = sizeof(InstanceData) * instanceCapacity;
        for (uint32_t i = 0; i < m_Config.FramesInFlight; i++)
        {
            m_Frames[i].InstanceBuffer = Rhi::UniqueHandle<Rhi::BufferHandle>(
                *m_RhiDevice, m_RhiDevice->CreateBuffer(Rhi::BufferDesc{
                                  .Size = size,
                                  .Usage = Rhi::BufferUsage::Vertex,
                                  .Access = Rhi::MemoryAccess::CpuToGpu,
                                  .DebugName = std::format("Instance Buffer Frame {}", i)}));
        }

        m_InstanceCapacity = instanceCapacity;
    }

    /**
     * Every frame's buffer is replaced at once, not just the one being filled:
     * growing them one at a time would leave the other frame short by exactly
     * the same amount and grow again on the very next frame, for two device
     * waits instead of one.
     *
     * The wait is what makes the replacement legal. These are vertex buffers
     * that frames still in flight have bound, and destroying one while a
     * submitted command buffer still refers to it is invalid
     * (VUID-vkDestroyBuffer-buffer-00922); the current frame's fence has been
     * waited on by this point, but nothing covers the others. Growth is rare
     * enough that a device-wide wait beats tracking per-buffer lifetimes.
     */
    void GrowInstanceBuffers(uint32_t neededInstances)
    {
        const uint32_t newCapacity = std::max(neededInstances, m_InstanceCapacity * 2u);

        m_RhiDevice->WaitIdle();
        CreateInstanceBuffers(newCapacity);

        LogMsg(LogSeverity::Info, LogRenderer, "Instance buffer grown to {} instances.",
               newCapacity);
    }

    void UpdateInstanceBuffer(uint32_t frameIndex)
    {
        const std::vector<InstanceData>& instanceDatas = m_ModelManager.GetInstanceDatas();
        if (instanceDatas.empty())
            return;

        if (instanceDatas.size() > m_InstanceCapacity)
            GrowInstanceBuffers(static_cast<uint32_t>(instanceDatas.size()));

        memcpy(m_RhiDevice->GetMappedData(m_Frames[frameIndex].InstanceBuffer.Get()),
               instanceDatas.data(), sizeof(InstanceData) * instanceDatas.size());
    }

    void CreateRenderTargets()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "CreateRenderTargets()");

        const auto makeTarget = [this](Rhi::Format format, const std::string& name)
        {
            return Texture(
                *m_RhiDevice,
                Rhi::TextureDesc{.Format = format,
                                 .Extent = {SwapchainExtent().width, SwapchainExtent().height, 1u},
                                 .Usage = Rhi::TextureUsage::ColorAttachment |
                                          Rhi::TextureUsage::Sampled,
                                 .DebugName = name},
                Rhi::TextureViewDimension::Texture2D);
        };

        for (size_t i = 0; i < m_Config.FramesInFlight; i++)
        {
            m_Frames[i].OpaqueTexture =
                makeTarget(m_OpaqueImageFormat, std::format("Frame_{} Opaque Image", i));
            m_Frames[i].AccumTexture =
                makeTarget(m_AccumImageFormat, std::format("Frame_{} Accum Image", i));
            m_Frames[i].RevealageTexture =
                makeTarget(m_RevealageImageFormat, std::format("Frame_{} Revealage Image", i));
        }
    }

    void CreateQuadBuffers()
    {
        LogMsg(LogSeverity::Info, LogRenderer, "CreateQuadBuffers()");

        std::array<QuadVertex, 4> vertices = {{{.Pos = {-1.f, -1.f}, .TexCoord{0.f, 0.f}},
                                               {.Pos = {-1.f, 1.f}, .TexCoord{0.f, 1.f}},
                                               {.Pos = {1.f, 1.f}, .TexCoord{1.f, 1.f}},
                                               {.Pos = {1.f, -1.f}, .TexCoord{1.f, 0.f}}}};

        assert(vertices.size() == 4);

        std::array<uint32_t, 6> indices = {0, 1, 2, 0, 2, 3};

        const auto createUploaded =
            [this](Rhi::BufferUsage usage, auto& contents, const char* debugName)
        {
            Rhi::UniqueHandle<Rhi::BufferHandle> buffer(
                *m_RhiDevice, m_RhiDevice->CreateBuffer(
                                  Rhi::BufferDesc{.Size = std::span(contents).size_bytes(),
                                                  .Usage = usage | Rhi::BufferUsage::CopyDst,
                                                  .Access = Rhi::MemoryAccess::GpuOnly,
                                                  .DebugName = debugName}));

            m_UploadContext->UploadBuffer(buffer.Get(), 0u, std::as_bytes(std::span(contents)));
            return buffer;
        };

        m_QuadVertexBuffer =
            createUploaded(Rhi::BufferUsage::Vertex, vertices, "Quad Vertex Buffer");
        m_QuadIndexBuffer = createUploaded(Rhi::BufferUsage::Index, indices, "Quad Index Buffer");

        // Not routed through ResourceManager, so nothing else is going to flush
        // these.
        m_UploadContext->Flush();
    }

private:
    /**
     * Declared first because the device's creation reads them, and members are
     * initialised in declaration order.
     */
    IPlatform& m_Platform;
    const Paths& m_Paths;

    /** The UI stack, built by the app and borrowed for the run. */
    IUiBackend* m_pUiBackend;
    RunSpec m_Spec;
    EngineConfig m_Config;
    IJobSystem& m_JobSystem;

    /**
     * Owned by main() rather than by the device, because the counts are read for
     * --strict-validation after the device has been destroyed.
     */
    Rhi::Diagnostics& m_Diagnostics;

    /**
     * Owns the instance, debug messenger, surface, physical and logical devices,
     * graphics queue and the VMA allocator. Declared ahead of every GPU resource
     * below so that it is destroyed after all of them — the ordering the old
     * hand-arranged member list was maintaining by hand.
     */
    std::unique_ptr<Rhi::IDevice> m_RhiDevice;

    /**
     * After the device and before every resource that is loaded through it: the
     * context holds staging buffers of its own, and destroying it after the
     * device would release them into nothing.
     */
    std::unique_ptr<Rhi::IUploadContext> m_UploadContext;
    std::unique_ptr<Rhi::IPipelineCache> m_PipelineCache;

    /**
     * Declared before the registry it is handed to, so that it outlives the
     * models whose materials hold descriptor sets from its allocator.
     */
    std::unique_ptr<MaterialFactory> m_MaterialFactory;

    /**
     * Every asset the run loads comes from here. Declared after the upload
     * context so that it is destroyed before it: the loaders hold references to
     * both it and the device.
     */
    std::unique_ptr<AssetRegistry> m_Assets;

    /**
     * Borrowed from m_RhiDevice, which outlives them. References rather than
     * copies so that the ~100 call sites still read as they did, and so that
     * there is exactly one owner. Each of these disappears as the corresponding
     * resource type moves behind IDevice.
     */
    vk::raii::PhysicalDevice& m_PhysicalDevice;

    /**
     * The frame loop's fence, and the last value handed to it. Monotonic, so it
     * is never reset -- each frame signals the next value and each slot
     * remembers its own.
     */
    Rhi::UniqueHandle<Rhi::FenceHandle> m_FrameFence;
    uint64_t m_FrameSubmitCount = 0u;

    std::unique_ptr<Rhi::IPresentTarget> m_PresentTarget;
    Rhi::UniqueHandle<Rhi::PipelineLayoutHandle> m_OpaquePipelineLayout;
    Rhi::UniqueHandle<Rhi::PipelineLayoutHandle> m_TransparentPipelineLayout;
    Rhi::UniqueHandle<Rhi::PipelineLayoutHandle> m_CompositePipelineLayout;
    Rhi::UniqueHandle<Rhi::GraphicsPipelineHandle> m_OpaquePipeline;
    Rhi::UniqueHandle<Rhi::GraphicsPipelineHandle> m_TransparentPipeline;
    Rhi::UniqueHandle<Rhi::GraphicsPipelineHandle> m_CompositePipeline;

    /**
     * Kept for the run: a pipeline may be rebuilt when the swapchain format
     * changes, and rebuilding it needs the modules it was made from.
     */
    std::vector<Rhi::UniqueHandle<Rhi::ShaderModuleHandle>> m_ShaderModules;
    GlobalBuffer m_GlobalBuffer = {};
    Rhi::UniqueHandle<Rhi::SamplerHandle> m_TextureSampler;
    Rhi::UniqueHandle<Rhi::BindGroupLayoutHandle> m_GlobalLayout;
    Rhi::UniqueHandle<Rhi::BindGroupLayoutHandle> m_CompositeLayout;
    Rhi::UniqueHandle<Rhi::BindGroupLayoutHandle> m_DepthLayout;
    Rhi::Format m_DepthFormat = Rhi::Format::Undefined;
    static constexpr Rhi::Format m_OpaqueImageFormat = Rhi::Format::RGBA16Float;
    static constexpr Rhi::Format m_AccumImageFormat = Rhi::Format::RGBA16Float;
    static constexpr Rhi::Format m_RevealageImageFormat = Rhi::Format::R8Unorm;
    Rhi::UniqueHandle<Rhi::BufferHandle> m_QuadVertexBuffer;
    Rhi::UniqueHandle<Rhi::BufferHandle> m_QuadIndexBuffer;

    /**
     * Set when the present target could not be rebuilt yet. The frame loop
     * retries and draws nothing until it clears.
     */
    bool m_bSwapchainOutOfDate = false;

    std::vector<FrameData> m_Frames;

    /**
     * Instances every frame's buffer has room for. A starting size, not a
     * ceiling — see GrowInstanceBuffers.
     */
    uint32_t m_InstanceCapacity = 0u;

    std::unique_ptr<SceneGraph> m_SceneGraph = nullptr;

    /**
     * Holds no models of its own — it is handed the scene each frame — so it
     * needs no lifetime beyond this class's.
     */
    ModelManager m_ModelManager;

    std::unique_ptr<Camera> m_Camera = nullptr;
    std::shared_ptr<Cubemap> m_Skybox = nullptr;
    std::unique_ptr<CloudSystem> m_CloudSystem = nullptr;

    uint32_t m_FrameIndex = 0;
    bool m_bIsFocused = true;

    /** Set by a scripted capture request, cleared by the frame that honours it. */
    bool m_bCaptureThisFrame = false;
    bool m_bCursorVisible = true;
    /**
     * Set in main() before anything else runs, so startupMs covers argument
     * parsing and the window as well as device creation and the scene's
     * uploads — the two paths a windowed and a headless run differ on.
     */
    std::chrono::steady_clock::time_point m_ProcessStart;

    /**
     * The simulation's clock, chosen by --fixed-dt. Owned rather than injected:
     * which one it is follows from the run description the engine already has,
     * and nothing else in the process has an opinion about it.
     */
    std::unique_ptr<Core::IClock> m_Clock;
    uint64_t m_FrameCounter = 0;
    float m_RunTime = 0.f;
    float m_DeltaTime = 0.f;
    float m_DisplayFrameTime = 0.f;
    float m_DisplayFPS = 0.f;
    bool m_bShutdown = false;

    /**
     * Barriers recorded for the current frame, split by the thread that records
     * them: the opaque and transparent passes are recorded on job threads, so
     * each owns its own counters rather than sharing one set. Everything else
     * is recorded on the main thread and shares the third.
     */
    Rhi::BarrierCounts m_OpaqueBarrierCounts;
    Rhi::BarrierCounts m_TransparentBarrierCounts;
    Rhi::BarrierCounts m_MainThreadBarrierCounts;
    Rhi::BarrierCounts m_LoggedBarrierCounts;

    /** Used in WriteReport() */
    uint32_t m_OpaqueDrawCallCount = 0;
    uint32_t m_OpaqueBatchCount = 0;
    uint32_t m_OpaqueInstanceCount = 0;
    uint32_t m_TransparentDrawCallCount = 0;
    uint32_t m_TransparentBatchCount = 0;
    uint32_t m_TransparentInstanceCount = 0;
    /**
     * Wall clock per frame, and the same minus everything the frame spent
     * blocked. Frame 0 is held separately rather than mixed in: it pays for
     * first use of every pipeline and the first acquire, so averaging it with
     * the rest describes neither.
     */
    std::vector<float> m_FrameMs;
    std::vector<float> m_CpuMs;
    float m_FirstFrameMs = 0.f;
    float m_FirstFrameCpuMs = 0.f;
    float m_StartupMs = 0.f;

    /** Time this frame spent blocked, accumulated across its blocking calls. */
    float m_FrameWaitMs = 0.f;
    Rhi::UniqueHandle<Rhi::BufferHandle> m_ScreenshotStagingBuffer;
    bool m_bScreenshotBufferReady = false;
};

// Anchored here rather than defaulted in their headers, so each vtable has one
// home rather than one per translation unit that sees the class.
IEngine::~IEngine() = default;
IUiBackend::~IUiBackend() = default;

std::unique_ptr<IEngine> CreateEngine(const EngineDesc& desc)
{
    return std::make_unique<Engine>(*desc.pPlatform, *desc.pPaths, desc.pUiBackend, desc.Spec,
                                    desc.Config, *desc.pJobSystem, *desc.pDiagnostics,
                                    desc.ProcessStart);
}

void RequestStop()
{
    g_bShouldClose = true;
}

bool StopRequested()
{
    return g_bShouldClose;
}

} // namespace Hikari::Engine
