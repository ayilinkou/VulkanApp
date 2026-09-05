#include <editor/VulkanUiBackend.h>

#include <algorithm>

#include <SDL3/SDL.h>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"

#include <core/Log.h>

#include <rhi/vulkan/VulkanNative.h>

namespace Hikari::Editor
{

namespace
{
constexpr Core::LogCategory LogUi("UI Backend");
} // namespace

VulkanUiBackend::~VulkanUiBackend()
{
    Shutdown();
}

void VulkanUiBackend::Init(const Engine::UiBackendDesc& desc)
{
    Core::LogMsg(Core::LogSeverity::Info, LogUi, "Init()");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    const vk::Format targetFormat = Rhi::Vulkan::GetNativeFormat(desc.TargetFormat);
    vk::PipelineRenderingCreateInfo pipelineRenderingInfo = {
        .colorAttachmentCount = 1u, .pColorAttachmentFormats = &targetFormat};

    // ImGui's backend takes raw handles by value, and there is no neutral shape
    // for that short of wrapping ImGui itself — which is why this file is the
    // one that reaches for them rather than the engine.
    const Rhi::Vulkan::NativeDevice native = Rhi::Vulkan::GetNative(*desc.pDevice);

    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.ApiVersion = native.ApiVersion;
    initInfo.Instance = native.Instance;
    initInfo.PhysicalDevice = native.PhysicalDevice;
    initInfo.Device = native.Device;
    initInfo.QueueFamily = native.GraphicsQueueFamily;
    initInfo.Queue = native.GraphicsQueue;
    initInfo.DescriptorPool = VK_NULL_HANDLE;
    initInfo.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE;

    // The backend asserts MinImageCount >= 2 and ImageCount >= MinImageCount
    // (imgui_impl_vulkan.cpp:1298-1299), and an offscreen target makes one image
    // per frame in flight — so a run with one of those would trip both.
    //
    // ImageCount sizes only ImGui's own vertex/index ring and its unused-texture
    // delay, so over-provisioning costs a little memory and nothing else. Under-
    // provisioning is the hazard: the ring is reused every ImageCount frames,
    // and a ring shorter than the frames in flight would be overwritten while an
    // earlier frame was still reading it.
    initInfo.MinImageCount = 2u;
    initInfo.MinAllocationSize = 1024 * 1024;
    initInfo.ImageCount = std::max(2u, desc.RingSize);
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineCache = Rhi::Vulkan::GetNativePipelineCache(*desc.pPipelineCache);
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = pipelineRenderingInfo;
    initInfo.Allocator = nullptr;
    initInfo.CheckVkResultFn = nullptr;

    // The platform backend is the only half that needs a window. Skipping it
    // leaves ImGui with no platform backend at all, which is a supported
    // configuration: what it supplies is io.DisplaySize and io.DeltaTime, and a
    // windowless caller sets both by hand instead.
    m_bHasPlatformBackend = desc.pNativeWindowHandle != nullptr;
    if (m_bHasPlatformBackend)
        ImGui_ImplSDL3_InitForVulkan(static_cast<SDL_Window*>(desc.pNativeWindowHandle));

    ImGui_ImplVulkan_Init(&initInfo);
    m_bInitialised = true;
}

void VulkanUiBackend::Shutdown()
{
    if (!m_bInitialised)
        return;

    Core::LogMsg(Core::LogSeverity::Info, LogUi, "Shutdown()");

    ImGui_ImplVulkan_Shutdown();
    if (m_bHasPlatformBackend)
        ImGui_ImplSDL3_Shutdown();

    ImGui::DestroyContext();
    m_bInitialised = false;
}

void VulkanUiBackend::NewFrame()
{
    ImGui_ImplVulkan_NewFrame();
    if (m_bHasPlatformBackend)
        ImGui_ImplSDL3_NewFrame();
}

void VulkanUiBackend::Render(Rhi::ICommandList& commandList)
{
    // Null when nothing built a UI frame — an extra frame drawn purely to stage
    // a capture, say. The caller's rendering scope still opens and closes, so
    // the pass costs what it always does; there is simply nothing to draw into
    // it.
    ImDrawData* pDrawData = ImGui::GetDrawData();
    if (pDrawData == nullptr)
        return;

    ImGui_ImplVulkan_RenderDrawData(pDrawData, Rhi::Vulkan::GetNative(commandList));
}

void VulkanUiBackend::OnTargetRecreated(uint32_t imageCount, Rhi::Format targetFormat)
{
    // The pipeline is rebuilt because a recreate may have changed the format;
    // the count is pushed because ImGui's backend cached it at init.
    const vk::Format nativeFormat = Rhi::Vulkan::GetNativeFormat(targetFormat);
    vk::PipelineRenderingCreateInfo renderingInfo = {.colorAttachmentCount = 1u,
                                                     .pColorAttachmentFormats = &nativeFormat};

    ImGui_ImplVulkan_PipelineInfo pipelineInfo{};
    pipelineInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    pipelineInfo.PipelineRenderingCreateInfo = renderingInfo;
    ImGui_ImplVulkan_CreateMainPipeline(&pipelineInfo);

    // Clamped for the same reason Init clamps: the backend asserts a minimum of
    // two, which a single-image target would break.
    ImGui_ImplVulkan_SetMinImageCount(std::max(2u, imageCount));
}

void VulkanUiBackend::ProcessPlatformEvent(const void* pEvent)
{
    if (m_bHasPlatformBackend)
        ImGui_ImplSDL3_ProcessEvent(static_cast<const SDL_Event*>(pEvent));
}

} // namespace Hikari::Editor
