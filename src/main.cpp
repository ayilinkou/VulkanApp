#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vulkan/vulkan_core.h>

#include "SDL3/SDL.h"
#include "SDL3/SDL_events.h"
#include "SDL3/SDL_video.h"
#include "SDL3/SDL_vulkan.h"

#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"

#include "glm/glm.hpp"

constexpr uint32_t WIDTH = 1920;
constexpr uint32_t HEIGHT = 1080;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

std::atomic<bool> gbShouldClose = false;

void HandleSIGINT(int)
{
    gbShouldClose = true;
    std::cout << "\n";
}

struct Vertex
{
    glm::vec2 Pos;
    glm::vec3 Color;

    static vk::VertexInputBindingDescription GetBindingDescription()
    {
        return {.binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = vk::VertexInputRate::eVertex};
    }

    static std::array<vk::VertexInputAttributeDescription, 2>
    GetAttributeDescription()
    {
        return {{{.location = 0,
                  .binding = 0,
                  .format = vk::Format::eR32G32Sfloat,
                  .offset = offsetof(Vertex, Pos)},
                 {.location = 1,
                  .binding = 0,
                  .format = vk::Format::eR32G32B32Sfloat,
                  .offset = offsetof(Vertex, Color)}}};
    }
};

const std::vector<Vertex> vertices = {{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                      {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
                                      {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};

std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool bEnableValidationLayers = false;
#else
constexpr bool bEnableValidationLayers = true;
#endif

static VKAPI_ATTR vk::Bool32 VKAPI_CALL
DebugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
              vk::DebugUtilsMessageTypeFlagsEXT type,
              const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
              void* pUserData)
{
    std::cerr << "validation layer: type " << to_string(type)
              << " msg: " << pCallbackData->pMessage << std::endl;

    return vk::False;
}

class SDLException : public std::runtime_error
{
public:
    SDLException(const std::string& message)
        : std::runtime_error(std::format("{} {}", message, SDL_GetError()))
    {
    }
};

void InitSDL()
{
    if (!SDL_Init(SDL_INIT_VIDEO))
        throw SDLException("Failed to initialise SDL!");

    std::cout << "SDL video driver: " << SDL_GetCurrentVideoDriver() << "\n";

    if (!SDL_Vulkan_LoadLibrary(nullptr))
        throw SDLException("Failed to load Vulkan library!");
}

SDL_Window* CreateSDLWindow()
{
    // hidden to hide the window while initialisation is taking place
    SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                            SDL_WINDOW_HIDDEN | SDL_WINDOW_BORDERLESS;
    SDL_Window* window = SDL_CreateWindow("Vulkan App", WIDTH, HEIGHT, flags);
    if (window == nullptr)
        throw SDLException("Failed to create window!");

    SDL_SetWindowFullscreen(window, true);

    return window;
}

void ShutdownSDL(SDL_Window* pWindow)
{
    if (pWindow)
        SDL_DestroyWindow(pWindow);

    SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
}

static std::vector<char> ReadFile(const std::string filename)
{
    // std::ios::ate starts to read at end of file so that we can get the size
    // of the buffer
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        std::runtime_error("Failed to open file!");

    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();

    return buffer;
}

// Chooses an ideal swapchain format if available, if not picks the first
// one.
vk::SurfaceFormatKHR
ChooseSwapchainFormat(const std::vector<vk::SurfaceFormatKHR>& formats)
{
    if (formats.empty())
        throw std::runtime_error("No surface formats available!");

    const auto formatIt = std::ranges::find_if(
        formats,
        [](const auto& format)
        {
            return format.format == vk::Format::eB8G8R8A8Srgb &&
                   format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });

    return formatIt != formats.end() ? *formatIt : formats[0];
}

// Chooses mailbox presentation mode if available. Falls back to FIFO.
vk::PresentModeKHR
ChoosePresentMode(const std::vector<vk::PresentModeKHR>& modes)
{
    if (modes.empty())
        throw std::runtime_error("No swapchain presentation modes available!");

    const auto modeIt =
        std::ranges::find_if(modes, [](const auto& mode)
                             { return mode == vk::PresentModeKHR::eMailbox; });
    return modeIt != modes.end() ? *modeIt : vk::PresentModeKHR::eFifo;
}

vk::Extent2D
ChooseSwapchainExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                      SDL_Window* window)
{
    // Some window managers allow resolutions which don't match the window. They
    // symbol this with max value of a uint32_t.
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;

    // This has to be used rather than the raw window width and height as high
    // DPI displays might not match screen coordinates and pixels.
    int width, height;
    SDL_GetWindowSizeInPixels(window, &width, &height);

    return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width,
                                 capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height,
                                 capabilities.maxImageExtent.height)};
}

// Tries to get at least 3 images.
uint32_t ChooseSwapMinImageCount(const vk::SurfaceCapabilitiesKHR& capabilities)
{
    uint32_t minCount = std::max(3u, capabilities.minImageCount);

    // maxImageCount == 0 indicates that there is no maximum
    if ((0 < capabilities.maxImageCount) &&
        (capabilities.maxImageCount < minCount))
        minCount = capabilities.maxImageCount;
    return minCount;
}

class App
{
public:
    App() {}
    App(SDL_Window* pWindow) : m_pWindow(pWindow) {}
    ~App() {}

    void Run()
    {
        Init();

        SDL_ShowWindow(m_pWindow);

        while (!gbShouldClose)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            SDL_Event event;
            while (SDL_PollEvent(&event))
            {
                switch (event.type)
                {
                case SDL_EVENT_QUIT:
                    gbShouldClose = true;
                    break;
                case SDL_EVENT_WINDOW_FOCUS_GAINED:
                    m_bIsFocused = true;
                    std::cout << "Focus gained.\n";
                    break;
                case SDL_EVENT_WINDOW_FOCUS_LOST:
                    m_bIsFocused = false;
                    std::cout << "Focus lost.\n";
                    break;
                case SDL_EVENT_WINDOW_RESIZED:
                    RecreateSwapchain();
                    break;
                }
            }

            DrawFrame();
        }

        m_Device.waitIdle();
        Shutdown();
    }

private:
    void Init()
    {
        InitVulkan();
        std::cout << "Init() succeeded.\n";
    }

    void InitVulkan()
    {
        CreateInstance();
        SetupDebugMessenger();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        CreateSwapchain();
        CreateSwapchainImageViews();
        CreateGraphicsPipeline();
        CreateVertexBuffer();
        CreateCommandPool();
        CreateCommandBuffer();
        CreateSyncObjects();
    }

    void Shutdown() {}

    void DrawFrame()
    {
        // Semaphores coordinate GPU to GPU synchronisation, for example
        // ordering work between queues. They get reset automatically after the
        // waiting operation begins.
        //
        // Fences coordinate CPU to GPU synchronisation, for times when
        // the CPU needs to know that the GPU has finished a task. Must be
        // explicitely reset by the host.

        auto fenceResult = m_Device.waitForFences(*m_DrawFences[m_FrameIndex],
                                                  vk::True, UINT64_MAX);
        if (fenceResult != vk::Result::eSuccess)
            throw std::runtime_error("Failed to wait for fence!");

        auto [result, imageIndex] = m_Swapchain.acquireNextImage(
            UINT64_MAX, *m_PresentCompleteSemaphores[m_FrameIndex], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR)
        {
            RecreateSwapchain();
            return;
        }
        else if (result != vk::Result::eSuccess &&
                 result != vk::Result::eSuboptimalKHR)
        {
            assert(result == vk::Result::eTimeout ||
                   result == vk::Result::eNotReady);
            throw std::runtime_error("Failed to acquire next swapchain image!");
        }

        m_Device.resetFences(*m_DrawFences[m_FrameIndex]);
        RecordCommandBuffer(imageIndex);

        vk::PipelineStageFlags waitDestinationStageFlags(
            vk::PipelineStageFlagBits::eColorAttachmentOutput);
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*m_PresentCompleteSemaphores[m_FrameIndex],
            .pWaitDstStageMask = &waitDestinationStageFlags,
            .commandBufferCount = 1,
            .pCommandBuffers = &*m_CommandBuffers[m_FrameIndex],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*m_RenderCompleteSemaphores[imageIndex]};

        m_GraphicsQueue.submit(submitInfo, *m_DrawFences[m_FrameIndex]);

        const vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*m_RenderCompleteSemaphores[imageIndex],
            .swapchainCount = 1,
            .pSwapchains = &*m_Swapchain,
            .pImageIndices = &imageIndex};

        result = m_GraphicsQueue.presentKHR(presentInfo);
        if ((result == vk::Result::eSuboptimalKHR) ||
            (result == vk::Result::eErrorOutOfDateKHR))
        {
            RecreateSwapchain();
        }
        else if (result != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to present image!");
        }

        m_FrameIndex = (m_FrameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void CreateInstance()
    {
        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName = "Vulkan App",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = vk::ApiVersion14};

        // extensions
        Uint32 countInstanceExtensions;
        const char* const* instanceExtensions =
            SDL_Vulkan_GetInstanceExtensions(&countInstanceExtensions);

        if (countInstanceExtensions == 0)
            throw std::runtime_error("No available instance extensions found!");

        std::vector<const char*> requiredExtensions(countInstanceExtensions);
        memcpy(requiredExtensions.data(), instanceExtensions,
               countInstanceExtensions * sizeof(const char*));
        requiredExtensions.push_back(vk::EXTDebugUtilsExtensionName);

        auto extensionProperties =
            m_Context.enumerateInstanceExtensionProperties();

        auto unsupportedExtensionIt = std::ranges::find_if(
            requiredExtensions,
            [&extensionProperties](auto const& requiredExtension)
            {
                return std::ranges::none_of(
                    extensionProperties,
                    [requiredExtension](auto const& extensionProperty)
                    {
                        return strcmp(extensionProperty.extensionName,
                                      requiredExtension) == 0;
                    });
            });

        if (unsupportedExtensionIt != requiredExtensions.end())
            throw std::runtime_error("Required extension not supported: " +
                                     std::string(*unsupportedExtensionIt));

        // layers
        std::vector<const char*> requiredLayers;
        if (bEnableValidationLayers)
        {
            requiredLayers.assign(validationLayers.begin(),
                                  validationLayers.end());
        }

        auto layerProperties = m_Context.enumerateInstanceLayerProperties();

        auto unsupportedLayerIt = std::ranges::find_if(
            requiredLayers,
            [&layerProperties](auto const& requiredLayer)
            {
                return std::ranges::none_of(
                    layerProperties,
                    [requiredLayer](auto const& layerProperty)
                    {
                        return strcmp(layerProperty.layerName, requiredLayer) ==
                               0;
                    });
            });

        if (unsupportedLayerIt != requiredLayers.end())
            throw std::runtime_error("Required layer not supported: " +
                                     std::string(*unsupportedLayerIt));

        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = (uint32_t)requiredLayers.size(),
            .ppEnabledLayerNames = requiredLayers.data(),
            .enabledExtensionCount = (uint32_t)requiredExtensions.size(),
            .ppEnabledExtensionNames = requiredExtensions.data()};

        m_Instance = vk::raii::Instance(m_Context, createInfo);
    }

    void SetupDebugMessenger()
    {
        if (!bEnableValidationLayers)
            return;

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo);
        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);

        vk::DebugUtilsMessengerCreateInfoEXT createInfo{
            .messageSeverity = severityFlags,
            .messageType = messageTypeFlags,
            .pfnUserCallback = &DebugCallback};

        m_DebugMessenger = m_Instance.createDebugUtilsMessengerEXT(createInfo);
    }

    bool IsPhysicalDeviceSuitable(const vk::raii::PhysicalDevice& device)
    {
        auto properties = device.getProperties();

        bool bSupportsVulkan13 = properties.apiVersion >= vk::ApiVersion13;

        auto queueFamilies = device.getQueueFamilyProperties();
        bool bSupportsGraphicsQ = std::ranges::any_of(
            queueFamilies, [](const auto& qfp)
            { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

        std::vector<const char*> requiredExtensions = {
            vk::KHRSwapchainExtensionName};
        auto availableExtensions = device.enumerateDeviceExtensionProperties();
        bool bSupportsAllExtensions = std::ranges::all_of(
            requiredExtensions,
            [&availableExtensions](const auto& requiredExtension)
            {
                return std::ranges::any_of(
                    availableExtensions,
                    [requiredExtension](const auto& availableExtension)
                    {
                        return strcmp(availableExtension.extensionName,
                                      requiredExtension) == 0;
                    });
            });

        auto features = device.getFeatures2<
            vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
        bool bSupportsAllFeatures =
            features.get<vk::PhysicalDeviceVulkan13Features>()
                .dynamicRendering &&
            features.get<vk::PhysicalDeviceVulkan13Features>()
                .synchronization2 &&
            features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
                .extendedDynamicState;

        if (bSupportsVulkan13 && bSupportsGraphicsQ && bSupportsAllExtensions &&
            bSupportsAllFeatures)
            return true;
        return false;
    }

    void CreateSurface()
    {
        VkSurfaceKHR rawSurface;
        if (!SDL_Vulkan_CreateSurface(m_pWindow, *m_Instance, nullptr,
                                      &rawSurface))
            throw SDLException("Failed to create Vulkan surface!");

        m_Surface = vk::raii::SurfaceKHR(m_Instance, rawSurface);
    }

    void PickPhysicalDevice()
    {
        auto devices = m_Instance.enumeratePhysicalDevices();
        const auto deviceIt =
            std::ranges::find_if(devices, [&](const auto& device)
                                 { return IsPhysicalDeviceSuitable(device); });

        if (deviceIt == devices.end())
            std::runtime_error("Failed to find a suitable GPU!");

        m_PhysicalDevice = *deviceIt;
    }

    void CreateLogicalDevice()
    {
        std::vector<vk::QueueFamilyProperties> qfProperties =
            m_PhysicalDevice.getQueueFamilyProperties();

        for (size_t qfpIndex = 0; qfpIndex < qfProperties.size(); qfpIndex++)
        {
            if ((qfProperties[qfpIndex].queueFlags &
                 vk::QueueFlagBits::eGraphics) !=
                    static_cast<vk::QueueFlags>(0) &&
                m_PhysicalDevice.getSurfaceSupportKHR(qfpIndex, m_Surface))

            {
                m_QueueIndex = static_cast<uint32_t>(qfpIndex);
                break;
            }
        }

        if (m_QueueIndex == ~0)
            std::runtime_error(
                "Could not find a queue for graphics and presenting!");

        float queuePriority = 0.5f;
        vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = m_QueueIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority};

        vk::PhysicalDeviceFeatures deviceFeatures;

        vk::StructureChain<vk::PhysicalDeviceFeatures2,
                           vk::PhysicalDeviceVulkan11Features,
                           vk::PhysicalDeviceVulkan13Features,
                           vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
            featureChain = {
                {},
                {.shaderDrawParameters = true},
                {.synchronization2 = true, .dynamicRendering = true},
                {.extendedDynamicState = true}};

        std::vector<const char*> requiredDeviceExtensions = {
            vk::KHRSwapchainExtensionName};

        vk::DeviceCreateInfo deviceCreateInfo{
            .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = (uint32_t)requiredDeviceExtensions.size(),
            .ppEnabledExtensionNames = requiredDeviceExtensions.data()};

        m_Device = vk::raii::Device(m_PhysicalDevice, deviceCreateInfo);
        m_GraphicsQueue = vk::raii::Queue(m_Device, m_QueueIndex, 0);
    }

    void CreateSwapchain()
    {
        vk::SurfaceCapabilitiesKHR capabilities =
            m_PhysicalDevice.getSurfaceCapabilitiesKHR(*m_Surface);
        const std::vector<vk::SurfaceFormatKHR> formats =
            m_PhysicalDevice.getSurfaceFormatsKHR(*m_Surface);
        m_SwapchainSurfaceFormat = ChooseSwapchainFormat(formats);
        const std::vector<vk::PresentModeKHR> presentModes =
            m_PhysicalDevice.getSurfacePresentModesKHR(*m_Surface);
        m_SwapchainExtent = ChooseSwapchainExtent(capabilities, m_pWindow);

        std::cout << "Swapchain Extent: " << m_SwapchainExtent.width << "x"
                  << m_SwapchainExtent.height << "\n";

        vk::SwapchainCreateInfoKHR createInfo{
            .surface = *m_Surface,
            .minImageCount = ChooseSwapMinImageCount(capabilities),
            .imageFormat = m_SwapchainSurfaceFormat.format,
            .imageColorSpace = m_SwapchainSurfaceFormat.colorSpace,
            .imageExtent = m_SwapchainExtent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = ChoosePresentMode(presentModes),
            .clipped = true,
            .oldSwapchain = nullptr};

        m_Swapchain = vk::raii::SwapchainKHR(m_Device, createInfo);
        m_SwapImages = m_Swapchain.getImages();

        std::cout << "Swapchain image count: " << m_SwapImages.size() << "\n";
    }

    void CreateSwapchainImageViews()
    {
        assert(m_SwapImageViews.empty());

        vk::ImageViewCreateInfo createInfo{
            .viewType = vk::ImageViewType::e2D,
            .format = m_SwapchainSurfaceFormat.format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

        for (const vk::Image& image : m_SwapImages)
        {
            createInfo.image = image;
            m_SwapImageViews.emplace_back(m_Device, createInfo);
        }
    }

    [[nodiscard]] vk::raii::ShaderModule
    CreateShaderModule(const std::vector<char>& shaderCode) const
    {
        vk::ShaderModuleCreateInfo createInfo{
            .codeSize = shaderCode.size() * sizeof(char),
            .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data())};
        vk::raii::ShaderModule shaderModule(m_Device, createInfo);

        return shaderModule;
    }

    void CreateGraphicsPipeline()
    {
        vk::raii::ShaderModule shaderModule =
            CreateShaderModule(ReadFile("shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = shaderModule,
            .pName = "vertMain"};

        vk::PipelineShaderStageCreateInfo fragCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = shaderModule,
            .pName = "fragMain"};

        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
            vertCreateInfo, fragCreateInfo};

        auto bindingDesc = Vertex::GetBindingDescription();
        auto attributeDesc = Vertex::GetAttributeDescription();
        vk::PipelineVertexInputStateCreateInfo vertexInput{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDesc,
            .vertexAttributeDescriptionCount = 2,
            .pVertexAttributeDescriptions = attributeDesc.data()};
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList};

        vk::Viewport viewport{
            .x = 0.f,
            .y = 0.f,
            .width = static_cast<float>(m_SwapchainExtent.width),
            .height = static_cast<float>(m_SwapchainExtent.height),
            .minDepth = 0.f,
            .maxDepth = 1.f};
        vk::Rect2D scissor{.offset = vk::Offset2D{0, 0},
                           .extent = m_SwapchainExtent};
        std::vector<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()};
        vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1,
                                                          .scissorCount = 1};

        vk::PipelineRasterizationStateCreateInfo rasterState{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.f};

        vk::PipelineMultisampleStateCreateInfo multisampleState{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False};

        vk::PipelineColorBlendAttachmentState attachmentState{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR |
                              vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB |
                              vk::ColorComponentFlagBits::eA};
        vk::PipelineColorBlendStateCreateInfo blendState{
            .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &attachmentState};

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 0, .pushConstantRangeCount = 0};
        m_PipelineLayout =
            vk::raii::PipelineLayout(m_Device, pipelineLayoutInfo);

        vk::StructureChain<vk::GraphicsPipelineCreateInfo,
                           vk::PipelineRenderingCreateInfo>
            pipelineCreateInfoChain = {
                {.stageCount = static_cast<uint32_t>(shaderStages.size()),
                 .pStages = shaderStages.data(),
                 .pVertexInputState = &vertexInput,
                 .pInputAssemblyState = &inputAssembly,
                 .pViewportState = &viewportState,
                 .pRasterizationState = &rasterState,
                 .pMultisampleState = &multisampleState,
                 .pColorBlendState = &blendState,
                 .pDynamicState = &dynamicState,
                 .layout = m_PipelineLayout,
                 .renderPass = nullptr},
                {.colorAttachmentCount = 1,
                 .pColorAttachmentFormats = &m_SwapchainSurfaceFormat.format}};

        m_GraphicsPipeline = vk::raii::Pipeline(
            m_Device, nullptr,
            pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    }

    void CreateCommandPool()
    {
        vk::CommandPoolCreateInfo createInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = m_QueueIndex};
        m_CommandPool = vk::raii::CommandPool(m_Device, createInfo);
    }

    void CreateCommandBuffer()
    {
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = m_CommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
        m_CommandBuffers = vk::raii::CommandBuffers(m_Device, allocInfo);
    }

    void RecordCommandBuffer(uint32_t imageIndex)
    {
        m_CommandBuffers[m_FrameIndex].reset();

        vk::CommandBufferBeginInfo beginInfo{};
        m_CommandBuffers[m_FrameIndex].begin(beginInfo);

        TransitionImageLayout(
            imageIndex, vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal, {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput);

        vk::ClearValue clearColor = vk::ClearColorValue(0.f, 0.f, 0.f, 1.f);
        vk::RenderingAttachmentInfo attachmentInfo = {
            .imageView = m_SwapImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clearColor};

        vk::RenderingInfo renderingInfo = {
            .renderArea = {.offset = {0, 0}, .extent = m_SwapchainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachmentInfo};

        m_CommandBuffers[m_FrameIndex].beginRendering(renderingInfo);

        m_CommandBuffers[m_FrameIndex].bindPipeline(
            vk::PipelineBindPoint::eGraphics, *m_GraphicsPipeline);
        m_CommandBuffers[m_FrameIndex].bindVertexBuffers(0, *m_VertexBuffer,
                                                         {0});
        m_CommandBuffers[m_FrameIndex].setViewport(
            0, vk::Viewport(
                   0.f, 0.f, static_cast<float>(m_SwapchainExtent.width),
                   static_cast<float>(m_SwapchainExtent.height), 0.f, 1.f));
        m_CommandBuffers[m_FrameIndex].setScissor(
            0, vk::Rect2D(vk::Offset2D(0, 0), m_SwapchainExtent));

        m_CommandBuffers[m_FrameIndex].draw(vertices.size(), 1, 0, 0);

        m_CommandBuffers[m_FrameIndex].endRendering();

        TransitionImageLayout(
            imageIndex, vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite, {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe);

        m_CommandBuffers[m_FrameIndex].end();
    }

    void TransitionImageLayout(uint32_t imageIndex, vk::ImageLayout oldLayout,
                               vk::ImageLayout newLayout,
                               vk::AccessFlags2 srcAccessMask,
                               vk::AccessFlags2 dstAccessFlags,
                               vk::PipelineStageFlags2 srcStageMask,
                               vk::PipelineStageFlags2 dstStageMask)
    {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessFlags,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_SwapImages[imageIndex],
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1}};

        vk::DependencyInfo info = {.dependencyFlags = {},
                                   .imageMemoryBarrierCount = 1,
                                   .pImageMemoryBarriers = &barrier};
        m_CommandBuffers[m_FrameIndex].pipelineBarrier2(info);
    }

    void CreateSyncObjects()
    {
        assert(m_RenderCompleteSemaphores.empty() &&
               m_PresentCompleteSemaphores.empty() && m_DrawFences.empty());

        for (size_t i = 0; i < m_SwapImages.size(); i++)
        {
            m_RenderCompleteSemaphores.emplace_back(
                vk::raii::Semaphore(m_Device, vk::SemaphoreCreateInfo()));
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            m_PresentCompleteSemaphores.emplace_back(
                vk::raii::Semaphore(m_Device, vk::SemaphoreCreateInfo()));
            m_DrawFences.emplace_back(vk::raii::Fence(
                m_Device, {.flags = vk::FenceCreateFlagBits::eSignaled}));
        }
    }

    void RecreateSwapchain()
    {
        std::cout << "Recreating swapchain...\n";

        int width, height;
        SDL_GetWindowSizeInPixels(m_pWindow, &width, &height);
        while (width == 0 || height == 0)
        {
            SDL_GetWindowSizeInPixels(m_pWindow, &width, &height);
            SDL_Event event;
            SDL_WaitEvent(&event);
        }

        m_Device.waitIdle();

        m_SwapImageViews.clear();
        m_Swapchain = nullptr;

        m_Device.waitIdle();

        CreateSwapchain();
        CreateSwapchainImageViews();
    }

    void CreateVertexBuffer()
    {
        vk::BufferCreateInfo bufferInfo{
            .size = sizeof(vertices[0]) * vertices.size(),
            .usage = vk::BufferUsageFlagBits::eVertexBuffer,
            .sharingMode = vk::SharingMode::eExclusive};
        m_VertexBuffer = vk::raii::Buffer(m_Device, bufferInfo);

        vk::MemoryRequirements memRequirements =
            m_VertexBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo = {
            .allocationSize = memRequirements.size,
            .memoryTypeIndex =
                FindMemoryType(memRequirements.memoryTypeBits,
                               vk::MemoryPropertyFlagBits::eHostVisible |
                                   vk::MemoryPropertyFlagBits::eHostCoherent)};

        m_VertexBufferMemory = vk::raii::DeviceMemory(m_Device, allocInfo);
        m_VertexBuffer.bindMemory(*m_VertexBufferMemory, 0);

        void* data = m_VertexBufferMemory.mapMemory(0, bufferInfo.size);
        memcpy(data, vertices.data(), bufferInfo.size);
        m_VertexBufferMemory.unmapMemory();
    }

    uint32_t FindMemoryType(uint32_t typeFilter,
                            vk::MemoryPropertyFlags properties)
    {
        vk::PhysicalDeviceMemoryProperties memProperties =
            m_PhysicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) ==
                    properties)
                return i;
        }
        throw std::runtime_error("Failed to find a suitable memory type!");
    }

private:
    vk::raii::Context m_Context;
    vk::raii::Instance m_Instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT m_DebugMessenger = nullptr;
    vk::raii::SurfaceKHR m_Surface = nullptr;
    vk::raii::PhysicalDevice m_PhysicalDevice = nullptr;
    vk::raii::Device m_Device = nullptr;
    vk::raii::Queue m_GraphicsQueue = nullptr;
    vk::raii::SwapchainKHR m_Swapchain = nullptr;
    vk::raii::PipelineLayout m_PipelineLayout = nullptr;
    vk::raii::Pipeline m_GraphicsPipeline = nullptr;
    vk::raii::CommandPool m_CommandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> m_CommandBuffers;
    vk::raii::Buffer m_VertexBuffer = nullptr;
    vk::raii::DeviceMemory m_VertexBufferMemory = nullptr;

    vk::SurfaceFormatKHR m_SwapchainSurfaceFormat;
    vk::Extent2D m_SwapchainExtent;
    std::vector<vk::Image> m_SwapImages;
    std::vector<vk::raii::ImageView> m_SwapImageViews;
    uint32_t m_QueueIndex = ~0;

    std::vector<vk::raii::Semaphore> m_PresentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> m_RenderCompleteSemaphores;
    std::vector<vk::raii::Fence> m_DrawFences;
    uint32_t m_FrameIndex = 0;

    SDL_Window* m_pWindow = nullptr;
    bool m_bIsFocused = true;
};

int main()
{
    std::signal(SIGINT, HandleSIGINT);

    SDL_Window* pWindow = nullptr;

    try
    {
        InitSDL();
        pWindow = CreateSDLWindow();

        App app(pWindow);
        app.Run();
    }
    catch (const SDLException& e)
    {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "SDL error: %s", e.what());
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "SDL Error", e.what(),
                                 nullptr);
        return EXIT_FAILURE;
    }
    catch (const vk::SystemError& e)
    {
        std::cerr << "Vulkan error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    ShutdownSDL(pWindow);

    std::cout << "Exiting gracefully..." << std::endl;
    return EXIT_SUCCESS;
}
