#include <algorithm>
#include <cstdlib>
#include <exception>
#include <format>
#include <iostream>
#include <stdexcept>

#include "SDL3/SDL_error.h"
#include "SDL3/SDL_log.h"
#include "SDL3/SDL_messagebox.h"
#include "SDL3/SDL_stdinc.h"
#include "SDL3/SDL_video.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan_core.h>

#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"

class SDLException : public std::runtime_error
{
public:
    SDLException(const std::string& message)
        : std::runtime_error(std::format("{} {}", message, SDL_GetError()))
    {
    }
};

class App
{
public:
    App() {}
    ~App() {}

    void Run()
    {
        Init();

        SDL_ShowWindow(pWindow);

        bool bDone = false;
        while (!bDone)
        {
            SDL_Event event;
            while (SDL_PollEvent(&event))
            {
                if (event.type == SDL_EVENT_QUIT)
                    bDone = true;
            }
        }

        Shutdown();
    }

private:
    void Init()
    {
		InitSDL();
        InitVulkan();
    }

    void InitVulkan() { CreateInstance(); }

    void InitSDL()
    {
        if (!SDL_Init(SDL_INIT_VIDEO))
            throw SDLException("Failed to initialise SDL!");

        if (!SDL_Vulkan_LoadLibrary(nullptr))
            throw SDLException("Failed to load Vulkan library!");

        // hidden to hide the window while initialisation is taking place
        SDL_WindowFlags flags =
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN;
        pWindow = SDL_CreateWindow("Vulkan App", 640, 480, flags);
        if (pWindow == nullptr)
            throw SDLException("Failed to create window!");
    }

    void Shutdown()
    {
        SDL_DestroyWindow(pWindow);
        SDL_Quit();
    }

    void CreateInstance()
    {
        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName = "Vulkan App",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = vk::ApiVersion14};

        Uint32 countInstanceExtensions;
        const char* const* instanceExtensions =
            SDL_Vulkan_GetInstanceExtensions(&countInstanceExtensions);

        if (countInstanceExtensions == 0)
            throw std::runtime_error("No available instance extensions found!");

        std::vector<const char*> extensions(countInstanceExtensions);
        memcpy(extensions.data(), instanceExtensions,
               countInstanceExtensions * sizeof(const char*));

        auto extensionProperties =
            context.enumerateInstanceExtensionProperties();

        std::cout << "Available extensions:\n";
        for (const auto& extensionProp : extensionProperties)
        {
            std::cout << '\t' << extensionProp.extensionName << '\n';
        }

        for (int i = 0; i < countInstanceExtensions; i++)
        {
            if (std::ranges::none_of(
                    extensionProperties, [ext = extensions[i]](auto const& prop)
                    { return strcmp(prop.extensionName, ext) == 0; }))
                throw std::runtime_error(
                    "Required SDL extension not supported: " +
                    std::string(extensions[i]));
        }

        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = (uint32_t)extensions.size(),
            .ppEnabledExtensionNames = extensions.data()};

        instance = vk::raii::Instance(context, createInfo);
    }

private:
    vk::raii::Instance instance = nullptr;
    vk::raii::Context context;

    SDL_Window* pWindow = nullptr;
};

int main()
{
    try
    {
        App app;
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

    return EXIT_SUCCESS;
}
