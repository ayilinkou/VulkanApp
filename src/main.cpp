#include <cstdlib>
#include <format>
#include <iostream>
#include <stdexcept>

#include "SDL3/SDL_error.h"
#include "SDL3/SDL_log.h"
#include "SDL3/SDL_messagebox.h"
#include "SDL3/SDL_video.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

class SDLException : public std::runtime_error
{
public:
    SDLException(const std::string &message)
        : std::runtime_error(std::format("{} {}", message, SDL_GetError()))
    {
    }
};

class App
{
public:
    App() {}
    ~App() {}

    void Init()
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

    void Run()
    {
        SDL_ShowWindow(pWindow);

        int frame = 0;
        bool bDone = false;
        while (!bDone)
        {
            SDL_Event event;
            while (SDL_PollEvent(&event))
            {
                if (event.type == SDL_EVENT_QUIT)
                    bDone = true;
            }

            std::cout << "Frame " << ++frame << "\n";
        }

        Shutdown();
    }

private:
    SDL_Window *pWindow = nullptr;
};

int main()
{
    try
    {
        App app;
        app.Init();
        app.Run();
    }
    catch (const SDLException &e)
    {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "%s", e.what());
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error", e.what(),
                                 nullptr);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
