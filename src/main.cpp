#include <iostream>

#include <SDL3/SDL.h>

int main() {

    SDL_Window *window;
    bool done = false;

    SDL_Init(SDL_INIT_VIDEO);

    window = SDL_CreateWindow(
        "Vulkan App",
        640,
        480,
        SDL_WINDOW_VULKAN
    );

    if (window == NULL)
	{
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Could not create window: %s\n", SDL_GetError());
        return 1;
    }

	int frame = 0;
    while (!done)
	{
        SDL_Event event;
        while (SDL_PollEvent(&event))
		{
            if (event.type == SDL_EVENT_QUIT)
                done = true;
        }

		std::cout << "Frame " << ++frame << "\n";
    }

    SDL_DestroyWindow(window);

    SDL_Quit();
    return 0;
}
