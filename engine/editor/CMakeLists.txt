# The UI stack. Above Engine rather than below it: an app builds the backend and
# hands it to the engine, which calls it through IUiBackend and never names
# ImGui's Vulkan integration itself.
engine_module(
  Editor
  SOURCES
  src/VulkanUiBackend.cpp
  LINK_LIBRARIES
  Engine::Core
  Engine::Platform
  Engine::RHI
  Engine::Engine
  SDL3::SDL3
  Vulkan::Vulkan
  imgui::imgui)
