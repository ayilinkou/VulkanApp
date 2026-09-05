engine_module(
  Engine SOURCES src/ParseEngineOptions.cpp LINK_LIBRARIES Engine::Core
  Engine::Platform Engine::RHI glm::glm-header-only)
