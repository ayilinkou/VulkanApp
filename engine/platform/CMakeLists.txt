engine_module(Platform SOURCES src/SdlPlatform.cpp src/HeadlessPlatform.cpp src/Paths.cpp
                              src/CommandLine.cpp src/InputScript.cpp src/ScriptedInputSource.cpp
              LINK_LIBRARIES Engine::Core SDL3::SDL3)

# The last-resort content root, so a freshly built tree runs before anything
# has been installed alongside the executable.
target_compile_definitions(Platform
                           PRIVATE HIKARI_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
