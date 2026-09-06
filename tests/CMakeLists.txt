engine_test(core_tests SOURCES unit/core/SwapbackArrayTests.cpp
  unit/core/ClockTests.cpp
  unit/core/LogTests.cpp unit/core/JobSystemTests.cpp
  unit/core/HandleTests.cpp unit/core/TimerTests.cpp LIBS Engine::Core)

target_include_directories(core_tests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)

target_compile_definitions(
  core_tests PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/")

# Separate from core_tests so that Core's tests keep linking nothing but Core.
engine_test(platform_tests SOURCES unit/platform/PathsTests.cpp
  unit/platform/CommandLineTests.cpp
  unit/platform/InputScriptTests.cpp unit/platform/HeadlessPlatformTests.cpp
  LIBS Engine::Platform)

engine_test(asset_tests SOURCES unit/asset/AssetCacheTests.cpp LIBS Engine::Asset)

engine_test(engine_tests SOURCES unit/engine/ParseEngineOptionTests.cpp
  LIBS Engine::Engine)

engine_test(rhi_tests SOURCES unit/rhi/ConversionTests.cpp
  unit/rhi/DiagnosticsTests.cpp unit/rhi/FormatTests.cpp
  unit/rhi/OwnershipTransferTests.cpp unit/rhi/QueueFamilyTests.cpp
  unit/rhi/SwapchainUtilTests.cpp unit/rhi/UniqueHandleTests.cpp
  LIBS Engine::RHI)

# The conversion tables and the queue-family selection live in src/vulkan/, not
# in the module's public include path — they are backend-internal by design
# (plan D1), so testing them means reaching into the module deliberately rather
# than promoting them to public headers just to make them reachable.
target_include_directories(rhi_tests
                           PRIVATE ${CMAKE_SOURCE_DIR}/engine/rhi/src)

# Needs a real GPU, so it is labelled "gpu" and left out of the "unit" run that
# CI performs: on a machine with no Vulkan ICD every case here skips, and a run
# of nothing but skips looks identical to a run of tests that were passing
# vacuously. tests/scripts/run_gpu_tests.sh is what executes them.
engine_test(rhi_gpu_tests SOURCES gpu/rhi/DeviceTests.cpp
  gpu/rhi/PresentTargetTests.cpp gpu/rhi/UploadRoundTripTests.cpp
  gpu/rhi/ValidationCoverageTests.cpp support/RhiDeviceListener.cpp LIBS Engine::RHI LABEL gpu)

# The same reach into the module's own sources that rhi_tests makes, plus the
# shared fixture: submitting a command list needs a queue, and the RHI hands out
# neither.
target_include_directories(
  rhi_gpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support
                        ${CMAKE_SOURCE_DIR}/engine/rhi/src)

# The scene matrix: real runs of the engine, headless, asserting on the RunReport
# they return. Labelled "scene" rather than "gpu" so that ctest -L gpu keeps its
# under-a-minute meaning and a red step says which kind of thing broke.
engine_test(scene_tests SOURCES scene/SceneLaunchTests.cpp LIBS Engine::Engine
            Engine::Editor LABEL scene)

# Beside the shaders rather than in tests/: the engine loads them from its own
# directory, so a test that runs the engine has to live where a binary does.
set_target_properties(scene_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                             ${HIKARI_EXE_DIR})
add_dependencies(scene_tests CompileShadersTarget)

target_include_directories(scene_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)

target_compile_definitions(
  scene_tests
  PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/"
          # The shipped scene is loaded from the repository's own content root,
          # which is the point of that case: it is the only one testing geometry
          # anybody actually ships.
          HIKARI_CONTENT_DIR="${CMAKE_SOURCE_DIR}/content/"
          # The smoke case launches the real binary, so it needs to know where
          # the build put it.
          HIKARI_HEADLESS_BINARY="$<TARGET_FILE:HikariHeadless>")

add_dependencies(scene_tests HikariHeadless)
