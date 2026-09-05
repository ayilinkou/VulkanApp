engine_test(core_tests SOURCES unit/core/SwapbackArrayTests.cpp
  unit/core/LogTests.cpp unit/core/JobSystemTests.cpp
  unit/core/HandleTests.cpp unit/core/TimerTests.cpp LIBS Engine::Core)

target_include_directories(core_tests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)

target_compile_definitions(
  core_tests PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/")

# Separate from core_tests so that Core's tests keep linking nothing but Core.
engine_test(platform_tests SOURCES unit/platform/PathsTests.cpp
  unit/platform/CommandLineTests.cpp unit/platform/HeadlessPlatformTests.cpp
  LIBS Engine::Platform)

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
  support/RhiDeviceListener.cpp LIBS Engine::RHI LABEL gpu)

# The same reach into the module's own sources that rhi_tests makes, plus the
# shared fixture: submitting a command list needs a queue, and the RHI hands out
# neither.
target_include_directories(
  rhi_gpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support
                        ${CMAKE_SOURCE_DIR}/engine/rhi/src)
