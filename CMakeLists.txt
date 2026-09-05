include(cmake/Warnings.cmake)
include(cmake/HeaderSelfContainment.cmake)
include(cmake/EngineModule.cmake)

cmake_minimum_required(VERSION 4.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# Only generate Debug and Release configurations
set(CMAKE_CONFIGURATION_TYPES
    "Debug;Release"
    CACHE STRING "" FORCE)

if(CMAKE_EXPORT_COMPILE_COMMANDS AND CMAKE_GENERATOR MATCHES "Ninja|Makefiles")
  file(
    CREATE_LINK "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
    RESULT compile_commands_link
    SYMBOLIC)
  if(NOT compile_commands_link STREQUAL "0")
    message(
      STATUS "compile_commands.json link skipped: ${compile_commands_link}")
  endif()
endif()

project(
  HikariEngine
  VERSION 1.0
  LANGUAGES CXX)

option(ENGINE_BUILD_TESTS "Build unit tests" ON)
option(ENABLE_SANITIZERS "Enable ASan (and UBSan when available)" OFF)

if(ENABLE_SANITIZERS)
  if(MSVC)
    add_compile_options(/fsanitize=address)
    add_link_options(/fsanitize=address /INCREMENTAL:NO)

    # ASan is incompatible with /RTC1, which CMake's Debug config injects by
    # default
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

    # MSVC STL auto-enables container-overflow annotations under
    # /fsanitize=address, which changes std::string/vector/optional ABI. vcpkg's
    # prebuilt libs (Catch2, etc.) aren't built with this, so leaving it on
    # causes LNK2038 ABI-mismatch errors when linking against them. Disable to
    # match their ABI.
    add_compile_definitions(_DISABLE_STL_ANNOTATION)
  else()
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer
                        -fno-sanitize-recover=undefined -g)
    add_link_options(-fsanitize=address,undefined)
  endif()
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  message(STATUS "ccache found, enabling compiler cache: ${CCACHE_PROGRAM}")
endif()

set(SOURCES
    src/main.cpp
    src/Model.cpp
    src/ModelData.cpp
    src/Camera.cpp
    src/Material.cpp
    src/PBRMaterial.cpp
    src/Texture.cpp
    src/Cubemap.cpp
    src/MaterialFactory.cpp
    src/ResourceManager.cpp
    src/TextureLoader.cpp
    src/CubemapLoader.cpp
    src/ModelLoader.cpp
    src/ModelManager.cpp
    src/Entity.cpp
    src/SceneComponent.cpp
    src/Transform.cpp
    src/Node.cpp
    src/Mesh.cpp
    src/CloudSystem.cpp
    src/XmlParser.cpp)

find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(pugixml CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(
  ImGuiFileDialog
  GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog.git
  GIT_TAG v0.6.7)
FetchContent_MakeAvailable(ImGuiFileDialog)
target_link_libraries(ImGuiFileDialog PRIVATE imgui::imgui)
set_target_properties(ImGuiFileDialog PROPERTIES COMPILE_WARNING_AS_ERROR OFF)

if(MSVC)
  target_compile_options(ImGuiFileDialog PRIVATE /w)
else()
  target_compile_options(ImGuiFileDialog PRIVATE -w)
endif()

# Must exist before any engine_module() call, since each module attaches its
# own header check to it.
engine_header_self_containment_init()

add_subdirectory(engine/core)
add_subdirectory(engine/platform)
add_subdirectory(engine/rhi)
add_subdirectory(engine/engine)

# clang-format's output is not stable across major versions, so the version is
# pinned rather than left to whatever is on PATH. Without the pin, the six CI
# configurations run two different clang-formats (the Ubuntu image's and the
# Windows image's LLVM) and disagree with each other and with developers — which
# is exactly how a commit lands green on Windows and red on Linux.
#
# The pinned version lives in .clang-format-version so that this file and the CI
# workflow read the same number. Bumping it means bumping that file and
# reformatting the tree in the same commit.
#
# The file list and the invocation live in cmake/Format.cmake, which runs without
# a configured tree; these targets are a convenience over it, not a second
# implementation. Checking it here as well is worth the duplicate warning: a
# developer meets it once at configure time rather than at the moment a
# formatting diff appears.
include(${CMAKE_SOURCE_DIR}/cmake/ClangFormatUtil.cmake)
hikari_find_clang_format(CLANG_FORMAT_EXE CLANG_FORMAT_VERSION)

if(CLANG_FORMAT_EXE)
  add_custom_target(
    format
    COMMAND ${CMAKE_COMMAND} -DFIX=ON -P ${CMAKE_SOURCE_DIR}/cmake/Format.cmake
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-format -i over source files..."
    VERBATIM)

  add_custom_target(
    format-check
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/cmake/Format.cmake
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking clang-format compliance over source files..."
    VERBATIM)
else()
  message(
    STATUS
      "clang-format not found — 'format' and 'format-check' targets unavailable"
  )
endif()

if(ENGINE_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 CONFIG REQUIRED)
  include(cmake/Testing.cmake)
  add_subdirectory(tests)
endif()

# The application's own headers. Engine modules attach their own checks from
# engine_module(), so this covers src/ only.
file(GLOB_RECURSE APP_HEADERS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.h)

# pch.h is deliberately excluded — it exists specifically to be force-included
# ahead of everything else and isn't meant to stand alone.
list(REMOVE_ITEM APP_HEADERS ${CMAKE_SOURCE_DIR}/src/pch.h)

engine_header_self_containment(
  App
  HEADERS
  ${APP_HEADERS}
  LINK_LIBRARIES
  Engine::Core
  Engine::Platform
  Engine::RHI
  SDL3::SDL3
  Vulkan::Vulkan
  GPUOpen::VulkanMemoryAllocator
  glm::glm-header-only
  assimp::assimp
  imgui::imgui
  pugixml::pugixml
  ImGuiFileDialog
  INCLUDE_DIRECTORIES
  src
  ${Stb_INCLUDE_DIR}
  COMPILE_DEFINITIONS
  VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
  GLM_FORCE_DEPTH_ZERO_TO_ONE
  GLM_ENABLE_EXPERIMENTAL
  $<$<CONFIG:Debug>:DEBUG>)

add_executable(HikariEngine)

# Where the executable lands, and — via cmake/Shaders.cmake — the shaders it
# loads. Both are derived from this one variable so that they cannot drift
# apart: Paths looks for shaders beside the executable, and a mismatch would
# only show up at runtime.
#
# Multi-configuration generators build every configuration out of a single
# build tree, so the configuration has to be part of the path there; the Ninja
# presets already get one build tree per configuration. Spelling the
# configuration out with a generator expression rather than letting CMake
# append it is what keeps the value here equal to the value Shaders.cmake uses
# — the automatic append happens to the target property only.
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
  set(HIKARI_EXE_DIR ${CMAKE_BINARY_DIR}/$<CONFIG>)
else()
  set(HIKARI_EXE_DIR ${CMAKE_BINARY_DIR})
endif()

set_target_properties(HikariEngine PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                              ${HIKARI_EXE_DIR})

target_precompile_headers(HikariEngine PRIVATE src/pch.h)

target_sources(HikariEngine PRIVATE ${SOURCES})

target_link_libraries(
  HikariEngine
  PRIVATE Engine::Core
          Engine::Platform
          Engine::RHI
          Engine::Engine
          SDL3::SDL3
          Vulkan::Vulkan
          GPUOpen::VulkanMemoryAllocator
          glm::glm-header-only
          assimp::assimp
          imgui::imgui
          pugixml::pugixml
          ImGuiFileDialog
          SanitizerShims)

target_include_directories(HikariEngine PRIVATE src ${Stb_INCLUDE_DIR})

target_compile_definitions(
  HikariEngine
  PRIVATE VULKAN_HPP_NO_STRUCT_CONSTRUCTORS GLM_FORCE_DEPTH_ZERO_TO_ONE
          GLM_ENABLE_EXPERIMENTAL $<$<CONFIG:Debug>:DEBUG>
          # Named in the run report, because frame times from a debug build and
          # a release one are not comparable and nothing else in the file says
          # which produced them. Derived rather than passed in by the preset:
          # $<CONFIG> is correct under multi-config generators, where there is
          # no single CMAKE_BUILD_TYPE, and for a tree configured without a
          # preset at all.
          HIKARI_BUILD_CONFIG="$<LOWER_CASE:$<CONFIG>>$<$<BOOL:${ENABLE_SANITIZERS}>:+asan>")

if(MSVC)
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    # MSBuild batches TUs into one cl invocation, so /MP is what parallelises.
    target_compile_options(HikariEngine PRIVATE /MP)
  else()
    # Ninja parallelises itself; /Z7 embeds debug info in each .obj and avoids
    # serialising all TUs through mspdbsrv writing a shared PDB.
    set_target_properties(HikariEngine PROPERTIES MSVC_DEBUG_INFORMATION_FORMAT
                                                  "$<$<CONFIG:Debug>:Embedded>")
  endif()

  target_compile_options(HikariEngine PRIVATE $<$<CONFIG:Debug>:/Od>
                                              $<$<CONFIG:Release>:/O2>)

  target_link_options(
    HikariEngine PRIVATE $<$<CONFIG:Debug>:/DEBUG> # optional but useful for crash
                                                   # reports
    $<$<CONFIG:Release>:/DEBUG>)
else()
  target_compile_options(
    HikariEngine
    PRIVATE $<$<CONFIG:Debug>:
            -O0
            -g
            >
            $<$<CONFIG:Release>:
            -O3
            >)

  # lld links noticeably faster than bfd, which matters most for the ASan/UBSan
  # build (large instrumented runtime + debug info to link).
  include(CheckLinkerFlag)
  check_linker_flag(CXX "-fuse-ld=lld" LLD_LINKER_AVAILABLE)
  if(LLD_LINKER_AVAILABLE)
    target_link_options(HikariEngine PRIVATE -fuse-ld=lld)
  endif()
endif()

engine_set_warnings(HikariEngine)

include(cmake/Shaders.cmake)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT
                                                            HikariEngine)
set_target_properties(HikariEngine PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY
                                              "${CMAKE_SOURCE_DIR}")
