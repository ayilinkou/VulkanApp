# Both shader tools come from vcpkg, which is what keeps the Vulkan SDK off the
# list of things a build needs. vcpkg's toolchain appends its tools directories
# to CMAKE_PROGRAM_PATH, and find_program searches CMake variables ahead of the
# environment's PATH, so these resolve to the versions vcpkg.json pins rather
# than to whichever copy a developer happens to have installed.
find_program(SLANGC_EXE slangc)
if(NOT SLANGC_EXE)
  message(FATAL_ERROR "slangc not found! It is provided by the shader-slang vcpkg port.")
endif()

# The Vulkan specification makes valid SPIR-V the application's responsibility,
# not the driver's — VUID-VkShaderModuleCreateInfo-pCode-08736 — so a driver is
# free to assume validity and skip checking. Invalid SPIR-V is then undefined
# behaviour that renders correctly on the driver it was written against and
# fails somewhere else.
#
# The validation layers do call spirv-val at runtime, but only on the modules a
# run actually creates, and only where a Vulkan ICD exists. CI has none, so the
# GPU tests skip there and the layers never load: this is the only shader
# correctness check CI can run. slangc does not stand in for it — it accepts
# modules spirv-val rejects.
#
# Fatal rather than optional, because a check that silently disappears on one of
# the six CI configurations is worse than no check: the build stays green and
# the coverage is imaginary. Depending on the spirv-tools port's "tools" feature
# rather than on an SDK install is what makes it present everywhere.
find_program(SPIRV_VAL_EXE spirv-val)
if(NOT SPIRV_VAL_EXE)
  message(FATAL_ERROR "spirv-val not found! It is provided by the spirv-tools vcpkg port.")
endif()

# Compiles .slang sources to SPIR-V beside the executable that loads them.
#
# The output directory is HIKARI_EXE_DIR/shaders — set next to
# add_executable(HikariEngine), so the two cannot drift — rather than a fixed
# path in the source tree, and that is load-bearing. Debug and Release
# compile the same sources with different flags (-O0 -g1 against -O3 -g0), so a
# shared output directory means the two configurations overwrite each other's
# work. Worse, they do it silently: each build directory judges the .spv
# up to date from its own records, and once both have built once, neither
# rebuilds — leaving whichever configuration ran last in place and every
# subsequent build a no-op. A debug session then runs optimized, stripped
# shaders while reporting success.
#
# It is deliberately <exe dir>/shaders and not <exe dir>/content/shaders.
# Paths tries <exe dir>/content as a content root candidate before the source
# tree, so creating that directory here would make the build directory look
# like a content root — an incomplete one, with no models, scenes or textures —
# and asset loading would resolve to it and fail.
function(add_slang_shader_target target)
  cmake_parse_arguments("SHADER" "" "" "SOURCES" ${ARGN})

  set(shaders_source_dir ${CMAKE_SOURCE_DIR}/engine/engine/src/shaders)
  set(shaders_out_dir ${HIKARI_EXE_DIR}/shaders)

  set(spv_outputs "")
  foreach(shader ${SHADER_SOURCES})
    file(RELATIVE_PATH rel_path ${shaders_source_dir} ${shader})
    set(output_file ${shaders_out_dir}/${rel_path})
    string(REPLACE ".slang" ".spv" output_file ${output_file})

    # Depfiles are build bookkeeping, so they stay in the build tree rather
    # than in the directory that gets deployed next to the executable. Keyed by
    # configuration for the same reason the SPIR-V is: the multi-config
    # generators build every configuration out of one build directory.
    set(depfile ${CMAKE_CURRENT_BINARY_DIR}/shader_deps/$<CONFIG>/${rel_path}.d)

    if(shader MATCHES "\\.comp\\.slang$")
      set(entry_points -entry main)
    else()
      set(entry_points -entry vertMain -entry fragMain)
    endif()

    # slangc reports exactly the files each shader pulled in — including the
    # C++ headers shared with the engine, which a *.slangh glob would miss — so
    # editing one header rebuilds only the shaders that include it.
    add_custom_command(
      OUTPUT ${output_file}
      COMMAND ${CMAKE_COMMAND} -E echo "Compiling ${rel_path}"
      COMMAND ${CMAKE_COMMAND} -E make_directory ${shaders_out_dir}
      COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/shader_deps/$<CONFIG>
      COMMAND
        ${SLANGC_EXE} ${shader} -target spirv -profile spirv_1_4
        -emit-spirv-directly -warnings-as-errors all -fvk-use-entrypoint-name ${entry_points} -o
        ${output_file} -depfile ${depfile} $<IF:$<CONFIG:Debug>,-g1,-g0>
        $<IF:$<CONFIG:Debug>,-O0,-O3>
      # Same command as the compile, so validation runs exactly when a shader
      # recompiles and a failure fails the build. The target environment is
      # stated rather than left at spirv-val's universal default, which would
      # miss the Vulkan-specific rules; it matches VulkanDevice's kApiVersion.
      COMMAND ${SPIRV_VAL_EXE} --target-env vulkan1.4 ${output_file}
      DEPENDS ${shader}
      DEPFILE ${depfile}
      COMMENT "Compiling shader ${rel_path}"
      VERBATIM)

    list(APPEND spv_outputs ${output_file})
  endforeach()

  add_custom_target(${target} ALL DEPENDS ${spv_outputs})
endfunction()

file(GLOB_RECURSE shader_slang_sources CONFIGURE_DEPENDS
     ${CMAKE_SOURCE_DIR}/engine/engine/src/shaders/*.slang)

add_slang_shader_target(CompileShadersTarget SOURCES ${shader_slang_sources})
