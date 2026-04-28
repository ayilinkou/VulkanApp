cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# Only generate Debug and Release configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

project(VulkanApp VERSION 1.0)

set(SOURCES
	src/main.cpp
)

find_package(SDL3 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glm CONFIG REQUIRED)

find_program(SLANGC_EXECUTABLE slangc)
if(NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR "slangc not found!")
endif()

add_executable(VulkanApp)

target_sources(VulkanApp PRIVATE
	${SOURCES}
)

target_link_libraries(VulkanApp PRIVATE
	SDL3::SDL3
	Vulkan::Vulkan
	glm::glm
)

target_include_directories(VulkanApp PRIVATE
	src/
)

target_compile_definitions(VulkanApp PRIVATE
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)

if(MSVC)
		target_compile_options(VulkanApp PRIVATE
			/W3            # Warning level
			/MP            # Multi-processor compilation
			/WX            # Warnings as errors

			$<$<CONFIG:Debug>:
				/Od        # No optimisation
				/Zi        # Debug info (PDB)
			>

			$<$<CONFIG:Release>:
				/O2        # Full optimisation
			>
		)

		target_link_options(VulkanApp PRIVATE
			$<$<CONFIG:Debug>:
				/DEBUG
			>

			# optional but useful for crash reports
			$<$<CONFIG:Release>:
				/DEBUG
			>
		)
	else()
		target_compile_options(VulkanApp PRIVATE
			-Wall
			-Wextra
			-Wpedantic

            $<$<CONFIG:Debug>:
                -O0
                -g
            >

            $<$<CONFIG:Release>:
                -O3
            >
		)
	endif()

set(SHADER_SLANG_SOURCES
	${CMAKE_CURRENT_LIST_DIR}/src/shaders/shader.slang
)

function (add_slang_shader_target TARGET)
  cmake_parse_arguments ("SHADER" "" "" "SOURCES" ${ARGN})

  set (SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
  set (ENTRY_POINTS -entry vertMain -entry fragMain)

  add_custom_command (
          OUTPUT  ${SHADERS_DIR}/slang.spv
          COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
          COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
          WORKING_DIRECTORY ${SHADERS_DIR}
          DEPENDS ${SHADER_SOURCES}
          COMMENT "Compiling Slang Shaders"
          VERBATIM
  )

  add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

add_slang_shader_target(CompileShadersTarget SOURCES ${SHADER_SLANG_SOURCES})
add_dependencies(VulkanApp CompileShadersTarget)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT VulkanApp)
set_target_properties(VulkanApp PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
