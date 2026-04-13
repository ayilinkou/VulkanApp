cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(VulkanApp VERSION 1.0)

set(SOURCES
	src/main.cpp
)

add_executable(VulkanApp)

target_sources(VulkanApp PRIVATE
	${SOURCES}
)

target_include_directories(VulkanApp PRIVATE
	src/
)

target_compile_options(VulkanApp PRIVATE
    $<$<CONFIG:Debug>:-O0 -g>
    $<$<CONFIG:Release>:-O3>
)
