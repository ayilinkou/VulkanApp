cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(VulkanApp VERSION 1.0)

set(SOURCES
	src/main.cpp
)

find_package(SDL3 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(VulkanApp)

target_sources(VulkanApp PRIVATE
	${SOURCES}
)

target_link_libraries(VulkanApp PRIVATE
	SDL3::SDL3
	Vulkan::Vulkan
)

target_include_directories(VulkanApp PRIVATE
	src/
)

target_compile_definitions(VulkanApp PRIVATE
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)

target_compile_options(VulkanApp PRIVATE
    $<$<CONFIG:Debug>:-O0 -g>
    $<$<CONFIG:Release>:-O3>
)
