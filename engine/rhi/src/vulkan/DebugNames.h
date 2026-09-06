#pragma once

#include <cstdint>

#include "vulkan/vulkan_raii.hpp"

namespace Hikari::Rhi::Vulkan
{

/**
 * Attaches a human-readable name to a Vulkan object, so that validation
 * messages and capture tools name it instead of printing a handle value.
 *
 * Compiles to nothing unless DEBUG is defined, which is why the RHI target
 * defines it PUBLIC in Debug configurations. That is load-bearing rather than
 * tidy: this is a template, so its body is instantiated in whichever
 * translation unit calls it. If the module's own sources disagreed with the
 * application's about DEBUG, the same instantiation would have two different
 * definitions — an ODR violation, with the linker free to keep either. The
 * visible symptom would be debug names going missing from some objects and not
 * others, which is a miserable thing to chase.
 */
template <typename T>
inline void SetVkDebugName([[maybe_unused]] vk::raii::Device& device, [[maybe_unused]] T handle,
                           [[maybe_unused]] vk::ObjectType objectType,
                           [[maybe_unused]] const char* name)
{
#ifdef DEBUG
    // convert vk:: C++ types into C types
    // eg. vk::Image -> VkImage
    using CType = decltype(static_cast<typename T::CType>(handle));

    vk::DebugUtilsObjectNameInfoEXT nameInfo{
        .objectType = objectType,
        .objectHandle = reinterpret_cast<uint64_t>(static_cast<CType>(handle)),
        .pObjectName = name};
    device.setDebugUtilsObjectNameEXT(nameInfo);
#endif
}
} // namespace Hikari::Rhi::Vulkan
