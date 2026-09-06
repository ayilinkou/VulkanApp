#pragma once

#include "vulkan/vulkan_raii.hpp"

#include <rhi/RhiTypes.h>

namespace Hikari::Rhi::Vulkan
{
/**
 * What a TextureViewHandle resolves to.
 *
 * A struct wrapping one vk::raii::ImageView rather than the raii type itself,
 * because Core::HandlePool requires a default-constructible payload and
 * vk::raii::ImageView has no default constructor — only one taking nullptr.
 * The wrapper is what supplies it, and keeping the raii type is what makes
 * releasing a pool slot destroy the view.
 */
struct VulkanTextureView
{
    vk::raii::ImageView View = nullptr;

    /**
     * Kept because the layout a *sampled* view must be in depends on it:
     * a colour view reads from SHADER_READ_ONLY_OPTIMAL and a depth one from
     * DEPTH_READ_ONLY_OPTIMAL. Nothing in the neutral binding description says
     * which, and nothing should -- it is a Vulkan layout rule, so the backend
     * derives it from what the view already is.
     */
    TextureAspect Aspect = TextureAspect::Color;
};
} // namespace Hikari::Rhi::Vulkan
