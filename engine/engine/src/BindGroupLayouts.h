#pragma once

#include <array>

#include <rhi/BindGroup.h>

/**
 * Every bind group layout the renderer creates, written down once.
 *
 * This exists to be pinned. The binding model is deliberately narrow -- scoped
 * to the layouts that exist rather than generalised (RHI plan D14) -- and D21
 * makes that a rule something enforces rather than a rule someone remembers:
 * BindGroupLayoutInventoryTests asserts this table's exact shape, so a fifth
 * layout or a fourth binding on an existing one cannot land without editing a
 * test expectation, which needs a conversation first.
 *
 * If this table starts changing every other stage, that is the signal it has
 * outlived its purpose rather than that the rule should be relaxed.
 */
namespace EngineBindGroups
{
using Hikari::Rhi::BindGroupLayoutBinding;
using Hikari::Rhi::BindingType;
using Hikari::Rhi::ShaderStage;

/** Per-frame camera and light constants, read by everything that draws. */
inline constexpr std::array<BindGroupLayoutBinding, 1> kGlobal{BindGroupLayoutBinding{
    .Slot = 0u,
    .Type = BindingType::UniformBuffer,
    .Visibility = ShaderStage::Vertex | ShaderStage::Pixel | ShaderStage::Compute}};

/**
 * The opaque, accumulation and revealage targets, plus the cloud output and the
 * sampler that reads it.
 *
 * Four textures and one sampler rather than three textures and one combined
 * image sampler: D3D12 keeps samplers in a separate heap and has no combined
 * descriptor at all (D22). Only the cloud target is sampled -- the other three
 * are fetched by texel and need no sampler.
 */
inline constexpr std::array<BindGroupLayoutBinding, 5> kComposite{
    BindGroupLayoutBinding{
        .Slot = 0u, .Type = BindingType::Texture, .Visibility = ShaderStage::Pixel},
    BindGroupLayoutBinding{
        .Slot = 1u, .Type = BindingType::Texture, .Visibility = ShaderStage::Pixel},
    BindGroupLayoutBinding{
        .Slot = 2u, .Type = BindingType::Texture, .Visibility = ShaderStage::Pixel},
    BindGroupLayoutBinding{
        .Slot = 3u, .Type = BindingType::Texture, .Visibility = ShaderStage::Pixel},
    BindGroupLayoutBinding{
        .Slot = 4u, .Type = BindingType::Sampler, .Visibility = ShaderStage::Pixel}};

/**
 * The depth buffer, read by the transparent pass and by the cloud dispatch --
 * which is why visibility spans compute as well as pixel, and why a
 * graphics-only assumption about stages would not have survived this layout.
 */
inline constexpr std::array<BindGroupLayoutBinding, 1> kDepth{
    BindGroupLayoutBinding{.Slot = 0u,
                           .Type = BindingType::Texture,
                           .Visibility = ShaderStage::Pixel | ShaderStage::Compute}};
} // namespace EngineBindGroups
