#include <catch2/catch_test_macros.hpp>

#include "BindGroupLayouts.h"

/**
 * Pins the renderer's bind group layouts to exactly the shapes it has.
 *
 * This is the enforcement half of RHI plan D21. The curated BindingType enum and
 * its default-free switches keep the *vocabulary* honest -- a new kind fails the
 * build until every backend maps it -- but nothing in that stops the model
 * growing: a fifth layout, or a fourth texture on an existing one, compiles
 * perfectly. This test is what makes growth visible, because widening the model
 * means editing an expectation here, and editing an existing test's expectation
 * needs asking first.
 *
 * It is deliberately dumb. It asserts counts and kinds rather than anything
 * clever, because its value is entirely in failing when someone adds a binding
 * without having argued for it.
 */
using namespace Hikari::Rhi;

TEST_CASE("The renderer declares exactly four bind group layouts", "[engine][binding]")
{
    // Four of six. CloudSystem owns the other two -- a dispatch set and a
    // noise-bake set, both holding storage images the binding vocabulary cannot
    // yet describe. Each joins this file when it moves behind the neutral API
    // rather than escaping it, and this count rises with them.
    CHECK(EngineBindGroups::kGlobal.size() == 1u);
    CHECK(EngineBindGroups::kComposite.size() == 5u);
    CHECK(EngineBindGroups::kDepth.size() == 1u);
    CHECK(EngineBindGroups::kMaterial.size() == 4u);
}

TEST_CASE("Every material texture is optional, and the sampler is not", "[engine][binding]")
{
    // The load-bearing assertion of the whole file. A material with no normal
    // map leaves that slot empty; if these stopped being optional, nothing would
    // fail to build and no validation message would appear -- the untextured
    // models would simply start reading a descriptor that was never written.
    for (uint32_t slot = 0u; slot < 3u; slot++)
    {
        INFO("slot " << slot);
        CHECK(EngineBindGroups::kMaterial[slot].Type == BindingType::Texture);
        CHECK(EngineBindGroups::kMaterial[slot].bOptional);
    }

    // Always present, so not optional: every material samples through it.
    CHECK(EngineBindGroups::kMaterial[3].Type == BindingType::Sampler);
    CHECK_FALSE(EngineBindGroups::kMaterial[3].bOptional);
}

TEST_CASE("The global layout is one uniform buffer visible everywhere", "[engine][binding]")
{
    const BindGroupLayoutBinding& binding = EngineBindGroups::kGlobal[0];
    CHECK(binding.Slot == 0u);
    CHECK(binding.Type == BindingType::UniformBuffer);
    CHECK(binding.Visibility ==
          (ShaderStage::Vertex | ShaderStage::Pixel | ShaderStage::Compute));
}

TEST_CASE("The composite layout is four textures and a separate sampler", "[engine][binding]")
{
    for (uint32_t slot = 0u; slot < 4u; slot++)
    {
        INFO("slot " << slot);
        CHECK(EngineBindGroups::kComposite[slot].Slot == slot);
        CHECK(EngineBindGroups::kComposite[slot].Type == BindingType::Texture);
    }

    // The sampler is its own binding, not folded into the texture it reads.
    // D3D12 has no combined image sampler and cannot be given one (D22).
    CHECK(EngineBindGroups::kComposite[4].Slot == 4u);
    CHECK(EngineBindGroups::kComposite[4].Type == BindingType::Sampler);
}

TEST_CASE("The depth layout is visible to compute as well as pixel", "[engine][binding]")
{
    // The cloud dispatch reads it. A layout description that assumed graphics
    // stages would have been wrong here, which is why visibility is per binding.
    const BindGroupLayoutBinding& binding = EngineBindGroups::kDepth[0];
    CHECK(binding.Type == BindingType::Texture);
    CHECK(binding.Visibility == (ShaderStage::Pixel | ShaderStage::Compute));
}
