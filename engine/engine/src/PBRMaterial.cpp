#include "PBRMaterial.h"

#include <array>
#include <span>

#include "assimp/material.h"

#include "AssetRegistry.h"

#include "BindGroupLayouts.h"

using namespace Hikari;

PBRMaterial::PBRMaterial(Rhi::IDevice& rhiDevice, Rhi::BindGroupLayoutHandle materialLayout,
                         Rhi::SamplerHandle sampler, aiMaterial* mat,
                         const std::string& texturesParentFolder, AssetRegistry& assets)
    : Material(mat)
{
    LoadTextures(mat, texturesParentFolder, assets);
    CreateBindGroup(rhiDevice, materialLayout, sampler);
}

void PBRMaterial::LoadTextures(aiMaterial* mat, const std::string& texturesParentFolder,
                               AssetRegistry& assets)
{
    aiString texturePath;

    mat->Get(AI_MATKEY_TWOSIDED, m_bTwoSided);
    m_MatData.bTwoSided = m_bTwoSided;

    mat->Get(AI_MATKEY_OPACITY, m_Opacity);
    m_MatData.Opacity = m_Opacity;

    // use BASE_COLOR if available, DIFFUSE as fallback
    // prefer texture, get value if texture not available
    aiColor4D baseColor;
    aiColor3D diffuse;
    if (mat->GetTexture(aiTextureType::aiTextureType_BASE_COLOR, 0, &texturePath) == AI_SUCCESS ||
        mat->GetTexture(aiTextureType::aiTextureType_DIFFUSE, 0, &texturePath) == AI_SUCCESS)
    {
        std::string path = texturesParentFolder + texturePath.C_Str();
        m_Albedo = assets.LoadTexture(path, Rhi::Format::RGBA8Srgb);
        m_MatData.bHasAlbedoTex = (m_Albedo != nullptr);
    }
    else if (mat->Get(AI_MATKEY_BASE_COLOR, baseColor) == AI_SUCCESS)
    {
        m_MatData.Albedo = {baseColor.r, baseColor.g, baseColor.b, baseColor.a};
    }
    else if (mat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse) == AI_SUCCESS)
    {
        m_MatData.Albedo = {diffuse.r, diffuse.g, diffuse.b, m_Opacity};
    }

    if (mat->GetTexture(aiTextureType::aiTextureType_NORMALS, 0, &texturePath) == AI_SUCCESS)
    {
        std::string path = texturesParentFolder + texturePath.C_Str();
        m_Normal = assets.LoadTexture(path, Rhi::Format::RGBA8Unorm);
        m_MatData.bHasNormalTex = (m_Normal != nullptr);
    }

    if (mat->GetTexture(aiTextureType::aiTextureType_GLTF_METALLIC_ROUGHNESS, 0, &texturePath) ==
        AI_SUCCESS)
    {
        std::string path = texturesParentFolder + texturePath.C_Str();
        m_MetallicRoughness = assets.LoadTexture(path, Rhi::Format::RGBA8Unorm);
        m_MatData.bHasMetallicRoughnessTex = (m_MetallicRoughness != nullptr);
    }

    mat->Get(AI_MATKEY_METALLIC_FACTOR, m_MatData.Metallic);
    mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, m_MatData.Roughness);
}

void PBRMaterial::CreateBindGroup(Rhi::IDevice& rhiDevice,
                                  Rhi::BindGroupLayoutHandle materialLayout,
                                  Rhi::SamplerHandle sampler)
{
    // A slot is filled only when the material has that map. The layout marks all
    // three optional, so an absent one stays empty and the shader branches on a
    // push constant rather than reading a placeholder.
    //
    // A fixed array rather than a vector: the layout bounds this at four, so the
    // count is known at compile time and a heap allocation per material buys
    // nothing. It also avoids a GCC 13 false positive -- at -O3 it inlines a
    // vector's push_back far enough to lose track of the reserved buffer and
    // reports -Wstringop-overflow against a destination it thinks may be null.
    std::array<Rhi::BindGroupBinding, EngineBindGroups::kMaterial.size()> bindings{};
    size_t count = 0u;

    const auto bind = [&](uint32_t slot, const std::shared_ptr<Texture>& texture)
    {
        if (texture)
        {
            bindings[count++] = Rhi::BindGroupBinding{
                .Slot = slot, .Type = Rhi::BindingType::Texture, .View = texture->GetView()};
        }
    };

    bind(TextureBinding::Albedo, m_Albedo);
    bind(TextureBinding::Normal, m_Normal);
    bind(TextureBinding::MetallicRoughness, m_MetallicRoughness);

    bindings[count++] =
        Rhi::BindGroupBinding{.Slot = 3u, .Type = Rhi::BindingType::Sampler, .Sampler = sampler};

    m_BindGroup = Rhi::UniqueHandle<Rhi::BindGroupHandle>(
        rhiDevice, rhiDevice.CreateBindGroup(Rhi::BindGroupDesc{
                       .Layout = materialLayout,
                       .Bindings = std::span(bindings).first(count),
                       .DebugName = std::format("{} Material Bind Group", m_Name)}));
}
