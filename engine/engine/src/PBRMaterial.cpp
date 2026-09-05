#include "PBRMaterial.h"

#include "assimp/material.h"

#include "AssetRegistry.h"

#include <rhi/vulkan/DebugNames.h>
#include <rhi/vulkan/VulkanNative.h>

using namespace Hikari;
using namespace Hikari::Rhi::Vulkan;

PBRMaterial::PBRMaterial(Rhi::IDevice& rhiDevice, DescriptorAllocator& descriptorAllocator,
                         vk::raii::DescriptorSetLayout& setLayout, Rhi::SamplerHandle sampler,
                         aiMaterial* mat, const std::string& texturesParentFolder,
                         AssetRegistry& assets)
    : Material(mat)
{
    LoadTextures(mat, texturesParentFolder, assets);
    CreateDescriptorSet(rhiDevice, descriptorAllocator, setLayout, sampler);
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

void PBRMaterial::CreateDescriptorSet(Rhi::IDevice& rhiDevice,
                                      DescriptorAllocator& descriptorAllocator,
                                      vk::raii::DescriptorSetLayout& materialSetLayout,
                                      Rhi::SamplerHandle sampler)
{
    // Descriptor writes still take raw Vulkan objects: the binding model is the
    // one part of Stage 5 that stays Vulkan-shaped (plan D7), so this is where
    // the handles are resolved back.
    vk::raii::Device& device = Rhi::Vulkan::GetDevice(rhiDevice);
    const vk::Sampler vkSampler = Rhi::Vulkan::GetSampler(rhiDevice, sampler);

    const auto viewOf = [&rhiDevice](const std::shared_ptr<Texture>& texture)
    {
        return texture ? Rhi::Vulkan::GetImageView(rhiDevice, texture->GetView()) : vk::ImageView{};
    };

    m_DescriptorSet = descriptorAllocator.Allocate(*materialSetLayout);
    SetVkDebugName(device, *m_DescriptorSet, vk::ObjectType::eDescriptorSet,
                   std::format("{} Material Descriptor Set", m_Name).c_str());

    vk::DescriptorImageInfo albedoInfo{.sampler = vkSampler,
                                       .imageView = viewOf(m_Albedo),
                                       .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo normalInfo{.sampler = vkSampler,
                                       .imageView = viewOf(m_Normal),
                                       .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo metallicRoughnessInfo{.sampler = vkSampler,
                                                  .imageView = viewOf(m_MetallicRoughness),
                                                  .imageLayout =
                                                      vk::ImageLayout::eShaderReadOnlyOptimal};

    std::vector<vk::WriteDescriptorSet> writeDescriptors;
    if (m_Albedo)
    {
        vk::WriteDescriptorSet albedoWriteSet{.dstSet = m_DescriptorSet,
                                              .dstBinding = TextureBinding::Albedo,
                                              .dstArrayElement = 0u,
                                              .descriptorCount = 1u,
                                              .descriptorType =
                                                  vk::DescriptorType::eCombinedImageSampler,
                                              .pImageInfo = &albedoInfo};
        writeDescriptors.push_back(albedoWriteSet);
    }

    if (m_Normal)
    {
        vk::WriteDescriptorSet normalWriteSet{.dstSet = m_DescriptorSet,
                                              .dstBinding = TextureBinding::Normal,
                                              .dstArrayElement = 0u,
                                              .descriptorCount = 1u,
                                              .descriptorType =
                                                  vk::DescriptorType::eCombinedImageSampler,
                                              .pImageInfo = &normalInfo};
        writeDescriptors.push_back(normalWriteSet);
    }

    if (m_MetallicRoughness)
    {
        vk::WriteDescriptorSet metallicRoughnessWriteSet{
            .dstSet = m_DescriptorSet,
            .dstBinding = TextureBinding::MetallicRoughness,
            .dstArrayElement = 0u,
            .descriptorCount = 1u,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &metallicRoughnessInfo};
        writeDescriptors.push_back(metallicRoughnessWriteSet);
    };

    device.updateDescriptorSets(writeDescriptors, {});
}
