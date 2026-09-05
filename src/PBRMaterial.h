#pragma once

#include <memory>
#include <string>

#include "glm/glm.hpp"

#include <rhi/Handles.h>
#include <rhi/IDevice.h>
#include <rhi/vulkan/DescriptorAllocator.h>

#include "Material.h"
#include "Texture.h"

struct aiMaterial;

class AssetRegistry;

class PBRMaterial : public Material
{
public:
    PBRMaterial(Hikari::Rhi::IDevice& rhiDevice,
                Hikari::Rhi::Vulkan::DescriptorAllocator& descriptorAllocator,
                vk::raii::DescriptorSetLayout& setLayout, Hikari::Rhi::SamplerHandle sampler,
                aiMaterial* mat, const std::string& texturesParentFolder, AssetRegistry& assets);

    virtual void* GetPushConstantData() override { return &m_MatData; }

private:
    void LoadTextures(aiMaterial* mat, const std::string& texturesParentFolder,
                      AssetRegistry& assets);
    void CreateDescriptorSet(Hikari::Rhi::IDevice& rhiDevice,
                             Hikari::Rhi::Vulkan::DescriptorAllocator& descriptorAllocator,
                             vk::raii::DescriptorSetLayout& setLayout,
                             Hikari::Rhi::SamplerHandle sampler);

public:
    struct MaterialData
    {
        glm::vec4 Albedo{1.f, 0.f, 1.f, 1.f};
        float Metallic = 0.f;
        float Roughness = 1.f;
        float AO = 1.f;
        float Opacity = 1.f; // might be redundant, can pack into albedo
        int bHasAlbedoTex = false;
        int bHasNormalTex = false;
        int bHasMetallicRoughnessTex = false;
        int bTwoSided = false;
    };

private:
    std::shared_ptr<Texture> m_Albedo = nullptr;
    std::shared_ptr<Texture> m_Normal = nullptr;
    std::shared_ptr<Texture> m_MetallicRoughness = nullptr;

    MaterialData m_MatData{};
};
