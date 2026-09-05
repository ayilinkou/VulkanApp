#pragma once

#include <string>

#include <rhi/Handles.h>
#include <rhi/IDevice.h>
#include <rhi/vulkan/DescriptorAllocator.h>

#include "Material.h"
#include "PBRMaterial.h"

struct aiMaterial;

class AssetRegistry;

class MaterialFactory
{
public:
    static void Init(Hikari::Rhi::IDevice& rhiDevice, Hikari::Rhi::SamplerHandle sampler);
    static void Shutdown();

    static MaterialFactory* Get() { return s_Instance; }

    /**
     * The registry is passed through rather than held: a material's textures
     * load through whichever registry asked for the model, and this factory is
     * still a singleton shared by all of them until it too is injected.
     */
    [[nodiscard]] PBRMaterial* CreatePBRMaterial(aiMaterial* mat,
                                                 const std::string& texturesParentFolder,
                                                 AssetRegistry& assets);

    vk::DescriptorSetLayout GetDescriptorSetLayout() const { return *m_SetLayout; }

private:
    MaterialFactory(Hikari::Rhi::IDevice& rhiDevice, Hikari::Rhi::SamplerHandle sampler);

    void CreateDescriptorSetLayout();

private:
    static MaterialFactory* s_Instance;

    vk::raii::DescriptorSetLayout m_SetLayout = nullptr;

    Hikari::Rhi::IDevice& m_RhiDevice;

    /**
     * Descriptor pools and set layouts stay Vulkan objects for the whole of
     * Stage 5 — the binding model is deliberately not abstracted (plan D7) — so
     * the factory keeps a device reference to build them from.
     */
    vk::raii::Device& m_Device;

    Hikari::Rhi::SamplerHandle m_Sampler;

    /**
     * Declared after m_Device because it is constructed from that reference,
     * and members are initialized in declaration order.
     */
    Hikari::Rhi::Vulkan::DescriptorAllocator m_DescriptorAllocator;
};
