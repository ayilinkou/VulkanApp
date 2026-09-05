#pragma once

#include <string>

#include <rhi/Handles.h>
#include <rhi/IDevice.h>
#include <rhi/vulkan/DescriptorAllocator.h>

#include "Material.h"
#include "PBRMaterial.h"

struct aiMaterial;

class AssetRegistry;

/**
 * Builds materials and owns the descriptor machinery they are allocated from.
 *
 * Engine-owned rather than registry-owned, though a model's materials are built
 * during a load: its descriptor set layout is a renderer input, wanted by the
 * pipeline layouts as much as by the materials. Handing it out from the asset
 * layer would put a Vulkan descriptor set layout on that layer's public surface
 * for the sake of one caller on the other side of the line.
 */
class MaterialFactory
{
public:
    MaterialFactory(Hikari::Rhi::IDevice& rhiDevice, Hikari::Rhi::SamplerHandle sampler);
    ~MaterialFactory();

    MaterialFactory(const MaterialFactory&) = delete;
    MaterialFactory& operator=(const MaterialFactory&) = delete;
    MaterialFactory(MaterialFactory&&) = delete;
    MaterialFactory& operator=(MaterialFactory&&) = delete;

    /**
     * The registry is passed through rather than held: a material's textures
     * load through whichever registry asked for the model, and this factory
     * serves whichever one that is.
     */
    [[nodiscard]] PBRMaterial* CreatePBRMaterial(aiMaterial* mat,
                                                 const std::string& texturesParentFolder,
                                                 AssetRegistry& assets);

    vk::DescriptorSetLayout GetDescriptorSetLayout() const { return *m_SetLayout; }

private:
    void CreateDescriptorSetLayout();

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
