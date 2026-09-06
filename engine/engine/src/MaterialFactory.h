#pragma once

#include <string>

#include <rhi/BindGroup.h>
#include <rhi/Handles.h>
#include <rhi/IDevice.h>
#include <rhi/UniqueHandle.h>

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

    Hikari::Rhi::BindGroupLayoutHandle GetLayout() const { return m_Layout.Get(); }

private:
    void CreateBindGroupLayout();

    Hikari::Rhi::IDevice& m_RhiDevice;
    Hikari::Rhi::SamplerHandle m_Sampler;
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::BindGroupLayoutHandle> m_Layout;
};
