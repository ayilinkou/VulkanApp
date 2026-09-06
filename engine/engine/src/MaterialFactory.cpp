#include "MaterialFactory.h"

#include <array>

#include <core/Log.h>

#include "BindGroupLayouts.h"
#include "Texture.h"

using namespace Hikari;
using namespace Hikari::Core;

constexpr LogCategory LogMaterialFactory("Material Factory");

/** A material set binds one combined image sampler per texture slot. */
/**
 * A starting size, not a ceiling — the allocator adds pools when a scene brings
 * more materials than this. Sized so that no scene shipped today pays for a
 * second pool.
 */
MaterialFactory::MaterialFactory(Rhi::IDevice& rhiDevice, Rhi::SamplerHandle sampler)
    : m_RhiDevice(rhiDevice), m_Sampler(sampler)
{
    LogMsg(LogSeverity::Info, LogMaterialFactory, "Constructed");
    CreateBindGroupLayout();
}

MaterialFactory::~MaterialFactory()
{
    LogMsg(LogSeverity::Info, LogMaterialFactory, "Destroyed");
}

void MaterialFactory::CreateBindGroupLayout()
{
    m_Layout = Rhi::UniqueHandle<Rhi::BindGroupLayoutHandle>(
        m_RhiDevice, m_RhiDevice.CreateBindGroupLayout(Rhi::BindGroupLayoutDesc{
                         .Bindings = EngineBindGroups::kMaterial, .DebugName = "Material Layout"}));
}

PBRMaterial* MaterialFactory::CreatePBRMaterial(aiMaterial* mat,
                                                const std::string& texturesParentFolder,
                                                AssetRegistry& assets)
{
    return new PBRMaterial(m_RhiDevice, m_Layout.Get(), m_Sampler, mat, texturesParentFolder,
                           assets);
}
