#include "MaterialFactory.h"

#include <array>

#include <core/Log.h>
#include <rhi/vulkan/DebugNames.h>
#include <rhi/vulkan/VulkanNative.h>

#include "Texture.h"

using namespace Hikari;
using namespace Hikari::Core;
using namespace Hikari::Rhi::Vulkan;

constexpr LogCategory LogMaterialFactory("Material Factory");

/** A material set binds one combined image sampler per texture slot. */
constexpr std::array kMaterialDescriptorsPerSet = {vk::DescriptorPoolSize{
    .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = TextureBinding::COUNT}};

/**
 * A starting size, not a ceiling — the allocator adds pools when a scene brings
 * more materials than this. Sized so that no scene shipped today pays for a
 * second pool.
 */
constexpr uint32_t kInitialMaterialSetCapacity = 100u;

MaterialFactory* MaterialFactory::s_Instance = nullptr;

MaterialFactory::MaterialFactory(Rhi::IDevice& rhiDevice, Rhi::SamplerHandle sampler)
    : m_RhiDevice(rhiDevice), m_Device(Rhi::Vulkan::GetDevice(rhiDevice)), m_Sampler(sampler),
      m_DescriptorAllocator(m_Device, kMaterialDescriptorsPerSet, kInitialMaterialSetCapacity,
                            "Material Factory")
{
    CreateDescriptorSetLayout();
}

void MaterialFactory::Init(Rhi::IDevice& rhiDevice, Rhi::SamplerHandle sampler)
{
    LogMsg(LogSeverity::Info, LogMaterialFactory, "Init()");

    if (s_Instance)
        throw std::runtime_error("MaterialFactory singleton is already initialised!");
    s_Instance = new MaterialFactory(rhiDevice, sampler);
}

void MaterialFactory::Shutdown()
{
    LogMsg(LogSeverity::Info, LogMaterialFactory, "Shutdown()");

    if (!s_Instance)
        throw std::runtime_error("Attempting to shutdown MaterialFactory when it is already null!");

    delete s_Instance;
    s_Instance = nullptr;
}

void MaterialFactory::CreateDescriptorSetLayout()
{
    std::array matBindings = {
        vk::DescriptorSetLayoutBinding(TextureBinding::Albedo,
                                       vk::DescriptorType::eCombinedImageSampler, 1,
                                       vk::ShaderStageFlagBits::eFragment),
        vk::DescriptorSetLayoutBinding(TextureBinding::Normal,
                                       vk::DescriptorType::eCombinedImageSampler, 1,
                                       vk::ShaderStageFlagBits::eFragment),
        vk::DescriptorSetLayoutBinding(TextureBinding::MetallicRoughness,
                                       vk::DescriptorType::eCombinedImageSampler, 1,
                                       vk::ShaderStageFlagBits::eFragment)};

    std::array<vk::DescriptorBindingFlags, 3> bindingFlags = {
        vk::DescriptorBindingFlagBits::ePartiallyBound,
        vk::DescriptorBindingFlagBits::ePartiallyBound,
        vk::DescriptorBindingFlagBits::ePartiallyBound};

    vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
        .bindingCount = static_cast<uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data()};

    vk::DescriptorSetLayoutCreateInfo matCreateInfo{.pNext = &flagsInfo,
                                                    .bindingCount =
                                                        static_cast<uint32_t>(matBindings.size()),
                                                    .pBindings = matBindings.data()};

    m_SetLayout = vk::raii::DescriptorSetLayout(m_Device, matCreateInfo);
    SetVkDebugName(m_Device, *m_SetLayout, vk::ObjectType::eDescriptorSetLayout,
                   "Material Factory Descriptor Set Layout");
}

PBRMaterial* MaterialFactory::CreatePBRMaterial(aiMaterial* mat,
                                                const std::string& texturesParentFolder,
                                                AssetRegistry& assets)
{
    return new PBRMaterial(m_RhiDevice, m_DescriptorAllocator, m_SetLayout, m_Sampler, mat,
                           texturesParentFolder, assets);
}
