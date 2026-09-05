#include "CloudSystem.h"
#include <core/Log.h>
#include <rhi/BarrierPresets.h>
#include <rhi/ICommandList.h>
#include <rhi/vulkan/CommandListUtil.h>
#include <rhi/vulkan/ComputePipelineBuilder.h>
#include <rhi/vulkan/DebugNames.h>
#include <rhi/vulkan/VulkanNative.h>

using namespace Hikari;
using namespace Hikari::Core;
using namespace Hikari::Platform;
using namespace Hikari::Rhi::Vulkan;

inline constexpr LogCategory LogCloudSystem{"Cloud System"};

const uint32_t CloudSystem::s_NOISE_RES = 128u;

CloudSystem::CloudSystem(CloudSystemCreateInfo createInfo)
    : m_RhiDevice(createInfo.RhiDevice), m_Device(Rhi::Vulkan::GetDevice(createInfo.RhiDevice)),
      m_FramesInFlight(createInfo.FramesInFlight)
{
    Init(createInfo);
}

void CloudSystem::Init(const CloudSystemCreateInfo& createInfo)
{
    LogMsg(LogSeverity::Info, LogCloudSystem, "Init()");

    CreateTextureSampler();
    CreateOutputTextures(createInfo.SwapchainWidth, createInfo.SwapchainHeight);
    CreateNoiseTexture();
    CreateDescriptorPool();
    CreateDescriptorSetLayout();
    CreatePipeline(createInfo.ContentPaths, createInfo.PipelineCache, createInfo.GlobalSetLayout,
                   createInfo.DepthSetLayout);
    AllocateDescriptorSets();
    WriteDescriptorSets();

    CreateBakeDescriptorPool();
    CreateBakeDescriptorSetLayout();
    CreateBakePipeline(createInfo.ContentPaths, createInfo.PipelineCache);
    AllocateAndWriteBakeDescriptorSet();
    BakeNoiseTexture(createInfo.CommandPool, createInfo.ComputeQueue);
}

void CloudSystem::Resize(uint32_t width, uint32_t height)
{
    CreateOutputTextures(width, height);
    WriteDescriptorSets();
}

void CloudSystem::CreateOutputTextures(uint32_t width, uint32_t height)
{
    LogMsg(LogSeverity::Info, LogCloudSystem, "CreateOutputImages()");

    m_OutputTextures.clear();

    m_Width = width;
    m_Height = height;

    const uint32_t resFactor = 4u;
    m_OutputWidth = std::max(1u, width / resFactor);
    m_OutputHeight = std::max(1u, height / resFactor);

    for (uint32_t i = 0; i < m_FramesInFlight; ++i)
    {
        m_OutputTextures.emplace_back(
            m_RhiDevice,
            Rhi::TextureDesc{.Format = Rhi::Format::RGBA16Float,
                             .Extent = {m_OutputWidth, m_OutputHeight, 1u},
                             .Usage = Rhi::TextureUsage::Storage | Rhi::TextureUsage::Sampled,
                             .DebugName = std::format("Frame_{} Cloud Output Image", i)},
            Rhi::TextureViewDimension::Texture2D);
    }
}

void CloudSystem::CreateDescriptorSetLayout()
{
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{
        {{0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
         {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute}}};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    m_SetLayout = vk::raii::DescriptorSetLayout(m_Device, layoutInfo);
}

void CloudSystem::CreateBakeDescriptorSetLayout()
{
    std::array<vk::DescriptorSetLayoutBinding, 1> bindings{{
        {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
    }};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    m_BakeSetLayout = vk::raii::DescriptorSetLayout(m_Device, layoutInfo);
}

void CloudSystem::CreatePipeline(const Paths& paths, Rhi::IPipelineCache& pipelineCache,
                                 vk::raii::DescriptorSetLayout& globalSetLayout,
                                 vk::raii::DescriptorSetLayout& depthSetLayout)
{
    std::array setLayouts = {*globalSetLayout, *depthSetLayout, *m_SetLayout};
    std::array<vk::PushConstantRange, 1> pushRanges = {
        vk::PushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                              .offset = 0,
                              .size = sizeof(CloudPushConstants)}};

    auto [layout, pipeline] = ComputePipelineBuilder(m_Device)
                                  .Shader(paths.Shader("clouds.comp.spv").string())
                                  .Layout(setLayouts, pushRanges)
                                  .DebugName("Clouds")
                                  .Cache(pipelineCache)
                                  .Build();

    m_PipelineLayout = std::move(layout);
    m_Pipeline = std::move(pipeline);
}

void CloudSystem::CreateBakePipeline(const Paths& paths, Rhi::IPipelineCache& pipelineCache)
{
    std::array<vk::DescriptorSetLayout, 1> setLayouts = {*m_BakeSetLayout};
    std::array<vk::PushConstantRange, 1> pushRanges = {
        vk::PushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                              .offset = 0,
                              .size = sizeof(BakeConstants)}};

    auto [layout, pipeline] = ComputePipelineBuilder(m_Device)
                                  .Shader(paths.Shader("bakePerlinWorley.comp.spv").string())
                                  .Layout(setLayouts, pushRanges)
                                  .DebugName("Bake Perlin Worley")
                                  .Cache(pipelineCache)
                                  .Build();

    m_BakePipelineLayout = std::move(layout);
    m_BakePipeline = std::move(pipeline);
}

void CloudSystem::CreateDescriptorPool()
{
    std::array<vk::DescriptorPoolSize, 2> poolSizes{
        {{vk::DescriptorType::eStorageImage, m_FramesInFlight},
         {vk::DescriptorType::eCombinedImageSampler, m_FramesInFlight}}};

    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = m_FramesInFlight,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    m_DescriptorPool = vk::raii::DescriptorPool(m_Device, poolInfo);
}

void CloudSystem::CreateBakeDescriptorPool()
{
    std::array<vk::DescriptorPoolSize, 1> poolSizes{{
        {vk::DescriptorType::eStorageImage, 1},
    }};

    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    m_BakeDescriptorPool = vk::raii::DescriptorPool(m_Device, poolInfo);
}

void CloudSystem::AllocateDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(m_FramesInFlight, *m_SetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *m_DescriptorPool,
        .descriptorSetCount = m_FramesInFlight,
        .pSetLayouts = layouts.data(),
    };
    m_DescriptorSets = vk::raii::DescriptorSets(m_Device, allocInfo);
}

void CloudSystem::AllocateAndWriteBakeDescriptorSet()
{
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *m_BakeDescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*m_BakeSetLayout,
    };
    m_BakeDescriptorSet = std::move(vk::raii::DescriptorSets(m_Device, allocInfo).front());

    vk::DescriptorImageInfo noiseImageInfo{
        .imageView = Rhi::Vulkan::GetImageView(m_RhiDevice, m_PerlinWorley.GetView()),
        .imageLayout = vk::ImageLayout::eGeneral,
    };

    vk::WriteDescriptorSet writeDescSet{
        .dstSet = *m_BakeDescriptorSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = &noiseImageInfo,
    };
    m_Device.updateDescriptorSets(writeDescSet, {});
}

void CloudSystem::WriteDescriptorSets()
{
    LogMsg(LogSeverity::Info, LogCloudSystem, "WriteDescriptorSets()");

    for (uint32_t i = 0u; i < m_FramesInFlight; i++)
    {
        vk::DescriptorImageInfo storageImageInfo{
            .imageView = Rhi::Vulkan::GetImageView(m_RhiDevice, m_OutputTextures[i].GetView()),
            .imageLayout = vk::ImageLayout::eGeneral,
        };
        vk::DescriptorImageInfo perlinWorleyImageInfo{
            .sampler = Rhi::Vulkan::GetSampler(m_RhiDevice, m_TextureSampler.Get()),
            .imageView = Rhi::Vulkan::GetImageView(m_RhiDevice, m_PerlinWorley.GetView()),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

        std::array<vk::WriteDescriptorSet, 2> writeDescSet{
            vk::WriteDescriptorSet{.dstSet = *m_DescriptorSets[i],
                                   .dstBinding = 0,
                                   .dstArrayElement = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &storageImageInfo},
            vk::WriteDescriptorSet{
                .dstSet = *m_DescriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &perlinWorleyImageInfo,
            }};

        m_Device.updateDescriptorSets(writeDescSet, {});
    }
}

Rhi::BarrierCounts CloudSystem::RecordDispatch(vk::raii::CommandBuffer& cmd, uint32_t frameIndex,
                                               vk::raii::DescriptorSet& globalSet,
                                               vk::raii::DescriptorSet& depthSet)
{
    std::unique_ptr<Rhi::ICommandList> list = Rhi::Vulkan::WrapCommandList(m_RhiDevice, *cmd);
    list->Begin();

    Rhi::BarrierCounts barrierCounts =
        list->Barrier(Rhi::BarrierPresets::UndefinedToUnorderedAccess().On(
            m_OutputTextures[frameIndex].GetHandle()));

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *m_Pipeline);
    std::array<vk::DescriptorSet, 3> sets = {*globalSet, *depthSet, *m_DescriptorSets[frameIndex]};
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_PipelineLayout, 0, sets, {});

    cmd.pushConstants<CloudPushConstants>(*m_PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                          m_CloudData);

    cmd.dispatch((m_OutputWidth + 7) / 8, (m_OutputHeight + 7) / 8, 1);

    barrierCounts += list->Barrier(Rhi::BarrierPresets::UnorderedAccessToShaderResource().On(
        m_OutputTextures[frameIndex].GetHandle()));

    list->End();

    return barrierCounts;
}

void CloudSystem::CreateNoiseTexture()
{
    m_PerlinWorley =
        Texture(m_RhiDevice,
                Rhi::TextureDesc{.Dimension = Rhi::TextureDimension::Texture3D,
                                 .Format = Rhi::Format::R8Unorm,
                                 .Extent = {s_NOISE_RES, s_NOISE_RES, s_NOISE_RES},
                                 .Usage = Rhi::TextureUsage::Storage | Rhi::TextureUsage::Sampled,
                                 .DebugName = "Perlin Worley Image"},
                Rhi::TextureViewDimension::Texture3D);
}

void CloudSystem::BakeNoiseTexture(vk::raii::CommandPool& commandPool,
                                   vk::raii::Queue& computeQueue)
{
    LogMsg(LogSeverity::Info, LogCloudSystem, "Baking perlin worley texture ({}x{}x{})",
           s_NOISE_RES, s_NOISE_RES, s_NOISE_RES);

    vk::raii::CommandBuffer cmd = BeginSingleTimeCommand(m_Device, commandPool);
    std::unique_ptr<Rhi::ICommandList> list = Rhi::Vulkan::WrapCommandList(m_RhiDevice, *cmd);

    list->Barrier(Rhi::BarrierPresets::UndefinedToUnorderedAccess().On(m_PerlinWorley.GetHandle()));

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *m_BakePipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_BakePipelineLayout, 0,
                           *m_BakeDescriptorSet, {});

    BakeConstants bc{.Resolution = s_NOISE_RES, .WorleyPointsPerCell = 1};
    cmd.pushConstants<BakeConstants>(*m_BakePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                     bc);

    cmd.dispatch(s_NOISE_RES / 4, s_NOISE_RES / 4,
                 s_NOISE_RES / 4); // matches numthreads(4,4,4)

    // The bake's result is read by another dispatch, not by the main pass, so
    // the destination stage is compute rather than the preset's default.
    list->Barrier(
        Rhi::BarrierPresets::UnorderedAccessToShaderResource(Rhi::PipelineStage::ComputeStage)
            .On(m_PerlinWorley.GetHandle()));

    // TODO: move to a read only image
    EndSingleTimeCommand(cmd, computeQueue);
}

void CloudSystem::CreateTextureSampler()
{
    LogMsg(LogSeverity::Info, LogCloudSystem, "CreateTextureSampler()");

    m_TextureSampler = Rhi::UniqueHandle<Rhi::SamplerHandle>(
        m_RhiDevice,
        m_RhiDevice.CreateSampler(Rhi::SamplerDesc{.DebugName = "Cloud System Texture Sampler"}));
}
