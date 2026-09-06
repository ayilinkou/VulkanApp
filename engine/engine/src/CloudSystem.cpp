#include "CloudSystem.h"
#include <core/Log.h>
#include <rhi/BarrierPresets.h>
#include <rhi/ICommandList.h>
#include <rhi/vulkan/CommandListUtil.h>
#include <rhi/vulkan/VulkanNative.h>

#include <platform/FileSystem.h>

#include "BindGroupLayouts.h"

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
    CreateBindGroupLayouts();
    CreatePipeline(createInfo.ContentPaths, createInfo.PipelineCache, createInfo.GlobalSetLayout,
                   createInfo.DepthSetLayout);
    CreateBakePipeline(createInfo.ContentPaths, createInfo.PipelineCache);
    CreateBindGroups();
    BakeNoiseTexture(createInfo.CommandPool, createInfo.ComputeQueue);
}

void CloudSystem::Resize(uint32_t width, uint32_t height)
{
    CreateOutputTextures(width, height);

    // The groups name the output textures, and a bind group is immutable, so new
    // targets mean new groups (RHI plan D20). Safe here because the caller
    // resizes only after waiting for the device to go idle.
    m_BindGroups.clear();
    CreateBindGroups();
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

void CloudSystem::CreateBindGroupLayouts()
{
    m_SetLayout = Rhi::UniqueHandle<Rhi::BindGroupLayoutHandle>(
        m_RhiDevice,
        m_RhiDevice.CreateBindGroupLayout(Rhi::BindGroupLayoutDesc{
            .Bindings = EngineBindGroups::kCloudDispatch, .DebugName = "Cloud Dispatch Layout"}));

    m_BakeSetLayout = Rhi::UniqueHandle<Rhi::BindGroupLayoutHandle>(
        m_RhiDevice,
        m_RhiDevice.CreateBindGroupLayout(Rhi::BindGroupLayoutDesc{
            .Bindings = EngineBindGroups::kCloudBake, .DebugName = "Cloud Bake Layout"}));
}

void CloudSystem::CreatePipeline(const Paths& paths, Rhi::IPipelineCache& pipelineCache,
                                 Rhi::BindGroupLayoutHandle globalLayout,
                                 Rhi::BindGroupLayoutHandle depthLayout)
{
    const std::array bindGroupLayouts{globalLayout, depthLayout, m_SetLayout.Get()};
    const std::array pushRanges{Rhi::PushConstantRange{.Stages = Rhi::ShaderStage::Compute,
                                                       .Size = sizeof(CloudPushConstants)}};

    m_PipelineLayout = Rhi::UniqueHandle<Rhi::PipelineLayoutHandle>(
        m_RhiDevice, m_RhiDevice.CreatePipelineLayout(
                         Rhi::PipelineLayoutDesc{.BindGroupLayouts = bindGroupLayouts,
                                                 .PushConstantRanges = pushRanges,
                                                 .DebugName = "Clouds Layout"}));

    m_Pipeline = Rhi::UniqueHandle<Rhi::ComputePipelineHandle>(
        m_RhiDevice,
        m_RhiDevice.CreateComputePipeline(
            Rhi::ComputePipelineDesc{.Layout = m_PipelineLayout.Get(),
                                     .Shader = {LoadShader(paths, "clouds.comp"), "main"},
                                     .DebugName = "Clouds"},
            pipelineCache));
}

void CloudSystem::CreateBakePipeline(const Paths& paths, Rhi::IPipelineCache& pipelineCache)
{
    const std::array bindGroupLayouts{m_BakeSetLayout.Get()};
    const std::array pushRanges{
        Rhi::PushConstantRange{.Stages = Rhi::ShaderStage::Compute, .Size = sizeof(BakeConstants)}};

    m_BakePipelineLayout = Rhi::UniqueHandle<Rhi::PipelineLayoutHandle>(
        m_RhiDevice, m_RhiDevice.CreatePipelineLayout(
                         Rhi::PipelineLayoutDesc{.BindGroupLayouts = bindGroupLayouts,
                                                 .PushConstantRanges = pushRanges,
                                                 .DebugName = "Bake Perlin Worley Layout"}));

    m_BakePipeline = Rhi::UniqueHandle<Rhi::ComputePipelineHandle>(
        m_RhiDevice,
        m_RhiDevice.CreateComputePipeline(
            Rhi::ComputePipelineDesc{.Layout = m_BakePipelineLayout.Get(),
                                     .Shader = {LoadShader(paths, "bakePerlinWorley.comp"), "main"},
                                     .DebugName = "Bake Perlin Worley"},
            pipelineCache));
}

/**
 * The compiled shader named `name`, kept alive for the run.
 *
 * Same arrangement as the renderer's: the caller resolves the file and the
 * device says which kind it reads (plan D24).
 */
Rhi::ShaderModuleHandle CloudSystem::LoadShader(const Paths& paths, const std::string& name)
{
    const std::string file = std::format("{}.{}", name, m_RhiDevice.GetCaps().ShaderExtension);
    const std::vector<char> code = Platform::ReadFile(paths.Shader(file).string());

    m_ShaderModules.push_back(Rhi::UniqueHandle<Rhi::ShaderModuleHandle>(
        m_RhiDevice, m_RhiDevice.CreateShaderModule(Rhi::ShaderModuleDesc{
                         .Bytes = std::as_bytes(std::span(code)), .DebugName = file})));

    return m_ShaderModules.back().Get();
}

void CloudSystem::CreateBindGroups()
{
    for (uint32_t i = 0u; i < m_FramesInFlight; i++)
    {
        const std::array bindings{
            Rhi::BindGroupBinding{.Slot = 0u,
                                  .Type = Rhi::BindingType::UnorderedAccessTexture,
                                  .View = m_OutputTextures[i].GetView()},
            Rhi::BindGroupBinding{
                .Slot = 1u, .Type = Rhi::BindingType::Texture, .View = m_PerlinWorley.GetView()},
            Rhi::BindGroupBinding{
                .Slot = 2u, .Type = Rhi::BindingType::Sampler, .Sampler = m_TextureSampler.Get()}};

        m_BindGroups.push_back(Rhi::UniqueHandle<Rhi::BindGroupHandle>(
            m_RhiDevice, m_RhiDevice.CreateBindGroup(Rhi::BindGroupDesc{
                             .Layout = m_SetLayout.Get(),
                             .Bindings = bindings,
                             .DebugName = std::format("Cloud Dispatch Frame {}", i)})));
    }

    const std::array bakeBindings{
        Rhi::BindGroupBinding{.Slot = 0u,
                              .Type = Rhi::BindingType::UnorderedAccessTexture,
                              .View = m_PerlinWorley.GetView()}};

    m_BakeBindGroup = Rhi::UniqueHandle<Rhi::BindGroupHandle>(
        m_RhiDevice,
        m_RhiDevice.CreateBindGroup(Rhi::BindGroupDesc{
            .Layout = m_BakeSetLayout.Get(), .Bindings = bakeBindings, .DebugName = "Cloud Bake"}));
}

/**
 * Records the cloud dispatch into a list the caller owns, began and will end.
 *
 * The barriers around it are the point of returning counts: the output volume is
 * written here and sampled by the composite pass, so both transitions belong to
 * this pass rather than to whoever submits it.
 */
Rhi::BarrierCounts CloudSystem::RecordDispatch(Rhi::ICommandList& list, uint32_t frameIndex,
                                               Rhi::BindGroupHandle globalGroup,
                                               Rhi::BindGroupHandle depthGroup)
{
    Rhi::BarrierCounts barrierCounts =
        list.Barrier(Rhi::BarrierPresets::UndefinedToUnorderedAccess().On(
            m_OutputTextures[frameIndex].GetHandle()));

    list.SetPipeline(m_Pipeline.Get());
    list.SetComputeBindGroup(m_PipelineLayout.Get(), 0u, globalGroup);
    list.SetComputeBindGroup(m_PipelineLayout.Get(), 1u, depthGroup);
    list.SetComputeBindGroup(m_PipelineLayout.Get(), 2u, m_BindGroups[frameIndex].Get());

    list.PushConstants(m_PipelineLayout.Get(), Rhi::ShaderStage::Compute, 0u,
                       std::as_bytes(std::span(&m_CloudData, 1)));

    // matches numthreads(8,8,1)
    list.Dispatch((m_OutputWidth + 7) / 8, (m_OutputHeight + 7) / 8, 1u);

    barrierCounts += list.Barrier(Rhi::BarrierPresets::UnorderedAccessToShaderResource().On(
        m_OutputTextures[frameIndex].GetHandle()));

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

    list->SetPipeline(m_BakePipeline.Get());
    list->SetComputeBindGroup(m_BakePipelineLayout.Get(), 0u, m_BakeBindGroup.Get());

    const BakeConstants bc{.Resolution = s_NOISE_RES, .WorleyPointsPerCell = 1};
    list->PushConstants(m_BakePipelineLayout.Get(), Rhi::ShaderStage::Compute, 0u,
                        std::as_bytes(std::span(&bc, 1)));

    // matches numthreads(4,4,4)
    list->Dispatch(s_NOISE_RES / 4, s_NOISE_RES / 4, s_NOISE_RES / 4);

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
