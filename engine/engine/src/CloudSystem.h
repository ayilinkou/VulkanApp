#pragma once

#include <cstdint>
#include <vector>

#include "glm/glm.hpp"
#include "vulkan/vulkan_raii.hpp"

#include <platform/Paths.h>

#include <rhi/Barrier.h>
#include <rhi/Handles.h>
#include <rhi/ICommandList.h>
#include <rhi/IDevice.h>
#include <rhi/PipelineCache.h>
#include <rhi/UniqueHandle.h>

#include "Texture.h"

struct CloudSystemCreateInfo
{
    Hikari::Rhi::IDevice& RhiDevice;
    Hikari::Rhi::IPipelineCache& PipelineCache;
    const Hikari::Platform::Paths& ContentPaths;
    vk::DescriptorSetLayout GlobalSetLayout;
    vk::DescriptorSetLayout DepthSetLayout;
    vk::raii::CommandPool& CommandPool;
    vk::raii::Queue& ComputeQueue;
    uint32_t SwapchainWidth;
    uint32_t SwapchainHeight;
    uint32_t FramesInFlight;
};

class CloudSystem
{
private:
    struct CloudPushConstants
    {
        glm::vec3 WindVelocity = {0.05f, 0.f, 0.03f};
        float MinHeight = 1500.f;
        float MaxHeight = 4000.f;
        float Coverage = 0.2f;
        float Anisotropy = 0.3f;
        float BoundaryDisplacement = 300.f;
        uint32_t ViewStepCount = 64u;
        uint32_t SunStepCount = 6u;
    };

    struct BakeConstants
    {
        uint32_t Resolution;
        uint32_t WorleyPointsPerCell;
    };

public:
    CloudSystem(CloudSystemCreateInfo createInfo);

    /**
     * Returns the barriers recorded, so the caller can account for them in the
     * frame's totals.
     */
    Hikari::Rhi::BarrierCounts RecordDispatch(Hikari::Rhi::ICommandList& list, uint32_t frameIndex,
                                              vk::DescriptorSet globalSet,
                                              vk::DescriptorSet depthSet);
    void Resize(uint32_t width, uint32_t height);

    Hikari::Rhi::TextureViewHandle GetOutputView(uint8_t frameIndex) const
    {
        return m_OutputTextures[frameIndex].GetView();
    }

private:
    void Init(const CloudSystemCreateInfo& createInfo);

    void CreateOutputTextures(uint32_t width, uint32_t height);
    void CreateNoiseTexture();
    void CreateDescriptorSetLayout();
    void CreateBakeDescriptorSetLayout();
    void CreatePipeline(const Hikari::Platform::Paths& paths,
                        Hikari::Rhi::IPipelineCache& pipelineCache,
                        vk::DescriptorSetLayout globalSetLayout,
                        vk::DescriptorSetLayout depthSetLayout);
    void CreateBakePipeline(const Hikari::Platform::Paths& paths,
                            Hikari::Rhi::IPipelineCache& pipelineCache);
    void CreateDescriptorPool();
    void CreateBakeDescriptorPool();
    void AllocateDescriptorSets();
    void AllocateAndWriteBakeDescriptorSet();
    void WriteDescriptorSets();
    void BakeNoiseTexture(vk::raii::CommandPool& commandPool, vk::raii::Queue& computeQueue);
    void CreateTextureSampler();

private:
    static const uint32_t s_NOISE_RES;

    /**
     * Declared before every GPU resource below so that it outlives them: the
     * handles they hold are released through it.
     */
    Hikari::Rhi::IDevice& m_RhiDevice;

    /**
     * Borrowed from m_RhiDevice. Still needed because pipelines and descriptors
     * stay Vulkan-shaped for the whole of Stage 5 (plan D7, D8).
     */
    vk::raii::Device& m_Device;

    vk::raii::DescriptorSetLayout m_SetLayout = nullptr;
    vk::raii::DescriptorSetLayout m_BakeSetLayout = nullptr;
    vk::raii::DescriptorPool m_DescriptorPool = nullptr;
    vk::raii::DescriptorPool m_BakeDescriptorPool = nullptr;
    vk::raii::PipelineLayout m_PipelineLayout = nullptr;
    vk::raii::PipelineLayout m_BakePipelineLayout = nullptr;
    vk::raii::Pipeline m_Pipeline = nullptr;
    vk::raii::Pipeline m_BakePipeline = nullptr;
    std::vector<vk::raii::DescriptorSet> m_DescriptorSets;
    vk::raii::DescriptorSet m_BakeDescriptorSet = nullptr;
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::SamplerHandle> m_TextureSampler;

    std::vector<Texture> m_OutputTextures;
    Texture m_PerlinWorley;

    const uint32_t m_FramesInFlight;
    uint32_t m_Width;
    uint32_t m_Height;
    uint32_t m_OutputWidth;
    uint32_t m_OutputHeight;

    CloudPushConstants m_CloudData{};
};
