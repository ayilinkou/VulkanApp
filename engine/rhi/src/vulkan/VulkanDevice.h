#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "vulkan/vulkan_raii.hpp"

#include <core/HandlePool.h>

#include <rhi/BufferDesc.h>
#include <rhi/DeviceDesc.h>
#include <rhi/Diagnostics.h>
#include <rhi/Handles.h>
#include <rhi/IDevice.h>
#include <rhi/IPresentTarget.h>
#include <rhi/PipelineCache.h>
#include <rhi/RhiTypes.h>
#include <rhi/SamplerDesc.h>
#include <rhi/TextureDesc.h>
#include <rhi/TextureViewDesc.h>
#include <rhi/UploadContext.h>

#include "vulkan/DescriptorAllocator.h"
#include "vulkan/OwnershipTransfer.h"
#include "vulkan/QueueFamilies.h"
#include "vulkan/VulkanAllocator.h"
#include "vulkan/VulkanBindGroup.h"
#include "vulkan/VulkanBuffer.h"
#include "vulkan/VulkanFence.h"
#include "vulkan/VulkanPipeline.h"
#include "vulkan/VulkanSampler.h"
#include "vulkan/VulkanSemaphore.h"
#include "vulkan/VulkanTexture.h"
#include "vulkan/VulkanTextureView.h"

namespace Hikari::Rhi::Vulkan
{
class VulkanDevice final : public IDevice
{
public:
    explicit VulkanDevice(const DeviceDesc& desc);
    ~VulkanDevice() override;

    const DeviceCaps& GetCaps() const override { return m_Caps; }
    Diagnostics& GetDiagnostics() override { return *m_pDiagnostics; }
    void WaitIdle() override;

    BufferHandle CreateBuffer(const BufferDesc& desc) override;
    void Destroy(BufferHandle handle) override;
    void* GetMappedData(BufferHandle handle) override;
    uint32_t GetLiveBufferCount() const override { return m_Buffers.Size(); }

    TextureHandle CreateTexture(const TextureDesc& desc) override;
    void Destroy(TextureHandle handle) override;
    TextureViewHandle CreateTextureView(const TextureViewDesc& desc) override;
    void Destroy(TextureViewHandle handle) override;
    SamplerHandle CreateSampler(const SamplerDesc& desc) override;
    void Destroy(SamplerHandle handle) override;
    const TextureDesc* GetTextureDesc(TextureHandle handle) const override;
    uint32_t GetLiveTextureCount() const override { return m_Textures.Size(); }
    uint32_t GetLiveTextureViewCount() const override { return m_TextureViews.Size(); }
    uint32_t GetLiveSamplerCount() const override { return m_Samplers.Size(); }

    [[nodiscard]] std::unique_ptr<IUploadContext>
    CreateUploadContext(const UploadContextDesc& desc) override;

    [[nodiscard]] std::unique_ptr<ICommandAllocator>
    CreateCommandAllocator(const CommandAllocatorDesc& desc) override;

    BindGroupLayoutHandle CreateBindGroupLayout(const BindGroupLayoutDesc& desc) override;
    void Destroy(BindGroupLayoutHandle handle) override;
    BindGroupHandle CreateBindGroup(const BindGroupDesc& desc) override;
    void Destroy(BindGroupHandle handle) override;
    uint32_t GetLiveBindGroupLayoutCount() const override { return m_BindGroupLayouts.Size(); }
    uint32_t GetLiveBindGroupCount() const override { return m_BindGroups.Size(); }

    /** The set a handle names, for the transitional binding path. */
    vk::DescriptorSet GetDescriptorSet(BindGroupHandle handle) const;
    vk::DescriptorSetLayout GetDescriptorSetLayout(BindGroupLayoutHandle handle) const;

    PipelineLayoutHandle CreatePipelineLayout(const PipelineLayoutDesc& desc) override;
    void Destroy(PipelineLayoutHandle handle) override;
    ShaderModuleHandle CreateShaderModule(const ShaderModuleDesc& desc) override;
    void Destroy(ShaderModuleHandle handle) override;
    GraphicsPipelineHandle CreateGraphicsPipeline(const GraphicsPipelineDesc& desc,
                                                  IPipelineCache& cache) override;
    void Destroy(GraphicsPipelineHandle handle) override;

    /** Raw objects for the transitional recording path. */
    vk::PipelineLayout GetPipelineLayout(PipelineLayoutHandle handle) const;
    vk::Pipeline GetPipeline(GraphicsPipelineHandle handle) const;

    FenceHandle CreateFence(const FenceDesc& desc) override;
    void Destroy(FenceHandle handle) override;
    uint32_t GetLiveFenceCount() const override { return m_Fences.Size(); }
    void WaitForFence(FenceHandle handle, uint64_t value) override;
    void Submit(const SubmitDesc& desc) override;

    [[nodiscard]] std::unique_ptr<IPipelineCache>
    CreatePipelineCache(const PipelineCacheDesc& desc) override;

    [[nodiscard]] std::unique_ptr<IPresentTarget>
    CreatePresentTarget(const PresentTargetDesc& desc) override;

    /**
     * Binary semaphores, for the present path only — IDevice deliberately does
     * not expose these (see SemaphoreHandle in <rhi/Handles.h>). SwapchainTarget
     * creates them through the device rather than owning vk::raii::Semaphore
     * itself so that a handle can be resolved from outside the module, which is
     * what lets the application keep recording its own submit.
     */
    SemaphoreHandle CreateSemaphore(std::string_view debugName);
    void Destroy(SemaphoreHandle handle);
    vk::Semaphore GetSemaphore(SemaphoreHandle handle) const;

    /**
     * Gives an image the device did not allocate a pool slot, so that barriers,
     * views and copies can name it by handle like any other texture. Destroying
     * the returned handle releases the slot and does not touch the image.
     *
     * Exists for the swapchain, whose images belong to the presentation engine
     * rather than to us. SwapchainTarget is the only caller and the only one
     * there should be: a handle to memory the device did not allocate cannot be
     * destroyed, resized or aliased like a real one, and the target is what
     * keeps that distinction from escaping.
     */
    TextureHandle RegisterExternalTexture(vk::Image image, const TextureDesc& desc);

    /**
     * The Vulkan objects behind a handle, or a null object if it is stale. These
     * back the accessors in <rhi/vulkan/VulkanNative.h>, and are also what
     * VulkanCommandList resolves handles through.
     */
    vk::Buffer GetBuffer(BufferHandle handle) const;
    vk::Image GetImage(TextureHandle handle) const;
    vk::ImageView GetImageView(TextureViewHandle handle) const;
    vk::Sampler GetSampler(SamplerHandle handle) const;

    /**
     * Reports a handle that resolved to nothing, from the places that cannot
     * throw over it — recording a barrier or a copy, where the caller's own
     * command list is already half-built.
     */
    void ReportStaleHandle(std::string_view what) const;

    /**
     * Everything below is reachable only through <rhi/vulkan/VulkanNative.h>,
     * which is the one sanctioned way for code outside this module to see a
     * Vulkan handle. These are non-const references because callers still create
     * Vulkan objects from them; that shrinks as resource creation moves in here.
     */
    vk::raii::Instance& GetInstance() { return m_Instance; }
    vk::raii::PhysicalDevice& GetPhysicalDevice() { return m_PhysicalDevice; }
    vk::raii::Device& GetDevice() { return m_Device; }
    vk::raii::SurfaceKHR& GetSurface() { return m_Surface; }
    vk::raii::Queue& GetGraphicsQueue() { return m_GraphicsQueue; }
    VmaAllocator GetAllocator() const { return m_Allocator; }
    uint32_t GetApiVersion() const { return kApiVersion; }

    /**
     * The queue family serving `role`, or QueueFamilies::kInvalid when the
     * device has none. Graphics and Copy are backed by created queues; compute
     * work is still submitted to the graphics queue, so that family is known
     * but idle.
     */
    uint32_t GetQueueFamily(QueueType role) const { return m_QueueFamilies.Get(role); }

    /**
     * The queue to submit `role`'s work to. Falls back to the graphics queue
     * for every role the device has no separate queue for, so a caller never
     * has to test IsDedicated before submitting — GetQueueFamily() is what it
     * must consult instead, because a command pool is tied to a family and the
     * two answers differ exactly when an ownership transfer is needed.
     */
    vk::raii::Queue& GetQueue(QueueType role);

    /**
     * Whether VK_KHR_maintenance8 was enabled, which is what allows a queue
     * family ownership transfer to name real pipeline stages instead of being
     * pinned to AllCommands.
     */
    bool IsMaintenance8Enabled() const { return m_bMaintenance8Enabled; }

    /**
     * What this device promises about handing a resource on from `srcFamily`.
     * The answer differs per family, so a caller passes the family it recorded
     * the releasing work on. This is the whole answer for a buffer; an image
     * needs the question below, which folds in how the image was created.
     */
    OwnershipTransferRules GetOwnershipTransferRules(uint32_t srcFamily) const;

    /**
     * Whether a texture filled on `srcFamily` must be explicitly released
     * before `dstFamily` can rely on its contents. A stale handle answers yes,
     * since the safe answer is the one that does more work.
     */
    bool RequiresOwnershipTransfer(TextureHandle handle, uint32_t srcFamily,
                                   uint32_t dstFamily) const;

private:
    void CreateInstance(const DeviceDesc& desc);
    void SetupDebugMessenger(const DeviceDesc& desc);
    void CreateSurface(const DeviceRequirements& requirements);
    void PickPhysicalDevice(const DeviceRequirements& requirements);
    void SelectOptionalExtensions(const DeviceDesc& desc);
    void FindQueueFamilies(const DeviceDesc& desc);
    void CreateLogicalDevice(const DeviceRequirements& requirements);

    bool IsPhysicalDeviceSuitable(const vk::raii::PhysicalDevice& device,
                                  const DeviceRequirements& requirements) const;

    /**
     * Called from the driver's debug callback, on whichever thread the driver
     * happens to be on.
     */
    void ReportDiagnostic(DiagnosticSeverity severity, std::string_view message) const;

    /**
     * Static so that it has C linkage-compatible calling convention while still
     * reaching the members above; the instance arrives via pUserData.
     */
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type,
        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

    /**
     * 1.4 rather than the 1.3 that IsPhysicalDeviceSuitable requires: the
     * instance-level version is a ceiling on what the loader will expose, while
     * the device check is the actual hard requirement.
     */
    static constexpr uint32_t kApiVersion = VK_API_VERSION_1_4;

    /**
     * Declaration order is destruction order reversed, and both matter here.
     * The allocator sits after the device so that it is destroyed first, and the
     * surface after the instance that has to outlive it.
     *
     * Diagnostics comes first of all, because m_DebugMessenger below is
     * destroyed second-to-last and the driver reports validation messages
     * raised while the allocator and the logical device are being torn down.
     * Anything the callback touches has to still be alive at that point.
     */
    std::unique_ptr<Diagnostics> m_OwnedDiagnostics;

    /** Either the caller's or m_OwnedDiagnostics; never null after construction. */
    Diagnostics* m_pDiagnostics = nullptr;

    /**
     * Built from the loader this module is linked against, rather than from the
     * one vk::raii::Context would dlopen by bare filename for itself.
     *
     * Those are not always the same library. A bare-name dlopen resolves through
     * the calling object's RUNPATH, which for the linked loader is its own
     * directory — but AddressSanitizer interposes dlopen, and the RUNPATH glibc
     * then consults is the sanitizer runtime's. A sanitizer build therefore
     * picked up whichever libvulkan.so.1 happened to be installed system-wide,
     * leaving two loaders in one process: the instance created through one, its
     * function pointers queried through the other, and every extension entry
     * point null. The first casualty was vkCreateDebugUtilsMessengerEXT, which
     * aborts in a debug build rather than failing quietly.
     */
    vk::raii::Context m_Context{&::vkGetInstanceProcAddr};
    vk::raii::Instance m_Instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT m_DebugMessenger = nullptr;
    vk::raii::SurfaceKHR m_Surface = nullptr;
    vk::raii::PhysicalDevice m_PhysicalDevice = nullptr;
    vk::raii::Device m_Device = nullptr;
    VulkanAllocator m_Allocator{};
    vk::raii::Queue m_GraphicsQueue = nullptr;

    /**
     * Null unless the copy family is a family of its own; GetQueue() is what
     * resolves that, so nothing else has to know.
     */
    vk::raii::Queue m_CopyQueue = nullptr;

    /**
     * After the allocator, so that every buffer is destroyed before the
     * allocator that owns their memory. Releasing a slot frees its VulkanBuffer,
     * so this is also what makes an un-destroyed buffer merely a leak reported
     * at shutdown rather than a crash during it.
     */
    Core::HandlePool<VulkanBuffer, BufferTag> m_Buffers;
    Core::HandlePool<VulkanTexture, TextureTag> m_Textures;

    /**
     * After m_Textures so that views are destroyed before the images they were
     * made from: a VkImageView outliving its VkImage is undefined behaviour
     * rather than something the driver diagnoses.
     */
    Core::HandlePool<VulkanTextureView, TextureViewTag> m_TextureViews;
    Core::HandlePool<VulkanSampler, SamplerTag> m_Samplers;

    /**
     * The present target is destroyed before the device that owns its
     * semaphores, so this only has to outlive the target — but it sits with the
     * other pools rather than after them because a semaphore depends on nothing
     * else here.
     */
    Core::HandlePool<VulkanSemaphore, SemaphoreTag> m_Semaphores;
    Core::HandlePool<VulkanFence, FenceTag> m_Fences;
    Core::HandlePool<VulkanBindGroupLayout, BindGroupLayoutTag> m_BindGroupLayouts;

    /**
     * Declared after the allocator that owns the pools their sets came from, so
     * the sets are freed before the pools they name are destroyed.
     */
    std::unique_ptr<DescriptorAllocator> m_BindGroupAllocator;
    Core::HandlePool<VulkanBindGroup, BindGroupTag> m_BindGroups;
    Core::HandlePool<VulkanPipelineLayout, PipelineLayoutTag> m_PipelineLayouts;
    Core::HandlePool<VulkanShaderModule, ShaderModuleTag> m_ShaderModules;
    Core::HandlePool<VulkanGraphicsPipeline, GraphicsPipelineTag> m_GraphicsPipelines;

    QueueFamilies m_QueueFamilies;

    /**
     * Optional extensions, resolved once at creation from what the device
     * supports and what DeviceDesc asked to be pretended away.
     */
    bool m_bMaintenance8Enabled = false;
    bool m_bMaintenance9Enabled = false;

    /**
     * optimalImageTransferToQueueFamilies per queue family, empty unless
     * maintenance9 was enabled. Indexed by the family releasing a resource.
     */
    std::vector<uint32_t> m_OptimalImageTransferToQueueFamilies;

    DeviceCaps m_Caps{};
};
} // namespace Hikari::Rhi::Vulkan
