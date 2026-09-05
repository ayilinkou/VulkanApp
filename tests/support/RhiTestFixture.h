#pragma once

#include <cstdlib>

#include <array>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <rhi/DeviceDesc.h>
#include <rhi/Diagnostics.h>
#include <rhi/IDevice.h>

/**
 * The device the GPU tests run against, created once per binary and shared by
 * every case in it.
 *
 * Shared rather than per-case because creating a device costs a few hundred
 * milliseconds and proves nothing after the first time — the tests that follow
 * are about what the device *does*. What that shares is deliberate and small:
 * resource pools and validation counters. Every test destroys the resources it
 * creates, and ValidationGuard resets the counters it reads, so a case still
 * starts from a known state.
 *
 * A machine with no Vulkan ICD is not a failure. Nothing here can run there, so
 * the cases skip with the reason attached rather than failing, which is why
 * these are labelled "gpu" and kept out of the run CI performs — see
 * cmake/Testing.cmake.
 *
 * The devices are torn down by a Catch2 listener (RhiDeviceListener.cpp) rather
 * than by the static that holds them. Leaving it to static destruction aborts
 * inside the validation layer: the layer's own globals are destroyed by its own
 * static destructors, and whichever runs first wins — here vkDestroyDevice
 * reached a layer that had already forgotten the device existed.
 */
namespace RhiTest
{
/**
 * The device arrangements the upload path has to work under.
 *
 * The last three are unreachable on any one machine without a lever, and that
 * is the point: whichever of them this GPU is, the other two are the ones most
 * hardware in the field takes. The extension names are Vulkan's because
 * DeviceDesc::DisabledOptionalExtensions is neutral in type and backend-specific
 * in content; a second backend's fixture would name its own.
 */
enum class DeviceConfig : uint8_t
{
    /**
     * Whatever this machine is. On a device with VK_KHR_maintenance9 and a
     * separate copy family, uploads cross queues with no ownership transfer at
     * all.
     */
    Default,

    /**
     * No maintenance9, so every uploaded resource is released by the copy family
     * and acquired by the graphics one. The path a device without the extension
     * takes, which today is nearly all of them.
     */
    OwnershipTransfer,

    /**
     * The same, and without maintenance8, so the release and acquire cannot name
     * a pipeline stage and fall back to AllCommands.
     */
    OwnershipTransferAllStages,

    /**
     * One family for every role, so there is no second queue to submit on and
     * nothing to hand over. What an integrated GPU exposes.
     */
    SingleQueue,
};

/**
 * A device and the diagnostics it reports into. Diagnostics is declared first so
 * that it is destroyed last: the debug messenger is torn down near the end of
 * the device's destructor and reports into this while it happens.
 */
struct DeviceInstance
{
    Hikari::Rhi::Diagnostics Diagnostics;
    std::unique_ptr<Hikari::Rhi::IDevice> pDevice;
};

namespace Detail
{
inline Hikari::Rhi::DeviceDesc MakeDesc(DeviceConfig config, Hikari::Rhi::Diagnostics& diagnostics)
{
    Hikari::Rhi::DeviceDesc desc;
    desc.ApplicationName = "HikariEngine RHI GPU tests";

    // The whole reason these tests exist is to be the place a validation error
    // is noticed, so they pay for the layer. Count rather than FailFast: a
    // failing assertion naming the message is more useful to whoever runs this
    // than an abort inside the driver.
    desc.bEnableValidation = true;
    desc.pDiagnostics = &diagnostics;

    // No window exists in a test binary, and none is needed: nothing here
    // presents.
    desc.Requirements.bPresent = false;

    switch (config)
    {
        case DeviceConfig::Default:
            break;
        case DeviceConfig::OwnershipTransfer:
            desc.DisabledOptionalExtensions = {"VK_KHR_maintenance9"};
            break;
        case DeviceConfig::OwnershipTransferAllStages:
            desc.DisabledOptionalExtensions = {"VK_KHR_maintenance8", "VK_KHR_maintenance9"};
            break;
        case DeviceConfig::SingleQueue:
            desc.bForceSingleQueue = true;
            break;
    }

    return desc;
}

/**
 * Why device creation failed, or empty while it has not been attempted. Held
 * per configuration because "no ICD" and "this device cannot do that" are
 * different answers and only the first is a reason to skip everything.
 */
struct Slot
{
    std::unique_ptr<DeviceInstance> pInstance;
    std::string FailureReason;
};

inline std::map<DeviceConfig, Slot>& Slots()
{
    static std::map<DeviceConfig, Slot> slots;
    return slots;
}
} // namespace Detail

/**
 * The shared device for `config`, or nullptr when one could not be created.
 * The failure is remembered, so a machine with no ICD attempts creation once
 * per configuration rather than once per test case.
 */
inline DeviceInstance* TryGetDevice(DeviceConfig config)
{
    Detail::Slot& slot = Detail::Slots()[config];

    if (slot.pInstance != nullptr || !slot.FailureReason.empty())
        return slot.pInstance.get();

    auto instance = std::make_unique<DeviceInstance>();

    try
    {
        instance->pDevice = Hikari::Rhi::CreateDevice(Detail::MakeDesc(config, instance->Diagnostics));
    }
    catch (const std::exception& e)
    {
        slot.FailureReason = e.what();
        return nullptr;
    }

    slot.pInstance = std::move(instance);
    return slot.pInstance.get();
}

/**
 * Whether a run without a device is a failure rather than a skip.
 *
 * Set by CI, which supplies an ICD on purpose and therefore learns nothing from
 * a green run of nothing: CTest reports a skipped case as not-failed, so an
 * environment that quietly stopped providing a device looks exactly like one
 * that never had to. A developer without a GPU still gets skips.
 */
inline bool DeviceRequiredByEnvironment()
{
    const char* value = std::getenv("HIKARI_TESTS_REQUIRE_DEVICE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

/**
 * The shared device for `config`, skipping the calling test case when there is
 * none — or failing it where the environment says a device was supposed to be
 * there.
 *
 * The skip happens here rather than at the call site because SKIP() throws —
 * that is what aborts the case — so the return below is unreachable whenever
 * the device is null. Catch2 records the skip against whichever test is
 * running, so this being a plain function rather than a macro costs nothing.
 */
inline Hikari::Rhi::IDevice& RequireDevice(DeviceConfig config = DeviceConfig::Default)
{
    DeviceInstance* pInstance = TryGetDevice(config);
    if (pInstance == nullptr)
    {
        const std::string reason = "No usable Vulkan device: " + Detail::Slots()[config].FailureReason;
        if (DeviceRequiredByEnvironment())
            FAIL(reason);

        SKIP(reason);
    }

    return *pInstance->pDevice;
}

/**
 * The diagnostics `config`'s device reports into. Only valid once RequireDevice
 * has returned for the same configuration.
 */
inline Hikari::Rhi::Diagnostics& DeviceDiagnostics(DeviceConfig config = DeviceConfig::Default)
{
    DeviceInstance* pInstance = TryGetDevice(config);
    REQUIRE(pInstance != nullptr);
    return pInstance->Diagnostics;
}

/**
 * Destroys every device created so far. Called once at the end of the run, from
 * a Catch2 listener, for the reason given at the top of this file.
 */
inline void ShutDownDevices()
{
    Detail::Slots().clear();
}

/** Every configuration, for the cases that have to pass under all of them. */
inline constexpr std::array kAllDeviceConfigs{
    DeviceConfig::Default,
    DeviceConfig::OwnershipTransfer,
    DeviceConfig::OwnershipTransferAllStages,
    DeviceConfig::SingleQueue,
};

inline const char* Describe(DeviceConfig config)
{
    switch (config)
    {
        case DeviceConfig::Default:
            return "default";
        case DeviceConfig::OwnershipTransfer:
            return "ownership transfer";
        case DeviceConfig::OwnershipTransferAllStages:
            return "ownership transfer, all stages";
        case DeviceConfig::SingleQueue:
            return "single queue";
    }

    return "unknown";
}
} // namespace RhiTest
