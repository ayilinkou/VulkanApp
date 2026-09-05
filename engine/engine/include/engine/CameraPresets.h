#pragma once

#include <iterator>

#include <glm/vec3.hpp>

namespace Hikari::Engine
{

/**
 * Hardcoded camera transforms selected via --camera-preset <N>, for
 * deterministic screenshots and reports. Rotation is (pitch, yaw, roll) in
 * degrees, matching Transform::Rotation.
 */
struct CameraPresetData
{
    glm::vec3 Position;
    glm::vec3 Rotation;
};

constexpr CameraPresetData kCameraPresets[] = {
    {{0.f, 2.f, 10.f}, {0.f, 0.f, 0.f}},    // 0: front view, eye height
    {{10.f, 2.f, 0.f}, {0.f, 90.f, 0.f}},   // 1: side view
    {{0.f, 20.f, 0.1f}, {-89.f, 0.f, 0.f}}, // 2: top-down view
};

constexpr int kNumCameraPresets = static_cast<int>(std::size(kCameraPresets));

} // namespace Hikari::Engine
