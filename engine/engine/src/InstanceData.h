#pragma once

#include <array>
#include <cstdint>

#include "glm/glm.hpp"

#include "vulkan/vulkan.hpp"

#include <rhi/Handles.h>

class Material;

struct InstanceData
{
    glm::mat4 ModelMatrix;
    glm::mat3x4 NormalMatrix;

    static constexpr uint32_t AttributeCount = 7;

    static constexpr vk::VertexInputBindingDescription GetBindingDescription()
    {
        return {.binding = 1,
                .stride = sizeof(InstanceData),
                .inputRate = vk::VertexInputRate::eInstance};
    }

    static constexpr std::array<vk::VertexInputAttributeDescription, AttributeCount>
    GetAttributeDescriptions()
    {
        return {{{.location = 4,
                  .binding = 1,
                  .format = vk::Format::eR32G32B32A32Sfloat,
                  .offset = sizeof(glm::vec4) * 0},
                 {.location = 5,
                  .binding = 1,
                  .format = vk::Format::eR32G32B32A32Sfloat,
                  .offset = sizeof(glm::vec4) * 1},
                 {.location = 6,
                  .binding = 1,
                  .format = vk::Format::eR32G32B32A32Sfloat,
                  .offset = sizeof(glm::vec4) * 2},
                 {.location = 7,
                  .binding = 1,
                  .format = vk::Format::eR32G32B32A32Sfloat,
                  .offset = sizeof(glm::vec4) * 3},
                 {.location = 8,
                  .binding = 1,
                  .format = vk::Format::eR32G32B32A32Sfloat,
                  .offset = sizeof(glm::vec4) * 4},
                 {.location = 9,
                  .binding = 1,
                  .format = vk::Format::eR32G32B32A32Sfloat,
                  .offset = sizeof(glm::vec4) * 5},
                 {.location = 10,
                  .binding = 1,
                  .format = vk::Format::eR32G32B32A32Sfloat,
                  .offset = sizeof(glm::vec4) * 6}}};
    }
};

struct MeshBatch
{
    uint32_t FirstInstance = 0u;
    uint32_t InstanceCount = 0u;
    uint32_t FirstIndex = 0u;
    uint32_t IndexCount = 0u;
    Material* pMaterial = nullptr;
    Hikari::Rhi::BufferHandle IndexBuffer;
    Hikari::Rhi::BufferHandle VertexBuffer;
};
