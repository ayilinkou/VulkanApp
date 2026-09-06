#pragma once

#include <array>
#include <cstdint>

#include "glm/glm.hpp"

#include <rhi/Pipeline.h>

#include <rhi/Handles.h>

class Material;

struct InstanceData
{
    glm::mat4 ModelMatrix;
    glm::mat3x4 NormalMatrix;

    static constexpr uint32_t AttributeCount = 7;

    static constexpr Hikari::Rhi::VertexBufferLayout GetBindingDescription()
    {
        return {.Slot = 1,
                .Stride = sizeof(InstanceData),
                .Rate = Hikari::Rhi::VertexInputRate::Instance};
    }

    static constexpr std::array<Hikari::Rhi::VertexAttribute, AttributeCount>
    GetAttributeDescriptions()
    {
        return {{{.Location = 4,
                  .Slot = 1,
                  .AttributeFormat = Hikari::Rhi::Format::RGBA32Float,
                  .Offset = sizeof(glm::vec4) * 0},
                 {.Location = 5,
                  .Slot = 1,
                  .AttributeFormat = Hikari::Rhi::Format::RGBA32Float,
                  .Offset = sizeof(glm::vec4) * 1},
                 {.Location = 6,
                  .Slot = 1,
                  .AttributeFormat = Hikari::Rhi::Format::RGBA32Float,
                  .Offset = sizeof(glm::vec4) * 2},
                 {.Location = 7,
                  .Slot = 1,
                  .AttributeFormat = Hikari::Rhi::Format::RGBA32Float,
                  .Offset = sizeof(glm::vec4) * 3},
                 {.Location = 8,
                  .Slot = 1,
                  .AttributeFormat = Hikari::Rhi::Format::RGBA32Float,
                  .Offset = sizeof(glm::vec4) * 4},
                 {.Location = 9,
                  .Slot = 1,
                  .AttributeFormat = Hikari::Rhi::Format::RGBA32Float,
                  .Offset = sizeof(glm::vec4) * 5},
                 {.Location = 10,
                  .Slot = 1,
                  .AttributeFormat = Hikari::Rhi::Format::RGBA32Float,
                  .Offset = sizeof(glm::vec4) * 6}}};
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
