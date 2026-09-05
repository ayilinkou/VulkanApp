#pragma once

#include <array>
#include <cstdint>

#include "vulkan/vulkan.hpp"

#include "glm/glm.hpp"
#include "glm/gtx/hash.hpp"

struct Vertex
{
    glm::vec3 Pos;
    glm::vec2 TexCoord;
    glm::vec3 Normal;
    glm::vec4 Tangent;

    static constexpr uint32_t AttributeCount = 4;

    static constexpr vk::VertexInputBindingDescription GetBindingDescription()
    {
        return {.binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex};
    }

    static constexpr std::array<vk::VertexInputAttributeDescription, AttributeCount>
    GetAttributeDescriptions()
    {
        return {{{.location = 0,
                  .binding = 0,
                  .format = vk::Format::eR32G32B32Sfloat,
                  .offset = offsetof(Vertex, Pos)},
                 {.location = 1,
                  .binding = 0,
                  .format = vk::Format::eR32G32Sfloat,
                  .offset = offsetof(Vertex, TexCoord)},
                 {.location = 2,
                  .binding = 0,
                  .format = vk::Format::eR32G32B32Sfloat,
                  .offset = offsetof(Vertex, Normal)},
                 {.location = 3,
                  .binding = 0,
                  .format = vk::Format::eR32G32B32A32Sfloat,
                  .offset = offsetof(Vertex, Tangent)}}};
    }

    constexpr bool operator==(const Vertex& other) const
    {
        return Pos == other.Pos && TexCoord == other.TexCoord && Normal == other.Normal &&
               Tangent == other.Tangent;
    }
};

struct QuadVertex
{
    glm::vec2 Pos;
    glm::vec2 TexCoord;

    static constexpr uint32_t AttributeCount = 2u;

    static constexpr vk::VertexInputBindingDescription GetBindingDescription()
    {
        return {
            .binding = 0, .stride = sizeof(QuadVertex), .inputRate = vk::VertexInputRate::eVertex};
    }

    static constexpr std::array<vk::VertexInputAttributeDescription, AttributeCount>
    GetAttributeDescription()
    {
        return {{{.location = 0,
                  .binding = 0,
                  .format = vk::Format::eR32G32Sfloat,
                  .offset = offsetof(QuadVertex, Pos)},
                 {.location = 1,
                  .binding = 0,
                  .format = vk::Format::eR32G32Sfloat,
                  .offset = offsetof(QuadVertex, TexCoord)}}};
    }
};

namespace std
{
template <>
struct hash<Vertex>
{
    size_t operator()(const Vertex& vertex) const
    {
        return ((hash<glm::vec3>()(vertex.Pos) ^ (hash<glm::vec3>()(vertex.Normal) << 1)) >> 1) ^
               (hash<glm::vec2>()(vertex.TexCoord) << 1);
    }
};
} // namespace std
