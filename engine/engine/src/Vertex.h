#pragma once

#include <array>
#include <cstdint>

#include <rhi/Pipeline.h>

#include "glm/glm.hpp"
#include "glm/gtx/hash.hpp"

struct Vertex
{
    glm::vec3 Pos;
    glm::vec2 TexCoord;
    glm::vec3 Normal;
    glm::vec4 Tangent;

    static constexpr uint32_t AttributeCount = 4;

    static constexpr Hikari::Rhi::VertexBufferLayout GetBindingDescription()
    {
        return {.Slot = 0, .Stride = sizeof(Vertex), .Rate = Hikari::Rhi::VertexInputRate::Vertex};
    }

    static constexpr std::array<Hikari::Rhi::VertexAttribute, AttributeCount>
    GetAttributeDescriptions()
    {
        return {{{.Location = 0,
                  .Slot = 0,
                  .AttributeFormat = Hikari::Rhi::Format::RGB32Float,
                  .Offset = offsetof(Vertex, Pos)},
                 {.Location = 1,
                  .Slot = 0,
                  .AttributeFormat = Hikari::Rhi::Format::RG32Float,
                  .Offset = offsetof(Vertex, TexCoord)},
                 {.Location = 2,
                  .Slot = 0,
                  .AttributeFormat = Hikari::Rhi::Format::RGB32Float,
                  .Offset = offsetof(Vertex, Normal)},
                 {.Location = 3,
                  .Slot = 0,
                  .AttributeFormat = Hikari::Rhi::Format::RGBA32Float,
                  .Offset = offsetof(Vertex, Tangent)}}};
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

    static constexpr Hikari::Rhi::VertexBufferLayout GetBindingDescription()
    {
        return {
            .Slot = 0, .Stride = sizeof(QuadVertex), .Rate = Hikari::Rhi::VertexInputRate::Vertex};
    }

    static constexpr std::array<Hikari::Rhi::VertexAttribute, AttributeCount>
    GetAttributeDescription()
    {
        return {{{.Location = 0,
                  .Slot = 0,
                  .AttributeFormat = Hikari::Rhi::Format::RG32Float,
                  .Offset = offsetof(QuadVertex, Pos)},
                 {.Location = 1,
                  .Slot = 0,
                  .AttributeFormat = Hikari::Rhi::Format::RG32Float,
                  .Offset = offsetof(QuadVertex, TexCoord)}}};
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
