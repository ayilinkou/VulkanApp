#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Drawable.h"
#include "Material.h"
#include "Mesh.h"
#include "Node.h"
#include <rhi/Handles.h>
#include <rhi/UniqueHandle.h>

struct aiMaterial;

class ModelData
{
public:
    ModelData(const std::string& path, std::vector<std::unique_ptr<Material>> materials,
              uint32_t meshCount);
    void Init(Hikari::Rhi::UniqueHandle<Hikari::Rhi::BufferHandle> vertexBuffer,
              Hikari::Rhi::UniqueHandle<Hikari::Rhi::BufferHandle> indexBuffer,
              std::unique_ptr<Node> rootNode);

    Mesh* RegisterMesh(aiMesh* mesh, uint32_t meshIndex, const glm::mat4& localTransform,
                       std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);

    Hikari::Rhi::BufferHandle GetVertexBuffer() const { return m_VertexBuffer.Get(); }
    Hikari::Rhi::BufferHandle GetIndexBuffer() const { return m_IndexBuffer.Get(); }

    const std::vector<Drawable>& GetDrawables() const { return m_Drawables; }

    const std::string& GetFilepath() const { return m_Path; }

private:
    std::vector<Mesh> m_Meshes;
    std::vector<std::unique_ptr<Material>> m_Materials;
    std::unordered_map<uint32_t, std::vector<glm::mat4>> m_MeshLocalTransforms;

    std::unique_ptr<Node> m_RootNode = nullptr;

    Hikari::Rhi::UniqueHandle<Hikari::Rhi::BufferHandle> m_VertexBuffer;
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::BufferHandle> m_IndexBuffer;

    std::vector<Drawable> m_Drawables;

    const std::string m_Path;
};
