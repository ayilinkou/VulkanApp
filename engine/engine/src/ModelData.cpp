#include "ModelData.h"

#include "assimp/mesh.h"

using namespace Hikari;

ModelData::ModelData(const std::string& path, std::vector<std::unique_ptr<Material>> materials,
                     uint32_t meshCount)
    : m_Materials(std::move(materials)), m_Path(path)
{
    m_Meshes.resize(meshCount);
}

void ModelData::Init(Rhi::UniqueHandle<Rhi::BufferHandle> vertexBuffer,
                     Rhi::UniqueHandle<Rhi::BufferHandle> indexBuffer,
                     std::unique_ptr<Node> rootNode)
{
    m_VertexBuffer = std::move(vertexBuffer);
    m_IndexBuffer = std::move(indexBuffer);
    m_RootNode = std::move(rootNode);

    for (const Mesh& mesh : m_Meshes)
    {
        Mesh* pMesh = const_cast<Mesh*>(&mesh);
        uint32_t meshIndex = mesh.GetMeshIndex();
        for (const glm::mat4& transform : m_MeshLocalTransforms.at(meshIndex))
        {
            m_Drawables.push_back(Drawable{.pMesh = pMesh,
                                           .pMat = mesh.GetMaterial(),
                                           .blendMode = mesh.GetMaterial()->GetBlendMode(),
                                           .Transform = transform});
        }
    }
}

Mesh* ModelData::RegisterMesh(aiMesh* mesh, uint32_t meshIndex, const glm::mat4& localTransform,
                              std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
{
    if (!m_Meshes[meshIndex].IsValid())
    {
        Material* pMat = m_Materials[mesh->mMaterialIndex].get();
        m_Meshes[meshIndex].Init(this, mesh, vertices, indices, pMat, meshIndex);
    }

    m_MeshLocalTransforms[meshIndex].push_back(localTransform);
    return &m_Meshes[meshIndex];
}
