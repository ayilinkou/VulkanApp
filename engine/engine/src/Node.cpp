#include "Node.h"

#include "assimp/scene.h"

#include "ModelData.h"

Node::Node() : m_AccumulatedModelLocal(glm::mat4(1.f)) {}

void Node::ProcessNode(ModelData* pModelData, aiNode* modelNode, const aiScene* scene,
                       const glm::mat4& parentAccumulatedModelLocal, std::vector<Vertex>& vertices,
                       std::vector<uint32_t>& indices)
{
    m_NodeName = modelNode->mName.C_Str();
    const glm::mat4 localTransform = ToMat4(modelNode->mTransformation);
    m_AccumulatedModelLocal = parentAccumulatedModelLocal * localTransform;

    m_Meshes.reserve(modelNode->mNumMeshes);
    for (size_t i = 0; i < modelNode->mNumMeshes; i++)
    {
        uint32_t meshIndex = modelNode->mMeshes[i];
        aiMesh* sceneMesh = scene->mMeshes[meshIndex];
        m_Meshes.push_back(pModelData->RegisterMesh(sceneMesh, meshIndex, m_AccumulatedModelLocal,
                                                    vertices,
                                                    indices)); // TODO: this is already transposed
                                                               // somehow
    }

    for (size_t i = 0; i < modelNode->mNumChildren; i++)
    {
        m_Children.emplace_back();
        m_Children.back().ProcessNode(pModelData, modelNode->mChildren[i], scene,
                                      m_AccumulatedModelLocal, vertices, indices);
    }
}

glm::mat4 Node::ToMat4(const aiMatrix4x4& matrix)
{
    // aiMatrix4x4 is row major, glm::mat4 is column major
    return glm::transpose(glm::make_mat4(&matrix.a1));
}
