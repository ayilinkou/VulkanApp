#include "Mesh.h"

#include "assimp/mesh.h"

#include "ModelData.h"
#include "Vertex.h"

void Mesh::Init(ModelData* pModelData, aiMesh* mesh, std::vector<Vertex>& vertices,
                std::vector<uint32_t>& indices, Material* pMaterial, uint32_t meshIndex)
{
    m_pModelData = pModelData;
    m_Name = mesh->mName.C_Str();
    uint32_t verticesOffset = (uint32_t)vertices.size();
    m_IndexOffset = (uint32_t)indices.size();
    m_Material = pMaterial;
    m_MeshIndex = meshIndex;

    for (size_t i = 0; i < mesh->mNumVertices; i++)
    {
        Vertex v;
        v.Pos = {mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z};
        if (mesh->mTextureCoords[0])
        {
            v.TexCoord = {mesh->mTextureCoords[0][i].x, 1.f - mesh->mTextureCoords[0][i].y};
        }

        if (!mesh->HasNormals())
        {
            throw std::runtime_error(
                std::format("Mesh {} does not have normals!", pModelData->GetFilepath().c_str()));
        }

        v.Normal = {mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z};

        if (!mesh->HasTangentsAndBitangents())
        {
            throw std::runtime_error(
                std::format("Mesh {} has no tangent basis!", pModelData->GetFilepath().c_str()));
        }

        glm::vec3 tangent = {mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z};
        glm::vec3 bitangent = {mesh->mBitangents[i].x, mesh->mBitangents[i].y,
                               mesh->mBitangents[i].z};
        float handedness = (glm::dot(glm::cross(v.Normal, tangent), bitangent) > 0.f) ? -1.f : 1.f;
        v.Tangent = glm::vec4(tangent, handedness);

        vertices.push_back(v);
    }

    for (size_t i = 0; i < mesh->mNumFaces; i++)
    {
        const aiFace& face = mesh->mFaces[i];
        for (size_t j = 0; j < face.mNumIndices; j++)
        {
            indices.push_back(static_cast<uint32_t>(face.mIndices[j] + verticesOffset));
        }
    }

    m_IndexCount = static_cast<uint32_t>(indices.size() - m_IndexOffset);
    m_bIsValid = true;
}
