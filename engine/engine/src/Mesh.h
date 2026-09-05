#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct aiMesh;
struct Vertex;

class ModelData;
class Material;

class Mesh
{
public:
    void Init(ModelData* pModelData, aiMesh* mesh, std::vector<Vertex>& vertices,
              std::vector<uint32_t>& indices, Material* pMaterial, uint32_t meshIndex);

    bool IsValid() const { return m_bIsValid; }

    uint32_t GetMeshIndex() const { return m_MeshIndex; }
    uint32_t GetIndexCount() const { return m_IndexCount; }
    uint32_t GetIndexOffset() const { return m_IndexOffset; }

    ModelData* GetModelData() const { return m_pModelData; }
    Material* GetMaterial() const { return m_Material; }

private:
    ModelData* m_pModelData = nullptr;
    std::string m_Name;
    bool m_bIsValid = false;

    uint32_t m_MeshIndex = 0u;
    uint32_t m_IndexCount = 0u;
    uint32_t m_IndexOffset = 0u;
    Material* m_Material = nullptr;
};
