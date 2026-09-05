#include "ModelManager.h"

#include <complex>
#include <stdexcept>

#include "Entity.h"
#include "ModelData.h"
#include "SceneGraph.h"

void ModelManager::CollectRenderables(const SceneGraph& scene)
{
    m_Drawables.clear();

    for (const std::unique_ptr<Entity>& entity : scene.Entities)
    {
        for (const Model* pModel : entity->GetComponents<Model>())
        {
            const std::vector<Drawable> drawables = pModel->GetDrawables();
            m_Drawables.insert(m_Drawables.end(), drawables.begin(), drawables.end());
        }
    }
}

void ModelManager::GenerateBatches(const SceneGraph& scene)
{
    m_InstanceDatas.clear();
    m_OpaqueBatches.clear();
    m_TransparentBatches.clear();

    CollectRenderables(scene);

    // sort by mesh and material
    std::sort(m_Drawables.begin(), m_Drawables.end());

    // when sorted
    // iterate and build batches and instance transforms
    size_t size = m_Drawables.size();
    uint32_t i = 0u;
    while (i < size)
    {
        Mesh* pMesh = m_Drawables[i].pMesh;
        ModelData* pModelData = pMesh->GetModelData();

        MeshBatch batch;
        batch.pMaterial = m_Drawables[i].pMat;
        batch.FirstInstance = i;
        batch.IndexBuffer = pModelData->GetIndexBuffer();
        batch.VertexBuffer = pModelData->GetVertexBuffer();
        batch.IndexCount = pMesh->GetIndexCount();
        batch.FirstIndex = pMesh->GetIndexOffset();

        while (i < size && pMesh == m_Drawables[i].pMesh && m_Drawables[i].pMat == batch.pMaterial)
        {
            InstanceData data;
            // float4x4 constructor already implicitely transposes, no don't
            // need to explicitely transpose here before sending to GPU
            data.ModelMatrix = m_Drawables[i].Transform;
            const glm::mat4 normalMatrix4 = glm::transpose(glm::inverse(m_Drawables[i].Transform));
            data.NormalMatrix = glm::mat3x4(normalMatrix4[0], normalMatrix4[1], normalMatrix4[2]);
            m_InstanceDatas.push_back(data);
            batch.InstanceCount++;
            i++;
        }

        BlendMode blendMode = batch.pMaterial->GetBlendMode();
        if (blendMode == BlendMode::Opaque)
            m_OpaqueBatches.push_back(batch);
        else if (blendMode == BlendMode::Transparent)
            m_TransparentBatches.push_back(batch);
        else
            throw std::runtime_error(std::format("BlendMode of type {} is not supported!",
                                                 static_cast<uint8_t>(blendMode)));
    }
}
