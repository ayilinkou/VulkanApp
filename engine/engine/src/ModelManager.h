#pragma once

#include <vector>

#include "InstanceData.h"
#include "Model.h"

struct SceneGraph;

/**
 * Turns the scene's models into the batches a frame draws.
 *
 * It holds no models. Every rebuild walks the scene it is handed, so a model
 * that has been added, removed or replaced is accounted for without anything
 * having told this class — which is what lets Model be inert data rather than
 * an object that registers itself somewhere at construction.
 */
class ModelManager
{
public:
    ModelManager() = default;

    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;
    ModelManager(ModelManager&&) = delete;
    ModelManager& operator=(ModelManager&&) = delete;

    /**
     * Rebuilds the batches from the scene as it stands now.
     *
     * Run every frame, which is what the registration it replaced amounted to
     * anyway: the drawables were re-flattened and re-sorted on every call.
     * Rebuilding only when the scene changes needs a way to know that it has,
     * and that dirty flag is a later step's — adding it here would make an
     * ownership change into a behaviour change as well.
     */
    void GenerateBatches(const SceneGraph& scene);

    const std::vector<MeshBatch>& GetOpaqueBatches() const { return m_OpaqueBatches; }
    const std::vector<MeshBatch>& GetTransparentBatches() const { return m_TransparentBatches; }
    const std::vector<InstanceData>& GetInstanceDatas() const { return m_InstanceDatas; }

private:
    /** Flattens every model in the scene into world-space drawables. */
    void CollectRenderables(const SceneGraph& scene);

    std::vector<InstanceData> m_InstanceDatas;
    std::vector<MeshBatch> m_OpaqueBatches;
    std::vector<MeshBatch> m_TransparentBatches;
    std::vector<Drawable> m_Drawables;
};
