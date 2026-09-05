#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "glm/glm.hpp"

#include "assimp/matrix4x4.h"

struct aiNode;
struct aiScene;
struct Vertex;

class ModelData;
class Mesh;

class Node
{
public:
    Node();

    void ProcessNode(ModelData* pModelData, aiNode* modelNode, const aiScene* scene,
                     const glm::mat4& parentAccumulatedModelLocal, std::vector<Vertex>& vertices,
                     std::vector<uint32_t>& indices);

private:
    static glm::mat4 ToMat4(const aiMatrix4x4& matrix);

private:
    std::vector<Node> m_Children;
    std::vector<Mesh*> m_Meshes;
    glm::mat4 m_AccumulatedModelLocal;

    std::string m_NodeName;
};
