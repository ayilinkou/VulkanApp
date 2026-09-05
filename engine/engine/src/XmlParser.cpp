#include "XmlParser.h"

#include <core/Timer.h>

#include <sstream>
#include <stdexcept>

#include "core/Log.h"
#include "pugixml.hpp"

using namespace Hikari::Core;

constexpr LogCategory LogXmlParser("XmlParser");

namespace XML
{
constexpr const char* Position = "position";
constexpr const char* Rotation = "rotation";
constexpr const char* Scale = "scale";
constexpr const char* Entity = "entity";
constexpr const char* Transform = "transform";
constexpr const char* Model = "model";
constexpr const char* Path = "path";
constexpr const char* Scene = "scene";
constexpr const char* Name = "name";
constexpr const char* Light = "light";
constexpr const char* Type = "type";
constexpr const char* Intensity = "intensity";
constexpr const char* Direction = "direction";
constexpr const char* Color = "color";
} // namespace XML

glm::vec3 XmlParser::ParseVec3(const std::string& str)
{
    std::istringstream iss(str);
    glm::vec3 v;
    iss >> v.x >> v.y >> v.z;
    return v;
}

Transform XmlParser::ParseTransform(const pugi::xml_node& node)
{
    auto posAtt = node.attribute(XML::Position);
    auto rotAtt = node.attribute(XML::Rotation);
    auto scaleAtt = node.attribute(XML::Scale);

    if (!posAtt || !rotAtt || !scaleAtt)
    {
        LogMsg(LogSeverity::Warning, LogXmlParser,
               "Found transform without all position, rotation and scale values "
               "when parsing scene! Falling back to default transform...");
        return Transform{};
    }

    Transform t;
    std::istringstream iss;
    iss = std::istringstream(posAtt.as_string());
    iss >> t.Position.x >> t.Position.y >> t.Position.z;
    iss = std::istringstream(rotAtt.as_string());
    iss >> t.Rotation.x >> t.Rotation.y >> t.Rotation.z;
    iss = std::istringstream(scaleAtt.as_string());
    iss >> t.Scale.x >> t.Scale.y >> t.Scale.z;
    return t;
}

std::unique_ptr<Model> XmlParser::ParseModel(const pugi::xml_node& node,
                                             const std::string& scenePath, AssetRegistry& assets)
{
    auto pathAtt = node.attribute(XML::Path);
    if (!pathAtt)
    {
        LogMsg(LogSeverity::Error, LogXmlParser,
               "Found model with no \"path\" value when parsing scene {}", scenePath);
        return nullptr;
    }

    const char* path = pathAtt.as_string();
    std::unique_ptr<Model> model = std::make_unique<Model>(path, assets);

    if (auto transformNode = node.child(XML::Transform))
    {
        Transform transform = ParseTransform(transformNode);
        model->GetTransform() = transform;
    }

    return model;
};

std::unique_ptr<Light> XmlParser::ParseLight(const pugi::xml_node& node,
                                             const std::string& scenePath)
{
    std::unique_ptr<Light> light = nullptr;
    LightType type = static_cast<LightType>(node.attribute(XML::Type).as_int());

    switch (type)
    {
        case LightType::Directional:
        {
            DirectionalLight* pDirLight = new DirectionalLight();
            if (auto dirAtt = node.attribute(XML::Direction))
            {
                glm::vec3 dir = ParseVec3(dirAtt.as_string());
                pDirLight->SetDirection(dir);
            }
            if (auto intensityAtt = node.attribute(XML::Intensity))
            {
                float intensity = intensityAtt.as_float();
                pDirLight->SetIntensity(intensity);
            }
            if (auto colorAtt = node.attribute(XML::Color))
            {
                glm::vec3 color = ParseVec3(colorAtt.as_string());
                pDirLight->SetColor(color);
            }
            return std::unique_ptr<Light>(pDirLight);
        }
        case LightType::Point:
        {
            PointLight* pPointLight = new PointLight();
            if (auto posAtt = node.attribute(XML::Position))
            {
                glm::vec3 pos = ParseVec3(posAtt.as_string());
                pPointLight->SetPosition(pos);
            }
            if (auto intensityAtt = node.attribute(XML::Intensity))
            {
                float intensity = intensityAtt.as_float();
                pPointLight->SetIntensity(intensity);
            }
            if (auto colorAtt = node.attribute(XML::Color))
            {
                glm::vec3 color = ParseVec3(colorAtt.as_string());
                pPointLight->SetColor(color);
            }
            return std::unique_ptr<Light>(pPointLight);
        }
        default:
            LogMsg(LogSeverity::Error, LogXmlParser, "Failed to load light type in scene: {}",
                   scenePath.c_str());
    }

    return nullptr;
};

NodeType XmlParser::TagToNodeType(std::string_view tag)
{
    if (tag == XML::Transform)
        return NodeType::Transform;
    if (tag == XML::Light)
        return NodeType::Light;
    if (tag == XML::Model)
        return NodeType::Model;
    return NodeType::Unknown;
}

/** Returns a nullptr if failed to load a scene. */
std::unique_ptr<SceneGraph> XmlParser::LoadScene(const std::string& path, AssetRegistry& assets)
{
    LogMsg(LogSeverity::Info, LogXmlParser, "Loading scene: {}", path.c_str());
    Timer timer("LoadScene()");

    std::unique_ptr<SceneGraph> scene = std::make_unique<SceneGraph>();

    pugi::xml_document doc;
    if (!doc.load_file(path.c_str()))
    {
        LogMsg(LogSeverity::Error, LogXmlParser, "Failed to load xml document for scene: {}",
               path.c_str());
        return nullptr;
    }

    for (pugi::xml_node node : doc.child(XML::Scene).children())
    {
        if (strcmp(XML::Entity, node.name()) == 0)
        {

            scene->Entities.push_back(std::make_unique<Entity>());
            Entity& entity = *scene->Entities.back();
            entity.SetName(node.attribute(XML::Name).as_string());

            for (pugi::xml_node comp : node.children())
            {
                switch (TagToNodeType(comp.name()))
                {
                    case NodeType::Transform:
                    {
                        Transform transform = ParseTransform(comp);
                        entity.GetTransform() = transform;
                        continue;
                    }
                    case NodeType::Light:
                    {
                        std::unique_ptr<Light> light = ParseLight(comp, path);
                        Light* pLight = light.get();
                        if (!pLight)
                            continue;

                        entity.AddComponent(std::move(light));

                        if (PointLight* pPointLight = dynamic_cast<PointLight*>(pLight))
                        {
                            scene->PointLights.push_back(pPointLight);
                        }
                        else if (DirectionalLight* pDirLight =
                                     dynamic_cast<DirectionalLight*>(pLight))
                        {
                            scene->DirLights.push_back(pDirLight);
                        }
                        continue;
                    }
                    case NodeType::Model:
                    {
                        std::unique_ptr<Model> model = ParseModel(comp, path, assets);
                        if (model.get())
                            entity.AddComponent(std::move(model));
                        continue;
                    }
                    default:
                        LogMsg(LogSeverity::Error, LogXmlParser,
                               "Unexpected component node \"{}\" found when "
                               "parsing scene: {}. Skipping...",
                               comp.name(), path.c_str());
                }
            }
        }
        else
        {
            LogMsg(LogSeverity::Error, LogXmlParser,
                   "Unexpected node \"{}\" found when parsing scene: {}. "
                   "Skipping...",
                   node.name(), path.c_str());
        }
    }
    return scene;
}

std::string XmlParser::Vec3ToString(glm::vec3 v)
{
    std::ostringstream oss;
    oss << v.x << " " << v.y << " " << v.z;
    return oss.str();
}

void XmlParser::WriteTransform(pugi::xml_node& parent, const Transform& t)
{
    pugi::xml_node transformNode = parent.append_child(XML::Transform);

    transformNode.append_attribute(XML::Position) = Vec3ToString(t.Position).c_str();
    transformNode.append_attribute(XML::Rotation) = Vec3ToString(t.Rotation).c_str();
    transformNode.append_attribute(XML::Scale) = Vec3ToString(t.Scale).c_str();
}

void XmlParser::WriteModel(pugi::xml_node& parent, Model* pModel)
{
    pugi::xml_node modelNode = parent.append_child(XML::Model);
    modelNode.append_attribute(XML::Path) = pModel->GetPath().c_str();
    WriteTransform(modelNode, pModel->GetTransform());
}

void XmlParser::WriteLight(pugi::xml_node& parent, Light* pLight)
{
    pugi::xml_node lightNode = parent.append_child(XML::Light);

    if (PointLight* pPointLight = dynamic_cast<PointLight*>(pLight))
    {
        PointLight::Data data = pPointLight->GetData();
        lightNode.append_attribute(XML::Type) = static_cast<int>(LightType::Point);
        lightNode.append_attribute(XML::Intensity) = data.Intensity;
        lightNode.append_attribute(XML::Position) = Vec3ToString(data.Pos);
        lightNode.append_attribute(XML::Color) = Vec3ToString(data.Color);
        return;
    }
    if (DirectionalLight* pDirLight = dynamic_cast<DirectionalLight*>(pLight))
    {
        DirectionalLight::Data data = pDirLight->GetData();
        lightNode.append_attribute(XML::Type) = static_cast<int>(LightType::Directional);
        lightNode.append_attribute(XML::Intensity) = data.Intensity;
        lightNode.append_attribute(XML::Direction) = Vec3ToString(data.Dir);
        lightNode.append_attribute(XML::Color) = Vec3ToString(data.Color);
        return;
    }
}

void XmlParser::SaveScene(const std::unique_ptr<SceneGraph>& sceneGraph, const std::string& path)
{
    LogMsg(LogSeverity::Info, LogXmlParser, "Saving scene: {}", path.c_str());

    pugi::xml_document doc;
    pugi::xml_node sceneNode = doc.append_child(XML::Scene);

    for (const std::unique_ptr<Entity>& entityPtr : sceneGraph->Entities)
    {
        const Entity& entity = *entityPtr.get();
        pugi::xml_node entityNode = sceneNode.append_child(XML::Entity);
        entityNode.append_attribute(XML::Name) = entity.GetName().c_str();
        WriteTransform(entityNode, entityPtr->GetTransform());

        if (const std::vector<Model*> models = entity.GetComponents<Model>(); !models.empty())
        {
            for (Model* pModel : models)
            {
                WriteModel(entityNode, pModel);
            }
        }

        if (const std::vector<Light*> lights = entity.GetComponents<Light>(); !lights.empty())
        {
            for (Light* pLight : lights)
            {
                WriteLight(entityNode, pLight);
            }
        }
    }

    if (!doc.save_file(path.c_str(), "\t"))
    {
        LogMsg(LogSeverity::Error, LogXmlParser, "Failed to save scene: {}", path.c_str());
    }
}
