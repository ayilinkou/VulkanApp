#pragma once

#include <cstdint>
#include <string>

#include <rhi/Handles.h>
#include <rhi/UniqueHandle.h>

struct aiMaterial;

enum class BlendMode : uint8_t
{
    Opaque,
    Transparent
};

class Material
{
public:
    Material() = delete;
    Material(aiMaterial* pMat);
    virtual ~Material() = default;

    virtual void* GetPushConstantData() = 0;
    Hikari::Rhi::BindGroupHandle GetBindGroup() const { return m_BindGroup.Get(); }
    const std::string& GetName() const { return m_Name; }
    bool IsTwoSided() const { return m_bTwoSided; }
    bool IsOpaque() const { return m_Opacity == 1.f; }

    BlendMode GetBlendMode() const { return m_BlendMode; }
    static BlendMode DetectBlendMode(aiMaterial* pMat);

protected:
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::BindGroupHandle> m_BindGroup;

    const std::string m_Name;

    const BlendMode m_BlendMode;
    bool m_bTwoSided = true;
    float m_Opacity = 1.f;
};
