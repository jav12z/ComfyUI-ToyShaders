from .nodes import ToyShaderNode, ShaderCodeInput

NODE_CLASS_MAPPINGS = {
    "ToyShaderNode": ToyShaderNode,
    "ShaderCodeInput": ShaderCodeInput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToyShaderNode": "👾 ToyShader Render",
    "ShaderCodeInput": "📝 Shader Code (GLSL)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']