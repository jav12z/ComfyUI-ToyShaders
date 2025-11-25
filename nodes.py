import torch
import torch.nn.functional as F
import numpy as np
import os
import folder_paths
from PIL import Image
from .renderer import ToyRenderer

# --- GENERADOR DE TEXTURAS INTERNO ---
class TextureBuilder:
    @staticmethod
    def get_texture_list():
        return [
            "none", 
            "Noise White (Mono)", 
            "Noise White (RGB)", 
            "Noise Blue (Approx)", 
            "Bayer 8x8 (Dithering)",
            "Clouds (Organic)",
            "Rusty Metal",
            "Wood (Procedural)",
            "Checkerboard",
        ]

    @staticmethod
    def create_texture(name, width=512, height=512):
        # Usamos PyTorch para generar texturas rápidamente
        w, h = width, height
        
        if name == "Noise White (Mono)":
            return torch.rand((h, w, 3)).numpy()
            
        elif name == "Noise White (RGB)":
            return torch.rand((h, w, 3)).numpy()

        elif name == "Noise Blue (Approx)":
            # Aproximación rápida: Ruido gaussiano con desenfoque negativo (high-pass)
            noise = torch.randn((h, w, 3))
            blurred = F.avg_pool2d(noise.permute(2,0,1).unsqueeze(0), 3, 1, 1).squeeze(0).permute(1,2,0)
            blue = noise - blurred
            return ((blue + 0.5).clamp(0, 1)).numpy()

        elif name == "Bayer 8x8 (Dithering)":
            # Matriz Bayer estándar 8x8
            bayer = torch.tensor([
                [ 0, 32,  8, 40,  2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44,  4, 36, 14, 46,  6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [ 3, 35, 11, 43,  1, 33,  9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47,  7, 39, 13, 45,  5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]
            ], dtype=torch.float32) / 64.0
            # Repetir hasta llenar la imagen
            return bayer.repeat(h//8 + 1, w//8 + 1)[:h, :w].unsqueeze(-1).repeat(1, 1, 3).numpy()

        elif name == "Clouds (Organic)":
            # Ruido perlin simple simulado con interpolación
            small_noise = torch.rand((1, 3, h//8, w//8))
            clouds = F.interpolate(small_noise, size=(h, w), mode='bicubic', align_corners=False)
            return clouds.squeeze(0).permute(1, 2, 0).numpy()

        elif name == "Rusty Metal":
            # Ruido marrón/naranja mezclado
            base = torch.rand((1, 1, h//2, w//2))
            detail = F.interpolate(base, size=(h, w), mode='bicubic')
            detail = detail.squeeze(0).permute(1, 2, 0)
            color = torch.tensor([0.55, 0.27, 0.07]).view(1, 1, 3) # Marrón óxido
            return (detail * color + torch.rand((h, w, 3)) * 0.2).clamp(0,1).numpy()

        elif name == "Wood (Procedural)":
            # Bandas sinusoidales distorsionadas
            y, x = torch.meshgrid(torch.linspace(0, 10, h), torch.linspace(0, 10, w))
            noise = F.interpolate(torch.rand((1, 1, h//16, w//16)), size=(h, w), mode='bicubic').squeeze()
            pattern = torch.sin(x * 2.0 + noise * 5.0)
            pattern = (pattern + 1) * 0.5
            wood_color = torch.tensor([0.6, 0.4, 0.2]).view(1, 1, 3)
            return (pattern.unsqueeze(-1) * wood_color).numpy()
            
        elif name == "Checkerboard":
            # Tablero de ajedrez útil para debugging UV
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
            check = ((x // 32) + (y // 32)) % 2
            return check.unsqueeze(-1).repeat(1, 1, 3).float().numpy()

        return np.full((h, w, 3), 1.0, dtype=np.float32)

# --- NODO DE ENTRADA DE TEXTO ---
class ShaderCodeInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "code": ("STRING", {
                    "multiline": True, 
                    "default": "// Pega aquí tu código de Shadertoy..."
                }),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "output_code"
    CATEGORY = "ToyShaders"

    def output_code(self, code):
        return (code,)


# --- NODO RENDERIZADOR ---
class ToyShaderNode:
    def __init__(self):
        self.renderer = ToyRenderer()

    @classmethod
    def INPUT_TYPES(s):
        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
        if not os.path.exists(shader_dir): os.makedirs(shader_dir)
        
        files = []
        for f in os.listdir(shader_dir):
            if f.endswith(".glsl") or f.endswith(".txt"):
                files.append(os.path.splitext(f)[0])
        files = sorted(list(set(files)))
        if not files: files = ["ninguno"]
        
        # Lista de texturas internas
        texture_options = TextureBuilder.get_texture_list()

        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame_count": ("INT", {"default": 24, "min": 1}),
                "fps": ("INT", {"default": 24, "min": 1}),
                
                "preset": (files,), 
                
                "speed": ("FLOAT", {"default": 1.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "step": 0.1}),
                
                "depth_var": ("STRING", {"default": "", "multiline": False, "placeholder": "Nombre variable depth (ej: d, t)..."}),
                "depth_scale": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 10.0, "step": 0.001}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                
                # --- NUEVOS CAMPOS: SELECCIÓN DE TEXTURA INTERNA ---
                "tex_0_builtin": (texture_options,),
                "tex_1_builtin": (texture_options,),
                "tex_2_builtin": (texture_options,),
                "tex_3_builtin": (texture_options,),
            },
            "optional": {
                "main_code": ("STRING", {"forceInput": True}),     
                "buffer_a_code": ("STRING", {"forceInput": True}),
                # ENTRADAS EXTERNAS (Tienen prioridad)
                "iChannel0": ("IMAGE",),
                "iChannel1": ("IMAGE",),
                "iChannel2": ("IMAGE",),
                "iChannel3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "DEPTH")
    FUNCTION = "generate_anim"
    CATEGORY = "ToyShaders"
    OUTPUT_NODE = True 

    def generate_anim(self, width, height, frame_count, fps, preset, speed, scale, depth_var, depth_scale, invert_depth, 
                      tex_0_builtin, tex_1_builtin, tex_2_builtin, tex_3_builtin,
                      main_code=None, buffer_a_code=None, 
                      iChannel0=None, iChannel1=None, iChannel2=None, iChannel3=None):
        
        frames_rgb = []
        frames_depth = []
        
        # --- LÓGICA DE TEXTURAS (PRIORIDAD: EXTERNA > INTERNA) ---
        final_textures = {}

        def resolve_texture(idx, external_tensor, builtin_name):
            # 1. Si hay input externo, usarlo
            if external_tensor is not None:
                img_np = external_tensor[0].cpu().numpy()
                return img_np
            # 2. Si hay seleccion interna, generarla
            if builtin_name != "none":
                print(f"🎨 Generando textura interna [{idx}]: {builtin_name}")
                return TextureBuilder.create_texture(builtin_name, width, height)
            return None

        final_textures[0] = resolve_texture(0, iChannel0, tex_0_builtin)
        final_textures[1] = resolve_texture(1, iChannel1, tex_1_builtin)
        final_textures[2] = resolve_texture(2, iChannel2, tex_2_builtin)
        final_textures[3] = resolve_texture(3, iChannel3, tex_3_builtin)

        # --- CÓDIGO ---
        final_main_code = None
        if main_code and isinstance(main_code, str) and len(main_code) > 10:
            print("📝 Usando código manual.")
            final_main_code = main_code
        else:
            if preset != "ninguno":
                print(f"📁 Cargando preset: {preset}")
                final_main_code = self.renderer.load_from_file(preset)
        
        if not final_main_code:
            final_main_code = """void mainImage(out vec4 c, in vec2 f){ c = vec4(0.5 + 0.5*sin(iTime), f/iResolution.xy, 1.0); }"""

        depth_variable_name = depth_var.strip()

        # --- RENDER LOOP ---
        for i in range(frame_count):
            time_val = i / float(fps)
            
            img_rgba = self.renderer.render_multipass(
                width, height, time_val, i, speed, scale, depth_scale, 
                final_main_code, buffer_a_code, 
                depth_var_name=depth_variable_name,
                input_textures=final_textures
            )
            
            img_float = img_rgba.astype(np.float32) / 255.0
            
            rgb = img_float[:, :, :3]
            alpha = img_float[:, :, 3] 

            if invert_depth:
                alpha = 1.0 - alpha
            
            depth_3ch = np.stack((alpha, alpha, alpha), axis=-1)
            
            frames_rgb.append(rgb)
            frames_depth.append(depth_3ch)

        tensor_rgb = torch.from_numpy(np.array(frames_rgb))
        tensor_depth = torch.from_numpy(np.array(frames_depth))
        
        ui_preview = self.save_preview(frames_rgb, fps)
        
        return {"ui": {"images": ui_preview}, "result": (tensor_rgb, tensor_depth)}

    def save_preview(self, frames_list, fps):
        output_dir = folder_paths.get_temp_directory()
        filename = f"toyshader_{os.urandom(4).hex()}.webp"
        file_path = os.path.join(output_dir, filename)
        preview_frames = [(f * 255).astype(np.uint8) for f in frames_list]
        if len(preview_frames) > 0:
            pil_images = [Image.fromarray(f) for f in preview_frames]
            pil_images[0].save(file_path, save_all=True, append_images=pil_images[1:], duration=int(1000/fps), loop=0, quality=80)
        return [{"filename": filename, "subfolder": "", "type": "temp"}]