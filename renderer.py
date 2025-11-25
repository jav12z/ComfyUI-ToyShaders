import moderngl
import numpy as np
import os
import re

class ToyRenderer:
    def __init__(self):
        self.ctx = None
        self.shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
        if not os.path.exists(self.shader_dir): os.makedirs(self.shader_dir)
        self.fallback_texture = None 

    def _init_context(self):
        if self.ctx is None:
            try:
                self.ctx = moderngl.create_context(standalone=True)
            except:
                self.ctx = moderngl.create_context(standalone=True, backend='egl')
            
            # Textura blanca de respaldo (2x2)
            data = np.full((2, 2, 4), 255, dtype='u1').tobytes()
            self.fallback_texture = self.ctx.texture((2, 2), 4, data)
            self.fallback_texture.repeat_x = True
            self.fallback_texture.repeat_y = True

    def load_from_file(self, filename):
        filepath_glsl = os.path.join(self.shader_dir, f"{filename}.glsl")
        if os.path.exists(filepath_glsl):
            with open(filepath_glsl, 'r', encoding='utf-8') as f: return f.read()
        filepath_txt = os.path.join(self.shader_dir, f"{filename}.txt")
        if os.path.exists(filepath_txt):
            with open(filepath_txt, 'r', encoding='utf-8') as f: return f.read()
        return None

    def _get_auto_fixes(self):
        # Corrección para funciones reservadas y compatibilidad
        return """
        float _hash33(vec3 p3) { p3 = fract(p3 * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
        
        // Sobrecarga estándar 2D
        vec4 _toyTexture(sampler2D s, vec2 c) { return texture(s, c); }
        
        // Sobrecarga con BIAS (3 argumentos) -> Ignoramos el bias
        vec4 _toyTexture(sampler2D s, vec2 c, float b) { return texture(s, c); }

        // Sobrecarga 3D (CubeMap fake)
        vec4 _toyTexture(sampler2D s, vec3 c) { float h = _hash33(floor(c * 50.0)); return vec4(vec3(h), 1.0); }
        
        // Sobrecarga 3D con BIAS -> Ignoramos bias
        vec4 _toyTexture(sampler2D s, vec3 c, float b) { float h = _hash33(floor(c * 50.0)); return vec4(vec3(h), 1.0); }

        #define texture2D texture
        #define textureCube texture
        """

    def _get_wrapper(self, body_code, depth_var_name=None):
        # HACK: Reemplazar texture( por _toyTexture(
        body_code = re.sub(r'\btexture\s*\(', '_toyTexture(', body_code)
        
        channels_def = ""
        for i in range(4):
            channels_def += f"uniform sampler2D iChannel{i};\n"

        header = """
        #version 330
        #ifdef GL_ES
        precision mediump float;
        #endif
        uniform float iTime;
        uniform vec3 iResolution;
        uniform float scale;
        uniform float speed;
        uniform int iFrame;
        uniform float depthScale; 
        uniform vec4 iMouse = vec4(0.0);
        uniform float iTimeDelta;
        uniform float iFrameRate;
        uniform vec3 iChannelResolution[4];
        """ + channels_def + self._get_auto_fixes() + """
        out vec4 fragColor;
        void mainImage(out vec4 fragColor, in vec2 fragCoord);
        """
        
        footer_logic = """
            vec2 center = iResolution.xy * 0.5;
            vec2 scaledCoord = (gl_FragCoord.xy - center) * scale + center;
            mainImage(fragColor, scaledCoord);
            if (fragColor.a == 0.0) fragColor.a = 1.0; 
        """

        if depth_var_name and depth_var_name.strip() != "":
            last_brace_index = body_code.rfind('}')
            if last_brace_index != -1:
                injection = f"    fragColor.a = 1.0 / (1.0 + max(0.0, float({depth_var_name})) * max(0.001, depthScale));\n"
                body_code = body_code[:last_brace_index] + injection + body_code[last_brace_index:]

        footer = f"void main() {{ {footer_logic} }}"
        return header + "\n" + body_code + "\n" + footer

    def _compile_source(self, full_source):
        vertex_code = "#version 330\nin vec2 in_vert; void main() { gl_Position = vec4(in_vert, 0.0, 1.0); }"
        try:
            return self.ctx.program(vertex_shader=vertex_code, fragment_shader=full_source)
        except Exception as e:
            print(f"❌ Shader compilation error:\n{e}")
            return None

    def _create_program(self, code, depth_var=None):
        if not depth_var or depth_var.strip() == "":
            return self._compile_source(self._get_wrapper(code, None))
        
        prog = self._compile_source(self._get_wrapper(code, depth_var_name=depth_var))
        if prog: return prog
        print(f"⚠️ Error variable depth '{depth_var}'.")
        return self._compile_source(self._get_wrapper(code, None))

    def _update_uniforms(self, prog, w, h, t, f, spd, scl, d_scale):
        if 'iResolution' in prog: prog['iResolution'].value = (w, h, w/h)
        if 'iTime' in prog:       prog['iTime'].value = t * spd
        if 'iFrame' in prog:      prog['iFrame'].value = f
        if 'speed' in prog:       prog['speed'].value = spd
        if 'scale' in prog:       prog['scale'].value = scl
        safe_scale = d_scale if d_scale > 0.0001 else 0.0001
        if 'depthScale' in prog:  prog['depthScale'].value = safe_scale
        if 'iChannelResolution' in prog: prog['iChannelResolution'].value = [(512.0, 512.0, 1.0)] * 4

    def _upload_texture(self, channel_idx, img_np):
        h, w, c = img_np.shape
        if img_np.dtype != np.uint8:
            data = (img_np * 255).astype('u1').tobytes()
        else:
            data = img_np.tobytes()
            
        tex = self.ctx.texture((w, h), c, data)
        tex.repeat_x = True
        tex.repeat_y = True
        tex.use(location=channel_idx)
        return tex

    def render_multipass(self, width, height, time, frame_idx, speed, scale, depth_scale, code_main, code_buffer_a=None, depth_var_name=None, input_textures=None):
        try:
            self._init_context() 
            self.ctx.gc()

            if not hasattr(self, 'vbo'):
                vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
                self.vbo = self.ctx.buffer(vertices)

            managed_textures = []
            
            for i in range(4):
                tex_to_bind = self.fallback_texture
                
                if input_textures and i in input_textures and input_textures[i] is not None:
                    try:
                        new_tex = self._upload_texture(i, input_textures[i])
                        managed_textures.append(new_tex)
                        tex_to_bind = new_tex
                    except Exception as e:
                        print(f"Error cargando textura {i}: {e}")

                tex_to_bind.use(location=i)

            prog_main = self._create_program(code_main, depth_var=depth_var_name)
            if not prog_main: return np.zeros((height, width, 4), dtype=np.uint8)

            prog_buffer = None
            if code_buffer_a:
                prog_buffer = self._create_program(code_buffer_a)

            fbo_main = self.ctx.simple_framebuffer((width, height), components=4)
            texture_a = None
            fbo_a = None

            # BUFFER A
            if prog_buffer:
                texture_a = self.ctx.texture((width, height), 4, dtype='f1')
                fbo_a = self.ctx.framebuffer(color_attachments=[texture_a])
                fbo_a.use()
                fbo_a.clear(0.0, 0.0, 0.0, 1.0)
                self._update_uniforms(prog_buffer, width, height, time, frame_idx, speed, scale, depth_scale)
                if 'iChannel0' in prog_buffer: prog_buffer['iChannel0'].value = 0
                
                vao_a = self.ctx.simple_vertex_array(prog_buffer, self.vbo, 'in_vert')
                try: vao_a.render(moderngl.TRIANGLE_STRIP)
                except: pass
                vao_a.release()

            # MAIN IMAGE
            fbo_main.use()
            fbo_main.clear(0.0, 0.0, 0.0, 1.0) 
            self._update_uniforms(prog_main, width, height, time, frame_idx, speed, scale, depth_scale)

            if texture_a:
                texture_a.use(location=0)
            
            for i in range(4):
                if f'iChannel{i}' in prog_main: prog_main[f'iChannel{i}'].value = i

            vao_main = self.ctx.simple_vertex_array(prog_main, self.vbo, 'in_vert')
            try: vao_main.render(moderngl.TRIANGLE_STRIP)
            except Exception as e: print(f"⚠️ Error render Main: {e}")
            
            data = fbo_main.read(components=4)
            image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
            image = np.flipud(image)
            image = np.ascontiguousarray(image)

            for t in managed_textures: t.release()
            if fbo_a: fbo_a.release()
            if texture_a: texture_a.release()
            if prog_buffer: prog_buffer.release()
            vao_main.release(); fbo_main.release(); prog_main.release()

            return image

        except Exception as e:
            print(f"🔥 CRASH EVITADO: {e}")
            return np.zeros((height, width, 4), dtype=np.uint8)