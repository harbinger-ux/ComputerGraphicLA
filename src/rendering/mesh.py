import ctypes
import os
import numpy as np
import OpenGL.GL as GL
from PIL import Image
from cglibs.buffer import UManager, VAO
from cglibs.shader import Shader
from cglibs.shading import emit_draw


class Mesh:
    LIGHT_POS = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    SHININESS = 32.0
    K_MATERIALS = np.array([
        [1.0, 1.0, 1.0],  # Diffuse modifier
        [0.5, 0.5, 0.5],  # Specular modifier
        [0.0, 0.0, 0.0],  # Ambient modifier
    ], dtype=np.float32)
    I_LIGHT = np.ones((3, 3), dtype=np.float32)

    def __init__(self, verts, colors, normals, uvs, indices):
        self.verts = np.asarray(verts, dtype=np.float32)
        self.base_colors = np.asarray(colors, dtype=np.float32).copy()
        self.normals = np.asarray(normals, dtype=np.float32)
        self.uvs = np.asarray(uvs, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.uint32)

        self.fill_count = len(self.indices)
        self.draw_as_lines = False
        self.shading = 'phong'

        self.vao = self._init_vao(self.verts, self.base_colors, self.normals, self.uvs, self.indices)
        self.programs = self._init_shaders()
        self.tex_id = self._init_checker_texture()

    @staticmethod
    def _init_vao(verts, colors, normals, uvs, indices):
        vao = VAO()
        vao.add_vbo(0, verts, ncomponents=3, dtype=GL.GL_FLOAT, offset=ctypes.c_void_p(0))
        vao.add_vbo(1, colors, ncomponents=3, dtype=GL.GL_FLOAT, offset=ctypes.c_void_p(0))
        vao.add_vbo(2, normals, ncomponents=3, dtype=GL.GL_FLOAT, offset=ctypes.c_void_p(0))
        vao.add_vbo(3, uvs, ncomponents=2, dtype=GL.GL_FLOAT, offset=ctypes.c_void_p(0))
        vao.add_ebo(indices)
        return vao

    @staticmethod
    def _init_shaders():
        programs = {}
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        shader_dir = os.path.join(src_dir, "shaders")

        for name in ['flat', 'gouraud', 'phong', 'interp']:
            s = Shader(
                os.path.join(shader_dir, f"{name}.vert"),
                os.path.join(shader_dir, f"{name}.frag"),
            )
            programs[name] = (s, UManager(s))

        s_blinn = Shader(
            os.path.join(shader_dir, "phong.vert"),
            os.path.join(shader_dir, "blinn.frag"),
        )
        s_tex = Shader(
            os.path.join(shader_dir, "phong_texture.vert"),
            os.path.join(shader_dir, "phong_texture.frag"),
        )
        programs['blinn'] = (s_blinn, UManager(s_blinn))
        programs['texture'] = (s_tex, UManager(s_tex))
        return programs

    @staticmethod
    def _init_checker_texture():
        tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)

        checker = np.zeros((8, 8, 4), dtype=np.uint8)
        for i in range(8):
            for j in range(8):
                checker[i, j] = [255, 255, 255, 255] if (i + j) % 2 == 0 else [100, 100, 100, 255]

        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, 8, 8, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, checker,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return tex_id

    @staticmethod
    def resolve_asset_path(filepath):
        cleaned = filepath.strip().strip('"').strip("'")
        if not cleaned:
            raise FileNotFoundError("Empty filepath provided.")

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [
            cleaned,
            os.path.join(project_root, cleaned),
            os.path.join(project_root, "assets", cleaned),
            os.path.join(project_root, "assets", os.path.basename(cleaned)),
            os.path.join(project_root, "src", cleaned),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

        raise FileNotFoundError(
            f"Cannot find '{cleaned}' (checked project root and assets/ folder)."
        )

    def reset_checker_texture(self):
        GL.glDeleteTextures([self.tex_id])
        self.tex_id = self._init_checker_texture()

    def load_texture_from_file(self, filepath):
        """Loads an external image (.png, .jpg, .webp) while preserving native transparency."""
        resolved_path = self.resolve_asset_path(filepath)

        try:
            with Image.open(resolved_path) as pil_img:
                pil_rgba = pil_img.convert("RGBA")
                flipped = pil_rgba.transpose(Image.FLIP_TOP_BOTTOM)
                rgba_image = np.ascontiguousarray(np.array(flipped, dtype=np.uint8))
        except Exception:
            raw = UManager.load_texture(resolved_path)
            if raw.dtype == np.uint16:
                raw = (raw >> 8).astype(np.uint8)
            else:
                raw = raw.astype(np.uint8)
            rgba_image = np.ascontiguousarray(np.flipud(raw))

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
            rgba_image.shape[1], rgba_image.shape[0], 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba_image,
        )
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return resolved_path

    def update_colors(self, uniform_rgb=None, use_gradient=False):
        if use_gradient and len(self.verts) > 0:
            vmin = self.verts.min(axis=0)
            vmax = self.verts.max(axis=0)
            span = np.where((vmax - vmin) < 1e-5, 1.0, vmax - vmin)
            colors = (0.15 + 0.85 * ((self.verts - vmin) / span)).astype(np.float32)
        elif uniform_rgb is not None:
            colors = np.full_like(self.base_colors, uniform_rgb, dtype=np.float32)
        else:
            colors = self.base_colors

        if 1 in self.vao.vbo:
            GL.glDeleteBuffers(1, [self.vao.vbo[1]])
        self.vao.add_vbo(1, colors, ncomponents=3, dtype=GL.GL_FLOAT, offset=ctypes.c_void_p(0))

    def draw(self, projection, view, model):
        active_shader = 'blinn' if self.shading == 'phong' else self.shading
        shader, uman = self.programs[active_shader]
        GL.glUseProgram(shader.render_idx)

        model_mat = model if model is not None else np.identity(4, np.float32)
        uman.upload_uniform_matrix4fv(projection, 'projection', transpose=True)
        uman.upload_uniform_matrix4fv(view @ model_mat, 'modelview', transpose=True)

        if active_shader in ['gouraud', 'blinn', 'texture']:
            uman.upload_uniform_vector3fv(self.LIGHT_POS, 'light_pos')
            uman.upload_uniform_scalar1f(self.SHININESS, 'shininess')
            uman.upload_uniform_matrix3fv(self.K_MATERIALS, 'K_materials', transpose=False)
            uman.upload_uniform_matrix3fv(self.I_LIGHT, 'I_light', transpose=False)

        if active_shader == 'texture':
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
            loc = GL.glGetUniformLocation(shader.render_idx, 'texture1')
            if loc != -1:
                GL.glUniform1i(loc, 0)
            uman.upload_uniform_scalar1f(0.0, 'phong_factor')
        else:
            GL.glDisable(GL.GL_BLEND)

        poly_mode = GL.GL_LINE if self.draw_as_lines else GL.GL_FILL
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, poly_mode)
        emit_draw(self.vao, GL.GL_TRIANGLES, self.fill_count, False)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        
        if GL.glIsEnabled(GL.GL_BLEND):
            GL.glDisable(GL.GL_BLEND)