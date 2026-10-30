"""Shared shading-mode names and F/G/P/I/B/T (+ optional L lines) keys."""
import os

import glfw
import numpy as np
import OpenGL.GL as GL

from .shader import Shader
from .buffer import UManager


def shading_files(shader_dir="./shaders"):
    """Map F/G/P/I names to (vert, frag) paths under ``shader_dir``."""
    return {
        'flat':    (os.path.join(shader_dir, "flat.vert"),    os.path.join(shader_dir, "flat.frag")),
        'gouraud': (os.path.join(shader_dir, "gouraud.vert"), os.path.join(shader_dir, "gouraud.frag")),
        'phong':   (os.path.join(shader_dir, "phong.vert"),   os.path.join(shader_dir, "phong.frag")),
        'interp':  (os.path.join(shader_dir, "interp.vert"),  os.path.join(shader_dir, "interp.frag")),
    }


def shading_files_beside(py_file):
    """Shaders in ``shaders/`` next to the calling ``.py`` file."""
    return shading_files(os.path.join(os.path.dirname(os.path.abspath(py_file)), "shaders"))


def shader_beside(py_file, filename):
    """``shaders/filename`` next to ``py_file``."""
    return os.path.join(os.path.dirname(os.path.abspath(py_file)), "shaders", filename)


def _shared(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders", name)


def load_fgpi(programs, src_py):
    """Compile F/G/P/I from ``src_py``'s ``shaders/`` folder."""
    for name, (vs, fs) in shading_files_beside(src_py).items():
        shader = Shader(vs, fs)
        programs[name] = (shader, UManager(shader))


def add_blinn(programs, phong_vert_py):
    """Blinn-Phong fragment + the phong vertex shader next to ``phong_vert_py``."""
    shader = Shader(shader_beside(phong_vert_py, "phong.vert"), _shared("blinn.frag"))
    programs['blinn'] = (shader, UManager(shader))


def add_texture(programs):
    """Shared textured Phong (attribute location 3 = UV)."""
    shader = Shader(_shared("phong_texture.vert"), _shared("phong_texture.frag"))
    programs['texture'] = (shader, UManager(shader))


def load_shape_programs(programs, src_py):
    """F/G/P/I from ``src_py``, plus shared Blinn and textured Phong."""
    load_fgpi(programs, src_py)
    add_blinn(programs, src_py)
    add_texture(programs)


# cwd-relative paths (slides / old snippets). Prefer shading_files_beside(__file__).
SHADING_FILES = shading_files()

_LABELS = {
    'flat':    "FLAT",
    'gouraud': "GOURAUD",
    'phong':   "PHONG",
    'interp':  "INTERP (vertex color)",
    'texture': "TEXTURE",
    'edge':    "EDGE",
    'blinn':   "BLINN-PHONG",
}


def handle_shading_key(obj, key):
    if key == glfw.KEY_L and hasattr(obj, 'draw_as_lines'):
        obj.draw_as_lines = not obj.draw_as_lines
        print("lines:", "ON (shape edges)" if obj.draw_as_lines else "OFF")
        return
    mapping = {
        glfw.KEY_F: 'flat',
        glfw.KEY_G: 'gouraud',
        glfw.KEY_P: 'phong',
        glfw.KEY_I: 'interp',
        glfw.KEY_T: 'texture',
        glfw.KEY_E: 'edge',
        glfw.KEY_B: 'blinn',
    }
    name = mapping.get(key)
    if name is None or name not in getattr(obj, 'programs', {}):
        return
    obj.shading = name
    print("Shading:", _LABELS.get(name, name.upper()))


def emit_draw(vao, fill_mode, fill_count, draw_as_lines, indexed=True):
    """Filled mesh, or GL_LINES of modeling edges (not the W triangulation)."""
    vao.activate()
    if draw_as_lines and getattr(vao, 'line_ebo', None) is not None:
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vao.line_ebo)
        GL.glDrawElements(GL.GL_LINES, vao.n_line_idx, GL.GL_UNSIGNED_INT, None)
        if vao.ebo is not None:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vao.ebo)
        return
    if indexed:
        GL.glDrawElements(fill_mode, fill_count, GL.GL_UNSIGNED_INT, None)
    else:
        GL.glDrawArrays(fill_mode, 0, fill_count)


def ring_lines(n, start=1):
    """n segments along a ring stored as start .. start+n (last copies first)."""
    out = []
    for i in range(n):
        out.extend([start + i, start + i + 1])
    return np.array(out, dtype=np.int32)


def prism_side_lines(n):
    """Verticals + rings for a side strip: column i is (bottom 2i, top 2i+1)."""
    out = []
    for i in range(n):
        out.extend([2 * i, 2 * i + 1])
    for i in range(n):
        out.extend([2 * i, 2 * (i + 1)])
        out.extend([2 * i + 1, 2 * (i + 1) + 1])
    return np.array(out, dtype=np.int32)


def triangle_edges(indices):
    """Unique edges of a triangle mesh (flat index array). For geodesic L."""
    seen = set()
    out = []
    faces = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
    for a, b, c in faces:
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (i, j) if i < j else (j, i)
            if key not in seen:
                seen.add(key)
                out.extend(key)
    return np.array(out, dtype=np.int32)


def sphere_grid_lines(slices, stacks):
    """Meridians + parallels on a UV sphere (not every triangle edge)."""
    cols = slices + 1
    out = []
    for i in range(slices):
        for j in range(stacks):
            a = j * cols + i
            out.extend([a, a + cols])
    for j in range(1, stacks):
        for i in range(slices):
            a = j * cols + i
            out.extend([a, a + 1])
    return np.array(out, dtype=np.int32)


def grid_lines(nu, nv):
    """Axis-aligned edges of an nu-by-nv vertex grid (u fastest). Height-field L."""
    out = []
    for j in range(nv):
        for i in range(nu - 1):
            a = j * nu + i
            out.extend([a, a + 1])
    for j in range(nv - 1):
        for i in range(nu):
            a = j * nu + i
            out.extend([a, a + nu])
    return np.array(out, dtype=np.int32)
