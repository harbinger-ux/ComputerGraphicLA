import numpy as np
import OpenGL.GL as GL
from cglibs.buffer import VAO


class AxisGrid:
    def __init__(self, gl_bounds):
        gl_xmin, gl_xmax, gl_ymin, gl_ymax, gl_zmin, gl_zmax = gl_bounds

        # Drop the floor slightly below the minimum bounding box to prevent Z-fighting
        epsilon = 0.015
        floor_y = gl_ymin - epsilon

        self._verts, self._colors, self._normals, self._uvs, self._indices = [], [], [], [], []
        self._idx = 0

        base_thick = 0.0025
        axis_thick = 0.01

        # Subtle floor grid (gray dashes)
        for t in np.linspace(0, 1, 21):
            x = gl_xmin + t * (gl_xmax - gl_xmin)
            z = gl_zmin + t * (gl_zmax - gl_zmin)
            self._add_dashed_line(2, gl_zmin, gl_zmax, x, floor_y, [0.6, 0.6, 0.6], 0.025, base_thick)
            self._add_dashed_line(0, gl_xmin, gl_xmax, floor_y, z, [0.6, 0.6, 0.6], 0.025, base_thick)

        # Bold colored axes (continuous solid boxes)
        self._add_box(
            [gl_xmin, floor_y - axis_thick, gl_zmax - axis_thick],
            [gl_xmax, floor_y + axis_thick, gl_zmax + axis_thick],
            [1.0, 0.2, 0.2],  # Math X (Red)
        )
        self._add_box(
            [gl_xmin - axis_thick, floor_y - axis_thick, gl_zmin],
            [gl_xmin + axis_thick, floor_y + axis_thick, gl_zmax],
            [0.2, 0.4, 1.0],  # Math Y (Blue)
        )
        self._add_box(
            [gl_xmin - axis_thick, floor_y, gl_zmax - axis_thick],
            [gl_xmin + axis_thick, gl_ymax, gl_zmax + axis_thick],
            [0.2, 0.8, 0.2],  # Math Z (Green)
        )

        # Subtle back-wall framing (gray dashes)
        self._add_dashed_line(0, gl_xmin, gl_xmax, gl_ymax, gl_zmin, [0.8, 0.8, 0.8], 0.05, base_thick)
        self._add_dashed_line(2, gl_zmin, gl_zmax, gl_xmin, gl_ymax, [0.8, 0.8, 0.8], 0.05, base_thick)
        self._add_dashed_line(1, floor_y, gl_ymax, gl_xmin, gl_zmin, [0.8, 0.8, 0.8], 0.05, base_thick)

        self.verts = np.array(self._verts, dtype=np.float32)
        self.colors = np.array(self._colors, dtype=np.float32)
        self.normals = np.array(self._normals, dtype=np.float32)
        self.uvs = np.array(self._uvs, dtype=np.float32)
        self.indices = np.array(self._indices, dtype=np.uint32)

        self.vao = VAO()
        self.vao.add_vbo(0, self.verts, 3, GL.GL_FLOAT)
        self.vao.add_vbo(1, self.colors, 3, GL.GL_FLOAT)
        self.vao.add_vbo(2, self.normals, 3, GL.GL_FLOAT)
        self.vao.add_vbo(3, self.uvs, 2, GL.GL_FLOAT)
        self.vao.add_ebo(self.indices)
        self.i_count = len(self.indices)

    def _add_box(self, p_min, p_max, color):
        x0, y0, z0 = p_min
        x1, y1, z1 = p_max
        v = np.array([
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        ], dtype=np.float32)
        faces = [(0, 1, 2, 3), (1, 5, 6, 2), (5, 4, 7, 6), (4, 0, 3, 7), (3, 2, 6, 7), (4, 5, 1, 0)]
        for face in faces:
            p0, p1, p2 = v[face[0]], v[face[1]], v[face[2]]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            if norm > 0:
                n = n / norm
            for i_v in face:
                self._verts.append(v[i_v])
                self._normals.append(n)
                self._colors.append(color)
                self._uvs.append([0, 0])
            self._indices.extend([
                self._idx, self._idx + 1, self._idx + 2,
                self._idx, self._idx + 2, self._idx + 3,
            ])
            self._idx += 4

    def _add_dashed_line(self, axis, start, end, fixed1, fixed2, color, dash_len, thickness):
        dist = abs(end - start)
        if dist < 1e-5:
            return
        segments = max(2, int(dist / dash_len))
        if segments % 2 != 0:
            segments += 1

        for i in range(0, segments, 2):
            c0 = start + (end - start) * (i / segments)
            c1 = start + (end - start) * ((i + 1) / segments)
            c_min, c_max = min(c0, c1), max(c0, c1)

            if axis == 0:    # X-axis dash (fixed1=y, fixed2=z)
                p_min = [c_min, fixed1 - thickness, fixed2 - thickness]
                p_max = [c_max, fixed1 + thickness, fixed2 + thickness]
            elif axis == 1:  # Y-axis dash (fixed1=x, fixed2=z)
                p_min = [fixed1 - thickness, c_min, fixed2 - thickness]
                p_max = [fixed1 + thickness, c_max, fixed2 + thickness]
            else:            # Z-axis dash (fixed1=x, fixed2=y)
                p_min = [fixed1 - thickness, fixed2 - thickness, c_min]
                p_max = [fixed1 + thickness, fixed2 + thickness, c_max]

            self._add_box(p_min, p_max, color)

    def draw(self, shader, uman, projection, view, model):
        GL.glUseProgram(shader.render_idx)
        uman.upload_uniform_matrix4fv(projection, 'projection', transpose=True)
        uman.upload_uniform_matrix4fv(view @ model, 'modelview', transpose=True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, self.i_count, GL.GL_UNSIGNED_INT, None)
        self.vao.deactivate()