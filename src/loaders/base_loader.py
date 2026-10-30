import numpy as np


class BaseMeshLoader:
    """Base utility class providing normal computation and normalization for 3D mesh loaders."""

    @staticmethod
    def compute_vertex_normals(verts_arr, indices_arr):
        normals = np.zeros_like(verts_arr, dtype=np.float32)
        if len(indices_arr) == 0:
            normals[:, 1] = 1.0
            return normals

        tris = indices_arr.reshape(-1, 3)
        v0 = verts_arr[tris[:, 0]]
        v1 = verts_arr[tris[:, 1]]
        v2 = verts_arr[tris[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        np.add.at(normals, tris[:, 0], face_normals)
        np.add.at(normals, tris[:, 1], face_normals)
        np.add.at(normals, tris[:, 2], face_normals)

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (normals / norms).astype(np.float32)

    @staticmethod
    def normalize_vertices(verts_arr, target_extent=2.0):
        if len(verts_arr) == 0:
            return verts_arr
        vmin = verts_arr.min(axis=0)
        vmax = verts_arr.max(axis=0)
        center = 0.5 * (vmin + vmax)
        max_span = float(np.max(vmax - vmin))
        if max_span < 1e-6:
            return (verts_arr - center).astype(np.float32)
        return ((verts_arr - center) * (target_extent / max_span)).astype(np.float32)

    @staticmethod
    def generate_planar_uvs(verts_arr):
        if len(verts_arr) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        xy = verts_arr[:, :2]
        min_xy = xy.min(axis=0)
        span = np.where((xy.max(axis=0) - min_xy) < 1e-6, 1.0, xy.max(axis=0) - min_xy)
        return ((xy - min_xy) / span).astype(np.float32)