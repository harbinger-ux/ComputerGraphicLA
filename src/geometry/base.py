import numpy as np


def finalize_mesh(verts, normals, uvs, indices, color_val=0.8):
    """Packs geometry lists into typed NumPy arrays expected by Mesh."""
    verts_arr = np.asarray(verts, dtype=np.float32)
    colors_arr = np.full((len(verts_arr), 3), color_val, dtype=np.float32)
    normals_arr = np.asarray(normals, dtype=np.float32)
    uvs_arr = np.asarray(uvs, dtype=np.float32)
    indices_arr = np.asarray(indices, dtype=np.uint32)
    return verts_arr, colors_arr, normals_arr, uvs_arr, indices_arr


def create_2d_planar_mesh(points_2d, indices=None):
    """
    Builds a 2D mesh in the XY plane (z = 0) with +Z normals and normalized UVs.
    If indices are omitted, performs fan triangulation from vertex 0 (for convex polygons).
    """
    pts = np.asarray(points_2d, dtype=np.float32)
    n_verts = len(pts)

    verts = np.column_stack([pts, np.zeros(n_verts, dtype=np.float32)])
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n_verts, 1))

    min_xy = pts.min(axis=0)
    span_xy = np.maximum(pts.max(axis=0) - min_xy, 1e-6)
    uvs = (pts - min_xy) / span_xy

    if indices is None:
        indices = []
        for i in range(1, n_verts - 1):
            indices.extend([0, i, i + 1])

    return finalize_mesh(verts, normals, uvs, indices)


def create_regular_polygon_2d(sides, radius=0.5, start_angle=np.pi / 2):
    angles = start_angle + np.arange(sides) * (2.0 * np.pi / sides)
    points_2d = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return create_2d_planar_mesh(points_2d)