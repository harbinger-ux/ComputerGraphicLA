import math
import numpy as np
from src.utils.math_parser import parse_formula


def create_surface(
    func_str="sin(sqrt(x**2 + y**2)) / sqrt(x**2 + y**2)",
    x_min=-10.0,
    x_max=10.0,
    y_min=-10.0,
    y_max=10.0,
    res=60,
    use_jet=True,
):
    np_func_str = parse_formula(func_str)

    # 1. Generate matrix grids for mathematical X and Y values
    x_vals = np.linspace(x_min, x_max, res + 1, dtype=np.float32)
    y_vals = np.linspace(y_min, y_max, res + 1, dtype=np.float32)
    X, Y = np.meshgrid(x_vals, y_vals)

    # 2. Vectorized Evaluation
    with np.errstate(all='ignore'):
        try:
            Z = eval(np_func_str, {"np": np, "math": math, "x": X, "y": Y})
            if np.isscalar(Z):
                Z = np.full_like(X, Z)
            Z[~np.isfinite(Z)] = 0.0
        except Exception as e:
            print(f"Mathematical evaluation failed: {e}")
            Z = np.zeros_like(X)

    min_z, max_z = float(np.min(Z)), float(np.max(Z))
    if max_z == min_z:
        max_z = min_z + 1.0

    # 3. Vectorized Mapping to OpenGL target volume (4.0 x 2.0 x 4.0)
    gl_X = (X - x_min) / (x_max - x_min) * 4.0 - 2.0
    gl_Z = (Y - y_min) / (y_max - y_min) * 4.0 - 2.0
    gl_Y = (Z - min_z) / (max_z - min_z) * 2.0 - 1.0
    verts = np.stack([gl_X, gl_Y, gl_Z], axis=-1).reshape(-1, 3)

    # 4. Vectorized UV Mapping
    u = np.linspace(0, 1, res + 1, dtype=np.float32)
    v = np.linspace(0, 1, res + 1, dtype=np.float32)
    U, V = np.meshgrid(u, v)
    uvs = np.stack([U, V], axis=-1).reshape(-1, 2)

    # 5. Vectorized Jet Color Mapping
    if use_jet:
        Z_flat = Z.reshape(-1)
        v_val = (Z_flat - min_z) / (max_z - min_z)
        r = np.clip(1.5 - np.abs(4.0 * v_val - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * v_val - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * v_val - 1.0), 0.0, 1.0)
        colors = np.stack([r, g, b], axis=-1).astype(np.float32)
    else:
        colors = np.full((verts.shape[0], 3), 0.8, dtype=np.float32)

    # 6. Vectorized Triangle Index Generation
    i_idx = np.arange(res)
    j_idx = np.arange(res)
    I, J = np.meshgrid(i_idx, j_idx, indexing='ij')

    k1 = I * (res + 1) + J
    k2 = (I + 1) * (res + 1) + J

    tri1 = np.stack([k1, k2, k1 + 1], axis=-1)
    tri2 = np.stack([k1 + 1, k2, k2 + 1], axis=-1)
    indices = np.concatenate([tri1, tri2], axis=-1).flatten().astype(np.uint32)

    # 7. Fast Normal Accumulation
    normals = np.zeros_like(verts)
    indices_arr = indices.reshape(-1, 3)
    v0, v1, v2 = verts[indices_arr[:, 0]], verts[indices_arr[:, 1]], verts[indices_arr[:, 2]]

    n = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, indices_arr[:, 0], n)
    np.add.at(normals, indices_arr[:, 1], n)
    np.add.at(normals, indices_arr[:, 2], n)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    bounds = (x_min, x_max, y_min, y_max, min_z, max_z)
    return verts, colors, normals, uvs, indices, bounds