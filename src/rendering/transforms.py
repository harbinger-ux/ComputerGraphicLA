import math
import numpy as np
from cglibs import transform as T


def translate(tx, ty, tz):
    return T.translate(tx, ty, tz).astype(np.float32)


def scale(sx, sy, sz):
    return T.scale(sx, sy, sz).astype(np.float32)


def rotate_x(angle_rad):
    return T.rotate((1.0, 0.0, 0.0), radians=angle_rad).astype(np.float32)


def rotate_y(angle_rad):
    return T.rotate((0.0, 1.0, 0.0), radians=angle_rad).astype(np.float32)


def rotate_z(angle_rad):
    return T.rotate((0.0, 0.0, 1.0), radians=angle_rad).astype(np.float32)


def compose_trs(translation_xyz, rotation_deg_xyz, scale_xyz):
    """Composes a 4x4 model matrix: M = T * Rz * Ry * Rx * S."""
    tx, ty, tz = translation_xyz
    rx, ry, rz = np.radians(rotation_deg_xyz)
    sx, sy, sz = scale_xyz
    return (
        translate(tx, ty, tz)
        @ rotate_z(rz)
        @ rotate_y(ry)
        @ rotate_x(rx)
        @ scale(sx, sy, sz)
    )


def align_y_to_vector(vec):
    """Returns a 4x4 rotation matrix aligning the unit +Y axis (0, 1, 0) with `vec`."""
    v = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        return np.identity(4, dtype=np.float32)

    u = v / norm
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    dot = float(np.clip(np.dot(y_axis, u), -1.0, 1.0))

    if dot > 0.99999:
        return np.identity(4, dtype=np.float32)
    if dot < -0.99999:
        return rotate_z(math.pi)

    axis = np.cross(y_axis, u)
    angle = math.acos(dot)
    return T.rotate(axis, radians=angle).astype(np.float32)


def get_initial_rotation():
    return rotate_x(np.radians(30.0)) @ rotate_y(np.radians(-45.0))