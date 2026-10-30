"""Draw a shared mesh with its own model matrix (p_clip = P V M p)."""
import numpy as np

from . import transform as T


class Placed:
    """One instance: translate, yaw about Y, then scale. Then the mesh."""

    def __init__(self, mesh, pos=(0.0, 0.0, 0.0), yaw=0.0, scale=1.0):
        self.mesh = mesh
        self.pos = np.array(pos, dtype=np.float32)
        self.yaw = float(yaw)
        if isinstance(scale, (int, float)):
            self.scl = (float(scale), float(scale), float(scale))
        else:
            self.scl = tuple(float(c) for c in scale)

    def model(self):
        sx, sy, sz = self.scl
        return T.translate(*self.pos) @ T.rotate((0.0, 1.0, 0.0), self.yaw) @ T.scale(sx, sy, sz)

    def draw(self, projection, view, model):
        M = self.model()
        if model is not None:
            M = model @ M
        self.mesh.draw(projection, view @ M, None)
