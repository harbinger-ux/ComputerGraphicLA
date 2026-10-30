import numpy as np


class SceneNode:
    """Hierarchical node to manage relative transformations for 3D objects."""

    def __init__(self, mesh=None):
        self.mesh = mesh
        self.transform = np.identity(4, dtype=np.float32)
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child

    def draw(self, proj, view, parent_transform):
        world_transform = parent_transform @ self.transform
        if self.mesh:
            self.mesh.draw(proj, view, world_transform)
        for child in self.children:
            child.draw(proj, view, world_transform)