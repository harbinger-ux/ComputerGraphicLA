import os
from src.loaders.obj_loader import ObjLoader
from src.loaders.ply_loader import PlyLoader


class ModelLoader:
    """Unified dispatcher for importing 3D mesh files (.obj and .ply)."""

    _LOADERS = {
        '.obj': ObjLoader,
        '.ply': PlyLoader,
    }

    @classmethod
    def load_model(cls, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        loader_cls = cls._LOADERS.get(ext)
        if loader_cls is None:
            supported = ", ".join(cls._LOADERS.keys())
            raise ValueError(f"Unsupported file extension '{ext}'. Supported formats: {supported}")
        return loader_cls.load_model(filepath)


def load_3d_model(filepath):
    """Wrapper function to align with the shape initialization flow."""
    return ModelLoader.load_model(filepath)