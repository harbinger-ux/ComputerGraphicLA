import numpy as np
from src.loaders.base_loader import BaseMeshLoader


class ObjLoader(BaseMeshLoader):
    """Wavefront .obj file importer."""

    @classmethod
    def load_model(cls, filepath):
        raw_verts, raw_uvs, raw_normals = [], [], []
        final_verts, final_uvs, final_normals, indices = [], [], [], []
        vertex_cache = {}
        next_index = 0
        has_normals = False
        has_uvs = False

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue
                prefix = parts[0]

                if prefix == 'v':
                    raw_verts.append([float(x) for x in parts[1:4]])
                elif prefix == 'vt':
                    raw_uvs.append([float(x) for x in parts[1:3]])
                elif prefix == 'vn':
                    raw_normals.append([float(x) for x in parts[1:4]])
                elif prefix == 'f':
                    face_verts = parts[1:]
                    for i in range(1, len(face_verts) - 1):
                        for v_str in (face_verts[0], face_verts[i], face_verts[i + 1]):
                            if v_str in vertex_cache:
                                indices.append(vertex_cache[v_str])
                            else:
                                v_data = v_str.split('/')
                                v_idx = int(v_data[0])
                                v_idx = v_idx - 1 if v_idx > 0 else len(raw_verts) + v_idx

                                vt_idx = -1
                                if len(v_data) > 1 and v_data[1]:
                                    vt_val = int(v_data[1])
                                    vt_idx = vt_val - 1 if vt_val > 0 else len(raw_uvs) + vt_val

                                vn_idx = -1
                                if len(v_data) > 2 and v_data[2]:
                                    vn_val = int(v_data[2])
                                    vn_idx = vn_val - 1 if vn_val > 0 else len(raw_normals) + vn_val

                                final_verts.append(raw_verts[v_idx])
                                if 0 <= vt_idx < len(raw_uvs):
                                    final_uvs.append(raw_uvs[vt_idx])
                                    has_uvs = True
                                else:
                                    final_uvs.append([0.0, 0.0])

                                if 0 <= vn_idx < len(raw_normals):
                                    final_normals.append(raw_normals[vn_idx])
                                    has_normals = True
                                else:
                                    final_normals.append([0.0, 1.0, 0.0])

                                vertex_cache[v_str] = next_index
                                indices.append(next_index)
                                next_index += 1

        n_verts = len(final_verts)
        verts_arr = cls.normalize_vertices(np.array(final_verts, dtype=np.float32))
        colors_arr = np.ones((n_verts, 3), dtype=np.float32) * 0.8
        indices_arr = np.array(indices, dtype=np.uint32)
        uvs_arr = np.array(final_uvs, dtype=np.float32) if has_uvs else cls.generate_planar_uvs(verts_arr)

        if has_normals:
            normals_arr = np.array(final_normals, dtype=np.float32)
        else:
            normals_arr = cls.compute_vertex_normals(verts_arr, indices_arr)

        return verts_arr, colors_arr, normals_arr, uvs_arr, indices_arr