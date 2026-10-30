import struct
import numpy as np
from src.loaders.base_loader import BaseMeshLoader


class PlyLoader(BaseMeshLoader):
    """Stanford .ply file importer supporting ASCII and binary (little/big endian) formats."""

    _PLY_TYPE_MAP = {
        'char': ('b', 1), 'int8': ('b', 1),
        'uchar': ('B', 1), 'uint8': ('B', 1),
        'short': ('h', 2), 'int16': ('h', 2),
        'ushort': ('H', 2), 'uint16': ('H', 2),
        'int': ('i', 4), 'int32': ('i', 4),
        'uint': ('I', 4), 'uint32': ('I', 4),
        'float': ('f', 4), 'float32': ('f', 4),
        'double': ('d', 8), 'float64': ('d', 8),
    }

    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            fmt, elements = cls._parse_header(f)
            if fmt == 'ascii':
                vertex_data, indices = cls._read_ascii_body(f, elements)
            else:
                endian = '<' if fmt == 'binary_little_endian' else '>'
                vertex_data, indices = cls._read_binary_body(f, elements, endian)

        return cls._build_buffers(vertex_data, indices)

    @classmethod
    def _parse_header(cls, f):
        magic = f.readline().decode('ascii', errors='ignore').strip()
        if magic != 'ply':
            raise ValueError(f"Invalid PLY file (missing 'ply' header): {magic}")

        fmt = 'ascii'
        elements = []
        current_element = None

        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header")
            text = line.decode('ascii', errors='ignore').strip()
            if not text or text.startswith('comment') or text.startswith('obj_info'):
                continue
            if text == 'end_header':
                break

            parts = text.split()
            keyword = parts[0]
            if keyword == 'format':
                fmt = parts[1]
            elif keyword == 'element':
                current_element = {
                    'name': parts[1],
                    'count': int(parts[2]),
                    'properties': [],
                }
                elements.append(current_element)
            elif keyword == 'property' and current_element is not None:
                if parts[1] == 'list':
                    current_element['properties'].append({
                        'kind': 'list',
                        'count_type': parts[2],
                        'item_type': parts[3],
                        'name': parts[4],
                    })
                else:
                    current_element['properties'].append({
                        'kind': 'scalar',
                        'type': parts[1],
                        'name': parts[2],
                    })

        return fmt, elements

    @classmethod
    def _read_ascii_body(cls, f, elements):
        vertex_data = {}
        indices = []

        for elem in elements:
            name, count, props = elem['name'], elem['count'], elem['properties']
            if name == 'vertex':
                prop_names = [p['name'] for p in props]
                rows = []
                for _ in range(count):
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    while not line or line.startswith('comment'):
                        line = f.readline().decode('ascii', errors='ignore').strip()
                    rows.append([float(x) for x in line.split()[:len(prop_names)]])
                arr = (
                    np.array(rows, dtype=np.float32)
                    if rows
                    else np.zeros((0, len(prop_names)), dtype=np.float32)
                )
                for idx, p_name in enumerate(prop_names):
                    vertex_data[p_name] = arr[:, idx]
                    if p_name in ('red', 'green', 'blue', 'r', 'g', 'b'):
                        vertex_data[f"{p_name}_type"] = props[idx]['type']

            elif name == 'face':
                for _ in range(count):
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    while not line or line.startswith('comment'):
                        line = f.readline().decode('ascii', errors='ignore').strip()
                    parts = [int(x) for x in line.split()]
                    n_poly = parts[0]
                    face_v = parts[1:1 + n_poly]
                    for i in range(1, len(face_v) - 1):
                        indices.extend([face_v[0], face_v[i], face_v[i + 1]])
            else:
                for _ in range(count):
                    f.readline()

        return vertex_data, indices

    @classmethod
    def _read_binary_body(cls, f, elements, endian):
        vertex_data = {}
        indices = []

        for elem in elements:
            name, count, props = elem['name'], elem['count'], elem['properties']
            if name == 'vertex' and all(p['kind'] == 'scalar' for p in props):
                fmt_chars = ''.join(cls._PLY_TYPE_MAP[p['type']][0] for p in props)
                row_struct = struct.Struct(endian + fmt_chars)
                raw_bytes = f.read(row_struct.size * count)
                unpacked = list(row_struct.iter_unpack(raw_bytes))
                arr = (
                    np.array(unpacked, dtype=np.float32)
                    if unpacked
                    else np.zeros((0, len(props)), dtype=np.float32)
                )
                for idx, p in enumerate(props):
                    p_name = p['name']
                    vertex_data[p_name] = arr[:, idx]
                    if p_name in ('red', 'green', 'blue', 'r', 'g', 'b'):
                        vertex_data[f"{p_name}_type"] = p['type']
            elif name == 'face':
                for _ in range(count):
                    for p in props:
                        if p['kind'] == 'list':
                            c_fmt, c_size = cls._PLY_TYPE_MAP[p['count_type']]
                            i_fmt, i_size = cls._PLY_TYPE_MAP[p['item_type']]
                            n_poly = struct.unpack(endian + c_fmt, f.read(c_size))[0]
                            face_v = struct.unpack(endian + (i_fmt * n_poly), f.read(i_size * n_poly))
                            if p['name'] in ('vertex_indices', 'vertex_index'):
                                for i in range(1, len(face_v) - 1):
                                    indices.extend([face_v[0], face_v[i], face_v[i + 1]])
                        else:
                            _, s_size = cls._PLY_TYPE_MAP[p['type']]
                            f.read(s_size)
            else:
                for _ in range(count):
                    for p in props:
                        if p['kind'] == 'list':
                            c_fmt, c_size = cls._PLY_TYPE_MAP[p['count_type']]
                            _, i_size = cls._PLY_TYPE_MAP[p['item_type']]
                            n_items = struct.unpack(endian + c_fmt, f.read(c_size))[0]
                            f.read(i_size * n_items)
                        else:
                            _, s_size = cls._PLY_TYPE_MAP[p['type']]
                            f.read(s_size)

        return vertex_data, indices

    @classmethod
    def _build_buffers(cls, vertex_data, indices):
        x = vertex_data.get('x', np.zeros(0, dtype=np.float32))
        y = vertex_data.get('y', np.zeros_like(x))
        z = vertex_data.get('z', np.zeros_like(x))
        verts_arr = cls.normalize_vertices(np.column_stack([x, y, z]).astype(np.float32))
        n_verts = len(verts_arr)

        indices_arr = np.array(indices, dtype=np.uint32)

        if {'red', 'green', 'blue'}.issubset(vertex_data.keys()):
            r, g, b = vertex_data['red'], vertex_data['green'], vertex_data['blue']
            c_type = vertex_data.get('red_type', 'uchar')
            colors_arr = np.column_stack([r, g, b]).astype(np.float32)
            if c_type in ('uchar', 'uint8') or colors_arr.max(initial=0.0) > 1.0:
                colors_arr /= 255.0
        elif {'r', 'g', 'b'}.issubset(vertex_data.keys()):
            r, g, b = vertex_data['r'], vertex_data['g'], vertex_data['b']
            c_type = vertex_data.get('r_type', 'uchar')
            colors_arr = np.column_stack([r, g, b]).astype(np.float32)
            if c_type in ('uchar', 'uint8') or colors_arr.max(initial=0.0) > 1.0:
                colors_arr /= 255.0
        else:
            colors_arr = np.full((n_verts, 3), 0.8, dtype=np.float32)

        u = next((vertex_data[k] for k in ('u', 's', 'texture_u') if k in vertex_data), None)
        v = next((vertex_data[k] for k in ('v', 't', 'texture_v') if k in vertex_data), None)
        if u is not None and v is not None:
            uvs_arr = np.column_stack([u, v]).astype(np.float32)
        else:
            uvs_arr = cls.generate_planar_uvs(verts_arr)

        if {'nx', 'ny', 'nz'}.issubset(vertex_data.keys()):
            normals_arr = np.column_stack([
                vertex_data['nx'], vertex_data['ny'], vertex_data['nz'],
            ]).astype(np.float32)
        else:
            normals_arr = cls.compute_vertex_normals(verts_arr, indices_arr)

        return verts_arr, colors_arr, normals_arr, uvs_arr, indices_arr