import os
import struct
import time
import numpy as np
from src.loaders import load_3d_model
from src.loaders.base_loader import BaseMeshLoader


class PlyOptimizer:
    """
    Optimizes high-poly .ply / .obj meshes using 3D Voxel Grid Vertex Clustering
    and exports the simplified mesh as a compact binary_little_endian .ply file.
    """

    @staticmethod
    def decimate_voxel_grid(verts, colors, normals, uvs, indices, grid_res=80):
        """
        Clusters vertices into a (grid_res x grid_res x grid_res) 3D grid,
        averages attributes per cell, removes degenerate faces, and recomputes normals.
        """
        n_orig_verts = len(verts)
        n_orig_faces = len(indices) // 3

        if n_orig_verts == 0 or n_orig_faces == 0:
            return verts, colors, normals, uvs, indices

        # 1. Quantize vertex coordinates into discrete 3D voxel indices
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        span = np.maximum(vmax - vmin, 1e-6)

        norm_coords = (verts - vmin) / span
        grid_coords = np.clip((norm_coords * grid_res).astype(np.int64), 0, grid_res - 1)

        # Pack 3D voxel indices (gx, gy, gz) into a single 1D spatial hash key
        stride = grid_res
        voxel_keys = (
            grid_coords[:, 0] * (stride * stride)
            + grid_coords[:, 1] * stride
            + grid_coords[:, 2]
        )

        # 2. Group vertices that fall into the same voxel cell
        _, inverse_map, counts = np.unique(
            voxel_keys, return_inverse=True, return_counts=True
        )
        n_new_verts = len(counts)
        counts_col = counts[:, None].astype(np.float32)

        # 3. Vectorized averaging of vertex positions, colors, and UVs per voxel
        new_verts = np.zeros((n_new_verts, 3), dtype=np.float32)
        np.add.at(new_verts, inverse_map, verts)
        new_verts /= counts_col

        new_colors = np.zeros((n_new_verts, 3), dtype=np.float32)
        np.add.at(new_colors, inverse_map, colors)
        new_colors /= counts_col

        new_uvs = np.zeros((n_new_verts, 2), dtype=np.float32)
        np.add.at(new_uvs, inverse_map, uvs)
        new_uvs /= counts_col

        # 4. Remap triangle face indices and prune degenerate triangles
        tris = indices.reshape(-1, 3)
        mapped_tris = inverse_map[tris]

        valid_mask = (
            (mapped_tris[:, 0] != mapped_tris[:, 1])
            & (mapped_tris[:, 1] != mapped_tris[:, 2])
            & (mapped_tris[:, 2] != mapped_tris[:, 0])
        )
        clean_tris = mapped_tris[valid_mask].astype(np.uint32)

        # Remove duplicate triangles created by collapsing adjacent quads
        if len(clean_tris) > 0:
            sorted_tris = np.sort(clean_tris, axis=1)
            _, unique_tri_idx = np.unique(sorted_tris, axis=0, return_index=True)
            clean_tris = clean_tris[np.sort(unique_tri_idx)]

        new_indices = clean_tris.flatten().astype(np.uint32)

        # 5. Recompute smooth vertex normals on the simplified topology
        new_normals = BaseMeshLoader.compute_vertex_normals(new_verts, new_indices)

        return new_verts, new_colors, new_normals, new_uvs, new_indices

    @staticmethod
    def save_binary_ply(filepath, verts, colors, normals, indices):
        """Saves mesh buffers as a compact binary_little_endian .ply file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        n_verts = len(verts)
        tris = indices.reshape(-1, 3).astype(np.int32)
        n_faces = len(tris)

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "comment Optimized via Voxel Grid Clustering\n"
            f"element vertex {n_verts}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            f"element face {n_faces}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )

        # Pack vertex array using a structured NumPy dtype for instant binary write
        v_dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
        ])
        v_buf = np.empty(n_verts, dtype=v_dtype)
        v_buf['x'], v_buf['y'], v_buf['z'] = verts[:, 0], verts[:, 1], verts[:, 2]
        v_buf['nx'], v_buf['ny'], v_buf['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]

        rgb_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        v_buf['r'], v_buf['g'], v_buf['b'] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]

        # Pack triangle faces: 1 byte count (3) + three 4-byte vertex indices
        f_dtype = np.dtype([
            ('count', 'u1'),
            ('i0', '<i4'), ('i1', '<i4'), ('i2', '<i4'),
        ])
        f_buf = np.empty(n_faces, dtype=f_dtype)
        f_buf['count'] = 3
        f_buf['i0'], f_buf['i1'], f_buf['i2'] = tris[:, 0], tris[:, 1], tris[:, 2]

        with open(filepath, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(v_buf.tobytes())
            f.write(f_buf.tobytes())

    @classmethod
    def optimize_file(cls, input_path, grid_res=80, output_path=None):
        """
        Loads input_path, decimates it on a 3D voxel grid, saves it to assets/<name>_opt.ply,
        and returns a dictionary of before/after comparison metrics.
        """
        t0 = time.perf_counter()
        verts, colors, normals, uvs, indices = load_3d_model(input_path)

        orig_verts = len(verts)
        orig_faces = len(indices) // 3
        orig_size_mb = os.path.getsize(input_path) / (1024.0 * 1024.0)

        opt_v, opt_c, opt_n, opt_uv, opt_idx = cls.decimate_voxel_grid(
            verts, colors, normals, uvs, indices, grid_res=grid_res
        )

        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            # Strip trailing _opt if user re-optimizes an already optimized file
            if base_name.endswith("_opt"):
                base_name = base_name[:-4]
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            output_path = os.path.join(project_root, "assets", f"{base_name}_opt.ply")

        cls.save_binary_ply(output_path, opt_v, opt_c, opt_n, opt_idx)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        opt_verts = len(opt_v)
        opt_faces = len(opt_idx) // 3
        opt_size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)

        stats = {
            "input_path": input_path,
            "output_path": output_path,
            "output_filename": os.path.basename(output_path),
            "orig_verts": orig_verts,
            "opt_verts": opt_verts,
            "orig_faces": orig_faces,
            "opt_faces": opt_faces,
            "orig_size_mb": orig_size_mb,
            "opt_size_mb": opt_size_mb,
            "reduction_pct": (1.0 - (opt_faces / max(orig_faces, 1))) * 100.0,
            "elapsed_ms": elapsed_ms,
        }
        return (opt_v, opt_c, opt_n, opt_uv, opt_idx), stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.mesh_optimizer <path_to_model.ply> [grid_res=80]")
        sys.exit(1)

    in_file = sys.argv[1]
    res = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    _, report = PlyOptimizer.optimize_file(in_file, grid_res=res)
    print(
        f"Optimized '{os.path.basename(report['input_path'])}' -> 'assets/{report['output_filename']}'\n"
        f"  Vertices : {report['orig_verts']:,} -> {report['opt_verts']:,}\n"
        f"  Triangles: {report['orig_faces']:,} -> {report['opt_faces']:,} (-{report['reduction_pct']:.1f}%)\n"
        f"  File Size: {report['orig_size_mb']:.2f} MB -> {report['opt_size_mb']:.2f} MB\n"
        f"  Time     : {report['elapsed_ms']:.1f} ms"
    )