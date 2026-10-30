import math
import numpy as np
from src.geometry.base import finalize_mesh


def create_cube():
    v = np.array([
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
    ], dtype=np.float32)
    faces = [(0, 1, 2, 3), (1, 5, 6, 2), (5, 4, 7, 6), (4, 0, 3, 7), (3, 2, 6, 7), (4, 5, 1, 0)]
    face_uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]

    verts, normals, uvs, indices = [], [], [], []
    idx = 0
    for face in faces:
        p0, p1, p2 = v[face[0]], v[face[1]], v[face[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        normal = normal / np.linalg.norm(normal)
        for i, v_idx in enumerate(face):
            verts.append(v[v_idx])
            normals.append(normal)
            uvs.append(face_uvs[i])
        indices.extend([idx, idx + 1, idx + 2, idx, idx + 2, idx + 3])
        idx += 4

    return finalize_mesh(verts, normals, uvs, indices)


def create_sphere(radius=0.5, sectors=36, stacks=18):
    verts, normals, uvs, indices = [], [], [], []
    for i in range(stacks + 1):
        lat = np.pi / 2.0 - i * np.pi / stacks
        z = radius * np.sin(lat)
        xy = radius * np.cos(lat)
        for j in range(sectors + 1):
            lon = j * 2.0 * np.pi / sectors
            x, y = xy * np.cos(lon), xy * np.sin(lon)
            verts.append([x, y, z])
            normals.append([x / radius, y / radius, z / radius])
            uvs.append([j / sectors, i / stacks])

    for i in range(stacks):
        for j in range(sectors):
            k1 = i * (sectors + 1) + j
            k2 = (i + 1) * (sectors + 1) + j
            if i != 0:
                indices.extend([k1, k2, k1 + 1])
            if i != stacks - 1:
                indices.extend([k1 + 1, k2, k2 + 1])

    return finalize_mesh(verts, normals, uvs, indices)


def create_cylinder(radius=0.5, height=1.0, sectors=36):
    verts, normals, uvs, indices = [], [], [], []
    for y_sign in [1.0, -1.0]:
        center_idx = len(verts)
        verts.append([0.0, y_sign * height / 2, 0.0])
        normals.append([0.0, y_sign, 0.0])
        uvs.append([0.5, 0.5])
        for i in range(sectors + 1):
            theta = i * 2.0 * np.pi / sectors
            verts.append([radius * np.cos(theta), y_sign * height / 2, radius * np.sin(theta)])
            normals.append([0.0, y_sign, 0.0])
            uvs.append([0.5 + 0.5 * np.cos(theta), 0.5 + 0.5 * np.sin(theta)])
            if i < sectors:
                indices.extend([
                    center_idx,
                    center_idx + i + (2 if y_sign > 0 else 1),
                    center_idx + i + (1 if y_sign > 0 else 2),
                ])
    wall_idx = len(verts)
    for i in range(sectors + 1):
        theta = i * 2.0 * np.pi / sectors
        x, z = radius * np.cos(theta), radius * np.sin(theta)
        verts.extend([[x, height / 2, z], [x, -height / 2, z]])
        normals.extend([[x / radius, 0.0, z / radius]] * 2)
        uvs.extend([[i / sectors, 1.0], [i / sectors, 0.0]])
        if i < sectors:
            idx = wall_idx + i * 2
            indices.extend([idx, idx + 2, idx + 1, idx + 1, idx + 2, idx + 3])

    return finalize_mesh(verts, normals, uvs, indices)


def create_truncated_cone(bottom_radius=0.5, top_radius=0.25, height=1.0, sectors=36):
    verts, normals, uvs, indices = [], [], [], []
    half_h = height / 2.0

    for y_sign, cap_radius in [(1.0, top_radius), (-1.0, bottom_radius)]:
        if cap_radius <= 1e-6:
            continue
        center_idx = len(verts)
        verts.append([0.0, y_sign * half_h, 0.0])
        normals.append([0.0, y_sign, 0.0])
        uvs.append([0.5, 0.5])
        for i in range(sectors + 1):
            theta = i * 2.0 * np.pi / sectors
            c, s = np.cos(theta), np.sin(theta)
            verts.append([cap_radius * c, y_sign * half_h, cap_radius * s])
            normals.append([0.0, y_sign, 0.0])
            uvs.append([0.5 + 0.5 * c, 0.5 + 0.5 * s])
            if i < sectors:
                i1 = center_idx + i + (2 if y_sign > 0 else 1)
                i2 = center_idx + i + (1 if y_sign > 0 else 2)
                indices.extend([center_idx, i1, i2])

    wall_idx = len(verts)
    dr = bottom_radius - top_radius
    slant = math.hypot(height, dr)
    nr_factor = height / slant
    ny = dr / slant

    for i in range(sectors + 1):
        theta = i * 2.0 * np.pi / sectors
        c, s = np.cos(theta), np.sin(theta)
        verts.extend([
            [top_radius * c, half_h, top_radius * s],
            [bottom_radius * c, -half_h, bottom_radius * s],
        ])
        normals.extend([[nr_factor * c, ny, nr_factor * s]] * 2)
        uvs.extend([[i / sectors, 1.0], [i / sectors, 0.0]])
        if i < sectors:
            idx = wall_idx + i * 2
            indices.extend([idx, idx + 2, idx + 1, idx + 1, idx + 2, idx + 3])

    return finalize_mesh(verts, normals, uvs, indices)


def create_cone(radius=0.5, height=1.0, sectors=36):
    return create_truncated_cone(
        bottom_radius=radius, top_radius=0.0, height=height, sectors=sectors
    )


def create_prism(radius=0.5, height=1.0, sides=3):
    verts, normals, uvs, indices = [], [], [], []
    half_h = height / 2.0
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)

    for y_sign in [1.0, -1.0]:
        center_idx = len(verts)
        verts.append([0.0, y_sign * half_h, 0.0])
        normals.append([0.0, y_sign, 0.0])
        uvs.append([0.5, 0.5])
        for i in range(sides + 1):
            theta = angles[i % sides]
            c, s = np.cos(theta), np.sin(theta)
            verts.append([radius * c, y_sign * half_h, radius * s])
            normals.append([0.0, y_sign, 0.0])
            uvs.append([0.5 + 0.5 * c, 0.5 + 0.5 * s])
            if i < sides:
                i1 = center_idx + i + (2 if y_sign > 0 else 1)
                i2 = center_idx + i + (1 if y_sign > 0 else 2)
                indices.extend([center_idx, i1, i2])

    for i in range(sides):
        t0, t1 = angles[i], angles[(i + 1) % sides]
        p0 = np.array([radius * np.cos(t0),  half_h, radius * np.sin(t0)], dtype=np.float32)
        p1 = np.array([radius * np.cos(t0), -half_h, radius * np.sin(t0)], dtype=np.float32)
        p2 = np.array([radius * np.cos(t1), -half_h, radius * np.sin(t1)], dtype=np.float32)
        p3 = np.array([radius * np.cos(t1),  half_h, radius * np.sin(t1)], dtype=np.float32)

        n = np.cross(p3 - p0, p1 - p0)
        n = n / np.linalg.norm(n)

        base = len(verts)
        verts.extend([p0, p1, p2, p3])
        normals.extend([n] * 4)
        uvs.extend([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        indices.extend([base, base + 2, base + 1, base, base + 3, base + 2])

    return finalize_mesh(verts, normals, uvs, indices)


def create_tetrahedron(radius=0.6):
    y_base = -radius / 3.0
    r_base = radius * (2.0 * math.sqrt(2.0) / 3.0)

    v = np.array([
        [0.0, radius, 0.0],
        [r_base * math.cos(0.0), y_base, r_base * math.sin(0.0)],
        [r_base * math.cos(2.0 * np.pi / 3.0), y_base, r_base * math.sin(2.0 * np.pi / 3.0)],
        [r_base * math.cos(4.0 * np.pi / 3.0), y_base, r_base * math.sin(4.0 * np.pi / 3.0)],
    ], dtype=np.float32)

    faces = [(0, 2, 1), (0, 3, 2), (0, 1, 3), (1, 2, 3)]
    face_uvs = [[0.5, 1.0], [0.0, 0.0], [1.0, 0.0]]

    verts, normals, uvs, indices = [], [], [], []
    for idx_f, (i0, i1, i2) in enumerate(faces):
        p0, p1, p2 = v[i0], v[i1], v[i2]
        n = np.cross(p1 - p0, p2 - p0)
        n = n / np.linalg.norm(n)
        verts.extend([p0, p1, p2])
        normals.extend([n] * 3)
        uvs.extend(face_uvs)
        base = idx_f * 3
        indices.extend([base, base + 1, base + 2])

    return finalize_mesh(verts, normals, uvs, indices)


def create_pyramid(base_size=0.9, height=0.9):
    """Square-base pyramid (explicitly listed in Section 3.2 Grading Scheme)."""
    hs = base_size / 2.0
    hh = height / 2.0
    v = np.array([
        [ 0.0,  hh,  0.0],  # 0: apex
        [-hs,  -hh,  hs],   # 1: front-left
        [ hs,  -hh,  hs],   # 2: front-right
        [ hs,  -hh, -hs],   # 3: back-right
        [-hs,  -hh, -hs],   # 4: back-left
    ], dtype=np.float32)

    side_faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)]
    tri_uvs = [[0.5, 1.0], [0.0, 0.0], [1.0, 0.0]]

    verts, normals, uvs, indices = [], [], [], []
    for i0, i1, i2 in side_faces:
        p0, p1, p2 = v[i0], v[i1], v[i2]
        n = np.cross(p1 - p0, p2 - p0)
        n = n / np.linalg.norm(n)
        base = len(verts)
        verts.extend([p0, p1, p2])
        normals.extend([n] * 3)
        uvs.extend(tri_uvs)
        indices.extend([base, base + 1, base + 2])

    base = len(verts)
    verts.extend([v[4], v[3], v[2], v[1]])
    normals.extend([[0.0, -1.0, 0.0]] * 4)
    uvs.extend([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    return finalize_mesh(verts, normals, uvs, indices)


def create_torus(inner_radius=0.2, outer_radius=0.5, rings=36, sectors=36):
    verts, normals, uvs, indices = [], [], [], []
    for i in range(rings + 1):
        u = i * 2.0 * np.pi / rings
        for j in range(sectors + 1):
            v = j * 2.0 * np.pi / sectors
            x = (outer_radius + inner_radius * np.cos(v)) * np.cos(u)
            y = (outer_radius + inner_radius * np.cos(v)) * np.sin(u)
            verts.append([x, y, inner_radius * np.sin(v)])
            normals.append([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)])
            uvs.append([i / rings, j / sectors])

    for i in range(rings):
        for j in range(sectors):
            k1 = i * (sectors + 1) + j
            k2 = (i + 1) * (sectors + 1) + j
            indices.extend([k1, k2, k1 + 1, k1 + 1, k2, k2 + 1])

    return finalize_mesh(verts, normals, uvs, indices)