import numpy as np
from src.geometry.base import create_2d_planar_mesh, create_regular_polygon_2d


def create_triangle(radius=0.55):
    return create_regular_polygon_2d(sides=3, radius=radius)


def create_rectangle(width=1.0, height=0.6):
    hw, hh = width / 2.0, height / 2.0
    points_2d = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]
    return create_2d_planar_mesh(points_2d)


def create_pentagon(radius=0.5):
    return create_regular_polygon_2d(sides=5, radius=radius)


def create_hexagon(radius=0.5):
    return create_regular_polygon_2d(sides=6, radius=radius, start_angle=0.0)


def create_circle(radius=0.5, sectors=64):
    return create_ellipse(rx=radius, ry=radius, sectors=sectors)


def create_ellipse(rx=0.6, ry=0.35, sectors=64):
    angles = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    ring = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
    points_2d = np.vstack([[0.0, 0.0], ring])

    indices = []
    for i in range(1, sectors):
        indices.extend([0, i, i + 1])
    indices.extend([0, sectors, 1])

    return create_2d_planar_mesh(points_2d, indices)


def create_trapezoid(bottom_width=1.0, top_width=0.5, height=0.7):
    hb, ht, hh = bottom_width / 2.0, top_width / 2.0, height / 2.0
    points_2d = [[-hb, -hh], [hb, -hh], [ht, hh], [-ht, hh]]
    return create_2d_planar_mesh(points_2d)


def create_star(outer_radius=0.55, inner_radius=0.22, points=5):
    total_pts = points * 2
    angles = (np.pi / 2.0) + np.arange(total_pts) * (np.pi / points)
    radii = np.where(np.arange(total_pts) % 2 == 0, outer_radius, inner_radius)
    ring = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    points_2d = np.vstack([[0.0, 0.0], ring])

    indices = []
    for i in range(1, total_pts):
        indices.extend([0, i, i + 1])
    indices.extend([0, total_pts, 1])

    return create_2d_planar_mesh(points_2d, indices)


def create_arrow():
    points_2d = [
        [-0.15, -0.50],  # 0: shaft bottom-left
        [ 0.15, -0.50],  # 1: shaft bottom-right
        [ 0.15,  0.05],  # 2: shaft top-right
        [-0.15,  0.05],  # 3: shaft top-left
        [ 0.38,  0.05],  # 4: head right wing
        [ 0.00,  0.55],  # 5: head tip
        [-0.38,  0.05],  # 6: head left wing
    ]
    indices = [
        0, 1, 2,
        0, 2, 3,
        6, 4, 5,
    ]
    return create_2d_planar_mesh(points_2d, indices)