from src.geometry.shapes_2d import (
    create_arrow,
    create_circle,
    create_ellipse,
    create_hexagon,
    create_pentagon,
    create_rectangle,
    create_star,
    create_trapezoid,
    create_triangle,
)
from src.geometry.shapes_3d import (
    create_cone,
    create_cube,
    create_cylinder,
    create_prism,
    create_pyramid,
    create_sphere,
    create_tetrahedron,
    create_torus,
    create_truncated_cone,
)

SHAPE_REGISTRY = {
    # 2D Shapes
    "Triangle": create_triangle,
    "Rectangle": create_rectangle,
    "Pentagon": create_pentagon,
    "Regular Hexagon": create_hexagon,
    "Circle": create_circle,
    "Ellipse": create_ellipse,
    "Trapezoid": create_trapezoid,
    "Star": create_star,
    "Arrow": create_arrow,
    # 3D Shapes
    "Cube": create_cube,
    "Sphere": create_sphere,
    "Cylinder": create_cylinder,
    "Torus": create_torus,
    "Cone": create_cone,
    "Truncated Cone": create_truncated_cone,
    "Prism": create_prism,
    "Tetrahedron": create_tetrahedron,
    "Pyramid": create_pyramid,
    # Parametric & External
    "Surface (z=f(x,y))": None,
    "Imported Model": None,
}