# HCMUT CO3059 — Assignment 1: 2D/3D Object Rendering & Molecular Visualization

An interactive 3D computer graphics engine built from scratch in Python using **PyOpenGL (OpenGL 3.3 Core Profile)**, **GLFW**, **Dear ImGui**, **NumPy**, and the course-provided **`cglibs`** framework for the Computer Graphics ($CO3059$) course at Ho Chi Minh City University of Technology (HCMUT).

---

## Features Overview

### Part 1: 2D & 3D Object Rendering
* **9 Procedural 2D Planar Shapes**: Triangle, Rectangle, Pentagon, Regular Hexagon, Circle, Ellipse, Trapezoid, Star, and Arrow (constructed in the $XY$ plane with $+Z$ normals and normalized $[0, 1]^2$ UV coordinates).
* **9 Procedural 3D Solid Primitives**: Cube, Sphere, Cylinder, Cone, Truncated Cone, Tetrahedron, Torus, Prism, and Pyramid (with analytical curved normals and sharp per-face polyhedral normals).
* **Vectorized Parametric Surface ($z = f(x, y)$)**: Real-time evaluation of user-entered mathematical formulas (e.g., $\frac{\sin(\sqrt{x^2 + y^2})}{\sqrt{x^2 + y^2}}$) with automatic $10^{-10}$ zero-division singularity protection, area-weighted smooth vertex normals, and optional **Jet Colormap** height coloring.
* **Custom `.obj` and `.ply` 3D Model Importers**:
  * **Wavefront `.obj`**: Supports arbitrary polygon fan-triangulation, topological `(v/vt/vn)` deduplication caching, automatic bounding-box normalization, and smooth normal generation.
  * **Stanford `.ply`**: Supports `ascii`, `binary_little_endian`, and `binary_big_endian` formats, including zero-copy vectorized NumPy buffer loading (`< 0.08s` for $>10\text{ MB}$ scans).
* **5 Programmable Shading Modes**:
  1. `flat`: Flat per-face provoking-vertex shading.
  2. `gouraud`: Per-vertex lighting with barycentric color interpolation (includes a **3D Vertex Color Gradient** toggle).
  3. `phong`: Per-fragment Blinn-Phong specular lighting.
  4. `texture`: External image texture mapping (`.png`, `.jpg`, `.webp`, `.bmp`) with native RGBA alpha blending (`GL_BLEND`) and a procedural $8 \times 8$ checkerboard fallback.
  5. `edge`: Wireframe (`GL_LINE`) polygon rasterization mode.
* **Interactive Geometric Transformations (TRS)**: Real-time ImGui HUD sliders for 3D Translation $(T_x, T_y, T_z)$, Euler Rotation $(R_x, R_y, R_z)$, Non-Uniform Scaling $(S_x, S_y, S_z)$, and Uniform Scaling relative to a 3D coordinate `AxisGrid`.

### Part 2: Atomic & Molecular Visualization
* **Chemically Accurate Bohr Atomic Model (Default View)**:
  * Supports 11 chemical elements (`H`, `He`, `Li`, `C`, `N`, `O`, `F`, `Ne`, `Na`, `S`, `Cl`).
  * Selecting an element automatically configures its atomic number $Z$, CPK color, nuclear radius, and exact ground-state $[K, L, M]$ electron shell distribution (e.g., Carbon $Z=6 \to [2, 4]$, Sodium $Z=11 \to [2, 8, 1]$).
  * Renders razor-thin 3D torus orbital paths (`inner_radius = 0.01`, `rings = 120`) with multi-axis 3D orbital rotation animations.
* **Ball-and-Stick Molecular Models ($\text{H}_2\text{O}$ & $\text{CO}_2$)**:
  * Hierarchical `SceneNode` graph representing atoms as CPK-colored spheres and chemical bonds as cylinders with harmonic bond-stretching vibration:
    $$L(t) = \|\mathbf{p}_B - \mathbf{p}_A\|_2 + A\sin(\omega_{vib} t)$$
* **Interactive 3D Molecular Editor**:
  * Change individual atom element types and adjust 3D $(x, y, z)$ coordinates in real time.
  * Add or delete bonded atoms dynamically; cylinder bonds automatically re-align to arbitrary 3D vectors using Rodrigues' rotation formula (`align_y_to_vector`).
  * Supports all 5 rendering modes (`flat`, `gouraud`, `phong`, `texture`, `edge`) across all atomic and molecular meshes.

### Bonus & Utility Features
* **Real-Time 3D Voxel Grid Mesh Optimizer (`PlyOptimizer`)**: Simplifies dense $>10\text{ MB}$ `.ply` 3D scans in milliseconds via 3D spatial hashing and vertex clustering, exporting compact `binary_little_endian` `assets/<name>_opt.ply` files with one-click GUI buttons to toggle between the original and optimized meshes.
* **Framebuffer Screenshot Exporter**: One-click **Save Screenshot** button on the HUD exports the current OpenGL viewport to `screenshots/` as a timestamped `.png` image.

---

## Project Structure

```text
project_root/
├── assets/                      # Sample textures (.png) and 3D models (.obj, .ply)
├── cglibs/                      # Course-provided OpenGL utility library
│   ├── buffer.py                # VAO, VBO, EBO, and UManager uniform/texture helpers
│   ├── camera.py                # Camera placement wrapper
│   ├── hud.py                   # On-screen key legend overlay (toggled via H)
│   ├── shader.py                # GLSL shader compiler and program linker
│   ├── shading.py               # Draw call emission (emit_draw) and key handlers
│   ├── texutil.py               # Procedural checker image writer
│   ├── transform.py             # Quaternions, 4x4 matrices, and Trackball math
│   └── viewer.py                # Base GLFW Viewer and TrackballViewer classes
├── screenshots/                 # Exported framebuffer PNG captures
└── src/
    ├── main.py                  # Application entry point & OpenGL context lifecycle
    ├── shaders/                 # GLSL 3.30 vertex (.vert) and fragment (.frag) shaders
    ├── app/
    │   └── viewer.py            # Assignment1Viewer: ImGui HUD, input routing, main loop
    ├── rendering/
    │   ├── grid.py              # AxisGrid: 3D coordinate axes and dashed bounding floor
    │   ├── mesh.py              # Mesh: VAO binding, shader selection, texture/color upload
    │   └── transforms.py        # Homogeneous 4x4 TRS matrices & Rodrigues vector alignment
    ├── geometry/
    │   ├── base.py              # Shared array finalization & 2D planar fan triangulation
    │   ├── shapes_2d.py         # 9 procedural 2D shapes in the XY plane
    │   ├── shapes_3d.py         # 9 procedural 3D solid primitives
    │   ├── surface.py           # Vectorized mathematical surface z = f(x,y) generator
    │   └── registry.py          # SHAPE_REGISTRY mapping GUI names to generator functions
    ├── loaders/
    │   ├── base_loader.py       # Vertex normal accumulation, bbox normalization, planar UVs
    │   ├── obj_loader.py        # Wavefront .obj parser with vertex deduplication cache
    │   ├── ply_loader.py        # Stanford .ply parser (ASCII & vectorized binary)
    │   └── model_loader.py      # Extension-based dispatcher (.obj / .ply)
    ├── scene/
    │   ├── scene_node.py        # SceneNode: Hierarchical scene graph node
    │   └── molecules.py         # MoleculeScene: Bohr model, H2O, CO2, custom atom editor
    └── utils/
        ├── math_parser.py       # Regex parser mapping math strings to safe NumPy expressions
        ├── mesh_optimizer.py    # PlyOptimizer: 3D Voxel Grid Vertex Clustering decimation
        └── screenshot.py        # OpenGL glReadPixels framebuffer PNG exporter
```

---

## Setup & Installation

### 1. Create and Activate a Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell / CMD)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install PyOpenGL PyOpenGL_accelerate glfw "imgui[glfw]" numpy Pillow opencv-python pandas
```

---

## Running the Application

### Launch the Main Interactive Viewer
Run the application from the project root directory:
```bash
python -m src.main
```

### Optional CLI Tool: Standalone `.ply` / `.obj` Mesh Optimizer
You can optimize a large `.ply` or `.obj` file either interactively inside the GUI or directly from the command line:
```bash
python -m src.utils.mesh_optimizer assets/your_model.ply 80
```
*(This clusters vertices on an $80 \times 80 \times 80$ voxel grid and writes `assets/your_model_opt.ply`.)*

---

## User Guide & Controls

### Camera Controls (Mouse)
* **Rotate Camera**: Left-click + Drag (Virtual Quaternion Trackball).
* **Pan Camera**: Right-click + Drag.
* **Zoom Camera**: Mouse Scroll Wheel.
* **Reset View**: Click the **Reset Camera** button at the bottom of the ImGui HUD.

### Using Part 1: Shapes, Surfaces, Textures & Imported Models
1. **Select a Shape**: Open the **Shape** dropdown to choose any 2D shape, 3D solid, `Surface (z=f(x,y))`, or `Imported Model`.
2. **Custom Mathematical Surface**:
   * Select `Surface (z=f(x,y))`, type any expression in terms of `x` and `y` (e.g., `cos(x) * sin(y)` or `sin(sqrt(x**2 + y**2)) / sqrt(x**2 + y**2)`), and press **Enter**.
3. **Import `.obj` or `.ply` Models & Optimize**:
   * Place your `.obj` or `.ply` file in `assets/` (or the project root).
   * Select `Imported Model` in the **Shape** dropdown, enter the filename (e.g., `LowpolyTree.obj` or `key.ply`), and click **Load 3D Model**.
   * To decimate a heavy mesh, adjust the **Voxel Grid Res** slider (`16`–`200`) and click **Optimize & Save to assets/**, then use the **Load Original** and **Load Optimized** buttons to compare triangle counts and visual quality.
4. **Apply External Textures**:
   * Place an image file (e.g., `texture1.png`) in `assets/` or the project root.
   * Enter the filename in the **Image Path** field under *Texture Mapping (External Image)* and click **Load Texture File**.
5. **Geometric Transformations (TRS)**:
   * Adjust the **Translate (X,Y,Z)**, **Rotate (X,Y,Z deg)**, **Scale (Sx,Sy,Sz)**, or **Uniform Scale** sliders to transform the active object in real time.

### Using Part 2: Atoms & Molecules
1. Click the **Part 2: Molecules** radio button at the top of the HUD.
2. **Bohr Model**:
   * Select an element from the **Atom Type** dropdown (`H`, `He`, `Li`, `C`, `N`, `O`, `F`, `Ne`, `Na`, `S`, `Cl`) to automatically populate its ground-state $[K, L, M]$ electron shells.
   * Adjust **Nucleus Pos** or **Orbit Speed**, or toggle **Enable Animation**.
3. **$\text{H}_2\text{O}$ & $\text{CO}_2$ Molecules**:
   * Select `H2O Molecule` or `CO2 Molecule` from the **Model** dropdown.
   * Change any atom's element type, drag its **Atom Pos** $(x, y, z)$ slider, or click **Add Bonded Atom** / **X** to modify the molecular topology in 3D.
4. **Part 2 Shading Modes**:
   * Use the **Shading** dropdown in Part 2 to switch the entire atomic/molecular scene between `flat`, `gouraud`, `phong`, `texture`, and `edge`.

---

## Third-Party Library Policy Compliance
* **Permitted**: `glfw` and `imgui` are used for windowing and GUI controls; `cglibs` and `numpy` are used for matrix/quaternion transformations.
* **Restricted (Implemented from Scratch)**: All 2D/3D mesh generators (`src/geometry/`), `.obj`/`.ply` file parsers (`src/loaders/`), voxel grid mesh decimators (`src/utils/mesh_optimizer.py`), and GLSL shader programs (`src/shaders/`) are implemented without any external 3D mesh libraries.
