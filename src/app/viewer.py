import os
import glfw
import imgui
import numpy as np
import OpenGL.GL as GL
from imgui.integrations.glfw import GlfwRenderer

from cglibs.texutil import write_checker
from cglibs.transform import Trackball
from cglibs.viewer import TrackballViewer
from src.geometry import SHAPE_REGISTRY, create_surface
from src.loaders import load_3d_model
from src.rendering import AxisGrid, Mesh, compose_trs, get_initial_rotation
from src.scene import MoleculeScene
from src.utils import PlyOptimizer, save_framebuffer_screenshot


class Assignment1Viewer(TrackballViewer):
    def __init__(self, *args, **kwargs):
        self.initial_distance = kwargs.get("distance", 7.2)
        super().__init__(*args, **kwargs)
        self.base_rotation = get_initial_rotation()
        self.show_help = False  # Press H to toggle cglibs HUD overlay

        self._init_imgui_and_callbacks()
        self._init_state()

    def _init_imgui_and_callbacks(self):
        self.tb_cb = {
            'btn': glfw.set_mouse_button_callback(self.win, None),
            'pos': glfw.set_cursor_pos_callback(self.win, None),
            'scl': glfw.set_scroll_callback(self.win, None),
            'key': glfw.set_key_callback(self.win, None),
        }

        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        imgui.create_context()
        self.impl = GlfwRenderer(self.win)
        GL.glBindVertexArray(0)

        self.ig_cb = {
            'btn': glfw.set_mouse_button_callback(self.win, self._route_btn),
            'pos': glfw.set_cursor_pos_callback(self.win, self._route_pos),
            'scl': glfw.set_scroll_callback(self.win, self._route_scl),
            'key': glfw.set_key_callback(self.win, self._route_key),
        }

    def _init_state(self):
        self.app_mode = 0
        self.status_msg = ""
        self.status_is_error = False

        # --- Part 1 State ---
        self.shape_names = list(SHAPE_REGISTRY.keys())
        self.modes = ["flat", "gouraud", "phong", "texture", "edge"]
        self.shape_idx = self.shape_names.index("Surface (z=f(x,y))")
        self.mode_idx = 2
        self.surface_func = "sin(sqrt(x**2 + y**2)) / sqrt(x**2 + y**2)"
        self.model_filepath = "LowpolyTree.obj"

        # Mesh Optimization & Comparison State
        self.voxel_grid_res = 80
        self.original_model_filepath = ""
        self.optimized_model_filepath = ""
        self.opt_stats = None

        write_checker("assets/checker_sample.png")
        if os.path.isfile("texture1.png") or os.path.isfile(os.path.join("assets", "texture1.png")):
            self.texture_filepath = "texture1.png"
        else:
            self.texture_filepath = "assets/checker_sample.png"
        self.use_custom_texture = False

        self.use_jet = True
        self.show_grid = True
        self.object_color = [0.80, 0.80, 0.80]
        self.use_vertex_gradient = False

        # Object Geometric Transformations (Translation, Rotation, Scaling)
        self.obj_translate = [0.0, 0.0, 0.0]
        self.obj_rotate = [0.0, 0.0, 0.0]
        self.obj_scale = [1.0, 1.0, 1.0]
        self.uniform_scale = 1.0

        static_grid_bounds = (-2.0, 2.0, -1.0, 1.0, -2.0, 2.0)
        self.grid_lines = AxisGrid(static_grid_bounds)
        self.active_mesh = None

        # --- Part 2 State ---
        self.molecule_scene = MoleculeScene()
        self.molecule_names = MoleculeScene.MOLECULE_NAMES
        self.molecule_idx = 0  # Default to Bohr Model
        self.mol_mode_idx = 2  # Default to Phong in Part 2
        self.mol_vertex_gradient = False
        self.animate = True
        self.anim_time = 0.0
        self.last_time = glfw.get_time()

        self.load_shape(self.shape_idx)
        self.molecule_scene.build(self.molecule_idx, self.anim_time)

    # =========================================================================
    # Input Routing (ImGui + cglibs TrackballViewer)
    # =========================================================================

    def _route_btn(self, w, b, a, m):
        if self.ig_cb['btn']:
            self.ig_cb['btn'](w, b, a, m)
        if not imgui.get_io().want_capture_mouse and self.tb_cb['btn']:
            self.tb_cb['btn'](w, b, a, m)

    def _route_pos(self, w, x, y):
        if self.ig_cb['pos']:
            self.ig_cb['pos'](w, x, y)
        if not imgui.get_io().want_capture_mouse and self.tb_cb['pos']:
            self.tb_cb['pos'](w, x, y)

    def _route_scl(self, w, x, y):
        if self.ig_cb['scl']:
            self.ig_cb['scl'](w, x, y)
        if not imgui.get_io().want_capture_mouse and self.tb_cb['scl']:
            self.tb_cb['scl'](w, x, y)

    def _route_key(self, w, key, scancode, action, mods):
        if self.ig_cb['key']:
            self.ig_cb['key'](w, key, scancode, action, mods)
        if not imgui.get_io().want_capture_keyboard and self.tb_cb['key']:
            self.tb_cb['key'](w, key, scancode, action, mods)

    # =========================================================================
    # State Helpers
    # =========================================================================

    def _set_status(self, msg, is_error=False):
        self.status_msg = msg
        self.status_is_error = is_error
        print(msg)

    def _reset_transforms_and_camera(self):
        self.obj_translate = [0.0, 0.0, 0.0]
        self.obj_rotate = [0.0, 0.0, 0.0]
        self.obj_scale = [1.0, 1.0, 1.0]
        self.uniform_scale = 1.0
        self.trackball = Trackball(distance=self.initial_distance)

    def _take_screenshot(self):
        try:
            prefix = "part1_shape" if self.app_mode == 0 else "part2_molecule"
            path = save_framebuffer_screenshot(self.win, prefix=prefix)
            self._set_status(f"Saved: {path}", is_error=False)
        except Exception as e:
            self._set_status(f"Screenshot error: {e}", is_error=True)

    def load_shape(self, idx):
        shape_name = self.shape_names[idx]
        try:
            if shape_name == "Surface (z=f(x,y))":
                shape_data = create_surface(self.surface_func, use_jet=self.use_jet, res=120)
            elif shape_name == "Imported Model":
                resolved_model = Mesh.resolve_asset_path(self.model_filepath)
                if not self.model_filepath.endswith("_opt.ply"):
                    self.original_model_filepath = os.path.basename(resolved_model)
                shape_data = load_3d_model(resolved_model)
            else:
                shape_data = SHAPE_REGISTRY[shape_name]()

            mesh_args = shape_data[:-1] if len(shape_data) == 6 else shape_data
            self.active_mesh = Mesh(*mesh_args)
            self._sync_mesh_appearance()
            self._apply_mode(self.modes[self.mode_idx])
            n_v = len(self.active_mesh.verts)
            n_f = self.active_mesh.fill_count // 3
            self._set_status(f"Loaded {shape_name} ({n_v:,} verts, {n_f:,} tris)", is_error=False)
        except Exception as e:
            self._set_status(f"{e}", is_error=True)

    def _optimize_current_model(self):
        try:
            source_file = self.original_model_filepath or self.model_filepath
            resolved_in = Mesh.resolve_asset_path(source_file)
            self.original_model_filepath = os.path.basename(resolved_in)

            mesh_args, stats = PlyOptimizer.optimize_file(
                resolved_in, grid_res=self.voxel_grid_res
            )
            self.opt_stats = stats
            self.optimized_model_filepath = stats["output_filename"]
            self.model_filepath = self.optimized_model_filepath

            self.active_mesh = Mesh(*mesh_args)
            self._sync_mesh_appearance()
            self._apply_mode(self.modes[self.mode_idx])
            self._set_status(
                f"Saved assets/{self.optimized_model_filepath} "
                f"(-{stats['reduction_pct']:.1f}% tris, {stats['opt_size_mb']:.2f} MB)",
                is_error=False,
            )
        except Exception as e:
            self._set_status(f"Optimization failed: {e}", is_error=True)

    def _sync_mesh_appearance(self):
        if not self.active_mesh:
            return
        shape_name = self.shape_names[self.shape_idx]
        if shape_name == "Surface (z=f(x,y))" and self.use_jet:
            self.active_mesh.update_colors(None, use_gradient=False)
        else:
            self.active_mesh.update_colors(
                uniform_rgb=self.object_color,
                use_gradient=self.use_vertex_gradient,
            )
        if self.use_custom_texture:
            try:
                resolved = self.active_mesh.load_texture_from_file(self.texture_filepath)
                self._set_status(f"Loaded texture: {os.path.basename(resolved)}", is_error=False)
            except Exception as e:
                self._set_status(f"Texture error: {e}", is_error=True)

    def _apply_mode(self, mode):
        if not self.active_mesh:
            return
        if mode == "edge":
            self.active_mesh.draw_as_lines = True
            self.active_mesh.shading = "flat"
        else:
            self.active_mesh.draw_as_lines = False
            self.active_mesh.shading = mode

    # =========================================================================
    # ImGui User Interface
    # =========================================================================

    def _render_ui(self):
        imgui.new_frame()
        imgui.begin("Assignment 1 Controls", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        imgui.set_window_font_scale(1.35)

        if imgui.radio_button("Part 1: Shapes", self.app_mode == 0):
            self.app_mode = 0
        imgui.same_line()
        if imgui.radio_button("Part 2: Molecules", self.app_mode == 1):
            self.app_mode = 1

        imgui.separator()

        if self.app_mode == 0:
            self._render_part1_controls()
        else:
            self._render_part2_controls()

        imgui.separator()
        if imgui.button("Reset Camera"):
            self._reset_transforms_and_camera()
        imgui.same_line()
        if imgui.button("Save Screenshot"):
            self._take_screenshot()

        if self.status_msg:
            if self.status_is_error:
                imgui.text_colored(self.status_msg, 0.85, 0.15, 0.15, 1.0)
            else:
                imgui.text_colored(self.status_msg, 0.10, 0.50, 0.15, 1.0)

        imgui.end()

    def _render_part1_controls(self):
        _, self.show_grid = imgui.checkbox("Show Axis Grid", self.show_grid)
        changed_shape, new_shape_idx = imgui.combo("Shape", self.shape_idx, self.shape_names)

        if changed_shape:
            self.shape_idx = new_shape_idx
            self.use_jet = (self.shape_names[self.shape_idx] == "Surface (z=f(x,y))")

        selected_name = self.shape_names[self.shape_idx]
        if selected_name == "Surface (z=f(x,y))":
            changed_func, self.surface_func = imgui.input_text(
                "Function", self.surface_func, 256, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            )
            changed_jet, self.use_jet = imgui.checkbox("Jet Color Mapping", self.use_jet)
            if changed_func or changed_shape or changed_jet:
                self.load_shape(self.shape_idx)

        elif selected_name == "Imported Model":
            enter_model, self.model_filepath = imgui.input_text(
                "Model (.obj/.ply)", self.model_filepath, 256, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            )
            if imgui.button("Load 3D Model") or enter_model or changed_shape:
                self.load_shape(self.shape_idx)

            if self.active_mesh:
                nv = len(self.active_mesh.verts)
                nf = self.active_mesh.fill_count // 3
                imgui.text(f"Active Mesh: {nv:,} vertices | {nf:,} triangles")

            imgui.separator()
            imgui.text("PLY / Mesh Optimization (Voxel Clustering)")
            _, self.voxel_grid_res = imgui.slider_int(
                "Voxel Grid Res", self.voxel_grid_res, 16, 200
            )
            if imgui.button("Optimize & Save to assets/"):
                self._optimize_current_model()

            if self.opt_stats:
                s = self.opt_stats
                imgui.text_colored(
                    f"Original : {s['orig_faces']:,} tris ({s['orig_size_mb']:.2f} MB) -> "
                    f"Optimized: {s['opt_faces']:,} tris ({s['opt_size_mb']:.2f} MB)",
                    0.15, 0.35, 0.75, 1.0
                )
                if imgui.button(f"Load Original ({self.original_model_filepath})"):
                    self.model_filepath = self.original_model_filepath
                    self.load_shape(self.shape_idx)
                imgui.same_line()
                if imgui.button(f"Load Optimized ({self.optimized_model_filepath})"):
                    self.model_filepath = self.optimized_model_filepath
                    self.load_shape(self.shape_idx)

        elif changed_shape:
            self.load_shape(self.shape_idx)

        # Shading & Appearance
        changed_mode, self.mode_idx = imgui.combo("Shading", self.mode_idx, self.modes)
        if changed_mode:
            self._apply_mode(self.modes[self.mode_idx])

        if selected_name != "Surface (z=f(x,y))" or not self.use_jet:
            changed_col, new_col = imgui.color_edit3("Object Color", *self.object_color)
            changed_grad, self.use_vertex_gradient = imgui.checkbox(
                "3D Vertex Color Gradient (for Gouraud)", self.use_vertex_gradient
            )
            if changed_col or changed_grad:
                self.object_color = list(new_col)
                self._sync_mesh_appearance()

        # External Texture Mapping Controls
        imgui.separator()
        imgui.text("Texture Mapping (External Image)")
        enter_tex, self.texture_filepath = imgui.input_text(
            "Image Path", self.texture_filepath, 256, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
        )
        if imgui.button("Load Texture File") or enter_tex:
            self.use_custom_texture = True
            self.mode_idx = self.modes.index("texture")
            self._apply_mode("texture")
            self._sync_mesh_appearance()
        imgui.same_line()
        if imgui.button("Default Checker"):
            self.use_custom_texture = False
            self.mode_idx = self.modes.index("texture")
            self._apply_mode("texture")
            if self.active_mesh:
                self.active_mesh.reset_checker_texture()
            self._set_status("Reset to default 8x8 checker texture", is_error=False)

        # Geometric Transformations (Translation, Rotation, Scaling)
        imgui.separator()
        imgui.text("Geometric Transformations (TRS)")
        _, t_vals = imgui.slider_float3("Translate (X,Y,Z)", *self.obj_translate, -3.0, 3.0, "%.2f")
        self.obj_translate = list(t_vals)

        _, r_vals = imgui.slider_float3("Rotate (X,Y,Z deg)", *self.obj_rotate, -180.0, 180.0, "%.1f")
        self.obj_rotate = list(r_vals)

        _, s_vals = imgui.slider_float3("Scale (Sx,Sy,Sz)", *self.obj_scale, 0.1, 3.0, "%.2f")
        self.obj_scale = list(s_vals)

        changed_u, self.uniform_scale = imgui.slider_float(
            "Uniform Scale", self.uniform_scale, 0.1, 3.0, "%.2f"
        )
        if changed_u:
            self.obj_scale = [self.uniform_scale] * 3

        if imgui.button("Reset Object Transform"):
            self.obj_translate = [0.0, 0.0, 0.0]
            self.obj_rotate = [0.0, 0.0, 0.0]
            self.obj_scale = [1.0, 1.0, 1.0]
            self.uniform_scale = 1.0

    def _render_part2_controls(self):
        changed_mol, self.molecule_idx = imgui.combo(
            "Model", self.molecule_idx, self.molecule_names
        )
        if changed_mol:
            self.molecule_scene.build(self.molecule_idx, self.anim_time)

        _, self.animate = imgui.checkbox("Enable Animation (Vibration/Orbit)", self.animate)

        # Part 2 Shading & Texture Controls (All 5 modes from Part 1)
        changed_mol_mode, self.mol_mode_idx = imgui.combo(
            "Shading##part2", self.mol_mode_idx, self.modes
        )
        if changed_mol_mode:
            self.molecule_scene.apply_shading_mode(self.modes[self.mol_mode_idx])

        changed_mol_grad, self.mol_vertex_gradient = imgui.checkbox(
            "3D Vertex Color Gradient (for Gouraud)##part2", self.mol_vertex_gradient
        )
        if changed_mol_grad:
            self.molecule_scene.apply_vertex_gradient(self.mol_vertex_gradient)

        if self.modes[self.mol_mode_idx] == "texture":
            enter_tex, self.texture_filepath = imgui.input_text(
                "Image Path##part2", self.texture_filepath, 256, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            )
            if imgui.button("Load Texture File##part2") or enter_tex:
                try:
                    resolved = self.molecule_scene.load_texture(self.texture_filepath)
                    self._set_status(f"Loaded texture: {os.path.basename(resolved)}", is_error=False)
                except Exception as e:
                    self._set_status(f"Texture error: {e}", is_error=True)
            imgui.same_line()
            if imgui.button("Default Checker##part2"):
                self.molecule_scene.reset_checker_texture()
                self._set_status("Reset Part 2 to default 8x8 checker texture", is_error=False)

        imgui.separator()
        current_name = self.molecule_names[self.molecule_idx]
        atom_types = MoleculeScene.ATOM_TYPES

        if current_name.startswith("Bohr Model"):
            imgui.text("Configure Bohr Atom Type & Position")
            b_cfg = self.molecule_scene.bohr_config

            cur_type_idx = atom_types.index(b_cfg['nucleus_type'])
            changed_nt, new_nt_idx = imgui.combo("Atom Type", cur_type_idx, atom_types)
            if changed_nt:
                b_cfg['nucleus_type'] = atom_types[new_nt_idx]
                self.molecule_scene.build(self.molecule_idx, self.anim_time)

            spec = MoleculeScene.ATOM_SPECS[b_cfg['nucleus_type']]
            shells_str = ", ".join(str(c) for c in spec['shells'])
            imgui.text_colored(
                f"Atomic Number Z = {spec['Z']} | Shells (K, L, M): [{shells_str}]",
                0.15, 0.35, 0.75, 1.0,
            )

            _, new_pos = imgui.slider_float3("Nucleus Pos", *b_cfg['nucleus_pos'], -2.0, 2.0, "%.2f")
            b_cfg['nucleus_pos'] = list(new_pos)
            _, b_cfg['orbit_speed'] = imgui.slider_float("Orbit Speed", b_cfg['orbit_speed'], 0.1, 3.0, "%.2f")
        else:
            imgui.text(f"Configure {current_name} Atoms & 3D Positions")
            m_cfg = self.molecule_scene.molecule_configs[current_name]
            _, m_cfg['vib_amp'] = imgui.slider_float("Vibration Amp", m_cfg['vib_amp'], 0.0, 0.5, "%.2f")

            rebuild_needed = False
            remove_idx = None

            for i, atom in enumerate(m_cfg['atoms']):
                t_idx = atom_types.index(atom['type'])
                changed_t, new_t = imgui.combo(f"Atom {i} Type##t{i}", t_idx, atom_types)
                if changed_t:
                    atom['type'] = atom_types[new_t]
                    rebuild_needed = True

                _, new_p = imgui.slider_float3(f"Atom {i} Pos##p{i}", *atom['pos'], -2.5, 2.5, "%.2f")
                atom['pos'] = list(new_p)

                if i > 0:
                    imgui.same_line()
                    if imgui.button(f"X##del{i}"):
                        remove_idx = i

            if remove_idx is not None:
                m_cfg['atoms'].pop(remove_idx)
                rebuild_needed = True

            if imgui.button("Add Bonded Atom"):
                m_cfg['atoms'].append({'type': 'H', 'pos': [0.0, 1.2, 0.0], 'bonded_to': 0})
                rebuild_needed = True
            imgui.same_line()
            if imgui.button("Reset Default Structure"):
                self.molecule_scene.reset_configs()
                rebuild_needed = True

            if rebuild_needed:
                self.molecule_scene.build(self.molecule_idx, self.anim_time)

    # =========================================================================
    # Scene Rendering
    # =========================================================================

    def _render_scene(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        proj, view, tb_model = self._draw_matrices()
        if tb_model is None or np.ndim(tb_model) < 2:
            tb_model = np.identity(4, dtype=np.float32)
        world_base = tb_model @ self.base_rotation

        if self.app_mode == 0:
            if self.show_grid and self.active_mesh:
                shader, uman = self.active_mesh.programs['interp']
                self.grid_lines.draw(shader, uman, proj, view, world_base)
            if self.active_mesh:
                obj_trs = compose_trs(self.obj_translate, self.obj_rotate, self.obj_scale)
                self.active_mesh.draw(proj, view, world_base @ obj_trs)
        else:
            self.molecule_scene.update(self.anim_time)
            self.molecule_scene.draw(proj, view, world_base)

        if self.show_help:
            self._ensure_hud()
            self._hud.draw(self.win)

    def run(self):
        self._set_window_title()

        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            self.impl.process_inputs()

            current_time = glfw.get_time()
            dt = current_time - self.last_time
            self.last_time = current_time
            if self.animate:
                self.anim_time += dt

            self._render_ui()
            self._render_scene()

            GL.glBindVertexArray(self.vao)
            imgui.render()
            self.impl.render(imgui.get_draw_data())
            GL.glBindVertexArray(0)

            glfw.swap_buffers(self.win)