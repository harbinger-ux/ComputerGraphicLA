import math
import numpy as np
from src.geometry.shapes_3d import create_cylinder, create_sphere, create_torus
from src.rendering.mesh import Mesh
from src.rendering.transforms import (
    align_y_to_vector,
    rotate_x,
    rotate_y,
    rotate_z,
    scale,
    translate,
)
from src.scene.scene_node import SceneNode


class MoleculeScene:
    """Manages hierarchical construction, configuration, and animation of Part 2 models."""

    MOLECULE_NAMES = ["Bohr Model", "H2O Molecule", "CO2 Molecule"]

    # Chemically accurate atomic numbers (Z), shell distributions (K, L, M), CPK colors, and radii
    ATOM_SPECS = {
        'H':  {'Z': 1,  'shells': [1],       'color': [0.90, 0.90, 0.90], 'radius': 0.25},
        'He': {'Z': 2,  'shells': [2],       'color': [0.85, 1.00, 1.00], 'radius': 0.28},
        'Li': {'Z': 3,  'shells': [2, 1],    'color': [0.80, 0.50, 1.00], 'radius': 0.42},
        'C':  {'Z': 6,  'shells': [2, 4],    'color': [0.15, 0.15, 0.15], 'radius': 0.50},
        'N':  {'Z': 7,  'shells': [2, 5],    'color': [0.18, 0.31, 0.97], 'radius': 0.45},
        'O':  {'Z': 8,  'shells': [2, 6],    'color': [0.90, 0.10, 0.10], 'radius': 0.40},
        'F':  {'Z': 9,  'shells': [2, 7],    'color': [0.56, 0.88, 0.31], 'radius': 0.38},
        'Ne': {'Z': 10, 'shells': [2, 8],    'color': [0.70, 0.89, 0.96], 'radius': 0.38},
        'Na': {'Z': 11, 'shells': [2, 8, 1], 'color': [0.67, 0.36, 0.95], 'radius': 0.52},
        'S':  {'Z': 16, 'shells': [2, 8, 6], 'color': [1.00, 0.78, 0.16], 'radius': 0.48},
        'Cl': {'Z': 17, 'shells': [2, 8, 7], 'color': [0.12, 0.94, 0.12], 'radius': 0.46},
    }
    ATOM_TYPES = list(ATOM_SPECS.keys())

    SHELL_DEFS = [
        ('shell1', 1.2),  # K shell (n=1, max 2e-)
        ('shell2', 2.0),  # L shell (n=2, max 8e-)
        ('shell3', 2.8),  # M shell (n=3, max 8e- for Z <= 18)
    ]

    def __init__(self):
        self.root = None
        self.anim_nodes = {}
        self.current_name = self.MOLECULE_NAMES[0]
        self.shading_mode = "phong"

        self.atom_meshes = {
            sym: self._get_colored_mesh(create_sphere, spec['color'], radius=spec['radius'])
            for sym, spec in self.ATOM_SPECS.items()
        }
        self.meshes = {
            'e': self._get_colored_mesh(create_sphere, [0.2, 0.5, 0.9], radius=0.08),
            'bond': self._get_colored_mesh(create_cylinder, [0.7, 0.7, 0.7], radius=0.08, height=1.0),
            'shell1': self._get_colored_mesh(
                create_torus, [0.5, 0.5, 0.5], inner_radius=0.01, outer_radius=1.2, rings=120
            ),
            'shell2': self._get_colored_mesh(
                create_torus, [0.5, 0.5, 0.5], inner_radius=0.01, outer_radius=2.0, rings=120
            ),
            'shell3': self._get_colored_mesh(
                create_torus, [0.5, 0.5, 0.5], inner_radius=0.01, outer_radius=2.8, rings=120
            ),
        }

        self.reset_configs()

    def _all_meshes(self):
        yield from self.atom_meshes.values()
        yield from self.meshes.values()

    def apply_shading_mode(self, mode):
        """Applies Part 1 shading modes (flat, gouraud, phong, texture, edge) to all Part 2 meshes."""
        self.shading_mode = mode
        for mesh in self._all_meshes():
            if mode == "edge":
                mesh.draw_as_lines = True
                mesh.shading = "flat"
            else:
                mesh.draw_as_lines = False
                mesh.shading = mode

    def apply_vertex_gradient(self, use_gradient):
        """Toggles 3D vertex color gradient (for Gouraud) or restores CPK element colors."""
        for mesh in self._all_meshes():
            mesh.update_colors(uniform_rgb=None, use_gradient=use_gradient)

    def load_texture(self, filepath):
        """Loads an external texture image onto all Part 2 meshes."""
        resolved_path = None
        for mesh in self._all_meshes():
            resolved_path = mesh.load_texture_from_file(filepath)
        return resolved_path

    def reset_checker_texture(self):
        """Resets all Part 2 meshes to the default 8x8 procedural checker texture."""
        for mesh in self._all_meshes():
            mesh.reset_checker_texture()

    def reset_configs(self):
        self.bohr_config = {
            'nucleus_type': 'C',
            'nucleus_pos': [0.0, 0.0, 0.0],
            'orbit_speed': 1.0,
        }

        h2o_angle = math.radians(127.75)
        h2o_L = 1.2
        hx = round(h2o_L * math.sin(h2o_angle), 3)
        hy = round(-h2o_L * math.cos(h2o_angle), 3)

        self.molecule_configs = {
            "H2O Molecule": {
                'vib_amp': 0.15,
                'vib_freq': 8.0,
                'atoms': [
                    {'type': 'O', 'pos': [0.0, 0.0, 0.0], 'bonded_to': -1},
                    {'type': 'H', 'pos': [hx, hy, 0.0], 'bonded_to': 0},
                    {'type': 'H', 'pos': [-hx, hy, 0.0], 'bonded_to': 0},
                ],
            },
            "CO2 Molecule": {
                'vib_amp': 0.20,
                'vib_freq': 6.0,
                'atoms': [
                    {'type': 'C', 'pos': [0.0, 0.0, 0.0], 'bonded_to': -1},
                    {'type': 'O', 'pos': [1.5, 0.0, 0.0], 'bonded_to': 0},
                    {'type': 'O', 'pos': [-1.5, 0.0, 0.0], 'bonded_to': 0},
                ],
            },
        }

    @staticmethod
    def _get_colored_mesh(func, color, **kwargs):
        v, c, n, uv, idx = func(**kwargs)
        c = np.ones_like(c) * np.array(color, dtype=np.float32)
        mesh = Mesh(v, c, n, uv, idx)
        mesh.shading = 'phong'
        return mesh

    def build(self, idx, anim_time=0.0):
        self.root = SceneNode()
        self.anim_nodes = {}
        self.current_name = self.MOLECULE_NAMES[idx]

        if self.current_name.startswith("Bohr Model"):
            self._build_bohr(anim_time)
        else:
            self._build_molecule(self.current_name, anim_time)

        self.apply_shading_mode(self.shading_mode)

    def _build_bohr(self, anim_time):
        nuc_type = self.bohr_config['nucleus_type']
        nuc_pos = self.bohr_config['nucleus_pos']
        shell_counts = self.ATOM_SPECS[nuc_type]['shells']

        nuc_node = self.root.add_child(SceneNode(self.atom_meshes[nuc_type]))
        nuc_node.transform = translate(*nuc_pos)
        self.anim_nodes['nucleus'] = nuc_node

        for (shell_key, radius), e_count in zip(self.SHELL_DEFS, shell_counts):
            if e_count <= 0:
                continue

            s_node = nuc_node.add_child(SceneNode(self.meshes[shell_key]))
            for i in range(e_count):
                rad = i * (2.0 * math.pi / e_count)
                e_node = s_node.add_child(SceneNode(self.meshes['e']))
                e_node.transform = translate(radius * math.cos(rad), radius * math.sin(rad), 0.0)
            self.anim_nodes[shell_key] = s_node

        self.update_bohr(anim_time)

    def _build_molecule(self, mol_name, anim_time):
        cfg = self.molecule_configs[mol_name]
        atoms = cfg['atoms']

        atom_nodes = []
        for i, atom in enumerate(atoms):
            node = SceneNode(self.atom_meshes[atom['type']])
            atom_nodes.append(node)
            self.anim_nodes[f'atom_{i}'] = node

        for i, atom in enumerate(atoms):
            parent_idx = atom.get('bonded_to', -1)
            if 0 <= parent_idx < len(atoms) and parent_idx != i:
                parent_node = atom_nodes[parent_idx]
                bond_node = parent_node.add_child(SceneNode(self.meshes['bond']))
                parent_node.add_child(atom_nodes[i])
                self.anim_nodes[f'bond_{i}'] = bond_node
            else:
                self.root.add_child(atom_nodes[i])

        self.update_molecule(mol_name, anim_time)

    def update_bohr(self, time):
        spd = float(self.bohr_config['orbit_speed'])
        if 'nucleus' in self.anim_nodes:
            self.anim_nodes['nucleus'].transform = translate(*self.bohr_config['nucleus_pos'])
        if 'shell1' in self.anim_nodes:
            self.anim_nodes['shell1'].transform = rotate_x(time * 2.0 * spd)
        if 'shell2' in self.anim_nodes:
            self.anim_nodes['shell2'].transform = rotate_y(time * 1.5 * spd) @ rotate_x(math.radians(45))
        if 'shell3' in self.anim_nodes:
            self.anim_nodes['shell3'].transform = rotate_z(time * 1.2 * spd) @ rotate_y(math.radians(-35))

    def update_molecule(self, mol_name, time):
        cfg = self.molecule_configs[mol_name]
        atoms = cfg['atoms']
        delta_L = cfg['vib_amp'] * math.sin(time * cfg['vib_freq'])

        for i, atom in enumerate(atoms):
            atom_node = self.anim_nodes.get(f'atom_{i}')
            if atom_node is None:
                continue

            parent_idx = atom.get('bonded_to', -1)
            pos = np.array(atom['pos'], dtype=np.float32)

            if 0 <= parent_idx < len(atoms) and parent_idx != i:
                parent_pos = np.array(atoms[parent_idx]['pos'], dtype=np.float32)
                rel_vec = pos - parent_pos
                base_dist = float(np.linalg.norm(rel_vec))

                if base_dist > 1e-5:
                    direction = rel_vec / base_dist
                    cur_len = max(0.15, base_dist + delta_L)
                    cur_rel = direction * cur_len
                else:
                    direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    cur_len = 0.01
                    cur_rel = np.zeros(3, dtype=np.float32)

                atom_node.transform = translate(*cur_rel)
                bond_node = self.anim_nodes.get(f'bond_{i}')
                if bond_node is not None:
                    mid = 0.5 * cur_rel
                    rot = align_y_to_vector(direction)
                    bond_node.transform = translate(*mid) @ rot @ scale(1.0, cur_len, 1.0)
            else:
                atom_node.transform = translate(*pos)

    def update(self, time):
        if self.current_name.startswith("Bohr Model"):
            self.update_bohr(time)
        elif self.current_name in self.molecule_configs:
            self.update_molecule(self.current_name, time)

    def draw(self, proj, view, model_transform):
        if self.root:
            self.root.draw(proj, view, model_transform)