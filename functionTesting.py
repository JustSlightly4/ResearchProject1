from ase.io import read
from ase.data import covalent_radii
from voxelgridC import VoxelGridC
from voxelgrid import VoxelGrid
import time
import numpy as np


def plot_3D(vgC, threshold=0.1, s=5, draw_cell=True):
        """
        Plot the VoxelGrid in real space using a scatter plot.
        Only voxels with value > threshold are plotted.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Create mesh of fractional coordinates
        nx, ny, nz = vgC.gpts
        ix, iy, iz = np.meshgrid(
            np.arange(nx) + 0.5,
            np.arange(ny) + 0.5,
            np.arange(nz) + 0.5,
            indexing='ij'
        )

        frac_coords = np.stack([ix / nx, iy / ny, iz / nz], axis=-1)  # (nx, ny, nz, 3)
        # Convert fractional coords to Cartesian using row-wise cell vectors
        real_coords = frac_coords @ vgC.cell  # (nx, ny, nz, 3)

        # Mask
        mask = vgC.grid > threshold
        xyz = real_coords[mask]
        values = vgC.grid[mask]

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        p = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    c=values, cmap='viridis', s=s)
        fig.colorbar(p, ax=ax, label='Voxel value')

        # Draw cell if requested
        if draw_cell:
            # Cell corners in fractional coords
            corners_frac = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1]
            ])
            corners = corners_frac @ vgC.cell  # shape (8, 3)

            # Define the 12 edges
            edges = [
                (0, 1), (0, 2), (0, 3),
                (1, 4), (1, 5),
                (2, 4), (2, 6),
                (3, 5), (3, 6),
                (4, 7), (5, 7), (6, 7)
            ]
            for i, j in edges:
                xs = [corners[i, 0], corners[j, 0]]
                ys = [corners[i, 1], corners[j, 1]]
                zs = [corners[i, 2], corners[j, 2]]
                ax.plot(xs, ys, zs, color='black')

        # Set equal aspect ratio
        all_coords = np.concatenate([xyz, corners]) if draw_cell else xyz
        xlim = [all_coords[:, 0].min(), all_coords[:, 0].max()]
        ylim = [all_coords[:, 1].min(), all_coords[:, 1].max()]
        zlim = [all_coords[:, 2].min(), all_coords[:, 2].max()]

        max_range = max(
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            zlim[1] - zlim[0]
        ) / 2.0

        mid_x = np.mean(xlim)
        mid_y = np.mean(ylim)
        mid_z = np.mean(zlim)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.tight_layout()
        plt.show()
        
def plot_2D(vgC, axis='z', index=None, position=None, threshold=0.1, draw_cell=True, real_space=True):
        """
        Plot a 2D slice of the VoxelGrid along a given axis.

        Parameters:
        - axis: 'x', 'y', or 'z' (which axis to slice along)
        - index: int, voxel index to slice at (mutually exclusive with position)
        - position: float, real-space coordinate along that axis (Å)
        - threshold: float, only show voxels with value > threshold
        - draw_cell: bool, whether to overlay the 2D projection of the unit cell (only in real_space mode)
        - real_space: bool, whether to plot in real-space coordinates (default True).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        ax_map = {'x': 0, 'y': 1, 'z': 2}
        if axis not in ax_map:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        ax_idx = ax_map[axis]

        if index is not None and position is not None:
            raise ValueError("Specify either `index` or `position`, not both")
        if position is not None:
            index = vgC.position_to_index(np.eye(3)[ax_idx] * position)[ax_idx]
        if index is None:
            index = vgC.gpts[ax_idx] // 2

        shape = vgC.grid.shape
        if not (0 <= index < shape[ax_idx]):
            raise IndexError(f"{axis}-index {index} out of bounds (0 to {shape[ax_idx] - 1})")

        # Slice axes
        axes = [0, 1, 2]
        axes.remove(ax_idx)
        ax1, ax2 = axes

        # Extract 2D slice
        slicers = [slice(None)] * 3
        slicers[ax_idx] = index
        slice_grid = vgC.grid[tuple(slicers)]

        n1, n2 = vgC.gpts[ax1], vgC.gpts[ax2]

        if real_space:
            i1 = (np.arange(n1) + 0.5) / n1
            i2 = (np.arange(n2) + 0.5) / n2
            coords = np.meshgrid(i1, i2, indexing='ij')
            frac_coords = np.stack(coords, axis=-1)
            xy = frac_coords @ vgC.cell[[ax1, ax2], :]
            xvals, yvals = xy[..., 0], xy[..., 1]
        else:
            xvals, yvals = np.meshgrid(np.arange(n1), np.arange(n2), indexing='ij')

        mask = slice_grid > threshold

        fig, ax = plt.subplots()
        sc = ax.scatter(xvals[mask], yvals[mask], c=slice_grid[mask], cmap='viridis', s=10)
        fig.colorbar(sc, ax=ax, label='Voxel value')

        if draw_cell and real_space:
            corners_frac = np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                [0, 0]
            ])
            corners_real = corners_frac @ vgC.cell[[ax1, ax2], :]
            ax.plot(corners_real[:, 0], corners_real[:, 1], 'k--', lw=1)

        ax.set_xlabel(f'{["x", "y", "z"][ax1]}' + (' [Å]' if real_space else ' (voxel)'))
        ax.set_ylabel(f'{["x", "y", "z"][ax2]}' + (' [Å]' if real_space else ' (voxel)'))
        ax.set_title(f'{axis.upper()} Slice at index {index}')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

#Standard Given Code from Szilvasi
def task1Py():
	# Load atoms from VASP POSCAR
	atoms = read("POSCAR_0")

	# Create voxel grid with resolution 0.3 Å
	vg = VoxelGrid(atoms.cell, resolution=0.3)

	# Add "outer shells" around atoms
	for atom in atoms:
		vg.add_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.7,
		value=1)

	# Subtract "inner cores" (set to 0 inside)
	for atom in atoms:
		vg.set_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.5,
		value=0)

	# Create generator of candidate sites (values between 2.0–3.0)
	gen = vg.sample_voxels_in_range(min_val=2.0, max_val=3.0, min_dist=1.2)

	# Draw 5 samples
	for attempt in range(5):
		pos = next(gen)
		print(pos)
	print("")
	
def task1C():
	# Load atoms from VASP POSCAR
	atoms = read("POSCAR_0")

	# Create voxel grid with resolution 0.3 Å
	vg = VoxelGridC(atoms.cell, resolution=0.3)

	# Add "outer shells" around atoms
	for atom in atoms:
		vg.add_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.7,
		value=1)

	# Subtract "inner cores" (set to 0 inside)
	for atom in atoms:
		vg.set_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.5,
		value=0)

	# Create generator of candidate sites (values between 2.0–3.0)
	gen = iter(vg.sample_voxels_in_range(min_val=2.0, max_val=3.0, min_dist=1.2))

	# Draw 5 samples
	for attempt in range(5):
		pos = next(gen)
		print(pos)
	print("")
		
def plot_3D_C():
	# Load atoms from VASP POSCAR
	atoms = read("POSCAR_0")

	# Create voxel grid with resolution 0.3 Å
	vg = VoxelGridC(atoms.cell, resolution=0.3)

	# Add "outer shells" around atoms
	for atom in atoms:
		vg.add_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.7,
		value=1)

	# Subtract "inner cores" (set to 0 inside)
	for atom in atoms:
		vg.set_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.5,
		value=0)

	plot_3D(vg)
	
def plot_3D_Py():
	# Load atoms from VASP POSCAR
	atoms = read("POSCAR_0")

	# Create voxel grid with resolution 0.3 Å
	vg = VoxelGrid(atoms.cell, resolution=0.3)

	# Add "outer shells" around atoms
	for atom in atoms:
		vg.add_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.7,
		value=1)

	# Subtract "inner cores" (set to 0 inside)
	for atom in atoms:
		vg.set_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.5,
		value=0)

	vg.plot_3D()
	
def plot_2D_C():
	# Load atoms from VASP POSCAR
	atoms = read("POSCAR_0")

	# Create voxel grid with resolution 0.3 Å
	vg = VoxelGridC(atoms.cell, resolution=0.3)

	# Add "outer shells" around atoms
	for atom in atoms:
		vg.add_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.7,
		value=1)

	# Subtract "inner cores" (set to 0 inside)
	for atom in atoms:
		vg.set_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.5,
		value=0)

	plot_2D(vg)
	
def plot_2D_Py():
	# Load atoms from VASP POSCAR
	atoms = read("POSCAR_0")

	# Create voxel grid with resolution 0.3 Å
	vg = VoxelGrid(atoms.cell, resolution=0.3)

	# Add "outer shells" around atoms
	for atom in atoms:
		vg.add_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.7,
		value=1)

	# Subtract "inner cores" (set to 0 inside)
	for atom in atoms:
		vg.set_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.5,
		value=0)

	vg.plot_2D()
		
def unit_test():
	# Load atoms from VASP POSCAR
	atoms = read("POSCAR_0")

	# Create voxel grid with resolution 0.3 Å
	vg = VoxelGrid(atoms.cell, resolution=0.3)
	vgC = VoxelGridC(atoms.cell, resolution=0.3)

	# Add "outer shells" around atoms
	for atom in atoms:
		vg.add_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.7,
		value=1)
		vgC.add_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.7,
		value=1)

	# Subtract "inner cores" (set to 0 inside)
	for atom in atoms:
		vg.set_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.5,
		value=0)
		vgC.set_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.5,
		value=0)
	
	result = np.array_equal(vg.cell, vgC.cell)
	print("Unit Test 1 (cell equal): " + str(result))  # True if all elements match exactly, False otherwise
	
	result = np.allclose(vg.cell_inv, vgC.cell_inv)
	print("Unit Test 2 (cell_inv equal): " + str(result))  # True if all elements match exactly, False otherwise
	
	result = np.array_equal(vg.gpts, vgC.gpts)
	print("Unit Test 3 (gpts equal): " + str(result))  # True if all elements match exactly, False otherwise
	
	result = np.array_equal(vg.resolution, vgC.resolution)
	print("Unit Test 4 (resolution equal): " + str(result))  # True if all elements match exactly, False otherwise
	
	result = np.allclose(vg.grid, vgC.grid)
	print("Unit Test 5 (grid equal): " + str(result))  # True if all elements match exactly, False otherwise
	
def main():
	unit_test()
	start = time.perf_counter()
	task1Py()
	end = time.perf_counter()
	print(f"Python Execution time: {end-start:.6f} seconds\n")
	
	start = time.perf_counter()
	task1C()
	end = time.perf_counter()
	print(f"C++ Execution time: {end-start:.6f} seconds\n")
	return 0
main()
