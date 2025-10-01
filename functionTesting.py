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
	gen = vg.sample_voxels_in_range(min_val=2.0, max_val=3.0, min_dist=1.2, seed=1)

	# Draw 5 samples
	for attempt in range(5):
		pos = next(gen)
		print(pos)
		
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
	gen = iter(vg.sample_voxels_in_range(min_val=2.0, max_val=3.0, min_dist=1.2, seed=1))

	# Draw 5 samples
	for attempt in range(5):
		pos = next(gen)
		print(pos)

#Print index (1, 0, 0)
def task2Py():
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
		
	print(str(vg.index_to_position(1, 0, 0)))
	
def task2C():
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
		
	print(str(vg.index_to_position(1, 0, 0)))
	
#Prints all indexs on the x-axis e.g. (i, 0, 0)
def task3Py():
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
	
	for i in range(vg.gpts[0]):
		print(str(vg.index_to_position(i, 0, 0)))
		
def task3C():
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
	
	for i in range(vg.gpts[0]):
		print(str(vg.index_to_position(i, 0, 0)))

#stuff
def task4Py():
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
	
	for atom in atoms:
		vg.mul_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.1)
	
	print(str(vg.index_to_position(1, 0, 0)))

def task4C():
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
	
	for atom in atoms:
		vg.mul_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 10)
	
	print(str(vg.index_to_position(1, 0, 0)))
	
def task5Py():
    # Identity cell scaled to 10 Å cube
    cell = [[10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]]

    # Build voxel grid with resolution 1.0 Å
    vg = VoxelGrid(cell, resolution=1.0)

    # Initialize all voxels to 2.0
    for i in range(vg.gpts[0]):
        for j in range(vg.gpts[1]):
            for k in range(vg.gpts[2]):
                vg.grid[i][j][k] = 2.0 + k

    # Center of the box, radius that definitely covers the middle voxel
    center = [5.0, 5.0, 5.0]
    radius = 3.0
    divisor = 2.0

    # Apply division
    vg.div_sphere(center=center, radius=radius, factor=divisor)
    
    print(str(vg.grid[4][4][3]))
    print(str(vg.grid[5][5][5]))
   
def task5C():
    # Identity cell scaled to 10 Å cube
    cell = [[10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]]

    # Build voxel grid with resolution 1.0 Å
    vg = VoxelGridC(cell, resolution=1.0)

    # Initialize all voxels to 2.0
    for i in range(vg.gpts[0]):
        for j in range(vg.gpts[1]):
            for k in range(vg.gpts[2]):
                vg.grid[i][j][k] = 2.0 + k

    # Center of the box, radius that definitely covers the middle voxel
    center = [5.0, 5.0, 5.0]
    radius = 3.0
    divisor = 2.0

    # Apply division
    vg.div_sphere(center=center, radius=radius, factor=divisor)
    
    print(str(vg.grid[4][4][3]))
    print(str(vg.grid[5][5][5]))
    
def task6Py():
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

def task6C():
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
	
def testValues():
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
	
	a = vg.grid.flat
	b = vgC.grid.flat

	for x, y in zip(a, b):
			print(f"{x}   {y}")
			
	print(str(np.allclose(vg.grid, vgC.grid)))  # True, allows tiny differences
	
	zero_count = np.count_nonzero(vg.grid == 0)
	print("Py Zero Count: " + str(zero_count))  # Output: 3	
	
	zero_count = np.count_nonzero(vgC.grid == 0)
	print("C++ Zero Count: " + str(zero_count))  # Output: 3
	
	
def maskTest():
	# Load atoms from VASP POSCAR
	atoms = read("POSCAR_0")

	# Create voxel grid with resolution 0.3 Å
	vg = VoxelGrid(atoms.cell, resolution=0.3)
	vgC = VoxelGridC(atoms.cell, resolution=0.3)
	
	# Add "outer shells" around atoms
	for atom in atoms:
		print(np.array_equal(vg.add_sphere(center=atom.position,
		radius=covalent_radii[atom.number] * 1.7,
		value=1), vgC.cached_sphere_mask(radius=covalent_radii[atom.number] * 1.7)))
		
	
def main():
	task6C()
	"""
	start = time.perf_counter()
	task1Py()
	end = time.perf_counter()
	print(f"Python Execution time: {end-start:.6f} seconds\n")
	
	start = time.perf_counter()
	task1C()
	end = time.perf_counter()
	print(f"C++ Execution time: {end-start:.6f} seconds\n")
	"""
	return 0
main()
