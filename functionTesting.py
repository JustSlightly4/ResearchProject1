from ase.io import read
from ase.data import covalent_radii
from voxelgridC import VoxelGridC
from voxelgrid import VoxelGrid
import time

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
                vg.grid[i][j][k] = 2.0

    # Center of the box, radius that definitely covers the middle voxel
    center = [5.0, 5.0, 5.0]
    radius = 3.0
    divisor = 2.0

    # Apply division
    vg.div_sphere(center=center, radius=radius, factor=divisor)
    
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
                vg.grid[i][j][k] = 2.0

    # Center of the box, radius that definitely covers the middle voxel
    center = [5.0, 5.0, 5.0]
    radius = 3.0
    divisor = 2.0

    # Apply division
    vg.div_sphere(center=center, radius=radius, factor=divisor)
    
    print(str(vg.grid[5][5][5]))

def main():
	
	start = time.perf_counter()
	task1Py()
	end = time.perf_counter()
	print(f"Python Execution time: {end-start:.6f} seconds\n")
	
	start = time.perf_counter()
	task1C()
	end = time.perf_counter()
	print(f"C++ Execution time: {end-start:.6f} seconds")
	
	return 0
main()
