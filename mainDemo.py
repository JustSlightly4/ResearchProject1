from ase.io import read
from ase.data import covalent_radii
#from voxelgrid import VoxelGrid   # import your class
from voxelgrid import VoxelGrid
import time
import os

def write_poscar_used(fileName, poscar):
    # Check if file exists and is empty
    if not os.path.exists(fileName) or os.stat(fileName).st_size == 0:
        with open(fileName, "w") as f:  # use "w" since it's empty
            f.write("POSCAR used: " + poscar + "\n")

def remove_average_lines(filename):
    # Read all lines
    with open(filename, "r") as f:
        lines = f.readlines()

    # Filter out lines containing "Average execution time"
    filtered = [line for line in lines if "Average execution time" not in line]

    # Overwrite the file with only the filtered lines
    with open(filename, "w") as f:
        f.writelines(filtered)

def write_avg_execution_time(filename):
    remove_average_lines(filename)
    times = []
    with open(filename, "r") as f:
        for line in f:
            if "Execution time:" in line:
                parts = line.strip().split()
                seconds = float(parts[2])  # extract the number
                times.append(seconds)

    if not times:
        raise ValueError("No execution times found in file.")

    avg = sum(times) / len(times)

    # Append the average to the file
    with open(filename, "a") as f:
        f.write(f"Average execution time: {avg:.6f} seconds\n")

def task(poscar):
	# Load atoms from VASP POSCAR
	atoms = read(poscar)

	# Create voxel grid with resolution 0.3 Å
	vg = VoxelGrid(atoms.cell, resolution=0.3)

	# Add "outer shells" around atoms
	for atom in atoms:
		vg.add_sphere(atom.position,
		covalent_radii[atom.number] * 1.7,
		1)

	# Subtract "inner cores" (set to 0 inside)
	for atom in atoms:
		vg.set_sphere(atom.position,
		covalent_radii[atom.number] * 1.5,
		0)

	# Create generator of candidate sites (values between 2.0–3.0)
	gen = iter(vg.sample_voxels_in_range(2.0, 3.0, 1.2))

	# Draw 5 samples
	for attempt in range(5):
		pos = next(gen)
		print(pos)
		
def task2(poscar):
	# Load atoms from VASP POSCAR
	atoms = read(poscar)

	# Create voxel grid with resolution 0.3 Å
	vg = VoxelGrid(atoms.cell, resolution=0.3)
	
	# Add "outer shells" around atoms
	for atom in atoms:
		vg.add_sphere(
			atom.position,
			covalent_radii[atom.number] * 1.7,
			1
		)

	# Subtract "inner cores" (set to 0 inside)
	for atom in atoms:
		vg.set_sphere(
			atom.position,
			covalent_radii[atom.number] * 1.5,
			0
		)
		
	print(str(vg.index_to_position(1, 0, 0)))
	print(str(vg.position_to_index([0.20884615, 0.41769231, 0.41769231])))

def main():
	#Settings
	fileName = "test.txt"
	poscar = "POSCAR_0"
	
	#Write the POSCAR used
	write_poscar_used(fileName, poscar)
	
	for i in range(1):
		#Create timers and run the task
		start = time.perf_counter()
		task(poscar)
		end = time.perf_counter()
		
		#Calculate total time and add it to text file
		elapsed = end - start
		print(f"Execution time: {elapsed:.6f} seconds")
		
		#Writes the execution time to text file
		with open(fileName, "a") as f:   # "w" = overwrite, "a" = append
			f.write(f"Execution time: {elapsed:.6f} seconds\n")
		
	#Averages all the times in the text file and writes it to the file
	#Also, removes any old average times already written to the text file
	write_avg_execution_time(fileName)

	return 0


if __name__ == "__main__":
    main()
