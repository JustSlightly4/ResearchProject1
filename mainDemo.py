from ase.io import read
from ase.data import covalent_radii
from voxelgrid import VoxelGrid   # import your class


def main():
    # Load atoms from VASP POSCAR
    atoms = read("POSCAR")

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

    return 0


if __name__ == "__main__":
    main()
