# Changelog

## 2.0.0

biobox 2.0 follows a correctness audit of version 1.1.5. More than 130 defects were fixed, each with a test that fails on the old code. Several fixes change the public API, and many change numerical results. Results obtained with 1.x and 2.x are not interchangeable, and scripts may need the edits listed under **Migrating from 1.x**.

### Migrating from 1.x

#### API changes

**Structures and molecules**

- **Transformations act on the current conformation only.** `translate` and `center_to_origin` now behave like `rotate`, `apply_transformation` and `align_axes`, in every class. In 1.1.5 they moved every frame. Call them once per frame (with `set_current`) if all frames must move.
- **Alternate locations and insertion codes** have their own data columns, `altloc` and `icode`. Atom names no longer carry the alternate location letter (1.1.5 read `"CA A"`), and all alternate locations are kept.
- **Residues are identified by chain, residue number and insertion code.** `atomselect(..., 52, ...)` selects residues 52 and 52A, while `"52A"` selects only 52A. `s2` returns the insertion code as an extra column.
- **New `formal_charge` column**, read from PDB columns 79-80 (0 when blank) and written back. Other importers set it to 0.
- `atomselect` raises for a non-numeric residue ID (unless `use_resname=True`) instead of matching nothing. It accepts numpy integers and numeric strings.
- `get_subset` accepts boolean masks and empty selections.
- **Two-character chain names** are written as their first character in column 22 and in full as segment identifier (columns 73-76), and read back from there. Files written by biobox 1.x, which put two-character chains in columns 22-23, no longer read back as two-character chains. Chain names of three or more characters raise.
- `properties['biomatrix']` is a dictionary `{biomolecule: [(chains, matrices), ...]}`, and `apply_biomatrix(biomolecule=1)` builds only the requested biological assembly, applying each operator to the chains it lists.
- `Structure.rmsf` and `Molecule.beta_factor_from_rmsf` no longer take a `step` argument.
- `Molecule.renumber_resid_keep_chains` no longer takes `atom_thresh`. It renumbers every residue (ligands and water included), honours `start_from` and clears insertion codes.
- `Molecule.reorder_resid` keeps the reordered chain in place, and raises if `idx` does not list every residue of the chain once.
- `Molecule.clean(remove_non_amino=True)` is pure Python and no longer takes a `path` to a shell script. Per residue it keeps the alternate location with the highest mean occupancy, and it returns a new molecule.
- `Molecule.get_fasta` writes `X` for unknown residues instead of raising, and maps common variants (MSE, HID/HIE/HIP, HSD/HSE/HSP, CYX/CYM, ASH, GLH, LYN, Amber terminal names).
- `Molecule.get_atoms_ccs` always returns a numpy array.
- `Molecule.__add__` and `Multimer.make_molecule` keep all data columns and give a 0-based `index`.
- An empty `Structure()` (and one after `clear()`) has 0 frames and 0 points.

**Assemblies and shapes**

- `Assembly.get_uxyz` returns a list (units may differ in size).
- `Assembly.contact_ratio` returns the fraction (0 to 1) of unit2's points inside unit1, and raises if unit1 has no `check_inclusion`.
- `check_inclusion` returns a boolean array for both `Sphere` and `Ellipsoid` (`Ellipsoid` returned a count).
- Convex shapes (`Prism`, `Cylinder`, `Cone`, `Sphere`, `Ellipsoid`) store their nominal dimensions in `properties` (in 1.1.5, the dimensions of the point centres), and raise when the point radius is too large for the shape. See **Changed results** for their new geometry.
- `Polyhedron.write_poly_architecture` expects one deformation coefficient per deformation class, like `generate_polyhedron`.

**Measures and paths**

- `Xlink._get_half_sphere` is now the public `Xlink.get_half_sphere`. `Xlink.distance_matrix` exposes all its options as `sphere_pts_surf`, `sphere_thresh` and `sphere_radii`.
- `Xlink.set_clashing_atoms` has a new `probe` argument (default 1.7 Å), placed before `points`. With `atoms_vdw=True`, `params` holds one exclusion radius per obstacle point.
- `sasa`'s `threshold` only selects which atoms are returned as surface atoms. The area and the mesh always count every exposed point. `sasa` raises on non-finite radii, and returns 0.0 for an empty structure.
- `Density.import_map` raises when a map cannot be loaded, instead of printing a message and leaving an empty map.
- `Molecule.get_dipole_density` (and `c_get_dipole_density`) raises `ValueError` for an `eqn` other than `'gauss'` or `'slater'`, and when no voxel fluctuates.
- `ccs` in executable mode (`use_lib=False`) raises for pseudo-atom structures with more than one radius.

#### Changed results (what to re-run)

- **Shortest solvent-accessible paths** (`Path.search_path`, `Xlink.distance_matrix`). The search uses Euclidean costs and an admissible heuristic, disconnected pairs return -1 (1.1.5 returned 0), line-of-sight shortcuts and smoothing no longer cut through obstacles, grid nodes are no longer offset by half a voxel, and the default global grid is centred on the middle of the obstacles' range. On HSP lysine pairs with smoothing, distances changed by a median of +0.65 Å (up to +7.2 Å).
- **Paths with `get_path=True` and `smooth=True`**: the filled path keeps every waypoint and spaces points by at most 1 Å, so smoothed distances change slightly (HSP: -0.90 to +0.67 Å).
- **Grids with `use_hull=True`** (the `Path.setup_global_search` default): the hull is swollen about its centroid, not the origin. **Grids requested with `boundaries` or `cloud`** are placed where requested (1.1.5 offset them).
- **The `atoms_vdw=True` clash model**: a grid point is blocked when closer to an atom than its van der Waals radius plus `probe`, in Å, for local and global grids alike and independently of the grid step. On HSP lysine pairs, local search links 22 pairs that 1.1.5 reported as buried, and global distances rise by 2.3 Å on average. The default model (`atoms_vdw=False`) is unchanged.
- **SASA** with the default threshold counts every exposed atom: HSP gives 10525 Å² instead of 9606 Å². This also changes `Assembly.get_buried`, which in addition now uses the units' real radii.
- **Element radii for PDB files without an element column**: elements are guessed from atom names, instead of every atom getting 1.8 Å. This affects SASA, CCS and densities for such files. Calcium has a radius (2.31 Å).
- **Convex shapes**: points are placed so that their spheres touch the nominal surface of the shape, surface and volume come from the Steiner formula for the body the points trace, and the CCS is a quarter of the surface of that body inflated by the gas radius (Vouk, Nature 162, 330, 1948). The analytical CCS now agrees with the CCS of the point cloud within 0.4% for every shape. Examples (gas radius 1 Å): `Cylinder(10, 20)` CCS 462 to 538 Å², `Prism(10, 20, 6)` 421 to 491 Å², `Cone(10, 30)` 282 to 385 Å². `Sphere(10)` is unchanged. Repeated `Sphere.squeeze` calls no longer compound.
- **Density maps from MRC2000/2014 files** take their origin from the header (EMD-1080 now loads at -135 Å on each axis, as with mrcfile). Mode 0 is read as signed 8-bit, and mode 6 is supported.
- **Simulated densities and electrostatics** (`get_density`, `get_vdw_density`, `get_electrostatics`): one shared grid whose stored origin is the grid corner, per-element maps built from that element's atoms only, charges summed per voxel, and a centred Coulomb kernel.
- **Biological assemblies** (`apply_biomatrix`): operators are applied as x' = Rx + t, only to the chains they list. 12 assemblies from 10 PDB entries match the RCSB assembly files.
- **Dipole densities**: kernels are centred on their voxel (an odd `vox_in_window` gives a 3-point kernel instead of 4), clipped at grid edges, and dipole-map cones are selected by the magnitude of the averaged dipole.
- **`Density.best_threshold`** places points at the tested threshold with the smallest mass error.
- **PDB output**: a TER record after the last ATOM record of every chain (taking a serial number, none after ligands, water or ion-only chains), 80-column atom lines with the formal charge, the MODEL serial in columns 11-14, a final END record, the radius in the beta column for `Structure` and `Assembly`, and hybrid-36 serials above 99999. `Multimer.write_pdb` writes one MODEL per frame. GRO and PQR output were also corrected.

### Fixed

Beyond the changes above, fixes include:

- **Reading files**: PDB and PQR files without END keep their charges, GRO imports have numeric columns, CASTEP `.md` imports have complete data columns, and MRC maps with a single section, mode 6 data or a "Chimera rotation" label load.
- **Molecules**: `same_residue` with lists, `guess_chain_split` on capped peptides (ACE/NME), `addall` with numpy arrays, `match_residue` with missing residues, and `pdb2pqr` histidine naming, terminal residues and its default forcefield path.
- **Structures**: `get_center` returns a copy, principal axes are taken about the centre of geometry and form a rotation matrix, `convex_hull` works, numpy index arrays are accepted, and numpy scalar radii are accepted.
- **Assemblies and polyhedra**: unit labels, `load_list` on non-empty assemblies, builders that translated twice or by label, `add_conformation` with multi-frame units, `Polyhedron.set_current`, `rmsd_distance_matrix` selections, and `get_neighbors`.
- **Measures**:
  - `sasa`, `rgyr` and `ccs` on a `Polyhedron`.
  - `saxs` no longer deletes the input PDB, and works on Windows.
  - CCS library names on macOS, and IMPACT parameter files kept out of the working directory.
  - Density predictions without a scan raise a clear error.
  - Path search no longer crashes on targets without accessible grid points.

### Removed

- `Xlink._get_sphere` (unused) and the private name `Xlink._get_half_sphere` (now `get_half_sphere`).
- `src/biobox/classes/remove_alt_conf.sh` (`clean` no longer needs it) and `src/biobox/lib/e_den_setup.py` (`setup.py` builds every extension).
- The unused text-map readers of `density_MRC.py`.

### Packaging and documentation

- Requires Python 3.10 or later and numpy 1.26 or later. Package metadata is in `pyproject.toml`, which declares the build requirements (setuptools 77 or later, Cython 3, numpy 2), so `pip install .` works without preinstalling them. Cython is no longer a runtime dependency.
- Licence declared as GPL-2.0-or-later, as stated in the source headers.
- Documentation moved to [biobox.readthedocs.io](https://biobox.readthedocs.io), with every docstring checked against the code and the examples run against this version.
- Continuous integration tests Python 3.10 to 3.14 on Linux, Windows and macOS, numpy 1.26, and the installed wheel.

### Known issues

- `Assembly.make_fiber` is experimental: units are placed at fractional rows, the composite fiber types (`'pmm'`, `'cmm'`) do not combine their transformations correctly, and `min_height` has no effect.
- `check_inclusion` of `Ellipsoid` and of a squeezed `Sphere` takes the semi-axes along x, y and z: after a rotation, the test is wrong.
- In path search, a target counts as buried (-2) when the squared distance to its closest accessible grid point exceeds `maxdist + step`. This is documented, and kept so that results remain comparable.
- With the default clash model, the accessibility threshold (maximum minus three standard deviations of the density) leaves about 0.4% of grid points at the protein edge accessible (HSP).
- `Molecule.get_mass_by_residue` does not add the water of each chain's termini.
