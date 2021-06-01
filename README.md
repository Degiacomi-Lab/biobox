BiobOx provides a collection of data structures and methods for loading, manipulating and analyzing atomistic and pseudoatomistic structures.

BiobOx main features:
* importing of PDB, PQR and GRO files, possibly containing multiple conformations (e.g. multi-PDB, gro trajectory)
* generation of coarse grain shapes composed of specific arrangements of pseudo-atoms
* loading and manipulation of density maps, including their transformation into a solid object upon isovalue definition
* assemblies of any points arrangement can be produced (i.e. densities converted in solid and geometric shapes can be equally treated).

Allowed operations on structures incude:
* generation of assemblies according to custom symmetries
* rototranslation and alignment on principal axes
* on ensembles: RMSD, RMSF, PCA and clustering
* calculation of CCS, SAXS, SASA, convex hull, s2 (for molecules), mass and volume estimation (for densities)
* atomselect for molecules and assemblies of molecules
* shortest solvent accessible path between atoms on molecule (using lazy Theta* or A*)
* density map simulation, 

See full documentation and examples [here](https://degiacom.github.io/biobox/).

## INSTALLATION AND REQUIREMENTS

BiobOx requires Python3.x and the following packages:
* numpy
* scipy
* pandas
* scikit-learn
* cython

install with: `python setup.py install` and make sure the folder where BiobOx is located is in your PYTHONPATH.

Optional external software:
* CCS calculation relies on a call to IMPACT (requires IMPACTPATH environment variable)
* SAXS simulations rely on a call to crysol, from ATSAS suite (requires ATSASPATH environment variable)

## CREDITS

* Importing of MRC format maps adapted from [CHIMERA](https://www.cgl.ucsf.edu/chimera/)
* Kabsch algorithm for RMSD calculation adapted from [Pymol](https://www.pymol.org/)
* A* implementation from [redblobgames](http://www.redblobgames.com)
