Biobox provides a collection of data structures and methods for loading, manipulating and analyzing atomistic and pseudoatomistic structures.

Biobox main features:
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
* density map simulation.

## INSTALLATION AND REQUIREMENTS

Biobox requires Python3.x and the following packages:
* numpy
* scipy
* pandas
* scikit-learn
* cython

Biobox can be installed with: `pip install biobox`. Biobox can otherwise be installed manually typing the followin command in the Biobox folder: `python setup.py install`. Please make sure the folder where Biobox is located is in your PYTHONPATH.

Optional external software:
* CCS calculation relies on a call to [IMPACT](
https://process.innovation.ox.ac.uk/software/) (requires definition of IMPACTPATH environment variable)
* SAXS simulations rely on a call to [crysol](https://www.embl-hamburg.de/biosaxs/crysol.html), from ATSAS suite (requires definition of ATSASPATH environment variable)

## USAGE

Documentation:
* Biobox's API is available at https://degiacom.github.io/biobox/

Tutorial:
* A Jupyter notebook presenting Biobox's main functionalities is available at
https://github.com/degiacom/biobox_notebook
* The notebook can be run directly in a browser via Binder: https://mybinder.org/v2/gh/degiacom/biobox_notebook/HEAD

## CITATION

When using Biobox in your work, please cite: [L. S. P. Rudden, S. C. Musson, J. L. P. Benesch, M. T. Degiacomi (2021). Biobox: a toolbox for biomolecular modelling, Bioinformatics](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btab785/6428530?login=true)
