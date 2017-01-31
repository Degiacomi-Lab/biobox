.. biobox documentation master file, created by
   sphinx-quickstart on Wed Dec 28 17:34:22 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


BiobOx's documentation
======================

BiobOx provides a collection of data structures and methods for loading, manipulating and analyzing atomistic and pseudo-atomistic structures.

BiobOx main features:

* importing of PDB, PQR and GRO files, possibly containing multiple conformations (e.g. multi PDB, gro trajectory)
* generation of coarse grain shapes composed of specific arrangements of pseudoatoms
* loading and manipulation of density maps, including their transformation into a solid object upon isovalue definition
* assemblies of any points arrangement can be produced (i.e. densities converted in solid and geometric shapes can be equally treated).

allowed operations on structures include:

*  rototranslation and alignment on principal axes
*  on ensembles: RMSD, RMSF, PCA and clustering
*  calculation of CCS, SAXS, SASA, convex hull, s2 (for molecules), mass and volume estimation
*  atomselect for molecules and assemblies of molecules
*  shortest physical paths between atoms on molecule using Theta* (or A*)
*  density map simulation


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   structure
   assembly
   measures   
   examples

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


