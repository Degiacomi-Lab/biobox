Examples
========

Selecting atoms from a (multi)PDB
---------------------------------

Let's load a molecule, and identify only the backbone atoms of chain A.

>>> M = biobox.Molecule()
>>> M.import_pdb("protein.pdb")
>>> pos, idx = M.atomselect("A", "*", ["CA","C","N","O"], get_index=True)

:func:`atomselect <molecule.Molecule.atomselect>` accepts as parameters single strings, lists or "*" as wildcard.
After this call, pos contains the coordinates of all selected atoms, and idx their indices.
Another way to select atoms, is to use the :func:`atomselect <molecule.Molecule.query>` method. The following call will yield the same result as the atomselect above.

>>> pos, idx = M.query('chain == "A" and name == ["CA","C","N","O"]', get_index=True)

The query methods follows the pandas query syntax, and allows to be more expressive. Any column stored in M.data (call M.data.columns) can be addressed.
Now that we have identified indices of interest, we can save a subset of the initial pdb in a new one, or to create a new :func:`Molecule <molecule.Molecule>` object containing only them.

>>> M.write_pdb("chainA.pdb", index=idx)
>>> M2 = M.get_subset(idx)

**multiple conformations** may be available in the PDB. By default, the first one is set as current.
Is is possible to set as current another one as follows: 

>>> M.set_current(2)
>>> pos2, idx2 = M.atomselect("A", "*", ["CA","C","N","O"], get_index=True)

After this new :func:`atomselect <molecule.Molecule.atomselect>` call, idx2 will be equal to idx1 (atom selected are still the same), but pos2 will be different from pos (atoms positions differ between different conformations).
Unless otherwise specified, :func:`get_subset <molecule.Molecule.get_subset>` selects all the alternative conformations from the atoms of interest.
:func:`get_subset <molecule.Molecule.get_subset>` can however also be instructed to select a subset of conformations, for instance: 

>>> M2 = M.get_subset(idx, conformations=[0,1,2])

This call will select only the conformations 0, 1 and 2 of atoms of interest.


protein conformations clustering 
--------------------------------

Suppose you have several PDB files of the same protein (same number of atoms), and you want to cluster them according to a hierarchical clustering.
We can for example add the coordinates of all pdb files to the same :func:`Molecule <molecule.Molecule>` instance (supposing that they all have the same amount of atoms):

>>> import glob
>>> files = glob.glob("*pdb")
>>> M = biobox.Molecule()
>>> M.import_pdb(files[0])
>>> for f in xrange(1, len(files)):
>>>     M2 = Molecule()
>>>     M2.import_pdb(f)
>>>     M2_xyz = M2.get_xyz()
>>>     M.add_xyz(M2_xyz)

In order to generate a hierarchical clustering of these conformations, we need a flattened RMSD distance matrix.
This can then be fed to scipy's Nearest Point Algorithm for clustering.
In this example, we will aggregate all structures having and RMSD smaller than 2 Angstrom.

>>> import scipy.cluster.hierarchy as SCH
>>> dist = M.rmsd_distance_matrix(flat=True)
>>> hierarchic_cluster = SCH.linkage(dist, method='single')
>>> flat_clusters = SCH.fcluster(hierarchic_cluster, 2.0, criterion='distance')


protein polyhedral assemblies 
-----------------------------

We want to produce several protein tetrahedral assemblies, and compare them to each other.
First, let's load our protein building block:

>>> M = biobox.Molecule()
>>> M.import_pdb("protein.pdb")

Now, let's create a :func:`Multimer <multimer.Multimer>` arranged according to a tetrahedral symmetry.
To do so, we have to load information about the tetrahedral scaffold BiobOx will exploit to align six monomers.
By default this information is stored in the file classes/polyhedron_database.dat, though the user can import his own database.

>>> P = biobox.Multimer()
>>> P.setup_polyhedron('Tetrahedron', M)
>>> P.generate_polyhedron(10,180,20,10)

Now, P contains six proteins arranged as a tetrahedron having a radius of 10 Angstrom.
Every subunit is rotated with respect of its specific position on the scaffold.
Rotation angles are defined with respect of the molecule's principal axes.
Here, we rotate by 180 degrees around the first principal axis, 20 around the second, and 10 around the third.
Let's now build two new polyhedra with different radii and rotation angles:

>>> P.generate_polyhedron(10,180,50,65, add_conformation=True)
>>> P.generate_polyhedron(12,185,40,60, add_conformation=True)

Since we set add_conformation=True, the atoms arrangement of the new multimers will be appended as new conformations.
With add_conformation=False (default) the previous subunits arrangements gets overwritten.

.. note:: assemblies' multiple conformations are treated by appending on each subunit its different conformation. BiobOx then sets on all subunits the same current position.

Now, we want to calculate the RMSD between the created multimers' alpha carbons. With these lines, dist_mat will contain the RMSD distance matrix between the multimers:

>>> idxs = P.atomselect("*", "*" ,"*", "CA", get_index=True)[1]
>>> dist_mat = P.rmsd_distance_matrix(points_indices=idxs)

Note that, as for the case of :func:`atomselect <molecule.Molecule>` objects, a :func:`query <molecule.Multimer.query>` method is also available. The same selection as the command above can be obtained with:

>>> idx = M.query('name == "CA"', get_index=True)[1]

To select atoms from some specific units, the following command can be issued:

>>> idx = M.query('unit == ["0", "3", "5"] and name == "CA"', get_index=True)[1]

Subunits can also be grouped, and different groups can be rotated differently.
In the following example, the tetrahedron's chains A, B, C and D, E, F form different groups that are rotated independently.

>>> import numpy as np
>>> P.conn_type = np.array([0, 0, 0, 1, 1, 1])
>>> P.generate_polyhedron(10, np.array([90,180]), np.array([0,0]), np.array([0,0]))

Note that when more than one edge type is provided, rotation angles should be in the form of a numpy array having the same length as the amount of different groups in connection (values in conn_type are used to index the angles arrays).

Polyhedral scaffolds are constituted of vertices connected by edges.
By altering the position of the vertices, the scaffolds can be deformed (e.g. useful to model near-symmetries).
In BiobOx, deformations are treated in terms of deformation vectors, i.e. unit-vectors indicating in which direction a vertex can move.
Here, we will allow the first vertex to move radially. We will then build a tetrahedron, where this vertex is displaced from its initial position by its deformation vector, scaled by a constant (here, 5).

>>> P.add_deformation(0)
>>> P.generate_polyhedron(10, np.array([90,180]), np.array([0,0]), np.array([0,0]), deformation=[5])

Note that :func:`add_deformation <polyhedron.Polyhedron.add_deformation>` also accepts user-defined deformation vectors.
To see how your scaffold looks like, a pdb file containing the vertices and an associated TCL script for `VMD <http://www.ks.uiuc.edu/Research/vmd>`_ (drawing colored edges, as a function of grouping) can be produced.

>>> P.write_poly_architecture("architecture", scale=10, deformation=[5])

This will generate two files architecture.pdb and architecture.tcl.
The initial unit-sized scaffold will scaled by 10, and the first vertex moved away radially.

.. seealso:: Protein polyhedral assemblies consistent with experimental data were generated in `Elisabeth's papers <http://awesome_link.com>`_


super coarse-grain modelling 
----------------------------

In this example, we will arrange a group of cylinders in a ring.
To do so, we have first to create a single collection of points arranged like a :func:`Cylinder <convex.Cylinder>`.
Unless otherwise specified (using the optional keyword radius), every point composing the cylinder (and any other convex point cloud) will have a radius of 1.4 Angstrom.
To simulate a smooth surface, one can either increase the points radius, or their density.
Here, we will use default values, and the resulting cylinder will then be rotated by 45 degrees along the x axis.

>>> cylinder_length = 20
>>> cylinder_radius = 10
>>> C = biobox.Cylinder(cylinder_length, cylinder_radius)
>>> C.rotate(45, 0, 0)

We will now create an assembly loading ten copies of our template cylinder, arrange them in a 30 Angstrom-wide circle, and save the resulting structure into a PDB file.

>>> A = biobox.Assembly()
>>> A.load(C, 10)
>>> A.make_circular_symmetry(30)
>>> A.write_pdb("assembly.pdb")

We can now assess some of the assembly's characteristics, for instance its height and width.
This can be done by extracting all the assembly's points coordinates in a unique numpy array.

>>> xyz = A.get_all_xyz()
>>> width = np.max(xyz[:, 0]) - np.min(xyz[:, 0])
>>> height = np.max(xyz[:, 2]) - np.min(xyz[:, 2])

An alternative way to measure assembly dimensions, it to profit of methods in :func:`Structure <structure.Structure>` class.
Here we collapse the Assembly's units coordinates in a single :func:`Structure <structure.Structure>` instance.

>>> S = A.make_structure()
>>> print S.get_size()

In case not all the subunits of the assembly are the same, a list of subunits can be loaded.
In this case, we will load a :func:`Sphere <convex.Sphere>` (and call it "S") as well as two identical cylinders (called "C1" and "C2").

>>> sphere_radius = 20
>>> cylinder_radius = 5
>>> cylinder_length = 50
>>> S = biobox.Sphere(sphere_radius)
>>> C = biobox.Cylinder(cylinder_radius, cylinder_length)
>>> A2 = biobox.Assembly()
>>> A2.load_list([S, C, C], ["S", "C1", "C2"])

Now, we will arrange the three loaded structures so that the bases of two cylinders are in touch with the sphere, and one cylinder is rotated by 45 degrees with respect to the other.

>>> A2.translate(0, 0, -cylinder_length/2.0-sphere_radius, ["C1", "C2"])
>>> A2.rotate(0.0, 45.0, 180.0, ["C2"])

As you can see, translations (and rotations) can be applied to units subsets.
In this case, we kept the sphere fixed, and only translated the cylinders, and then rotated just one of the two cylinders.

.. seealso:: this super-coarse grain approach was exploited to calculate the collision cross-section of curved chains of ellipsoids in `M. A. McDowell et al., Characterisation of Shigella Spa33 and Thermotoga FliM/N reveals a new model for C-ring assembly in T3SS, Molecular Microbiology, 2015 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4832279>`_ (Fig.3)
.. seealso:: a graphical representation of typical membrane protein arrangements was obtained combining super-coarse grain models and `VMD <http://www.ks.uiuc.edu/Research/vmd>`_-generated lipid bilayers, Fig.3 of `C. Bechara and C. V. Robinson, Different Modes of Lipid Binding to Membrane Proteins Probed by Mass Spectrometry, JACS, 2015 <http://pubs.acs.org/doi/abs/10.1021/jacs.5b00420>`_


density map cutoff via Collision Cross Section 
----------------------------------------------

Ion Mobility (IM) experiments report on a molecule's collision cross section (CCS).
Here we show how to relate IM data with a electron density 3D reconstruction obtained by Electron Microscopy (EM).

We first import a GroEL density map EMD-1457.mrc.

>>> D = biobox.Density()
>>> D.import_map("EMD-1457.mrc", "mrc")

Depending on which threshold value one selects, the resulting isosurface will have a certain volume and CCS.
We now compute the map's relationship between threshold, volume and CCS with 100 equally spaced threshold values.
This might take several minutes, depending on map size (by default, a scan between minimal and maximal map intensity is performed).
Obtained values will be returned in a numpy array containining as columns [threshold, volume, CCS].
This will also be stored in self.properties['scan'], for future usage.

>>> tvc = D.threshold_vol_ccs(low=0, sampling_points=100)

Let's predict the density CCS using a fitted mass-based threshold, and compare it the known CCS of 24500 A^2.
This requires providing the map's resolution (here, 5.4 Angstrom) and the mass of GroEL (801 kDa).
The procedure interrogates the data previously stored in D.properties['scan'].

>>> import numpy as np
>>> ccs_mass, fitted_mass_thresh = D.predict_ccs_from_mass(5.4, 801)
>>> error = 100 * (np.abs(ccs_mass - 24500)/24500)

Error should be typically less than 5%. Values greater than 8% indicate that the protein's conformation is likely different between EM and IM.
We can use fitted_mass_thresh to create a bead model, that can then be saved into a PDB.

>>> D.place_points(fitted_mass_thresh)
>>> D.write_pdb("model_ccs_mass.pdb")

.. seealso:: this method is described in `M. T. Degiacomi and J. L. P. Benesch, EMnIM: software for relating ion mobility mass spectrometry and electron microscopy data, Analyst, 2016 <http://pubs.rsc.org/en/Content/ArticleLanding/2016/AN/C5AN01636C>`_


lipids density around protein
-----------------------------

One of the questions one may want to answer when running a Molecular Dynamics (MD) simulation of a protein in a membrane is:
where do lipids spend most of their time?

Given an MD simulation, in a preprocessing step, align all frames around the protein, and save the resulting trajectory in a multi-PDB file.
First, we identify the position of every phosphorus atom (P):

>>> import numpy as np
>>> M = biobox.Molecule()
>>> M.import_pdb("trajectory.pdb")
>>> idx = M.atomselect("*", "*", "P", get_index=True)[1]

We then extract the coordinate of every selected atom at any time in the simulation, ignoring the first 20 frames (equilibration).

>>> crds = M.coordinates[20:, idx]
>>> atoms = np.reshape(crds, (crds.shape[0]*crds.shape[1], 3))

We finally generate a density map of the resulting collection of points, and save it in a DX file.

>>> S = biobox.Structure(atoms)
>>> D = S.get_density()
>>> D.write_dx("density.dx")

.. seealso:: The approach described here was used in `Landreh et al., Integrating mass spectrometry with MD simulations reveals the role of lipids in Na+/H+ antiporters,  Nature Communications, 2017 <http://www.nature.com/articles/ncomms13993>`_


calculating cross-linking distance 
----------------------------------

Cross-linking experiments report on the distance between the side chain of specific amino-acids.
This distance, measured by a cross-linker molecule, is however not a straight line, but a "shortest solvent accessible path". 

To identify in a structure which lysines may be cross-linked, we start loading it and identifying the location of all lysines' NZ atoms:

>>> M = biobox.Molecule()
>>> M.import_pdb("protein.pdb")
>>> idx = M.atomselect("*", "LYS", "NZ", use_resname=True, get_index=True)[1]

To calculate the path distance between all these atoms, we must first define which protein atoms should be used for clash detection.
Here, we select all backbone atoms as well as beta carbon ones. Furthermore, atoms buried in the protein core are also added (with densify=True).
This makes the protein core more "dense", reducing the likelihood that a path will find its way through the protein, instead of around it.

>>> XL = biobox.Xlink(M)
>>> XL.set_clashing_atoms(atoms=["CA", "C", "N", "O", "CB"], densify=True)

We then set up the grid used by the path detection algorithms.
Here, we use a local search, using a cubic moving grid of 18 Angstrom per side.
After this, the distance matrix path detection algorithm can be launched.
We will use a lazy Theta* method, with flexible side chains, and path smoothing as postprocessing.

>>> XL.setup_local_search(maxdist=18)
>>> distance_mat = XL.distance_matrix(idx, method="theta", smooth=True, flexible_sidechain=True)

distance_mat is the distance matrix between all lysines, sorted according to idx.
It will contain -1 for lysine's linking atoms too far to be encompassed by the moving grid, and -2 for failed path detection (e.g. because a linking atom is buried).

.. seealso:: this method is presented and benchmarked in `M. T. Degiacomi et al., Accommodating protein dynamics in the analysis of chemical cross-links, nice journal, 2017 <http://awesome_link.com>`_
