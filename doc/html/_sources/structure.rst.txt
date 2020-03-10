Single Structures
=================

The main data structure in BiobOx is the :func:`Structure <structure.Structure>` class, which handles collections of 3D points.
Points are stored in a MxNx3 **coordinates array**, where M is the number of alternative points arrangements, and N is the amount of points.

At any moment, one of the loaded points conformations is considered to be the active one (a.k.a. **current**).
Any rototranslation or measuring operation will be performed on the current structure only.
Some methods, e.g. :func:`rmsd <structure.Structure.rmsd>` allow comparing different conformations, independently from which is the current one.

The current conformation in the coordinates array can be changed by calling the :func:`set_current <structure.Structure.set_current>` method (altering the Structure's **current** attribute).
For comfort, the current conformation is accessible in the **points** Nx3 array, where:

>>> self.points = self.coordinates[self.current]

Points attributes are stored in a pandas **dataframe**. For instance, self.data["radius"] contains each point's radius.
A property of the whole point cloud can be stored in a properties dictionary (e.g. self.properties["center"]).

Several :func:`Structure <structure.Structure>` subclasses are available (:func:`Molecule <molecule.Molecule>`, :func:`Ellipsoid <convex.Ellipsoid>`, :func:`Cylinder <convex.Cylinder>`, :func:`Cone <convex.Cone>`, :func:`Sphere <convex.Sphere>`, :func:`Prism <convex.Prism>`, :func:`Density <density.Density>`, see below).


Structure
---------

.. automodule:: structure
   :members:


Molecule
--------

.. automodule:: molecule
   :members:
   :show-inheritance:

Convex Point Clouds
-------------------

The following classes allow generating clouds of points arranged according to specific (convex) geometries.
All these classes are subclass of :func:`Structure <structure.Structure>`.

.. automodule:: convex
   :members:
   :show-inheritance:

Density
-------

.. automodule:: density
   :members:
   :show-inheritance: