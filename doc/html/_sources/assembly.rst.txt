Assemblies
==========

Structures can be arranged in assemblies (class :func:`Assembly <assembly.Assembly>`). Within an assembly, Structures can be manipulated and their properties assessed either together or individually.

:func:`Assembly <assembly.Assembly>` has a subclass, :func:`Polyhedron <polyhedron.Polyhedron>`, providing a methodology to arrange structures on a polyhedral scaffold.
In turn, :func:`Polyhedron <polyhedron.Polyhedron>` has a :func:`Multimer <multimer.Multimer>` subclass, handling the case where assembles Structures are instances of the :func:`Molecule <molecule.Molecule>` class.

Assembly
--------

.. automodule:: assembly
   :members:


Polyhedron
----------

.. automodule:: polyhedron
   :members:
   :show-inheritance:

Multimer
--------

.. automodule:: multimer
   :members:
   :show-inheritance: