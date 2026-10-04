Assemblies
==========

Structures can be arranged in assemblies (class :func:`Assembly <biobox.classes.assembly.Assembly>`). Within an assembly, Structures can be manipulated and their properties assessed either together or individually.

:func:`Assembly <biobox.classes.assembly.Assembly>` has a subclass, :func:`Polyhedron <biobox.classes.polyhedron.Polyhedron>`, providing a methodology to arrange structures on a polyhedral scaffold.
In turn, :func:`Polyhedron <biobox.classes.polyhedron.Polyhedron>` has a :func:`Multimer <biobox.classes.multimer.Multimer>` subclass, handling the case where assembles Structures are instances of the :func:`Molecule <biobox.classes.molecule.Molecule>` class.

Assembly
--------

.. automodule:: biobox.classes.assembly
   :members:


Polyhedron
----------

.. automodule:: biobox.classes.polyhedron
   :members:
   :show-inheritance:

Multimer
--------

.. automodule:: biobox.classes.multimer
   :members:
   :show-inheritance: