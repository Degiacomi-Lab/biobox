Measures
========

measuring characteristics
-------------------------

.. automodule:: biobox.measures.calculators
   :members:
   :exclude-members: random_string, CCS, sasa_c


measuring distances
-------------------

Helper functions and classes are available to assess points' pairwise interactions between points. These are divided in two categories:

* straight line (i.e. euclidean distances, van der Waals interactions).
* paths (e.g. used to model cross-linking data) 


.. automodule:: biobox.measures.interaction
   :members:

.. automodule:: biobox.measures.path
   :members:
   :exclude-members: PriorityQueue
