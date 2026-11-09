Targets
=======

.. currentmodule:: skysurvey

A *target* is an astrophysical object as given by nature. Its parameters are
drawn from a *model* (a directed acyclic graph of random variables, see
:doc:`/advanced/build_a_new_model`) and, for transients, at a *rate* given per
unit volume and time. See :doc:`/quickstart/quickstart_target` for an
introduction.

Base classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Target
   Transient

Type Ia supernovae
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SNeIa
   SNeIaStretch
   SNeIaColor
   SNeIaMagnitude

Core-collapse supernovae
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SNeII
   SNeIIn
   SNeIIb
   SNeIb
   SNeIc
   SNeIcBL
   VincenziModels
   SnanaModels

Other transients and sources
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Kilonova
   target.stars.StableTarget
   target.stars.Star

Generic time-series transients
------------------------------

Use these to simulate any transient described by a spectral time series (for
instance any `sncosmo source
<https://sncosmo.readthedocs.io/en/stable/source-list.html>`_).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TSTransient
   MultiTemplateTSTransient

Collections of targets
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TargetCollection
   TransientCollection
   TSTransientCollection
   CompositeTransient

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_sncosmo_sourcenames
   get_kilonova_model
   read_possis_file
