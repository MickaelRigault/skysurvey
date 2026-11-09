DataSet
=======

.. currentmodule:: skysurvey

A :class:`DataSet` is what your survey would have observed: the simulated
lightcurves of the targets, together with the input targets and survey. See
:doc:`/quickstart/quickstart_survey_target_dataset` for an introduction.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DataSet

Lightcurve simulation
---------------------

Lower-level functions used by :meth:`DataSet.from_targets_and_survey`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   lightcurves.get_obsdata
   lightcurves.realize_lightcurves
