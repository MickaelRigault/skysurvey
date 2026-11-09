.. _api:

API reference
=============

This is the reference of the public ``skysurvey`` API, organised by concept.
Everything listed under ``skysurvey.`` is importable directly from the top-level
package (e.g. ``skysurvey.SNeIa``).

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: :octicon:`star` Targets
      :link: targets
      :link-type: doc

      Transients and other astrophysical sources: SNe Ia, core-collapse SNe,
      kilonovae, generic time series and collections.

   .. grid-item-card:: :octicon:`telescope` Surveys
      :link: surveys
      :link-type: doc

      Observing logs and footprints: generic, field-based and HEALPix surveys
      plus ready-to-use ZTF, LSST, DES and SNLS.

   .. grid-item-card:: :octicon:`graph` DataSet
      :link: dataset
      :link-type: doc

      Targets observed by a survey: simulated lightcurves and analysis tools.

   .. grid-item-card:: :octicon:`flame` Effects
      :link: effects
      :link-type: doc

      Propagation effects added to target templates (Milky Way and host dust,
      color scatter).

   .. grid-item-card:: :octicon:`pulse` Templates
      :link: templates
      :link-type: doc

      Wrappers around ``sncosmo`` models used to compute fluxes, magnitudes
      and lightcurves.

   .. grid-item-card:: :octicon:`tools` Utilities
      :link: utilities
      :link-type: doc

      Rates, lightcurve fitting, selection functions, sky geometry and mock
      surveys.

.. toctree::
   :hidden:
   :maxdepth: 2

   targets
   surveys
   dataset
   effects
   templates
   utilities
