Tutorials
=========

These tutorials introduce the three building blocks of skysurvey, one at a
time. Read them in order. If you are in a hurry, start with
:doc:`skysurvey_101`, then see the complete chain on a real survey, up to the
lightcurve fits, in :doc:`simulate_and_fit`.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: 1. Targets
      :link: quickstart_target
      :link-type: doc

      Draw a population of transients: templates, rates and parameter
      models.

   .. grid-item-card:: 2. Surveys
      :link: quickstart_survey
      :link-type: doc

      Describe the observations: footprints, pointings and observing
      conditions.

   .. grid-item-card:: 3. DataSets
      :link: quickstart_survey_target_dataset
      :link-type: doc

      Observe the targets with the survey and explore the simulated
      lightcurves.

Complete examples
-----------------

The full chain, from real survey logs to lightcurve fits:

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Survey logs to lightcurve fit
      :link: simulate_and_fit
      :link-type: doc

      ZTF logs, SNe Ia, lightcurves and SALT2 fits (sncosmo or saltjax),
      step by step and timed.

   .. grid-item-card:: LSST deep field
      :link: lsst_deepfield
      :link-type: doc

      One year of SNe Ia in an LSST deep drilling field (OpSim v5.1, camera
      rotation and CCD gaps), fitted and checked against the truth.

   .. grid-item-card:: Co-added detections
      :link: coadd_detections
      :link-type: doc

      Several visits per night: co-add them per band and night before
      counting detections (``join_how="coadd"``).

.. toctree::
   :hidden:
   :maxdepth: 1

   quickstart_target
   quickstart_survey
   quickstart_survey_target_dataset
   lsst_deepfield
   coadd_detections
