Surveys
=======

.. currentmodule:: skysurvey

A *survey* holds what has been observed, when and under which conditions
(``mjd``, ``band``, ``skynoise``, ``gain``, ``zp``), together with where the
camera was pointing. See :doc:`/quickstart/quickstart_survey` for an
introduction.

Generic surveys
---------------

:class:`Survey` accepts any observing pattern (pointings are matched to the sky
with HEALPix), while :class:`GridSurvey` is optimised for field-based surveys
(fields are shapely polygons).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Survey
   GridSurvey
   HealpixSurvey
   PolygonSurvey
   survey.core.BaseSurvey

Survey-specific classes
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ZTF
   LSST
   DES
   DESWide
   SNLS

Footprints and fields
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_footprint
   get_lsst_footprint
   get_des_footprint
   get_des_fields
   get_des_field_coordinates
   get_snls_footprint
   get_snls_field_coordinates
   survey.roman.get_roman_footprint

Observing logs
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   read_opsim
   get_weblogs
   register_snls_bandpasses
   survey.roman.parse_simlib
