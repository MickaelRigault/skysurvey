Templates
=========

.. currentmodule:: skysurvey

Templates wrap ``sncosmo`` models and give targets their spectral evolution:
they compute fluxes, magnitudes and lightcurves.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Template
   template.GridTemplate
   template.TemplateCollection
   template.get_sncosmo_model
   source.angular.AngularTimeSeriesSource

Blackbody transients
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tools.blackbody.get_blackbody_transient_source
   tools.blackbody.get_blackbody_transient_flux
   tools.blackbody.blackbody_nu
   tools.blackbody.blackbody_lambda
