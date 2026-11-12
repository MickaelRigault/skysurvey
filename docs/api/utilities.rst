Utilities
=========

.. currentmodule:: skysurvey

Rates
-----

Volumetric rates and redshift distributions (``skysurvey.target.rates``).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   target.rates.get_rate
   target.rates.get_ntargets
   target.rates.get_ntargets_per_shell
   target.rates.draw_redshift

Lightcurve fitting
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   lcfit.fit_salt
   lcfit.fit_salt.fit_salt_single
   lcfit.fit_salt_jax

Selection functions
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   selection.draw_from_selection
   selection.draw_selected_data

Sky coordinates and geometry
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tools.utils.random_radec
   tools.utils.random_radecz_skymap
   tools.utils.parse_skyarea
   tools.utils.surface_of_skyarea
   tools.projection.project_to_radec
   tools.projection.camera_to_sky
   tools.projection.sky_to_camera
   tools.projection.skyarea_to_skysurface
   tools.projection.spatialjoin_radec_to_fields

Noise and statistics
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tools.utils.apply_gaussian_noise
   tools.utils.get_skynoise_from_maglimit
   tools.utils.build_covariance
   tools.stats.skewed_gaussian_pdf
   tools.utils.random_uniform
   tools.utils.random_normal

Mock surveys
------------

Small synthetic surveys, handy for tests and tutorials.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   examples.mocksurvey.get_mocklogs
   examples.mocksurvey.get_mock_survey
   examples.mocksurvey.get_mock_gridsurvey

Configuration
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   config.get_band_color
