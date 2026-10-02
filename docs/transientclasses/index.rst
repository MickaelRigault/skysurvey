Transient catalogue
===================

The transient models shipped with skysurvey. Each page describes the template,
the rate and the parameter model of a class, with references.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Type Ia supernovae
      :link: sne_ia
      :link-type: doc

      :class:`~skysurvey.SNeIa`: SALT2 template with stretch, color and
      standardisation.

   .. grid-item-card:: Core-collapse supernovae
      :link: sne_cc
      :link-type: doc

      :class:`~skysurvey.SNeII`, :class:`~skysurvey.SNeIb`,
      :class:`~skysurvey.SNeIc`, ... from Vincenzi et al. (2019) templates.

   .. grid-item-card:: Kilonovae
      :link: kilonovae
      :link-type: doc

      :class:`~skysurvey.Kilonova`: angle-dependent POSSIS models.

Any other ``sncosmo`` source can be simulated with
:class:`~skysurvey.TSTransient`. To create your own class, see
:doc:`../howto/create_new_transient_class` and :doc:`../advanced/index`.

.. toctree::
   :hidden:
   :maxdepth: 1

   sne_ia
   sne_cc
   kilonovae
