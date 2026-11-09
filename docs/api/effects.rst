Effects
=======

.. currentmodule:: skysurvey

An :class:`Effect` modifies the spectral template of a target (e.g. dust
extinction) and comes with the model that draws its parameters. See
:doc:`/howto/add_mwebv`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Effect

Built-in effects
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   effects.milkyway.get_mwebv
   effects.scatter.ColorScatter_G10
   effects.scatter.ColorScatter_C11
