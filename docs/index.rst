skysurvey documentation
================

skysurvey_ is a generic package to simulate transients as observed by a
survey. It is a modern implementation of simsurvey_ that aims at
speed-up and simplify the code.



skysurvey
--------
The concept is simple. To simulate transient lightcurves you need three things:
    - a list of **target** properties as given by nature.
    - a **survey** observing logs providing, what has been observed when
      under which condition.
    - a **template** that can convert target properties into photometric points

skysurvey_ is providing that for you and should be able to produce
realistic lightcurves in a few minutes for typical multi year surveys
observing tens of thousands of transients.

The package is using the sncosmo_ for the **template** structure
(``sncosmo.Model``).

The ``Transient`` object is able to generate
realistic objects given a simple configuration dictionary.
Some ``Transient`` have already been implemented for you, such as
``SNIa``. 

The ``Survey`` object handle the observing logs. A pointing is
identified by a ``fieldid`` and each line corresponds to new a
pointing condition. A Survey has a ``fields`` attribute that contains
the survey field id footprint. Some survey have already been
implemented, such as ``ZTF`` for which the field footprint are pre-registered.
A healpix-based survey (``HealpixSurvey``) has also been implemented
where ``fieldid`` corresponds to the healpix ``ipix``.

Finally, the  ``DataSet`` takes a ``target`` and a ``survey`` and
knows how to match target with fieldid and thereby to create
lightcurves given the observing conditions of the survey.

.. toctree::
   :maxdepth: 1
   :caption: Getting Starting

   installation	     
   
.. toctree::
   :maxdepth: 1
   :caption: How to
	      
   quickstart/index
   advanced/index
   
.. toctree::
   :maxdepth: 2
   :caption: API documentation

   skysurvey
   
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. _simsurvey: https://simsurvey.readthedocs.io/en/latest/index.html
.. _skysurvey: https://github.com/MickaelRigault/skysurvey
