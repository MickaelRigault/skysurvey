========================
skysurvey documentation
========================

skysurvey_ is a generic package to simulate transients as observed by a
survey. It is a modern implementation of simsurvey_ that aims at
speed-up and simplify the code.


Concept
========

The concept is simple, to simulate transient lightcurves you need
three things:

1. a list of **target** properties as given by nature.
2. a **survey** observing data providing what has been observed when under which condition.
3. a **template** that can convert target properties into photometric points

With this logic, skysurvey_ produce realistic lightcurves in a few minutes for multi-years surveys
observing tens of thousands of transients. This constitute a **Dataset**.

Template
-----------

The package is using the sncosmo_ for the **template** structure
(``sncosmo.Model``).


Transient
-----------

**Data as given by nature.**

The ``Transient`` object is able to generate
realistic objects given a simple configuration dictionary.
Some ``Transient`` have already been implemented for you, such as
``SNIa``.

.. seealso::
   Tutorials:
   `quickstart with transient <quickstart/quickstart_target.ipynb>`_ •
   `create a new transient <quickstart/quickstart_target.ipynb>`_

Survey
-----------

**What has been observed when under which condition.**

The ``Survey`` object handle the observing logs. A pointing is
identified by a ``fieldid`` and each line corresponds to new a
pointing condition. A Survey has a ``fields`` attribute that contains
the survey field id footprint. Some survey have already been
implemented, such as ``ZTF`` for which the field footprint are pre-registered.
A healpix-based survey (``HealpixSurvey``) has also been implemented
where ``fieldid`` corresponds to the healpix ``ipix``.

.. seealso::
   Tutorials: 
   `quickstart with survey <quickstart/quickstart_survey.ipynb>`_  •
   `create a new survey <quickstart/quickstart_survey.ipynb>`_ •
   `match survey fields to target coordinates
   <quickstart/quickstart_survey.ipynb>`_ 

     
Dataset
-----------

**Join target and survey to create realisitc lightcurves.**

Finally, the  ``DataSet`` takes a ``target`` and a ``survey`` as input
and knows how to match target with fieldid and thereby how to create
lightcurves given the observing conditions of the survey.

.. seealso:: 
   Tutorials:
   `quickstart with dataset
   <quickstart/quickstart_survey_target_dataset.ipynb>`_  •
   `lightcurve fit <quickstart/quickstart_survey.ipynb>`_

Sharp start
============

You need to create a `Transient` object (or child of) and a `Survey`
object (or child of) and then to simulate how your survey would observe
your targets. This latter is called a `DataSet`.


Step 1: transients
------------------

**Draw the 'truth'**

..  code-block:: python
		 
    from skysurvey import target
    snia = target.SNeIa() # create a pre-defined SN Ia target object
    data = snia.draw(size=5000) # and draw 5000 of them (you have many options)
    data.head(5) # data also stored in snia.data
    # **tip**: you can also directly load and draw using
    #              snia = target.SNeIa.from_draw(size=5000)

    
you can also generate any kind of sncosmo "TimeSerieSource" based
transient (`see list here`_). Say you want the Type-IIb model "v19-2013df".
 
..  code-block:: python
		 
    from skysurvey import target
    snIIb = target.TSTransient.from_draw("v19-2013df", size=5000)


Step 2: survey
-----------------

**Provide what has been observed and when (here randomly drawn)**

..  code-block:: python
		 
    from skysurvey import survey
    # Say I what a ztf-fields survey, observing 1000 fields per day for 4 years
    # Let get the starting date from the data
    starting_date = snia.data["t0"].min()-50 # 50 days before the first target, no need to simulate a survey before that

    # and this is a much-simplified version of ZTF (independent random draws)
    ztf = survey.ZTF.from_random(size=365*4*1000, # number of observation 
                       bands=["ztfg","ztfr","ztfi"], # band to observed
                       mjd_range=[starting_date, starting_date+365*4], # timerange of observation
                       skynoise_range=[10,20], # sky noise
                     )
    ztf.data.head(5)

Step 3: dataset 
------------------

**And get the lightcurve you should have**

..  code-block:: python
		 
    from skysurvey import dataset
    dset = dataset.DataSet.from_targets_and_survey(snia, ztf) # this takes ~30s on a laptop for ~5000 targets
    dset.data
    # your survey (here ztf) is stored in dset.survey
    # your targets (here snia) is stored in dset.targets

   
Documentation
===============

Tutorials
------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Starting

   installation	     
   
.. toctree::
   :maxdepth: 2
   :caption: How to
	      
   quickstart/index
   advanced/index
   
.. toctree::
   :maxdepth: 2
   :caption: API documentation

   skysurvey
   
   
Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. _simsurvey: https://simsurvey.readthedocs.io/en/latest/index.html
.. _skysurvey: https://github.com/MickaelRigault/skysurvey
.. _sncosmo: https://sncosmo.readthedocs.io/en/stable/
.. _`see list here`: https://sncosmo.readthedocs.io/en/stable/source-list.html
