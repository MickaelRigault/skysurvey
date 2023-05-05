========================
skysurvey
========================

skysurvey_ is a python package made to simulate transients as they
would be observed by a survey.
It is a modern implementation of simsurvey_ that speeds-up and
simplifies the code. It also aim at replacing the simulation part of
the snana_ package.


Concept
========

To simulate transient lightcurves you need three things:

1.  **target** properties as given by nature.
2. a **survey** observing data providing what has been observed when
   and under which condition.
3. a **template** that can convert target properties into photometric points

Following this logic, skysurvey_ produces realistic lightcurves in a few minutes for multi-years surveys
observing tens of thousands of transients. In skysurvey_ the
simulated lightcurve data are stored in a **Dataset**.

Sharp start
============

You need to create a `Transient` object, a `Survey`
object and then to simulate how your survey would observe
your targets. This latter is called a `DataSet`.


Step 1: transients
------------------

**Draw the 'truth'**

..  code-block:: python
		 
    from skysurvey import target
    # let's use pre-built transients e.g. SNeIa, SNeII, SNeIb etc...
    snia = target.SNeIa() # create a pre-defined SN Ia target object
    data = snia.draw(size=5000) # and draw 5000 of them
    data.head(5) # data also stored in snia.data
    # **tip**: you can also directly load and draw using
    #              snia = target.SNeIa.from_draw(size=5000)
    


Step 2: survey
-----------------

**provide what has been observed when**

..  code-block:: python
		 
    from skysurvey import survey
    # Say I what a ztf-fields survey, observing 1000 fields per day for 4 years
    # Let get the starting date from the data

    # 50 days before the first target, no need to simulate a survey before that
    starting_date = snia.data["t0"].min()-50

    # and this is a much-simplified version of ZTF (independent random draws)
    ztf = survey.ZTF.from_random(size=365*4*1000, # number of observation 
                       bands=["ztfg","ztfr","ztfi"], # band to observed
                       mjd_range=[starting_date, starting_date+365*4],
		       # time range
                       skynoise_range=[10,20], # sky noise
                     )
    ztf.data.head(5)

Step 3: dataset 
------------------

**and get the lightcurve you should observe**

..  code-block:: python
		 
    from skysurvey import dataset
    # the following takes ~30s on a laptop for ~5000 targets
    dset = dataset.DataSet.from_targets_and_survey(snia, ztf) 
    dset.data
    # your survey (here ztf) is stored in dset.survey
    # your targets (here snia) is stored in dset.targets

   
Definitions
=======

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
``SNIa``, ``SNII`` or ``SNIc-BL`` . (see:  `quickstart with transient
<quickstart/quickstart_target.ipynb>`_ •
`built-in transient <builtin_transients.ipynb>`_ •
`create a new transient <quickstart/quickstart_target.ipynb>`_)

Survey
-----------

**What has been observed when under which condition.**

The ``Survey`` object handle the observing logs. A pointing is
identified by a ``fieldid`` and each line corresponds to new a
pointing condition. A Survey has a ``fields`` attribute that contains
the survey field id footprint. Some survey have already been
implemented, such as ``ZTF`` for which the field footprint are pre-registered.
A healpix-based survey (``HealpixSurvey``) has also been implemented
where ``fieldid`` corresponds to the healpix ``ipix``. (See
`quickstart with survey <quickstart/quickstart_survey.ipynb>`_  •
`create a new survey <quickstart/quickstart_survey.ipynb>`_ • `match
survey fields to target coordinates
<quickstart/quickstart_survey.ipynb>`_ ) 

     
Dataset
-----------

**Join target and survey to create realisitc lightcurves.**

Finally, the  ``DataSet`` takes a ``target`` and a ``survey`` as input
and knows how to match target with fieldid and thereby how to create
lightcurves given the observing conditions of the survey. (See
`quickstart with dataset
<quickstart/quickstart_survey_target_dataset.ipynb>`_  • 
`lightcurve fit <quickstart/quickstart_survey.ipynb>`_)


Tutorials
======

.. toctree::
   :maxdepth: 1
   :caption: Getting Starting

   installation
   builtin_transients
   combining_transients
   
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
============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. _simsurvey: https://simsurvey.readthedocs.io/en/latest/index.html
.. _skysurvey: https://github.com/MickaelRigault/skysurvey
.. _sncosmo: https://sncosmo.readthedocs.io/en/stable/
.. _`see list here`: https://sncosmo.readthedocs.io/en/stable/source-list.html
.. _snana: https://github.com/RickKessler/SNANA
