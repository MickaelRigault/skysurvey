========================
skysurvey
========================

skysurvey_ is a python package made to simulate transients as they
would be observed by a survey.
It is a modern implementation of simsurvey_ that speeds-up and
simplifies the code. It also aim at replacing the simulation part of
the snana_ package, while simplifying the user experience.


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
		 
    import skysurvey
    snia = skysurvey.SNeIa()
    data = snia.draw(size=50_000, tstart=56_000, tstop=56_100) # see options
    data.head(5) # also snia.data
    # **tip**: you can also directly load and draw using
    #              snia = target.SNeIa.from_draw(size=50_000)
    


Step 2: survey
-----------------

**provide what has been observed when**

..  code-block:: python
		 
    import numpy as np
    from skysurvey.tools import utils

    size = 10_000 # number of pointings

    # camera footprint in the sky, here a 2deg large square field.
    from shapely import geometry
    sq_footprint = geometry.box(-1, -1, +1, +1) 

    # Observing data
    ra, dec = utils.random_radec(size=size, ra_range=[200,250], dec_range=[-20,10])

    data = {}
    data["ra"] = ra
    data["dec"] = dec
    data["gain"] = 1
    data["zp"] = 30
    data["skynoise"] = np.random.normal(size=size, loc=300, scale=30)
    data["mjd"] = np.random.uniform(56_000-10, 56_100 + 10, size=size)
    data["band"] = np.random.choice(["desg","desr","desi"], size=size)

    # Build the survey
    survey = skysurvey.Survey.from_pointings(data, footprint=sq_footprint)
    survey.show()
    

Step 3: dataset 
------------------

**and get the lightcurve you should observe**
i.e., the dataset. The simulated lightcurves are in
dset.data, the input survey is stored in dset.survey, the input
targets is stored in dset.targets. This runs a ~10s.

..  code-block:: python
		 
    from skysurvey import dataset
    dset = dataset.DataSet.from_targets_and_survey(snia, survey)
    dset.data



Definitions
=======

Template
-----------

The package is pimilarly using the sncosmo_ for the **template** structure
(``sncosmo.Model``). The ``skysurvey.Template`` object does the
interface between sncosmo_ and skysurvey_ objects. It is unlikely that
you will need to interact directly with a ``skysurvey.Template`` but
rather with ``Target``.


Target
-----

**Data as given by nature.**

The ``Target`` object is able to generate
realistic objects given a simple configuration dictionary that connect
togehter their properties. This simple dictionary, called a ``model`` enable the easily generation of a any
complex inter-dependency of the Target parameters.
See the modeldag_ package for details.

Usual ``Target`` objects have been implemented for you, such as
``SNIa``, ``SNII``, ``SNIc-BL`` etc. and ``TSTransient`` that can
accept any sncosmo_ TimeSerie source (see *how-to*).

Survey
-----------

**What has been observed when under which condition.**

A ``Survey`` object handle the observing logs. There are two kinds of
*Survey* in skysurvey: ``Survey`` that accepts any observing pattern
and ``GridSurvey`` that handles surveys following pre-define pointing
parterns (such as poiting grid like ZTF, or deep fields like DES)
adding some usage simplifications, memory optimisation  and speed-up.

In both case, a pointing is identified by a **fieldid** and each line
of ``survey.data`` corresponds to new a pointing condition.

``(Grid)Survey`` has a ``fields`` attribute that contains
the survey fieldid footprint. To handle coordinates to observing
history association, ``Survey`` uses healpy_ while ``GridSurvey`` is
based on geopandas_ and shapely_ (so no pixel approximations).


Some survey have already been
implemented, such as ``ZTF`` (GridSurvey), ``DES`` (GridSurvey | 10
deep-fields) and ``LSST`` (Survey). That basically means that there
footprint have been pre-defined. 

     
Dataset
-----------

**Join target and survey to create realisitc lightcurves.**

The  ``DataSet`` takes a ``target`` and a ``survey`` as input
and knows how to match target with fieldid and thereby how to create
lightcurves given the observing conditions of the survey. 


Tutorials
======

.. toctree::
   :maxdepth: 1
   :caption: Getting Starting

   installation

.. toctree::
   :maxdepth: 2
   :caption: How to
	      
   quickstart/index
   howto/index   
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
.. _modeldag: https://github.com/MickaelRigault/modeldag
.. _sncosmo: https://sncosmo.readthedocs.io/en/stable/
.. _`see list here`: https://sncosmo.readthedocs.io/en/stable/source-list.html
.. _snana: https://github.com/RickKessler/SNANA
.. _shapely: https://shapely.readthedocs.io/en/stable/manual.html
.. _geopandas: https://geopandas.org/en/stable/gallery/index.html
.. _healpy: https://healpy.readthedocs.io/en/latest/
