========================
skysurvey
========================

skysurvey_ is a python package made to simulate astronomical targets as they
would be observed by a survey.

To simulate a realistic lightcurves you need two things:

1.  **target** properties as given by nature.
2. **survey** observing data providing what has been observed when
   and under which condition.

Joining these to create:

3. a **dataset**, i.e. simulated data of the targets observed by your survey.


Elements
======
	   
.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Targets
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      ``Target`` are objects as given by natures. You can generate
      realistic targets, building complex parametrisation thanks to
      the modeldag_ backend.
      skysurvey_ provides multi-predefined targets, such as SNeIa, SNII,
      or any sncosmo_ TimeSerie Source. 

   .. grid-item-card:: Survey
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Survey objects handle your observations. It can
      match sky positions with observing logs and provide observing
      statistics.  There are two kinds of
      surveys: ``Survey`` that accept any observing pattern and
      ``GridSurvey`` that are customed for field-based surveys.

   .. grid-item-card:: DataSet
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      ``DataSet`` corresponds the actual data you would have collected
      observing ``target`` (s) with your ``survey``. A dataset is easy
      and fast to load, and it contains analytical and visualisation tools.

.. grid:: 3

    .. grid-item-card:: :material-regular:`star;2em` Make Targets
      :columns: 12 6 6 4
      :link: quickstart/quickstart_target.html

    .. grid-item-card:: :material-regular:`scatter_plot;2em` Build a Survey
      :columns: 12 6 6 4
      :link: quickstart/quickstart_survey.html

    .. grid-item-card:: :material-regular:`timeline;2em`
			Create a DataSet
      :columns: 12 6 6 4
      :link: quickstart/quickstart_survey_target_dataset.html

.. image:: ./gallery/concept_image.png
		   


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

.. image:: ./gallery/lc_example.png



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
   
   


.. _simsurvey: https://simsurvey.readthedocs.io/en/latest/index.html
.. _skysurvey: https://github.com/MickaelRigault/skysurvey
.. _modeldag: https://github.com/MickaelRigault/modeldag
.. _sncosmo: https://sncosmo.readthedocs.io/en/stable/
.. _`see list here`: https://sncosmo.readthedocs.io/en/stable/source-list.html
.. _snana: https://github.com/RickKessler/SNANA
.. _shapely: https://shapely.readthedocs.io/en/stable/manual.html
.. _geopandas: https://geopandas.org/en/stable/gallery/index.html
.. _healpy: https://healpy.readthedocs.io/en/latest/
