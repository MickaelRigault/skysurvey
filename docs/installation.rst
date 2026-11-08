Installation
============

skysurvey requires Python 3.10 or later. Install it from the
`Python Package Index <https://pypi.org/project/skysurvey/>`_:

.. code-block:: bash

   pip install skysurvey

To get the latest development version, install it from GitHub:

.. code-block:: bash

   git clone https://github.com/MickaelRigault/skysurvey.git
   cd skysurvey
   pip install -e .

Dependencies
------------

The core dependencies (numpy, pandas, scipy, astropy, sncosmo, shapely,
geopandas, healpy, dustmaps, extinction, modeldag, ztffields) are installed
automatically. Some features need optional packages:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Needed for
   * - ``matplotlib``
     - All the ``show_*`` plotting methods.
   * - ``iminuit``
     - Lightcurve fitting (:func:`skysurvey.lcfit.fit_salt`, through ``sncosmo.fit_lc``).
   * - ``saltjax``
     - Fast batched SALT2 fitting with JAX (:func:`skysurvey.lcfit.fit_salt_jax`).
   * - ``ztfcosmo``
     - Loading the real ZTF observing logs with :meth:`skysurvey.ZTF.from_logs`.
   * - ``dask``, ``dask-geopandas``
     - Parallel processing of large surveys and datasets.
   * - ``afterglowpy``
     - The ``skysurvey.target.afterglow.Afterglow`` target (GRB afterglows).

.. note::

   Templates, bandpasses and dust maps are downloaded by ``sncosmo`` and
   ``dustmaps`` the first time you use them, and cached afterwards. The first
   run of a notebook can therefore take longer.

Check the installation
----------------------

.. code-block:: python

   import skysurvey
   print(skysurvey.__version__)

   snia = skysurvey.SNeIa.from_draw(size=100)
   snia.data.head()

Build this documentation
------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html   # the pages are written to docs/_build/html

Notebooks are stored with their outputs and are not executed at build time.
