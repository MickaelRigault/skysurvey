[![PyPI](https://img.shields.io/pypi/v/skysurvey.svg?style=flat-square)](https://pypi.python.org/pypi/skysurvey)
[![Documentation Status](https://readthedocs.org/projects/skysurvey/badge/?version=latest)](https://skysurvey.readthedocs.io/en/latest/?badge=latest)


# skysurvey 

Simulate transients within the sky

skysurvey relies on sncosmo for bandpass and lightcurve generations. (https://sncosmo.readthedocs.io/en/stable/)

**See documentation on [read the docs](https://skysurvey.readthedocs.io/en/latest/)**

# Install
```bash
pip install skysurvey
```
or 
```bash
git clone https://github.com/MickaelRigault/skysurvey.git
cd skysurvey
pip install .
```

# Quick Start
You need to create a `Target` and a `Survey` to then
simulate how your survey would observe your targets ; 
aka a `DataSet`. 

Here is a quick example:

## Step 1: targets (truth)
```python
import skysurvey
snia = skysurvey.SNeIa()
data = snia.draw(size=50_000, tstart=56_000, tstop=56_100, inplace=True) # see options
data.head(5) # also snia.data
```

## Step 2: Survey (pointing and observing conditions)
```python
import numpy as np
from skysurvey.tools import utils

size = 10_000

# footprint
from shapely import geometry
sq_footprint = geometry.box(-1, -1, +1, +1)

# Observing data
ra, dec = utils.random_radec(size=size, ra_range=[200,250], dec_range=[-20,10])

data = {}
data["ra"] = ra
data["dec"] = dec
data["gain"] = 1
data["zp"] = 30
data["skynoise"] = np.random.normal(size=size, loc=150, scale=20)
data["mjd"] = np.random.uniform(56_000-10, 56_100 + 10, size=size)
data["band"] = np.random.choice(["desg","desr","desi"], size=size)

# Build the survey
survey = skysurvey.Survey.from_pointings(data, footprint=sq_footprint)
survey.data
```

## Step 3: Dataset

And now let's build the dataset. The simulated lightcurves are in
dset.data, the input survey is stored in dset.survey, the input
targets is stored in dset.targets

```python
from skysurvey import dataset
dset = dataset.DataSet.from_targets_and_survey(snia, survey)
dset.data
```

