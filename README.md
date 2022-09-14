[![PyPI](https://img.shields.io/pypi/v/skysurvey.svg?style=flat-square)](https://pypi.python.org/pypi/skysurvey)
[![Documentation Status](https://readthedocs.org/projects/skysurvey/badge/?version=latest)](https://skysurvey.readthedocs.io/en/latest/?badge=latest)


# skysurvey 

modern version of simsurvey to simulate transients within the sky

_Under active development, things are likely to change_

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
python setup.py install
```

# Quick Start
You need to create a `Target` object (or child of) and a `Survey` object (or child of) and then to simulate how your survey would see your targets. This latter is called a `DataSet`. Here is a quick example:
## Step 1: targets (truth)
```python
from skysurvey import target
snia = target.SNeIa() # create a pre-defined SN Ia target object
data = snia.draw(size=5000) # and draw 5000 of them (you have many options)
data.head(5) # data also stored in snia.data
```

## Step 2: Survey (when you pointed and sky conditions)
```python
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
```

## Step 3: Dataset

And now let's build the dataset (computation split by fieldid)
```python
from survey import dataset
dset = dataset.DataSet.from_targets_and_survey(snia, ztf) # this takes ~30s on a laptop for ~5000 targets
dset.data
# your survey (here ztf) is stored in dset.survey
# your targets (here snia) is stored in dset.targets
```

