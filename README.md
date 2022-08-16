# skysurvey
modern version of simsurvey to simulate transients within the sky

_Under active development, things are likely to change_

skysurvey relies on sncosmo for bandpass and lightcurve generations. (https://sncosmo.readthedocs.io/en/stable/)

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
# Say I what a survey split in healpix pixel, observing 1000 fields per day for 4 years
# This is a healpix much-simplified version of ZTF. 
hpsurvey = survey.HealpixSurvey.from_random(nside=9, 
                                     size=365*4*1000, # number of observation 
                                     bands=["ztfg","ztfr","ztfi"], # observed bands
                                     mjd_range=[56000, 56000+365*4], # duration
                                     skynoise_range=[180,210], # typical skynoise
                                     gain_range=6, # typical gain (1 value means always the same)
                                     dec_range=[-30,90]) # limited in declination to the north.
hpsurvey.data.head(5)
```

## Step 3: Dataset
We will use dask for fasten distributed the computation (and memory usage) between available worker.
On you laptop it plays as a multiprocess/multithreading tool, but natively scale on a computer clusters.

If you don't want to, skip the dask part and set `use_dask=False` after. 
Careful `use_dask=True` is default.

See https://www.dask.org/

```python
# run Dask locally
from dask.distributed import Client
client = Client() # check localhost:8787 to see the computation live
```
And now let's build the dataset (computation split by fieldid)
```python
from survey import dataset
dset = dataset.DataSet.from_targets_and_survey(snia, hpsurvey, use_dask=True) # this takes ~1 min on a laptop for ~10000 targets
dset.data
# your survey (here hpsurvey) is stored in dset.survey
# your targets (here snia) is stored in dset.targets
```

