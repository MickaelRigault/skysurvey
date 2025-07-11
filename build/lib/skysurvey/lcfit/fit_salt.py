import warnings

import numpy as np
import sncosmo
import pandas

from .sncutils import sncosmo_fit_single

# =============== #
#  Single target  #
# =============== #


def fit_salt(dataset, free_param=['t0', 'x0', 'x1', 'c'],
             modelcov=True, keymap={}, 
             indexes=None, phase_range=[-10, +40], 
             progress_bar=False, client=None,
             as_future=False,
             **kwargs):
    """ fit salt model on given dataset. """
    results = {}
    if indexes is None:
        indexes = dataset.obs_index
        
    if progress_bar:
        from tqdm import tqdm
        indexes = tqdm(indexes)
    
    results = {index_: fit_salt_single(dataset, index_, 
                                        free_param=free_param,
                                        client=client, # uses client.submit within
                                        phase_range=phase_range, 
                                        modelcov=modelcov, keymap=keymap, 
                                        **kwargs)
                for index_ in indexes}

    if client is not None:
        if as_future: # fast output
            return results
        
        # this waits for the end of computation
        results = client.gather(results)
    
    return pandas.DataFrame(results).T.dropna()

def fit_salt_single(dataset, index, 
                    free_param=['t0', 'x0', 'x1', 'c'],
                    client=None, phase_range=[-10, 40], 
                    modelcov=True, keymap={},
                    bounds = {"t0": 3, "x1": 0.4, "c": 0.2},
                    in_scatter = {"t0": .5, "x1": 0.1, "c": 0.05},
                    warn=True,
                    **kwargs):
    """ This is a wrapper of fit_salt_single() that get data and model
    for the skysurvey.dataset.
    """
    target_model, target_data = _dataset_to_model_and_data_(dataset, index,
                                                            phase_range=phase_range)

    # add random noise in initial guess
    target_model.set(**{k: target_model.get(k) + np.random.normal(loc=0, scale=scatter)
                          for k, scatter in in_scatter.items()
                      })
    # create bounds
    bounds = {k: target_model.get(k) + np.array([-bound_, +bound_])
                  for k, bound_ in bounds.items()
             }
    # Failing input
    if len(target_data)==0:
        if warn:
            warnings.warn("no data in the target lightcurves")
        return
        
    if not np.any(target_data["flux"]/target_data["fluxerr"]>=5):
        if warn:        
            warnings.warn("no detection >5 in the target lightcurves")
        return 
    
    # and run the fit for this target.
    prop_to_run = dict(target_data=target_data,
                       target_model=target_model,
                       free_param=free_param,
                       modelcov=modelcov,
                       keymap=keymap,
                       bounds=bounds) | kwargs

    if client is not None:
        return client.submit(sncosmo_fit_single, **prop_to_run)
    else:
        return sncosmo_fit_single(**prop_to_run)    

#
# - Internal shortcut
#
def _dataset_to_model_and_data_(dataset, index, phase_range=None, time_key=None):
    """ """
    salt_keys = ['z', 't0', 'x0', 'x1', 'c']
    
    # get the sncosmo_model from this source
    this_model_ = dataset.targets.get_target_template(index).sncosmo_model
    this_model = sncosmo.Model(this_model_.source)
    if hasattr(this_model_, "effects") and this_model_.effects is not None and len(this_model_.effects)>0:
        # assuming it is dust for now.
        this_model.add_effect(this_model_.effects[0], name="mw", frame="obs")
        salt_keys.append("mwebv")
    
    # set values for current parameters
    this_model.set(**{k:this_model_.get(k) for k in salt_keys} )

    # get the simulated t0 as initial guess
    this_t0 = this_model.get("t0")

    # get lightcurve data.
    #  - phrase_range is cut made here to explicitly
    #    get them in rest-frame phase.
    #    phase_range is also an sncosmo.fit_lc option.
    this_data = dataset.data.xs(index).copy()
    if time_key is None:
        if "time" in this_data.columns:
            pass # ok
        elif "mjd" in this_data.columns:
            this_data = this_data.rename({"mjd": "time"}, axis=1)
        elif "jd" in this_data.columns:
            this_data = this_data.rename({"jd": "time"}, axis=1)
        else:
            raise ValueError("cannot parse time entry from input dataset, provide time_key.")

    # now time is "time", not "mjd" or something else
    
    if phase_range is not None:
        this_redshift = this_model.get("z")
        this_data = this_data[(this_data["time"]-this_t0
                              ).between(phase_range[0] * (1 + this_redshift),
                                     phase_range[1] * (1 + this_redshift))]

    return this_model, this_data

