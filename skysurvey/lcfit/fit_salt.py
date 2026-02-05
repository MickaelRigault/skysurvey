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
    """ 
    Fit a salt model on a given dataset. 
    
    Parameters
    ----------
    dataset: skysurvey.dataset.Dataset
        Dataset containing targets and their lightcurves.

    free_param: list
        Model parameters to vary in the fit. Default is ['t0', 'x0', 'x1', 'c'].

    modelcov: bool
        Include model covariance when calculating chisq. 
        If True, the fit is performed multiple times until convergence. Default is True.
        
    keymap: dict
        Change the key naming convention for lightcurve columns.

    indexes: iterable or None
        Subset of target indices to fit. If None, uses dataset.obs_index. Default is None.
    
    phase_range: list, None, optional
        Rest-frame phase range to be used for simulating 
        the lightcurves. If None, no cut is applied on time
        range for the logs. Default is [-10, +40].

    progress_bar: bool
        If True, display a progress bar over the target indices. Default is False.
    
    client: dask.distributed.Client or None
        If provided, submit fits to the Dask client for parallel execution. Default is None.
    
    as_future: bool
        If True and a Dask client is provided, return a dictionary of
        futures instead of waiting for completion. Default is False.

    **kwargs: 
        Additional keyword arguments passed to fit_salt_single.

    Return
    ------
    pandas.DataFrame or dict
        If client is None or as_future is False, returns a DataFrame of 
        flattened salt fit results. If client is provided and as_future is True,
        returns a dict.

    """

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
    """
    This is a wrapper of sncosmo_fit_single() that get data and model
    for a skysurvey.dataset.Dataset target.

    Parameters
    ----------
    dataset: skysurvey.dataset.Dataset
        Dataset containing the target and its lightcurves.

    index: hashable
        Target index identifying which lightcurve to fit.

    free_param: list
        Model parameters to vary in the fit. Default is ['t0', 'x0', 'x1', 'c'].
        
    client: dask.distributed.Client or None
        If provided, submit the fit to the Dask client. Default is None.

    phase_range: list, None, optional
        Rest-frame phase range to be used for simulating 
        the lightcurves. If None, no cut is applied on time
        range for the logs. Default is [-10, +40].

    modelcov: bool
        Include model covariance when calculating chisq. 
        If True, the fit is performed multiple times until convergence. Default is True.
        
    keymap: dict
        Change the key naming convention for lightcurve columns.

    bounds: dict
        Half-width bounds around the initial parameter values. Default is {"t0": 3, "x1": 0.4, "c": 0.2}.

    in_scatter: dict
        Gaussian scatter added to initial parameter guesses. Default is {"t0": .5, "x1": 0.1, "c": 0.05}.

    warn: bool
        If True, emit warnings when rejecting a target. Default is True.

    **kwargs:
        Additional keyword arguments passed to sncutils.sncosmo_fit_single.

    Return
    ------
    pandas.Series or dask.distributed.Future or None
        Flattened salt fit results for the target, a Dask future if
        client is provided, or None if the target is rejected.
    """
    target_model, target_data = _dataset_to_model_and_data_(dataset, index,
                                                            phase_range=phase_range)

    # add random noise in initial guess
    rng = np.random.default_rng()
    target_model.set(**{k: target_model.get(k) + rng.normal(loc=0, scale=scatter)
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
    """ 
    Extract a sncosmo model and lightcurve data for a skysurvey.dataset.Dataset target. 

    Parameters
    ----------
    dataset: skysurvey.dataset.Dataset
        Dataset containing the target and its lightcurve.

    index: hashable
        Target index identifying which lightcurve to fit.

    phase_range: list, None, optional
        Rest-frame phase range to be used for simulating 
        the lightcurves. If None, no cut is applied on time
        range for the logs. Default is None.

    time_key: str or None
        Column name to use as the time axis. If None, attempts to infer
        from "time", "mjd", or "jd". Default is None.

    Return
    ------
    sncosmo.Model, pandas.DataFrame
        salt model and lightcurve data for the target.
    """
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

