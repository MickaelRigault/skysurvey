"""Run single lightcurve fits and convert sncosmo results into pandas objects."""

import numpy as np
import sncosmo
import pandas


def sncosmo_results_to_dataframe(result, flatten=True):
    """Convert an sncosmo fit result into a pandas DataFrame or Series.

    Parameters
    ----------
    result : sncosmo.utils.Result
        Result object returned by :func:`sncosmo.fit_lc` (dict subclass with
        attribute access). It must provide:

        - ``result.param_names``
        - ``result.vparam_names``
        - ``result.parameters``
        - ``result["errors"]``
        - ``result["covariance"]``

    flatten : bool, optional
        If False, return a DataFrame. If True, return a flattened
        `pandas.Series`. The default is True.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        Fit results: values, errors and covariances of the parameters.
    """
    fitted = np.isin(result.param_names, result.vparam_names)
    df = pandas.DataFrame(np.asarray([result.parameters, fitted]).T, 
                          result.param_names, 
                          columns=["values", "fitted"])
    df.fitted = df.fitted.astype(bool)
    
    # - Error
    df = df.merge(pandas.Series(dict(result["errors"]), name="errors"), 
                      left_index=True, right_index=True, how="outer")
    
    # - Cov
    dcov = pandas.DataFrame(result["covariance"], columns=result.vparam_names, index=result.vparam_names)
    dcov.columns ="cov_"+dcov.columns
    
    # - merged
    data = df.merge(dcov,  left_index=True, right_index=True, how="outer")
    if not flatten:
        return data
    data = pandas.concat([pandas.Series({f"{name}": res_["values"],
                                            f"{name}_err": res_["errors"]} |\
                                            {f"{k}{name}":v for k, v in
                                                res_[res_.index.str.startswith("cov_")].items()
                                            }).dropna()
                              for name, res_ in data.T.items()])
    return data

def sncosmo_fit_single(target_data, target_model, free_param,
                        modelcov=True, keymap={},
                        **kwargs):
    """Fit an sncosmo model to a single lightcurve.

    Parameters
    ----------
    target_data : pandas.DataFrame
        Dataframe containing the lightcurve data. It must contain
        ["time", "band", "flux", "fluxerr", "zp", "zpsys"] (but see
        `keymap`).

    target_model : sncosmo.Model
        The model to fit.

    free_param : list
        Model parameters to vary in the fit.

    modelcov : bool, optional
        Include model covariance when calculating chisq. If True, the fit is
        performed multiple times until convergence. The default is True.

    keymap : dict, optional
        Change the key naming convention. For instance, to use
        ``fluxerr_tot`` for ``fluxerr``, use
        ``keymap={"fluxerr": "fluxerr_tot"}``. The default is {}.

    **kwargs
        Passed to :func:`sncosmo.fit_lc`.

    Returns
    -------
    pandas.Series
        Flattened fit results (see :func:`sncosmo_results_to_dataframe`).
    """
    # lightcurve parameters to enter the fit.
    lc_dict = {key: target_data[keymap.get(key, key)].values
               for key in ["time", "band", "flux", "fluxerr","zp", "zpsys"]
               }
               
    # run fit_lc from sncosmo
    result, fitted_model = sncosmo.fit_lc(lc_dict,
                                          model=target_model,
                                          vparam_names=free_param,  
                                          modelcov=modelcov,
                                          **kwargs
                                          )
    return sncosmo_results_to_dataframe(result)
