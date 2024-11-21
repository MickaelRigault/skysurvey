import numpy as np

import sncosmo
import pandas


def sncosmo_results_to_dataframe(result, flatten=True):
    """ """
    fitted = np.in1d(result.param_names, result.vparam_names)
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
    """ 
    Parameters
    ----------
    target_data: pandas.DataFrame
        dataframe containing the lightcurve data. It must contain
        "time", "band", "flux", "fluxerr","zp", "zpsys"]
        (but see keymap).

    target_model: sncosmo.Model
        The model to fit.

    free_param: list
        model parameters to vary in the fit. (all if None)

    modelcov: bool
        Include model covariance when calculating chisq. 
        If true, the fit is performed multiple times until convergence.
        
    keymap: dict
        Change the key naming convention. 
        For instance to use fluxerr_tot for fluxerr use:
        keymap = {"fluxerr": "fluxerr_tot"}

    kwargs goes to sncosmo.fit_lc()

    Return
    ------
    dict
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
