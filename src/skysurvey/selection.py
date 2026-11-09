import pandas
import numpy as np

def draw_from_selection(data, func, key, size=None, allow_duplicate=False, 
                        func_kw={}, **kwargs):
    """Draw samples from a dataset based on a selection probability function.

    This function applies a user-provided function to a specified key in the
    dataset to compute selection probabilities, then draws samples using these
    probabilities.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset. Will not be modified.

    func : callable
        Function to compute selection probabilities. Must accept the column
        specified by `key` and any additional keyword arguments in `func_kw`.

    key : str
        Column name in `data` to pass to `func`.

    size : int, optional
        Number of samples to draw. If None, this is estimated from the sum of
        the selection probabilities. The default is None.

    allow_duplicate : bool, optional
        If True, allows duplicate samples. If False, ensures unique entries.
        The default is False.

    func_kw : dict, optional
        Additional keyword arguments to pass to `func`. The default is {}.

    **kwargs
        Additional keyword arguments passed to :func:`draw_selected_data`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the selected samples.

    See Also
    --------
    draw_selected_data : Draw samples based on precomputed probabilities.
    """
    data = data.copy() # do not affect the data
    data["pobs"] = func(data[key], **func_kw)
    return draw_selected_data(data, pobs_key="pobs", 
                              size=size, allow_duplicate=allow_duplicate,
                              **kwargs)

def draw_selected_data(data, pobs_key="pobs", size=None, allow_duplicate=False, 
                       redshift_key="z", zstep=0.01,
                        *kwargs):
    """Draw samples from a dataset based on precomputed selection probabilities.

    This function selects samples either with or without replacement.
    If `allow_duplicate` is False and `size` is None, this bins data by
    redshift and samples per redshift bin given `pobs_key`.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset. Must contain a column with selection probabilities
        (specified by `pobs_key`).

    pobs_key : str, optional
        Column name in `data` containing the selection probabilities.
        The default is 'pobs'.

    size : int, optional
        Number of samples to draw. If None, this is estimated from the sum of
        the probabilities (`pobs_key`). The default is None.

    allow_duplicate : bool, optional
        If True, allows duplicate samples. If False, ensures unique entries.
        The default is False.

    redshift_key : str, optional
        Column name in `data` containing redshift values. Used for binning if
        `allow_duplicate` is False and `size` is None. The default is 'z'.

    zstep : float, optional
        Step size for redshift binning. Only used if `allow_duplicate` is
        False and `size` is None. The default is 0.01.

    *kwargs
        Additional positional arguments (unused in this function).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the selected samples.

    Notes
    -----
    - If `allow_duplicate` is False and `size` is None, the function bins the
      data by redshift and samples proportionally within each bin to ensure a
      representative sample.
    - Otherwise, the function samples `size` entries weighted by `pobs_key`,
      with replacement if `allow_duplicate` is True.
    """
    if allow_duplicate or size is not None:
        if size is None:
            size = int( data[pobs_key].sum() )
            
        # replace=True means duplicate allowed.
        selected_data = data.sample(size, weights=pobs_key, replace=allow_duplicate)
        
    else:
        data = data.copy()
        
        # do that binned by redshift.
        zrange = [data[redshift_key].min(), data[redshift_key].max()]
        zbins = np.arange(*zrange, zstep)
        data["zbins"] = pandas.cut(data["z"], zbins)

        # mean fraction of target observd in the given redshift bin
        grouped = data.groupby("zbins", observed=True)
        fobs = grouped[pobs_key].mean()
        
        # build selected one zbins at the time.
        sampled = [data.loc[index].sample(frac=fobs[name]) 
                   for name, index in grouped.groups.items()]
        selected_data = pandas.concat(sampled)

    return selected_data
