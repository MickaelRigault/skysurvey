"""Target template sources."""


# ============== #
#   SNCOSMO      #
# ============== #
import pandas
from sncosmo.models import _SOURCES

SNCOSMO_SOURCES_DF = pandas.DataFrame(_SOURCES.get_loaders_metadata())
def get_sncosmo_sourcenames(of_type=None, startswith=None, endswith=None):
    """Get the list of available sncosmo source names.

    Parameters
    ----------
    of_type : str or list of str, optional
        Source type name (or list of), e.g. 'SN II'. If None, all types
        are considered. The default is None.

    startswith : str, optional
        The source name should start with this (e.g. 'v19'). If None, no
        selection is applied. The default is None.

    endswith : str, optional
        The source name should end with this. If None, no selection is
        applied. The default is None.

    Returns
    -------
    list of str
        List of source names.

    Notes
    -----
    The `startswith` selection is only applied if `endswith` is also given.
    """
    import numpy as np
    
    sources = SNCOSMO_SOURCES_DF.copy()
    if of_type is not None:
        typenames = sources[sources["type"].isin(np.atleast_1d(of_type))]["name"]
    else:
        typenames = sources["name"]
        
    if endswith is not None:
        typenames = typenames[typenames.str.startswith(startswith)]
    
    if endswith is not None:
        typenames = typenames[typenames.str.endswith(endswith)]

    return list(typenames)

