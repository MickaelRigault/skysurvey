""" module for target template sources """


# ============== #
#   SNCOSMO      #
# ============== #
import pandas
from sncosmo.models import _SOURCES

SNCOSMO_SOURCES_DF = pandas.DataFrame(_SOURCES.get_loaders_metadata())
def get_sncosmo_sourcenames(of_type=None, startswith=None, endswith=None):
    """ get the list of available sncosmo source names

    Parameters
    ----------
    of_type: str, list
        type name (of list of). e.g SN II

    startswith: str
        the source name should start by this (e.g. v19)

    endswith: str
        the source name should end by this

    Returns
    -------
    list
        list of names

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

