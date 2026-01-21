import warnings

from .basesurvey import * # noqa: F403
from .ztf import * # noqa: F403
from .des import * # noqa: F403
from .snls import * # noqa: F403
from .lsst import * # noqa: F403

# shortcut
def get_footprint(which, **kwargs):
    """ 
    Get the footprint of a given survey.

    Parameters
    ----------
    which: str
        name of the survey (e.g. ztf, des, lsst).

    **kwargs goes to the get_{which}_footprint function.

    Returns
    -------
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon
    """
    which = which.lower()
    
    try:
        footprint = eval(f"{which}.get_{which}_footprint(**kwargs)")
    except Exception as e:
        warnings.warn(e)
        raise ValueError(f"{which} has no get_{which}_footprint() function")

    return footprint

#Andrade addition
from .lsst_comcam import from_dp1_parquet
