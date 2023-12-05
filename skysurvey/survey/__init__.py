from .basesurvey import *
from .ztf import *
from .des import *
from .snls import *
from .lsst import *
#from .healpix import *
#from .polygon import *

# real surveys


def get_footprint(which, **kwargs):
    """ """
    which = which.lower()
    
    try:
        footprint = eval(f"{which}.get_{which}_footprint(**kwargs)")
    except:
        raise ValueError(f"{which} has not get_{which}_footprint() function")

    return footprint
