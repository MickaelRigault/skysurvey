import numpy as np
from scipy import stats

def get_hostmass_rvs(size, c=0.90, loc=10.40, scale=0.60):
    """Draw a hostmass distribution from a loggamma function.

    Parameters
    ----------
    size : int
        The number of random variates to draw.
    c : float, optional
        The shape parameter of the loggamma function. The default is 0.90.
    loc : float, optional
        The location parameter of the loggamma function. The default is 10.40.
    scale : float, optional
        The scale parameter of the loggamma function. The default is 0.60.

    Returns
    -------
    ndarray
        The drawn random variates.
    """
    return stats.loggamma.rvs(size=size, c=c, loc=loc, scale=scale)
