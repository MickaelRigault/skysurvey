import numpy as np
from scipy import stats

def get_hostmass_rvs(size, c=0.90, loc=10.40, scale=0.60):
    """ draw a hostmass distribution from a loggamma function.

    """
    return stats.loggamma.rvs(size=size, c=c, loc=loc, scale=scale)
