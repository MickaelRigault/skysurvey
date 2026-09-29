"""
This module provides a model for host galaxy dust parameters.
"""

import scipy
import numpy as np


class ebv_distrib(scipy.stats.rv_continuous):
    """
    A class that implements a simple exponential probability density function
    to model the E(B-V) distribution for dust extinction for host galaxies.

    See :class:`scipy.stats.rv_continuous` for the full list of parameters.
    """

    def _pdf(self, x):
        """ Probability density function: exp(-x) for x > 0. """
        return np.heaviside(x, 0)*np.exp(-x)

ebv = ebv_distrib(name='ebv',a=0, b=100)

dust_model = {'hostebv': {"func": ebv.rvs, 
                        "kwargs": {"scale":0.17}},
                  
                  'hostr_v': {"func": scipy.stats.norm.rvs, 
                        "kwargs": {"loc":2, "scale":1.4}}
}