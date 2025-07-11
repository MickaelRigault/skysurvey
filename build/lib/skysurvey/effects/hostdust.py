import scipy
import numpy as np

__all__ = ["dust_model"]

class ebv_distrib(scipy.stats.rv_continuous):

    "E(B-V) distribution"

    def _pdf(self, x):

        return np.heaviside(x, 0)*np.exp(-x)

ebv = ebv_distrib(name='ebv',a=0, b=100)

dust_model = {'hostebv': {"func": ebv.rvs, 
                        "kwargs": {"scale":0.17}},
                  
                  'hostr_v': {"func": scipy.stats.norm.rvs, 
                        "kwargs": {"loc":2, "scale":1.4}}
}