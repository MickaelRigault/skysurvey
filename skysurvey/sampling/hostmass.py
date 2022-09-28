from .core import Sampling
from scipy import stats

__all__ = ["Sampling_host_mass"]

class Sampling_host_mass( Sampling ):
    _MIN = 6
    _MAX = 13
    _STEP = 1e-3
    
    @classmethod
    def getpdf_asymetric_gaussian(cls, xx=None,
                                  ksi=10, omega=1, alph=-3):
        """ asymetric gaussian """
        from scipy.ndimage import gaussian_filter1d
        # exponential decay center on cint
        if xx is None:
            xx = cls.get_default_xx()
         
        pdf = (2/omega)*stats.norm.pdf(xx, ksi, omega)*stats.norm.cdf(alph*(xx-ksi)/omega, 0, 1)
        return xx, pdf