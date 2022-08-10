
from .core import Sampling
from scipy import stats

__all__ = ["Sampling_c"]

class Sampling_c( Sampling ):
    _MIN = -0.3
    _MAX = 1.5
    _STEP = 1e-3
    
    @classmethod
    def getpdf_intrinsic_and_dust(cls, xx=None,
                                  cint=-0.05, sigmaint=0.05, tau=0.1):
        """ exponential decay convolved with and intrinsic gaussian color distribution. """
        from scipy.ndimage import gaussian_filter1d
        # exponential decay center on cint
        if xx is None:
            xx = cls.get_default_xx()
            
        expon = stats.expon.pdf(xx, loc=-0.05, scale=0.1)
        # applying gaussian filtering        
        #  - which require sigmaint in pixel.
        sigmaint_inpix = sigmaint/(xx[1]-xx[0]) # assuming constant step
        pdf = gaussian_filter1d(expon, sigmaint_inpix)
        return xx, pdf
