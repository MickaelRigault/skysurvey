

from .core import Sampling
from scipy import stats


__all__ = ["Sampling_x1"]

class Sampling_x1( Sampling ):
    _MIN = -4
    _MAX = 4
    _STEP = 1e-3
    
    @classmethod
    def getpdf_nicolas2021(cls, 
                    xx=None, 
                    mu1=0.33, sigma1=0.64, 
                    mu2=-1.50, sigma2=0.58, a=0.45,
                    redshift=None, fprompt=0.5):
        """ """
        if xx is None:
            xx = cls.get_default_xx()
            
        mode1 = stats.norm.pdf(xx, loc=mu1, scale=sigma1)
        mode2 = stats.norm.pdf(xx, loc=mu2, scale=sigma2)
        if redshift is not None:
            raise NotImplementedError("estimating fprompt from redshift is not implemented.")
            
        pdf = fprompt*mode1 + (1-fprompt)*(a*mode1 + (1-a)*mode2)
        return xx, pdf
    

    
