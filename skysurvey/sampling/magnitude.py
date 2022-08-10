from .core import Sampling
import numpy as np


__all__ = ["Sampling_mag", "Sampling_x0"]

class Sampling_mag( Sampling ):

    @classmethod
    def tripp1998(cls, x1, c,
                    size=None,
                    mabs=-19.3, sigmaint=0.10,
                    alpha=-0.14, beta=3.15):
        """ """
        mabs = np.random.normal(size=size, loc=mabs, scale=sigmaint)
        mobs = mabs + (x1*alpha + c*beta)
        return mobs
    
    
class Sampling_x0( Sampling_mag ):
    """ Converting mag->x0 """
    
    @classmethod
    def draw(cls, model, size=None, **kwargs):
        """ """
        mags = super().draw(model, size=size, **kwargs)
        
        return 10**(-0.4*mags)
