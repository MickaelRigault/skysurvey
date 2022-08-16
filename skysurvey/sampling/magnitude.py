from .core import Sampling
import numpy as np


__all__ = ["Sampling_magabs"]

class Sampling_magabs( Sampling ):

    @classmethod
    def tripp1998(cls, x1, c,
                    size=None,
                    mabs=-19.3, sigmaint=0.10,
                    alpha=-0.14, beta=3.15):
        """ """
        mabs = np.random.normal(size=size, loc=mabs, scale=sigmaint)
        mabs_notstandard = mabs + (x1*alpha + c*beta)
        return mabs_notstandard
