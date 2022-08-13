
import numpy as np
from .core import Sampling


__all__ = ["Sampling_radec"]

class Sampling_radec( Sampling ):

    @classmethod
    def random(cls, ra_range=[0,360], dec_range=[-90,90], size=None):
        """ """
        dec_sin_range = np.sin(np.asarray(dec_range)*np.pi/180)
        ra = np.random.uniform(*ra_range, size=size)
        dec = np.arcsin( np.random.uniform(*dec_sin_range, size=size) ) / (np.pi/180)
        return ra, dec
