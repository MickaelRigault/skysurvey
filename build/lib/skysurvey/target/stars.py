
import numpy as np
from .core import Target




class StableTarget( Target ):
    
    _KIND = "stable"
    _MODEL = dict( radec = {"func":"random",
                                "kwargs":dict(ra_range=[0, 360], dec_range=[-30, 90]),
                                "as":["ra","dec"]},
                    magobs = {"func": "random_magobs",
                                "kwargs": dict(zpmax=22.5)},
                   )
    
    @staticmethod
    def random_magobs(size=None, zpmax=22.5, scale=3):
        """ """
        exp_decay = np.random.exponential(scale=scale, size=size)
        return zpmax-exp_decay

    
class Star( StableTarget ):

    _KIND = "star"
    
    
