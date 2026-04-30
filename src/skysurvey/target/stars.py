"""
This module defines StableTarget and Star, representing time-independent point sources such as stars.
"""

import numpy as np

from .core import Target


class StableTarget( Target ):
    """
    A class to model targets with fixed, time-independent properties.

    Parameters
    ----------
    _KIND : str
        The transient type. Default is ``"stable"``.
    _MODEL : dict
        The model to use. The default is a dictionary with the following
        keys:
        
        - `radec`: The ra and dec of the target.
        - `magobs`: Randomly drawn observed magnitudes of the target, using :meth:`random_magobs`.
    """
    _KIND = "stable"
    _MODEL = dict( radec = {"func":"random",
                                "kwargs":dict(ra_range=[0, 360], dec_range=[-30, 90]),
                                "as":["ra","dec"]},
                    magobs = {"func": "random_magobs",
                                "kwargs": dict(zpmax=22.5)},
                   )
    
    @staticmethod
    def random_magobs(size=None, zpmax=22.5, scale=3, rng=None):
        """Draw random observed magnitudes from an exponential decay distribution.

        Parameters
        ----------
        size : int, optional
            Number of magnitudes to draw. Default is None.

        zpmax : float, optional
            Upper magnitude limit. Default is 22.5.

        scale : float, optional
            Scale parameter of the exponential distribution. Default is 3.

        rng : None, int, or (Bit)Generator, optional
            Seed for the random number generator. Default is None.

        Returns
        -------
        array
            Randomly drawn observed magnitudes.
        """
        rng = np.random.default_rng(rng)
        exp_decay = rng.exponential(scale=scale, size=size)
        return zpmax-exp_decay

    
class Star( StableTarget ):
    """
    A class to model stars, modelled as a stable point source.

    Parameters
    ----------
    _KIND : str
        The transient type. Default is ``"star"``.
    """
    _KIND = "star"