"""Time-independent point sources such as stars.

This module defines `StableTarget` and `Star`, representing time-independent
point sources such as stars.
"""

import numpy as np

from .core import Target


class StableTarget( Target ):
    """A class to model targets with fixed, time-independent properties.

    Attributes
    ----------
    _KIND : str, optional
        The target type. The default is 'stable'.

    _MODEL : dict, optional
        The model to use. The default is a dictionary with the following
        keys:

        - `radec`: The ra and dec of the target.
        - `magobs`: Randomly drawn observed magnitudes of the target, using
          :meth:`random_magobs`.
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
            Number of magnitudes to draw. If None, a single value is
            returned. The default is None.

        zpmax : float, optional
            Upper magnitude limit. The default is 22.5.

        scale : float, optional
            Scale parameter of the exponential distribution. The default is 3.

        rng : None, int, or numpy.random.Generator, optional
            Seed for the random number generator. The default is None.

        Returns
        -------
        float or numpy.ndarray
            Randomly drawn observed magnitudes.
        """
        rng = np.random.default_rng(rng)
        exp_decay = rng.exponential(scale=scale, size=size)
        return zpmax-exp_decay

    
class Star( StableTarget ):
    """A class to model stars, modelled as a stable point source.

    Attributes
    ----------
    _KIND : str, optional
        The target type. The default is 'star'.
    """
    _KIND = "star"