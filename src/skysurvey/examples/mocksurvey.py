"""
This module provides utility functions to generate mock surveys and observation logs from given parameters
for testing and demonstration purposes.
"""

import pandas
import numpy as np

from .. import Survey, GridSurvey
from ..tools import utils


# Matching coordinates in survey

def get_mocklogs(size = 10_000,
                    mjd_range = [58_900, 58_930],
                    skynoise = {"loc": 200, "scale":20},
                    bands = ["desg","desr","desi"],
                    zp = 30, gain=1, rng=None):
    """ Generate mock observation logs with random observing conditions.
    
    Parameters
    ----------
    size : int, optional
        Number of observations to generate. Default is 10_000.

    mjd_range : 2-element list, optional
        Time range [mjd_min, mjd_max] from which observation times are uniformly drawn.
        Default is [58_900, 58_930].

    skynoise : dict, optional
        Parameters for the Gaussian sky noise distribution, passed to `numpy.random.normal`
        as ``loc`` (mean) and ``scale`` (std). Default is {"loc": 200, "scale": 20}.

    bands : list of str, optional
        List of photometric bands to randomly assign to observations.
        Default is ["desg", "desr", "desi"] (DES's bands).

    zp : float, optional
        Zero point. Default is 30.
        
    gain : float, optional
        Detector gain. Default is 1.

    rng : None, int, or (Bit)Generator, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    pandas.DataFrame
        Mock observation log with columns: gain, zp, skynoise, mjd, band.
    
    """
    rng = np.random.default_rng(rng)
    
    data = {}
    data["gain"] = gain
    data["zp"] = zp
    data["skynoise"] = rng.normal(size=size, **skynoise)
    data["mjd"] = rng.uniform(*mjd_range, size=size)
    data["band"] = rng.choice(bands, size=size)

    data = pandas.DataFrame.from_dict(data)
    return data

def get_mock_survey(size=10_000, footprint = None,
                       nside=200,
                       ra_range = [200,250], dec_range=[-20,10],
                       **kwargs):
    """ Generate a mock Survey with random pointings over a given sky area.

    Parameters
    ----------
    size : int, optional
        Number of observations to generate. Default is 10_000.

    footprint : shapely.geometry, optional
        Camera footprint. If None, a circle of radius 2 degrees centered at (0,0)
        is used. Default is None.

    nside : int, optional
        HEALPix resolution parameter. Default is 200.

    ra_range : 2-element list, optional
        Right ascension range [min, max] in degrees for random pointings.
        Default is [200, 250].

    dec_range : 2-element list, optional
        Declination range [min, max] in degrees for random pointings.
        Default is [-20, 10].

    **kwargs
        Additional arguments passed to `get_mocklogs`.

    Returns
    -------
    Survey
        A Survey instance built from the randomly generated pointings.
    """
    # footprint
    if footprint is None:
        from shapely import geometry
        footprint = geometry.Point(0,0).buffer(2)

    data = get_mocklogs(size=size, **kwargs)
    ra, dec = utils.random_radec(size=size, ra_range=ra_range, dec_range=dec_range)
    data["ra"] = ra
    data["dec"] = dec
    # observing strategy

    return Survey.from_pointings(data, footprint=footprint, nside=nside)


def get_mock_gridsurvey(size=10_000, footprint = None, radec=None,
                        rng=None,
                       **kwargs):
    """ Get a default GridSurvey randomly drawn from the given parameters.

    Parameters
    ----------
    size : int, optional
        Number of observations to generate. Default is 10_000.

    footprint : shapely.geometry, optional
        Camera footprint. If None, a circle of radius 2 degrees centered at (0,0)
        is used. Default is None.

    radec : dict, optional
        Dictionary of field positions with the format
        ``{fieldid: {"ra": float, "dec": float}}``.
        If None, a default set of 5 DES-like fields is used. Default is None.

    rng : None, int, or (Bit)Generator, optional
        Seed for the random number generator. Default is None.

    **kwargs
        Additional arguments passed to `get_mocklogs`.

    Returns
    -------
    GridSurvey
        A GridSurvey instance built from the randomly generated pointings.
    """
    # footprint
    if footprint is None:
        from shapely import geometry
        footprint = geometry.Point(0,0).buffer(2)

    # DES fields
    if radec is None:
        radec = {'C1': {'dec': -27.11161, 'ra': 54.274292+180},
                 'C2': {'dec': -29.08839, 'ra': 54.274292+180},
                 'C3': {'dec': -28.10000, 'ra': 52.648417+180},
                 'E1': {'dec': -43.00961, 'ra': 7.8744167+180},
                 'E2': {'dec': -43.99800, 'ra': 9.5000000+180}}

    rng = np.random.default_rng(rng)
    data = get_mocklogs(size=size, rng=rng, **kwargs)
    data["fieldid"] = rng.choice(list(radec.keys()), size=len(data))

    return GridSurvey.from_pointings(data, radec, footprint=footprint)
