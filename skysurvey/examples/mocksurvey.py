

import pandas
import numpy as np

from .. import Survey, GridSurvey
from ..tools import utils


__all__ = ["get_mock_survey", "get_mock_gridsurvey"]


# Matching coordinates in survey

def get_mocklogs(size = 10_000,
                    mjd_range = [58_900, 58_930],
                    skynoise = {"loc": 200, "scale":20},
                    bands = ["desg","desr","desi"],
                    zp = 30, gain=1, rng=None):
    """ """
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
    """ get a default Survey randomly drawn from the given parameters

    
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
    """ get a default Survey randomly drawn from the given parameters

    
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
