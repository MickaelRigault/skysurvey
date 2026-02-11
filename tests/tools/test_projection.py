import numpy as np
from shapely import geometry
from skysurvey.tools import utils
from skysurvey.survey.polygon import project_to_radec
from skysurvey.tools.projection import radecmodel_to_skysurface

def test_project_to_radec():
    """ """
    footprint = geometry.Point(0,0).buffer(1)
    ra, dec = utils.random_radec(size=20, ra_range=[200,250], dec_range=[-20,10])
    _ = project_to_radec(footprint, ra, dec)
    assert len(_) == 20



def test_radecmodel_to_skysurface():
    """ """
    
    radec = {"func": utils.random_radec,
             "kwargs": {"dec_range":[0, 90], "ra_range":[0, 360]},
             "as": ["ra","dec"]
            }

    farea = radecmodel_to_skysurface(radec, ntrial=1e4, frac=True)
    assert np.isclose(farea, 0.5, rtol=0.1)

    farea = radecmodel_to_skysurface(radec, ntrial=1e4, frac=False)
    assert np.isclose(farea, 2*np.pi, rtol=0.1) # 4pi/2

    # update for full sky
    radec["kwargs"]= {"dec_range":[-90, 90], "ra_range":[0, 360]}
             
    farea = radecmodel_to_skysurface(radec, ntrial=1e4, frac=True)
    assert np.isclose(farea, 1., rtol=0.1) # 4pi/2    
