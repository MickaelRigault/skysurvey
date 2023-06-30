import numpy as np
from .survey import Survey


def get_lsst_footprint():
    """ A (3 5 5 5 3) ccd structure centered on 0 with a 9.6 deg2 area """
    from shapely import geometry
    lowleft = 0
    upright = 5
    corner_x = (1/5) * upright
    corner_y = (1/5) * upright

    footprint = np.asarray(
                [[corner_x, lowleft],
                 [upright-corner_x, lowleft],
                 [upright-corner_x, corner_y],
                 [upright, corner_y],
                 [upright, upright-corner_y],
                 [upright-corner_x, upright-corner_y],
                 [upright-corner_x, upright],
                 [corner_x, upright],
                 [corner_x, upright-corner_y],
                 [lowleft, upright-corner_y],
                 [lowleft, corner_y],
                 [corner_x, corner_y]
                ]) - upright/2.

    return geometry.Polygon(footprint * 0.675)
    



class LSST( Survey ):
    _FOOTPRINT = get_lsst_footprint()
