
from shapely import geometry
import numpy as np

def get_roman_footprint():
    """ 
    Get the Roman footprint.
    quick and dirty that have a 1011arcmin FoV

    Returns
    -------
    shapely.geometry.Polygon
    """
    roman_footprint = geometry.Polygon(
                    np.asarray([[0, 1.], [0.15, 1.],
                                  [0.15, 0.8],[0.3, 0.8],
                                  [0.3, 0.75],[0.70, 0.75],
                                  [0.70, 0.8],[0.85, 0.8],
                                  [0.85, 1.], [1,1.],
                                  [1, 0.25], [0.85, 0.25],
                                  [0.85, 0.1], [0.7, 0.1],
                                  [0.7, 0], [0.3, 0],
                                  [0.3, 0.1], [0.15, 0.1],
                                  [0.15, 0.25], [0., 0.25],
                                  [0,1.]
                                 ])*(1, 0.65)/1.304 # 0.281 deg^2
                )
    return roman_footprint
