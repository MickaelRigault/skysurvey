"""DES survey classes and DECam footprint utilities.

This module defines the `DES` and `DESWide` survey classes and utility
functions for the DECam footprint and the DES fields.
"""

import pandas
import geopandas
import numpy as np

from ztffields.projection import project_to_radec
from .basesurvey import Survey, GridSurvey


# ============= #
#  Top Level    #
# ============= #
def get_des_footprint(incl_focus=False, coef=(6.53,6.12)):
    """Get the DECam footprint (with or without the 'F' CCDs).

    See https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/characteristics.
    The footprint is north up, east right.

    Parameters
    ----------
    incl_focus : bool, optional
        If True, include the focus CCDs. The default is False.

    coef : tuple, optional
        Coefficients to convert from pixel to degree. The default is
        (6.53, 6.12).

    Returns
    -------
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        The DECam footprint.
    """
    from shapely import geometry
    from shapely.ops import unary_union

    if incl_focus:
        nccds = [2, 
                 8, 10, 10, 12, 12, 14, 
                 14, 12, 12, 10, 10, 8,
                 2
                ]
        offset = (7,7)
    else:
        nccds = [6, 8,
                     10, 12, 12, 14, 
                     14, 12, 12, 10,
                8, 6,
                ]
        offset = (7,6)

    ncenter = 7
    ccds = []
    for i,n_ in enumerate(nccds):
        left = int(n_/2)
        right = n_ -left
        row = np.asarray([[ncenter-left, i+1],[ncenter-left,i], [ncenter+right,i], [ncenter+right,i+1]])
        ccds.append(row)

    ccds = (np.asarray(ccds)-offset)/coef # 2.7 square degree excluding the focus and guiding
        
    return unary_union([geometry.Polygon(g) for g in ccds])


def get_des_field_coordinates(fieldid_name="fieldid"):
    """Get the radec location of the DES shallow (8) and deep (2) fields.

    Parameters
    ----------
    fieldid_name : str, optional
        Name of the fieldid index. If None, 'fieldid' is used. The default is
        'fieldid'.

    Returns
    -------
    pandas.DataFrame
        Dataframe indexed by field name with 'ra' and 'dec' columns.
    """
    if fieldid_name is None:
        fieldid_name = "fieldid"
    
    radec = {'C1': {'dec': -27.11161, 'ra': 54.274292},
             'C2': {'dec': -29.08839, 'ra': 54.274292},
             'C3': {'dec': -28.10000, 'ra': 52.648417},
             'E1': {'dec': -43.00961, 'ra': 7.8744167},
             'E2': {'dec': -43.99800, 'ra': 9.5000000},
             'S1': {'dec':   0.00000, 'ra': 42.820000},
             'S2': {'dec': -0.988389, 'ra': 41.194417},
             'X1': {'dec': -4.929500, 'ra': 34.475708},
             'X2': {'dec': -6.412111, 'ra': 35.664500},
             'X3': {'dec': -4.600000, 'ra': 36.450000}}
        
    data = pandas.DataFrame(radec).T
    data.index.name = fieldid_name
    return data

def get_des_fields(origin=180, incl_focus=False, fieldid_name=None):
    """Get the DES fields as a geopandas.GeoDataFrame.

    Parameters
    ----------
    origin : float, optional
        Offset (in degrees) added to the R.A. coordinates of the fields.
        The default is 180.

    incl_focus : bool, optional
        If True, include the focus CCDs in the footprint. The default is False.

    fieldid_name : str, optional
        Name of the fieldid index. If None, 'fieldid' is used. The default is
        None.

    Returns
    -------
    geopandas.GeoDataFrame
        The DES field geometries, indexed by field name.
    """
    footprint = get_des_footprint(incl_focus=incl_focus)
    radec = get_des_field_coordinates(fieldid_name=fieldid_name)    
    fields = geopandas.GeoDataFrame( geometry=project_to_radec(footprint, radec["ra"]+origin, radec["dec"]),
                                        index=radec.index)
    return fields


# ============= #
#  Classes      #
# ============= #

class DES( GridSurvey ):
    """The DES grid-based survey with predefined fields and DECam footprint.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Observing data. The default is None.

    fields : geopandas.GeoDataFrame, optional
        Field definitions. The default is None.

    footprint : shapely.geometry.Polygon, optional
        Footprint in the sky of the observing camera. The default is None.

    **kwargs
        Ignored.

    Attributes
    ----------
    _DEFAULT_FIELDS : geopandas.GeoDataFrame
        The standard DES field definitions loaded via :func:`get_des_fields`.

    _FOOTPRINT : shapely.geometry.Polygon
        The DECam camera footprint loaded via :func:`get_des_footprint`.
    """

    _DEFAULT_FIELDS = get_des_fields(fieldid_name="FIELD")
    _FOOTPRINT = get_des_footprint()
    
class DESWide( Survey ):
    """The DES wide-field survey with the DECam footprint.

    Parameters
    ----------
    footprint : shapely.geometry.Polygon, optional
        Footprint in the sky of the observing camera. The default is None.

    nside : int, optional
        HEALPix nside parameter. The default is 200.

    data : pandas.DataFrame, optional
        Observing data. The default is None.

    Attributes
    ----------
    _FOOTPRINT : shapely.geometry.Polygon
        The DECam camera footprint loaded via :func:`get_des_footprint`.
    """

    _FOOTPRINT = get_des_footprint()

