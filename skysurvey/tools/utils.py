import numpy as np
from shapely import geometry


def random_radec(size=None, skyarea=None,
                ra_range=[0,360], dec_range=[-90,90]):
    """ draw the sky positions

    Parameters
    ----------
    size: int, None
        number of draw

    ra_range: 2d-array
        = ignored if skyarea given =    
        right-accension boundaries (min, max)

    dec_range: 2d-array
        = ignored if skyarea given =
        declination boundaries
            
    skyarea: shapely.geometry.(Multi)Polyon
        area to consider. This will overwrite 
        if skyarea is given, ra_range, dec_range is ignored.
    Returns
    -------
    2d-array
        list of ra, list of dec.
    """
    # => MultiPolygon    
    if type(skyarea) is geometry.MultiPolygon:
        radecs = [random_radec(size=size, 
                               ra_range=ra_range,
                               dec_range=dec_range,                               
                               skyarea=skyarea_) 
                  for skyarea_ in skyarea.geoms]
        
        # at this stage they are already in.        
        ra = np.concatenate([radec_[0] for radec_ in radecs])
        dec = np.concatenate([radec_[1] for radec_ in radecs])
        # never twice the same
        indexes = np.random.choice(np.arange(len(ra)), size=size, replace=False)
        ra, dec = ra[indexes], dec[indexes] # limit to exact input request
        return ra, dec
    
    # => Polygon or no skyarea
    if skyarea is not None: # change the ra_range
        size_to_draw = size*2
        ramin, decmin, ramax, decmax = skyarea.bounds
        ra_range = [ramin, ramax]
        dec_range = [decmin, decmax]
    else:
        size_to_draw = size
        
    # => Draw RA, Dec    
    dec_sin_range = np.sin(np.asarray(dec_range)*np.pi/180)
    ra = np.random.uniform(*ra_range, size=size_to_draw)
    dec = np.arcsin( np.random.uniform(*dec_sin_range, size=size_to_draw) ) / (np.pi/180)
    
    if skyarea is not None:
        from shapely.vectorized import contains
        flag = contains(skyarea, ra, dec)
        ra, dec = ra[flag], dec[flag] # those in the polygon
        indexes = np.random.choice(np.arange(len(ra)), size=size, replace=False) # never twice the same
        ra, dec = ra[indexes], dec[indexes] 
    
    return ra, dec


def surface_of_skyarea(skyarea):
    """ convert input skyarea into deg**2
    """
    if  type(skyarea) is str and skyarea != "full":
        return None

    if "shapely" in str(type(skyarea)):
        return skyarea.area
    
    return skyarea
    

def parse_skyarea(skyarea):
    """ convert input skyarea into a geometry
    """
    if  type(skyarea) is str and skyarea != "full":
        return None

    
    return skyarea
    
