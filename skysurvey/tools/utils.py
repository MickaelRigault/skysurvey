import numpy as np
from shapely import geometry

# ================= #
#  Noise Generator  #
# ================= #

def build_covariance(as_dataframe=False, **kwargs):
    """ convert kwargs into covariance matrix

    Parameters
    ----------
    as_dataframe: bool
        should this return a np.array (False) or build a datraframe
        from it ? (True)

    kwargs: 
        the format is the following:
        {'{key1}': err,
         '{key2}': err,
         'cov_{key1}{key2}': covariance, # or 'cov_{key2}{key1}' both are looked for
        }
        
    Returns
    -------
    array or dataframe, param_names
        see as_dataframe

    """
    param_names, errors = np.stack([[l,v] for l,v in kwargs.items() if not l.startswith("cov")]).T
    cov_diag = np.diag(np.asarray(errors, dtype="float")**2)
    # not sure I need this for loop though...
    for i, ikey in enumerate(param_names):
        for j, jkey in enumerate(param_names):
            if j==i: continue
            cov_diag[i,j] = kwargs.get(f"cov_{ikey}{jkey}", kwargs.get(f"cov_{jkey}{ikey}", 0))
    
    if as_dataframe:
        cov_diag = pandas.DataFrame(cov_diag, 
                                    index=param_names, 
                                    columns=param_names)
    return cov_diag, param_names

def apply_gaussian_noise(target_or_data, **kwargs):
    """ apply random gaussian noise to the target
    
    pass the entries error and covariance as kwargs 
    following this format:
    {'{key1}': err,
     '{key2}': err,
     'cov_{key1}{key2}': covariance, # or 'cov_{key2}{key1}' both are looked for
     }
    
    Parameters
    ----------
    target_or_data: `skysurvey.Target` or pandas.DataFrame
        a target (of child of) or directly it's target.data
        This will affect what is returned.

    Returns
    -------
    target or dataframe
        according to input
    """
    import pandas
    if type(target_or_data) is pandas.DataFrame:
        data = target_or_data
        target = None
    else:
        target = target_or_data
        data = target.data
        
    
    # create the covariance matrix
    covmatrix, names = build_covariance(**kwargs)

    # create the noise
    noise = np.random.multivariate_normal( np.zeros( len(names)),
                                               covmatrix,
                                               size=( len(target.data), )
                                         )
    
    # create the noisy data form
    datanoisy = data.copy()
    datanoisy[[f"{k}_true" for k in names]] = datanoisy[names] # set truth
    datanoisy[names] += pandas.DataFrame(noise, index=data.index,
                                               columns=names)   # affect data

    # store the input noise information
    info = pandas.DataFrame(data=np.atleast_2d(list(kwargs.values())),
                       columns=list(kwargs.keys()))
    info= info.reindex(data.index, method="ffill")
    info.rename({k:f"{k}_err" for k in names}, axis=1, inplace=True)

    # to finally get the new data.
    newdata = datanoisy.merge(info, left_index=True, right_index=True)

    # returns a target or a dataframe according to input
    if target is not None:
        return target.__class__.from_data(newdata)
    
    return newdata

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
    
