"""
This module provides utility functions for drawing sky coordinates, and applying observational noise.
"""

import pandas
import healpy as hp
import numpy as np

from astropy.cosmology import Planck15
from scipy.stats import rv_discrete
from scipy.interpolate import InterpolatedUnivariateSpline as Spline1d
from shapely import geometry

try:
    from ligo.skymap.bayestar import rasterize 
    from ligo.skymap.io import read_sky_map
    import ligo.skymap.distance as ligodist
    LIGO_SKYMAP_IMPORTED = True
except ImportError:
    LIGO_SKYMAP_IMPORTED = False

def get_skynoise_from_maglimit(maglim, zp=30):
    """ Get the noise associated to the 5-sigma limit magnitude.

    Parameters
    ----------
    maglim : float
        5-sigma limiting magnitude.

    zp : float, optional
        Zero point. Default is 30.

    Returns
    -------
    float
        Sky noise.
    
    """
    flux_5sigma = 10**(-0.4*(maglim - zp))
    skynoise = flux_5sigma/5.
    return skynoise

# ================= #
#  Noise Generator  #
# ================= #
def build_covariance(as_dataframe=False, **kwargs):
    """ Convert kwargs into covariance matrix.

    Parameters
    ----------
    as_dataframe: bool
        should this return a np.array (False) or build a dataframe
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
        See `as_dataframe`.
    """
    param_names, errors = np.stack([[key,val] for key, val in kwargs.items()
                                        if not key.startswith("cov")]).T
    cov_diag = np.diag(np.asarray(errors, dtype="float")**2)
    # not sure I need this for loop though...
    for i, ikey in enumerate(param_names):
        for j, jkey in enumerate(param_names): 
            if j==i:
                continue 
            cov_diag[i,j] = kwargs.get(f"cov_{ikey}{jkey}", kwargs.get(f"cov_{jkey}{ikey}", 0))
    
    if as_dataframe:
        cov_diag = pandas.DataFrame(cov_diag, 
                                    index=param_names, 
                                    columns=param_names)
    return cov_diag, param_names

def apply_gaussian_noise(target_or_data, seed=None, **kwargs):
    """ Apply random gaussian noise to the target.
    
    Pass the entries error and covariance as kwargs 
    following this format:
    
        {'{key1}': err,
        '{key2}': err,
        'cov_{key1}{key2}': covariance, # or 'cov_{key2}{key1}' both are looked for
        }
    
    Parameters
    ----------
    target_or_data: ``skysurvey.Target`` or `pandas.DataFrame`
        a target (of child of) or directly it's `target.data`
        This will affect what is returned.

    Returns
    -------
    target or dataframe
        According to input.
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
    rng = np.random.default_rng(seed=seed)
    noise = rng.multivariate_normal( np.zeros( len(names)),
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
                ra_range=[0, 360], dec_range=[-90,90],
                rng=None):
    """ Draw the sky positions.

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
            
    skyarea: `shapely.geometry.(Multi)Polyon`
        Area to consider. Default is None.
        If skyarea is given, ra_range, dec_range is ignored.

    rng : None, int, `(Bit)Generator`, optional
        seed for the random number generator.
        (doc adapted from numpy's `np.random.default_rng` docstring. 
        See that documentation for details.)
        If None, an unpredictable entropy will be pulled from the OS.
        If an ``int``, (>0), it will set the initial `BitGenerator` state.
        If a `(Bit)Generator`, it will be returned as a `Generator` unaltered.

    Returns
    -------
    2d-array
        list of ra, list of dec.
    """
    rng = np.random.default_rng(rng)
    # => MultiPolygon    
    if type(skyarea) is geometry.MultiPolygon:
        radecs = [random_radec(size=size, 
                               ra_range=ra_range,
                               dec_range=dec_range,                               
                               skyarea=skyarea_,
                               rng=rng) 
                  for skyarea_ in skyarea.geoms]
        
        # at this stage they are already in.        
        ra = np.concatenate([radec_[0] for radec_ in radecs])
        dec = np.concatenate([radec_[1] for radec_ in radecs])
        # never twice the same
        indexes = rng.choice(np.arange(len(ra)), size=size, replace=False)
        ra, dec = ra[indexes], dec[indexes] # limit to exact input request
        return ra, dec

    
    # => Polygon or no skyarea
    if skyarea is not None: # change the ra_range
        default_skyrea = geometry.Polygon([ [ra_range[0], dec_range[0]],
                                            [ra_range[0], dec_range[1]],
                                            [ra_range[1], dec_range[1]],
                                            [ra_range[1], dec_range[0]]])
        skyarea = default_skyrea.intersection(skyarea)
        size_to_draw = size*4
        ramin, decmin, ramax, decmax = skyarea.bounds
        ra_range = [ramin, ramax]
        dec_range = [decmin, decmax]
    else:
        size_to_draw = size
        
    # => Draw RA, Dec    
    dec_sin_range = np.sin(np.asarray(dec_range)*np.pi/180)
    ra = rng.uniform(*ra_range, size=size_to_draw)
    dec = np.arcsin( rng.uniform(*dec_sin_range, size=size_to_draw) ) / (np.pi/180)
    
    if skyarea is not None:
        from shapely.vectorized import contains
        flag = contains(skyarea, ra, dec)
        ra, dec = ra[flag], dec[flag] # those in the polygon
        indexes = rng.choice(np.arange( len(ra) ), size=size, replace=False) # never twice the same
        ra, dec = ra[indexes], dec[indexes] 
    
    return ra, dec

def surface_of_skyarea(skyarea, incl_projection=True):
    """ Convert input skyarea into deg**2.

    Parameters
    ----------    
    skyarea: `shapely.geometry.(Multi)Polyon`
        Area to consider. 
    
    incl_projection : bool, optional
        If True, correct for spherical sky projection before computing the area, returning the true solid angle.
        If False, return the raw area in flat Ra/Dec space. 
        Default is True.

    Returns
    -------
    float 
        Area in deg**2 of the input skyarea, with or without sky projection correction.
    """
    if  type(skyarea) is str and skyarea != "full":
        return None

    if "shapely" in str(type(skyarea)):
        if not incl_projection:
            return skyarea.area
        else:
            # create a new skyarea deformed.
            ra, dec = np.asarray(skyarea.exterior.xy)
            dec = np.sin(dec / 180*np.pi) * 180/np.pi # keeps degree
            return geometry.Polygon( np.vstack([ra, dec]).T).area
    
    return skyarea
    
def parse_skyarea(skyarea):
    """ Pass through the skyarea as a shapely geometry.

    Parameters
    ----------    
    skyarea: `shapely.geometry.(Multi)Polyon`
        Area to consider.

    Returns
    -------
    `shapely.geometry.(Multi)Polygon`
        The input skyarea unchanged.
    """
    if  type(skyarea) is str and skyarea != "full":
        return None

    return skyarea
   

def random_radecz_skymap(size=None,skymap={},
                         filename=None,
                         do_3d=True,
                         nside=512,
                         ra_range=None,dec_range=None,
                         zcmb_range=None, cosmo=Planck15, batch_size=1000,
                         rng=None):
    """ Draw random (RA, Dec, redshift) coordinates from a 3D gravitational wave sky map.

    Parameters
    ----------
    size : int
        Number of samples to draw. Default is None.

    skymap : dict or astropy Table, optional
        Pre-loaded sky map in moc format (as returned by `ligo.skymap.io.read_sky_map`).
        Ignored if `filename` is provided. 

    filename : str, optional
        Path to a LIGO/Virgo sky map fits file. If provided, the sky map is loaded
        from this file. Default is None.

    do_3d : bool, optional
        If True, load the sky map with 3D distance information. Default is True.

    nside : int, optional
        HEALPix resolution parameter used to rasterize the sky map. Default is 512.

    ra_range : 2-element array, optional
        Right ascension range [min, max] in degrees to restrict the draw.
        Pixels outside this range have their probability set to 0. Default is None.

    dec_range : 2-element array, optional
        Declination range [min, max] in degrees to restrict the draw.
        Pixels outside this range have their probability set to 0. Default is None.

    zcmb_range : 2-element array, optional
        Redshift range [zmin, zmax] to restrict the distance sampling.
        If None, no redshift restriction is applied. Default is None.

    cosmo : `astropy.cosmology`, optional
        Cosmology used to convert luminosity distances to redshifts.
        Default is Planck15.

    batch_size : int, optional
        Number of pixels drawn per batch (to avoid memory issues for large `size`).
        Default is 1000.

    rng : None, int, or `(Bit)Generator`, optional
        Seed for the random number generator.
        If None, an unpredictable entropy will be pulled from the OS.
        If an int (>0), it sets the initial BitGenerator state.
        If a (Bit)Generator, it is returned unaltered.

    Returns
    -------
    ra : array
        Right ascension of the sampled positions, in degrees.
    dec : array
        Declination of the sampled positions, in degrees.
    zs : array
        Redshifts of the sampled positions.
    """

    if not LIGO_SKYMAP_IMPORTED:
        raise ImportError("ligo.skymap could not be imported. Please make sure it is installed.")

    if filename is not None:
        if do_3d:
            skymap = read_sky_map(filename, moc=True, distances=True)
    
            if "PROBDENSITY_SAMPLES" in skymap.columns:
                skymap.remove_columns(
                    [
                        f"{name}_SAMPLES"
                        for name in [
                            "PROBDENSITY",
                            "DISTMU",
                            "DISTSIGMA",
                            "DISTNORM",
                        ]
                    ]
                )
            else:
                skymap = read_sky_map(filename, moc=True, distances=False)

    skymap_raster = rasterize(
        skymap, order=hp.nside2order(nside)
    )
    if "DISTMU" in skymap_raster.columns:
        (
            skymap_raster["DISTMEAN"],
            skymap_raster["DISTSTD"],
            mom_norm,
        ) = ligodist.parameters_to_moments(
            skymap_raster["DISTMU"],
            skymap_raster["DISTSIGMA"],
        )

    prob = skymap_raster["PROB"]
    prob[~np.isfinite(skymap_raster["DISTMU"])] = 0.
    prob[skymap_raster["DISTMU"] < 0.] = 0.
    prob[prob < 0.] = 0.
    npix = len(prob)
    nside = hp.npix2nside(npix)

    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_map = np.rad2deg(phi)
    dec_map = np.rad2deg(0.5*np.pi - theta)

    if ra_range is not None:
        idx = np.where((ra_map < ra_range[0]) | (ra_map > ra_range[1]))[0]
        prob[idx] = 0.0

    if dec_range is not  None:
        idx = np.where((dec_map < dec_range[0]) | (dec_map > dec_range[1]))[0]
        prob[idx] = 0.0

    prob = prob / np.sum(prob)
    idx = np.where(prob<0)[0]
    distn = rv_discrete(values=(np.arange(npix), prob))
    ipix = distn.rvs(size=min(size, batch_size))
    while len(ipix) < size:
        ipix = np.append(ipix, distn.rvs(size=min(size-len(ipix), batch_size)))
    ra, dec = hp.pix2ang(nside, ipix, lonlat=True)

    # If no zcmb_range provided set the upper limit to 1e9 Mpc (z >> 1000)
    if zcmb_range is not None:
        z_tmp = np.linspace(zcmb_range[0], zcmb_range[1], 1000)
        dist_range = [cosmo.luminosity_distance(zcmb_range[0]).value,
                      cosmo.luminosity_distance(zcmb_range[1]).value]
    else:
        dist_range = [0, 1e9]
        z_tmp = np.linspace(0, 10, 1000)

    z_d = Spline1d(cosmo.luminosity_distance(z_tmp).value, z_tmp)

    dists = -np.ones(size)
    dists_in_range = np.zeros(size, dtype=bool)
    rng = np.random.default_rng(rng)
    while not np.all(dists_in_range):
        ipix_tmp = ipix[~dists_in_range]
        dists[~dists_in_range] = (skymap_raster["DISTMEAN"][ipix_tmp] +
                                  skymap_raster["DISTSTD"][ipix_tmp] *
                                  rng.normal(size=np.sum(~dists_in_range)))
        dists_in_range = (dists > dist_range[0]) & (dists < dist_range[1])

    zs = z_d(dists)

    return ra, dec, zs
