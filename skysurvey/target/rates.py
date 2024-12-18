import numpy as np
from astropy.cosmology import Planck18

from ..tools.utils import surface_of_skyarea

def draw_redshift(size, rate, zmin=0., zmax=2., zstep=1e-4, skyarea=None, **kwargs):
    """ random redshift draw following the given rate

    Parameters
    ----------
    size: int
        number of target to draw
    
    rate: func, float
        func: a function that takes as input an array or redshift "z"
        float: number of targets per Gpc3
        If float, get_volumetric_rate() is used.

    zmin, zmax: float
        redshift limits

    zstep: flaot
        sampling of the redshift.

    skyarea: None, str, float, geometry
        sky area (in deg**2).
        - None or 'full': 4pi
        - "extra-galactic": 4pi - (milky-way b<5)
        - float: area in deg**2
        - geometry: shapely.geometry.area is used (assumed in deg**2)

    kwargs goes to get_redshift_pdf() -> get_rate()

    Returns
    -------
    list
    """
    xx = np.arange(zmin, zmax, zstep)
    pdf = get_redshift_pdf(xx, rate=rate, keepsize=False, **kwargs)
    xx_eff = np.mean([xx[1:],xx[:-1]], axis=0)
    return np.random.choice(xx_eff, size=size, p=pdf/pdf.sum())

def get_redshift_pdf_func(rate, zmin=0, zmax=1., zstep=1e-3, 
                            kind="cubic", bounds_error=None, fill_value=np.nan,  
                            **kwargs):
    """ get a continuous function for the redshift rate pdf. 
    This is based on get_redshift_pdf() that returns np.diff of rate function and uses 
    scipy.interpolate.interp1d.

    Parameters
    ----------
    rate: func, float
        func: a function that takes as input an array or redshift "z"
        float: number of targets per Gpc3
        If float, get_volumetric_rate() is used.
 
    zmin, zmax, zstep: float
        define the grid binning where the pdf is evaluated 
        prior interpolation.

     kind: str, int
        # from scipy.interpolate.interp1d
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use.
        The string has to be one of 'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero',
        'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of
        zeroth, first, second or third order; 'previous' and 'next' simply
        return the previous or next value of the point; 'nearest-up' and
        'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5)
        in that 'nearest-up' rounds up and 'nearest' rounds down. Default
        is 'linear'.

    bounds_error: bool
        # from scipy.interpolate.interp1d
        If True, a ValueError is raised any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        By default, an error is raised unless ``fill_value="extrapolate"``.

    fill_value: array-like or (array-like, array_like) or "extrapolate"
        # from scipy.interpolate.interp1d
        - if a ndarray (or float), this value will be used to fill in for
          requested points outside of the data range. If not provided, then
          the default is NaN. The array-like must broadcast properly to the
          dimensions of the non-interpolation axes.
        - If a two-element tuple, then the first element is used as a
          fill value for ``x_new < x[0]`` and the second element is used for
          ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
          list or ndarray, regardless of shape) is taken to be a single
          array-like argument meant to be used for both bounds as
          ``below, above = fill_value, fill_value``. Using a two-element tuple
          or ndarray requires ``bounds_error=False``.
        - If "extrapolate", then points outside the data range will be
          extrapolated.

    Returns
    -------
    scipy.interpolate.interp1d
    """
    from scipy import interpolate
    z_grid = np.arange(zmin, zmax, zstep)
    rate_at_grid = get_redshift_pdf(z_grid, rate, **kwargs)
    return interpolate.interp1d(z_grid, rate_at_grid, kind=kind,
                                bounds_error=bounds_error,
                                fill_value=fill_value,
                                assume_sorted=True)


def get_rate(z, rate, skyarea=None, **kwargs):
    """ get the rate as a function of redshift

    z: array
        array of redshifts. 

    rate: float or func
        func: a function that takes as input an array or redshift "z"
        float: number of targets per Gpc3
        If float, get_volumetric_rate() is used.

    skyarea: None, str, float, geometry
        sky area (in deg**2).
        - None or 'full': 4pi
        - "extra-galactic": 4pi - (milky-way b<5)
        - float: area in deg**2
        - geometry: shapely.geometry.area is used (assumed in deg**2)

    **kwargs rate options.
        if rate is a float, these are that of get_volumetric_rate.
        (cosmology=Planck18, skyarea=None)
    """
    # specified rate function or volumetric rate ?
    if callable(rate): # function
        target_rate = rate(z, **kwargs)
    else: # volumetric
        target_rate = get_volumetric_rate(z, n_per_gpc3=rate, **kwargs)

    skyarea = surface_of_skyarea(skyarea) # in deg**2 or None
    if skyarea is not None:
        full_sky = 4*np.pi * (180/np.pi)**2 # 4pi in deg**2
        target_rate *= (skyarea/full_sky)

    return target_rate

def get_redshift_pdf(z, rate, skyarea=None, keepsize=True, **kwargs):
    """ get the redshift pdf given the rate (function or volumetric)
    
    z: array
        array of redshifts. 

    rate: float or func
        func: a function that takes as input an array or redshift "z"
        float: number of targets per Gpc3
        If float, get_volumetric_rate() is used.

    skyarea: None, str, float, geometry
        sky area (in deg**2).
        - None or 'full': 4pi
        - "extra-galactic": 4pi - (milky-way b<5)
        - float: area in deg**2
        - geometry: shapely.geometry.area is used (assumed in deg**2)

    keepsize: bool
        should this keep the size of the input z ? 
        If so, this that z is linear binned and add an extra step.
        This is because this func assume rate as 3d (not shell)
        

    **kwargs rate options.
        if rate is a float, these are that of get_volumetric_rate.
        (cosmology=Planck18)

    Returns
    -------
    array
    """
    if keepsize:
        step_ = z[-1]-z[-2]
        z = np.append(z, z[-1] + step_)
    
    target_rate = get_rate(z, rate, skyarea=skyarea, **kwargs)
    
    rates = np.diff(target_rate)
    return rates/np.nansum(rates)

def get_volumetric_rate(z, n_per_gpc3, cosmology=Planck18):
    """ number of target (per year) up to the given redshift

    Parameters
    ----------
    z: float
        redshift

    n_per_gpc3: float
        number of targets per Gpc3
    
    cosmology: astropy.Cosmology
        cosmology used to get the comiving_volume
    Returns
    -------
    float
    """
    volume = cosmology.comoving_volume(z).to("Gpc**3").value
    z_rate = volume * n_per_gpc3
    return z_rate
