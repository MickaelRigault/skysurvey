import numpy as np
from astropy.cosmology import Planck18


def draw_redshift(size, rate, zmin=0., zmax=2., zstep=1e-4,
                  flatten_ndim=True,
                  rng=None,
                  **kwargs):
    """Draw random redshift following the given rate.

    Parameters
    ----------
    size : int
        Number of target to draw.
    rate : float or callable
        If a float is given, it is assumed to be the number of targets per
        Gpc3, and `get_volumetric_rate()` is used. 
        If a callable is given, it is supposed to be a function of z that
        returns the volumetric rate as a function of redshift.
    zmin : float, optional
        Minimum redshift. The default is 0.
    zmax : float, optional
        Maximum redshift. The default is 2.
    zstep : float, optional
        Sampling of the redshift. The default is 1e-4.
    flatten_ndim : bool, optional
        [description]. The default is True.
    rng : None, int, (Bit)Generator, optional
        seed for the random number generator.
        (doc adapted from numpy's `np.random.default_rng` docstring. 
        See that documentation for details.)
        If None, an unpredictable entropy will be pulled from the OS.
        If an ``int``, (>0), it will set the initial `BitGenerator` state.
        If a `(Bit)Generator`, it will be returned as a `Generator` unaltered.
    **kwargs
        Goes to `get_redshift_pdf()` -> `get_rate()`.

    Returns
    -------
    list
        A list of redshifts.
    """
    xx = np.arange(zmin, zmax, zstep)
    pdf = get_redshift_pdf(xx, rate=rate, keepsize=False, **kwargs)
    xx_eff = np.mean([xx[1:],xx[:-1]], axis=0)

    rng = np.random.default_rng(rng)
    
    # normal pdf
    if np.ndim(pdf) == 1:
        return rng.choice(xx_eff, size=size, p=pdf/pdf.sum())

    # 2D rates
    if np.ndim(pdf) == 2:
        if not flatten_ndim:
            return [rng.choice(xx_eff, size=size, p=pdf_/pdf_.sum())
                        for pdf_ in pdf]
        
        xx_eff_flat = np.full_like(pdf, xx_eff).reshape(-1)
        pdf_flat = pdf.reshape(-1)
        return rng.choice(xx_eff_flat, size=size, p=pdf_flat/pdf_flat.sum())
        
    raise ValueError(f"ndim of pdf should be 1 or 2, not {np.ndim(pdf)=}")

def get_redshift_pdf_func(rate, zmin=0, zmax=1., zstep=1e-3, 
                            kind="cubic", bounds_error=None, fill_value=np.nan,  
                            **kwargs):
    """Get a continuous function for the redshift rate pdf.

    This is based on `get_redshift_pdf()` that returns `np.diff` of rate
    function and uses `scipy.interpolate.interp1d`.

    Parameters
    ----------
    rate : float or callable
        If a float is given, it is assumed to be the number of targets per
        Gpc3, and `get_volumetric_rate()` is used. 
        If a callable is given, it is supposed to be a function of z that
        returns the volumetric rate as a function of redshidt.
    zmin : float, optional
        Minimum redshift. The default is 0.
    zmax : float, optional
        Maximum redshift. The default is 1.
    zstep : float, optional
        Sampling of the redshift. The default is 1e-3.
    kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use. The string
        has to be one of 'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero',
        'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of
        zeroth, first, second or third order; 'previous' and 'next' simply
        return the previous or next value of the point; 'nearest-up' and
        'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5)
        in that 'nearest-up' rounds up and 'nearest' rounds down. Default
        is 'cubic'.
    bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        By default, an error is raised unless `fill_value="extrapolate"`.
    fill_value : array-like or (array-like, array_like) or "extrapolate", optional
        - if a ndarray (or float), this value will be used to fill in for
          requested points outside of the data range. If not provided, then
          the default is NaN. The array-like must broadcast properly to the
          dimensions of the non-interpolation axes.
        - If a two-element tuple, then the first element is used as a
          fill value for `x_new < x[0]` and the second element is used for
          `x_new > x[-1]`. Anything that is not a 2-element tuple (e.g.,
          list or ndarray, regardless of shape) is taken to be a single
          array-like argument meant to be used for both bounds as
          `below, above = fill_value, fill_value`. Using a two-element tuple
          or ndarray requires `bounds_error=False`.
        - If "extrapolate", then points outside the data range will be
          extrapolated.

    Returns
    -------
    scipy.interpolate.interp1d
        The redshift rate pdf.
    """
    from scipy import interpolate
    z_grid = np.arange(zmin, zmax, zstep)
    rate_at_grid = get_redshift_pdf(z_grid, rate, **kwargs)
    return interpolate.interp1d(z_grid, rate_at_grid, kind=kind,
                                bounds_error=bounds_error,
                                fill_value=fill_value,
                                assume_sorted=True)


def get_rate(z, rate, **kwargs):
    """Get the rate as a function of redshift.

    Parameters
    ----------
    z : array
        Array of redshifts.
    rate : float or callable
        If a float is given, it is assumed to be the number of targets per
        Gpc3, and `get_volumetric_rate()` is used. 
        If a callable is given, it is supposed to be a function of z that
        returns the volumetric rate as a function of redshift.
    **kwargs
        Rate options if rate is a function. 
        ignored otherwise.

    Returns
    -------
    rate
        the rate per Gpc, array (if func) or float
    """
    # specified rate function or volumetric rate ?
    if callable(rate): # function
        n_per_gpc3 = rate(z, **kwargs)
    else: # volumetric
        n_per_gpc3 = rate
        
    return n_per_gpc3

def get_redshift_pdf(z, rate, keepsize=True, cosmology=Planck18, normed=True, **kwargs):
    """Get the redshift pdf given the rate (function or volumetric).

    Parameters
    ----------
    z : array
        Array of redshifts.
    rate : float or callable
        If a float is given, it is assumed to be the number of targets per
        Gpc3, and `get_volumetric_rate()` is used. 
        If a callable is given, it is supposed to be a function of z that
        returns the volumetric rate as a function of redshift.
    keepsize : bool, optional
        Should this keep the size of the input `z`? If so, this `z` is
        linear binned and add an extra step. This is because this func
        assume rate as 3d (not shell). The default is True.
    cosmology : astropy.Cosmology, optional
        Cosmology used to get the comiving_volume. The default is
        `Planck18`.
    normed: bool
        should the sum of the pdf be forced to 1?
    **kwargs
        Rate options. ignored if `rate` is a float

    Returns
    -------
    array
        The redshift pdf.
    """
    if keepsize:
        step_ = z[-1]-z[-2]
        z = np.append(z, z[-1] + step_)
    
    n_per_gpc3 = get_rate(z, rate, **kwargs) # len(input_z) (+ 1 if keepsize)
    # volume
    volume = cosmology.comoving_volume(z).to("Gpc**3").value # len(input_z) (+ 1 if keepsize)
    shell = np.diff(volume) # len(input_z) -1 (+ 1 if keepsize)

    # what matters is the rate of that shell
    if len(np.atleast_1d(n_per_gpc3))==1: # float
        n_per_shell = n_per_gpc3
    else: # average within the shell
        n_per_shell = np.mean([n_per_gpc3[:-1], n_per_gpc3[1:]], axis=0) # len(input_z) -1 (+ 1 if keepsize)
    
    rates = n_per_shell * shell
    if normed:
        return rates/np.nansum(rates)
    
    return rates

def get_volumetric_rate(z, n_per_gpc3, cosmology=Planck18):
    """Get the number of target (per year) up to the given redshift.

    Parameters
    ----------
    z : float
        Redshift.
    n_per_gpc3 : float, array
        Number of targets per Gpc3.
        If array, it must broadcast with input `z`.
    cosmology : astropy.Cosmology, optional
        Cosmology used to get the comoving_volume. The default is
        `Planck18`.

    Returns
    -------
    float, array 
        The volumetric rate. The return type matches the type of input
        `n_per_gpc3`.
    """
    volume = cosmology.comoving_volume(z).to("Gpc**3").value
    z_rate = volume * n_per_gpc3
    return z_rate
