import numpy as np
from astropy.cosmology import Planck18


def draw_redshift(size, rate, zmin=0., zmax=2., zstep=1e-4,
                  cosmology=Planck18, rng=None, **kwargs):
    """Draw random redshift following the given rate.

    Parameters
    ----------
    size : int
        Number of target to draw.
    rate : float or callable
        If a float is given, it is assumed to be the number of targets per
        Gpc3. If a callable is given, it is supposed to be a function of z that
        returns the volumetric rate as a function of redshift.
    zmin : float, optional
        Minimum redshift. The default is 0.
    zmax : float, optional
        Maximum redshift. The default is 2.
    zstep : float, optional
        Sampling of the redshift. The default is 1e-5.
    cosmology: astropy.Cosmology, optional
        Cosmology to use to compute volume, as the rate are "volumetric rates".
    rng : None, int, (Bit)Generator, optional
        seed for the random number generator.
        (doc adapted from numpy's `np.random.default_rng` docstring. 
        See that documentation for details.)
        If None, an unpredictable entropy will be pulled from the OS.
        If an ``int``, (>0), it will set the initial `BitGenerator` state.
        If a `(Bit)Generator`, it will be returned as a `Generator` unaltered.
    **kwargs
        Goes to `get_ntargets_per_shell()` -> `get_rate()`.

    Returns
    -------
    list
        A list of redshifts.
    """
    # force number of target per redshift shell to be a float to avoid rounding errors.
    xx, pdf = get_ntargets_per_shell(zmin=zmin, zmax=zmax, zstep=zstep, rate=rate, astype="float",
                                          cosmology=cosmology, **kwargs)

    # sets the random number generator
    rng = np.random.default_rng(rng)

    # normal pdf
    if np.ndim(pdf) == 1:
        return rng.choice(xx, size=size, p=pdf/pdf.sum())

    # 2D rates | this could happend if rates is an array.
    elif np.ndim(pdf)==2:
        return [rng.choice(xx, size=size, p=pdf_/pdf_.sum()) for pdf_ in pdf]
    else:
        raise ValueError(f"ndim of pdf should be 1 or 2, not {np.ndim(pdf)=}")

def get_rate(z, rate, **kwargs):
    """Get the (volumetric) rate as a function of redshift.

    Parameters
    ----------
    z : array
        Array of redshifts.
    rate : float or callable
        If a float is given, it is assumed to be the number of targets per
        Gpc3. If a callable is given, it is supposed to be a function of z that
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
    
    
def get_ntargets_per_shell(zmax, rate, zmin=0, zstep=1e-5, cosmology=Planck18, astype="int", **kwargs):
    """ get the total number of target expected in the given volume

    Parameters
    ----------
    zmax : float
        outter redshift of the volume.
    rate : float, array or callable
        If a float is given, it is assumed to be the number of targets per
        Gpc3. If a callable is given, it is supposed to be a function of z that
        returns the volumetric rate as a function of redshift.
        If an array is given, if array broacasts with shell size, then it 
        multiplies shell, if not than an axes is added and pdf is (rates.shape, nbins)
    zmin: float
        inner redshift of the volume.
    cosmology : astropy.Cosmology, optional
        Cosmology used to get the comiving_volume. The default is
        `Planck18`.
    zstep: float
        binning of the redshift used for the computation.
    astype: bool
        type of the returned number of target per shell.
    **kwargs goes to get_rate()
    Returns
    -------
    zbins: array
        mid value of the redshift corresponding to the shell
    pdf: nd-array
        1d array if rate broadcast with shell, else nd-array with n the rate shape.
    """
    # initial binning
    bins_of_redshift = np.arange(zmin, zmax, step=zstep) # [ndim]

    # this define the volume of the universe
    volume = cosmology.comoving_volume( bins_of_redshift ).to("Gpc**3").value # len(input_z) (+ 1 if keepsize)
    # and this the shell of universe. This is used to compute cases of non-constante rates.
    shell = np.diff(volume) # [ndim-1]

    # this are the effective redshift of the shells
    bins_of_redshift_mid = np.mean([bins_of_redshift[1:], bins_of_redshift[:-1]], axis=0) # [ndim-1]

    # so this is the rate computed at the effective redshift of the shell
    # it basically assumes the rate to be constant within one shell.
    n_per_gpc3_of_shell = get_rate(bins_of_redshift_mid, rate, **kwargs) # [ndim-1]
    # the total number of target per shell is the volumetric_rate_per_shell * the shell_volume
    if np.ndim(n_per_gpc3_of_shell) == 0 or (np.ndim(n_per_gpc3_of_shell) == 1 and len(n_per_gpc3_of_shell) == len(shell)):
        ntargets_per_shell = n_per_gpc3_of_shell * shell
    else:
        ntargets_per_shell = np.atleast_1d(n_per_gpc3_of_shell)[:, None] * shell
    
    return bins_of_redshift_mid, ntargets_per_shell.astype(astype)
    

def get_ntargets(zmax, rate, zmin=0, cosmology=Planck18, zstep=1e-5, force_shell=False, astype="int", **kwargs):
    """ get the total number of target expected in the given volume

    Parameters
    ----------
    zmax : float
        outter redshift of the volume.
    rate : float or callable
        If a float is given, it is assumed to be the number of targets per
        Gpc3. If a callable is given, it is supposed to be a function of z that
        returns the volumetric rate as a function of redshift.
    zmin: float
        inner redshift of the volume.
    cosmology : astropy.Cosmology, optional
        Cosmology used to get the comiving_volume. The default is
        `Planck18`.
    zstep: float
        binning of the redshift used for the computation.
    force_shell: bool
        If the input rate is a constant, should this force the use of shell computation ?
    astype: bool
        type of the returned value.

    Returns
    -------
    ntargets: float, array
        number(s) of target.
    """
    # function or forced, hence shell computation
    if callable(rate) or force_shell: 
        bins_of_redshift_mid, ntargets_per_shell = get_ntargets_per_shell(zmax, rate, 
                                                                          zmin=zmin, zstep=zstep, 
                                                                          cosmology=cosmology,
                                                                          astype="float", # request astype comes at "return"
                                                                          **kwargs)
        ntargets = ntargets_per_shell.sum(axis=-1) # respects rate dimension

    # simple constant volumetric rate, so "V(zmax)-V(zmin) * Constant"
    else: 
        volume_zmax = cosmology.comoving_volume( zmax ).to("Gpc**3").value
        volume_zmin = cosmology.comoving_volume( zmin ).to("Gpc**3").value
        rate = np.atleast_1d(rate)
        if np.ndim(rate) == 1:
            ntargets = (volume_zmax-volume_zmin) * rate
        else:
            ntargets = (volume_zmax-volume_zmin) * rate[:,None]

    # squeeze() will be [float] => float
    return ntargets.astype(astype).squeeze()
