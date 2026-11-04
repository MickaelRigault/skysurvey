""" Functions to add peculiar velocities effects to SNe Ia model DAG (see dedicated notebooks)."""

from astropy.coordinates import SkyCoord
from astropy import constants
import astropy.units as u
import numpy as np

def zcmb_to_zhel(zcmb, ra, dec):
    """Convert CMB-frame redshifts to heliocentric redshifts.

    The conversion accounts for the motion of the Sun relative to the CMB
    rest frame, using the CMB dipole measured by Planck. The Sun's velocity
    is projected along the line of sight to each object and converted to a
    relativistic Doppler redshift, which is then combined with the CMB-frame
    redshift.

    Parameters
    ----------
    zcmb : float or array_like
        Redshift(s) in the CMB rest frame.

    ra : float or array_like
        Right ascension of the objects, in degrees (ICRS).

    dec : float or array_like
        Declination of the objects, in degrees (ICRS).

    Returns
    -------
    zhel : float or numpy.ndarray
        Heliocentric redshift(s), with the same shape as `zcmb`.

    See Also
    --------
    zhel_to_zcmb : Inverse conversion, heliocentric to CMB frame.
    """
    sun = SkyCoord(l=264.021*u.deg, b=48.253*u.deg, frame='galactic') # Dipole from Planck 2020
    SN_Ia = SkyCoord(ra=np.array(ra)*u.deg, dec=np.array(dec)*u.deg, frame='icrs')
    c = constants.c.value
    sep = sun.separation(SN_Ia)
    vsun = 369.82*1e3*np.cos(sep.radian)
    x = vsun/c
    zsun = np.sqrt((1-x)/(1+x))-1
    zhel = (1+zcmb)*(1+zsun)-1 # Carr et al. 2022
    return zhel

def zhel_to_zcmb(zhel, ra, dec):
    """Convert heliocentric redshifts to CMB-frame redshifts.

    The conversion removes the contribution of the Sun's motion relative to
    the CMB rest frame, using the CMB dipole measured by Planck. 

    Parameters
    ----------
    zhel : float or array_like
        Heliocentric redshift(s).

    ra : float or array_like
        Right ascension of the objects, in degrees (ICRS).

    dec : float or array_like
        Declination of the objects, in degrees (ICRS).

    Returns
    -------
    zcmb : float or numpy.ndarray
        Redshift(s) in the CMB rest frame, with the same shape as `zhel`.

    See Also
    --------
    zcmb_to_zhel : Inverse conversion, CMB frame to heliocentric.
    """
    sun = SkyCoord(l=264.021*u.deg, b=48.253*u.deg, frame='galactic') # Dipole from Planck 2020
    SN_Ia = SkyCoord(ra=np.array(ra)*u.deg, dec=np.array(dec)*u.deg, frame='icrs')
    c = constants.c.value
    sep = sun.separation(SN_Ia)
    vsun = 369.82*1e3*np.cos(sep.radian)
    x = vsun/c
    zsun = np.sqrt((1-x)/(1+x))-1
    zcmb = (1+zhel)/(1+zsun)-1 # from Carr et al. 2022
    return zcmb

def magobs_to_magobshel(magobs, z, zhel):
    """Convert observed magnitudes to heliocentric-frame observed magnitudes.

    Applies the correction to the observed magnitude that accounts for the
    difference between the redshift used in the CMB frame and the
    heliocentric redshift, through the factor (1 + zhel) / (1 + z).

    Parameters
    ----------
    magobs : float or array_like
        Observed magnitude(s) computed with the redshift `z`.

    z : float or array_like
        Redshift(s) in the frame in which `magobs` is defined (e.g. the
        CMB frame).

    zhel : float or array_like
        Heliocentric redshift(s).

    Returns
    -------
    magobshel : float or numpy.ndarray
        Observed magnitude(s) in the heliocentric frame, with the same
        shape as `magobs`.

    See Also
    --------
    magobshel_to_magobs : Inverse conversion.
    """
    return magobs + 5*np.log10((1+zhel)/(1+z)) # from Carreres et al. 2023

def magobshel_to_magobs(magobshel, z, zhel):
    """Convert heliocentric-frame observed magnitudes back to observed magnitudes.

    Removes the correction applied by `magobs_to_magobshel`, which accounts
    for the difference between the redshift `z` and the heliocentric
    redshift `zhel`, through the factor (1 + zhel) / (1 + z).

    Parameters
    ----------
    magobshel : float or array_like
        Observed magnitude(s) in the heliocentric frame.

    z : float or array_like
        Redshift(s) in the frame in which the returned `magobs` is defined
        (e.g. the CMB frame).

    zhel : float or array_like
        Heliocentric redshift(s).

    Returns
    -------
    magobs : float or numpy.ndarray
        Observed magnitude(s) computed with the redshift `z`, with the same
        shape as `magobshel`.

    See Also
    --------
    magobs_to_magobshel : Inverse conversion.
    """
    return magobshel - 5*np.log10((1+zhel)/(1+z)) # from Carreres et al. 2023