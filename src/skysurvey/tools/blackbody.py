"""Blackbody-related functions."""

# Units
import numpy as np
import warnings

from astropy import units as u
from astropy import constants

_fnu_units = u.erg / (u.cm**2 * u.s * u.Hz)
_flam_units = u.erg / (u.cm**2 * u.s * u.AA)



def get_blackbody_transient_source(phase, temperature, amplitude,
                                       lbda="1_000:10_000:1000j",
                                       zero_before=True, name="bb_transient"):
    """Get an evolving blackbody `sncosmo.TimeSeriesSource`.

    Parameters
    ----------
    phase : array_like
        Phase.

    temperature : float or array_like
        Temperature of the blackbody as a function of phase, in Kelvin.

    amplitude : float or array_like
        Amplitude of the blackbody peak (Wien's lbda_max), in flux units.
        If array_like, must have the same length as `phase`.

    lbda : str or array_like, optional
        Wavelength in Angstrom. If str, it is assumed to be in `np.r_` format.
        The default is "1_000:10_000:1000j".

    zero_before : bool, optional
        If True, flux is zero before the first phase; otherwise the first flux
        value is used. The default is True.

    name : str, optional
        Source name. The default is "bb_transient".

    Returns
    -------
    sncosmo.TimeSeriesSource
        The blackbody transient source.
    """
    from sncosmo import TimeSeriesSource
    if type(lbda) is str: # assumed r_ input
        lbda = eval(f"np.r_[{lbda}]")
    
    fluxes = get_blackbody_transient_flux(lbda, temperature=temperature, amplitude=amplitude)
    bb_source = TimeSeriesSource(phase=phase, wave=lbda, flux=fluxes, zero_before=zero_before, 
                                name=name)
    return bb_source 

def get_blackbody_transient_flux(lbda, temperature, amplitude, normed=True):
    r"""Provide a 2D flux grid assuming blackbody temperature and amplitude evolution.

    Parameters
    ----------
    lbda : float, array_like, or astropy.units.Quantity
        Wavelength. If not a Quantity, it is assumed to be in Angstrom.

    temperature : float, array_like, or astropy.units.Quantity
        Blackbody temperature. If not a Quantity, it is assumed to be in Kelvin.

    amplitude : float, array_like, or astropy.units.Quantity
        Amplitude of the blackbody. If array_like, must have the same length as
        `temperature`.

    normed : bool, optional
        If True, each blackbody flux is normalized to its peak value (given by
        Wien's lambda_max), and the returned array is dimensionless. If False,
        returns the flux with units. The default is True.

    Returns
    -------
    numpy.ndarray or astropy.units.Quantity
        2D blackbody monochromatic flux, scaled by its amplitude, with shape
        (len(temperature), len(lbda)). If `normed` is True, returns a
        dimensionless array normalized to peak flux. If `normed` is False,
        returns an `astropy.units.Quantity` in
        :math:`erg \; cm^{-2} s^{-1} \AA^{-1} sr^{-1}`.
    """
    normed_blackbody = blackbody_lambda(lbda, temperature=np.atleast_1d(temperature)[:,None],
                                       normed=normed)
    amplitude = np.atleast_1d(amplitude)

    return normed_blackbody*amplitude[:,None]

def blackbody_nu(freq, temperature):
    r"""Calculate blackbody flux per steradian, :math:`B_{\nu}(T)`.

    Parameters
    ----------
    freq : float, array_like, or astropy.units.Quantity
        Frequency, wavelength, or wave number. If not a Quantity, it is assumed
        to be in Hertz.

    temperature : float or astropy.units.Quantity
        Blackbody temperature. If not a Quantity, it is assumed to be in Kelvin.

    Returns
    -------
    flux : astropy.units.Quantity
        Blackbody monochromatic flux in
        :math:`erg \; cm^{-2} s^{-1} Hz^{-1} sr^{-1}`.

    Raises
    ------
    ValueError
        Invalid temperature.

    ZeroDivisionError
        Wavelength is zero (when converting to frequency).

    Warns
    -----
    UserWarning
        If `freq` contains invalid values (<= 0).

    Notes
    -----
    Use `numpy.errstate` to suppress Numpy warnings, if desired.

    Output values might contain ``nan`` and ``inf``.
    """
    # Convert to units for calculations | float64 required by astropy units
    with u.add_enabled_equivalencies(u.spectral() + u.temperature()):
        freq = u.Quantity(freq, u.Hz, dtype="float64")
        temp = u.Quantity(temperature, u.K, dtype="float64")

    # Check if input values are physically possible        
    if np.any(freq <= 0):
        warnings.warn("freq contains invalid values (<= 0)")

    # Calculate blackbody flux
    bb_nu = (2.0 * constants.h * freq ** 3 /
             (constants.c ** 2 * np.expm1(constants.h * freq / (constants.k_B * temp))))
    flux = bb_nu.to(_fnu_units, u.spectral_density(freq))

    return flux / u.sr  # Add per steradian to output flux unit

def blackbody_lambda(lbda, temperature, normed=True):
    r"""Like :func:`~skysurvey.tools.blackbody.blackbody_nu` but for :math:`B_{\lambda}(T)`.

    Parameters
    ----------
    lbda : float, array_like, or astropy.units.Quantity
        Wavelength. If not a Quantity, it is assumed to be in Angstrom.

    temperature : float or astropy.units.Quantity
        Blackbody temperature. If not a Quantity, it is assumed to be in Kelvin.

    normed : bool, optional
        If True, the blackbody flux is normalized to its peak value (given by
        Wien's lambda_max), and the returned array is dimensionless. If False,
        returns the flux with units. The default is True.

    Returns
    -------
    flux : numpy.ndarray or astropy.units.Quantity
        Blackbody monochromatic flux. If `normed` is True, returns a
        dimensionless array normalized to peak flux. If `normed` is False,
        returns an `astropy.units.Quantity` in
        :math:`erg \; cm^{-2} s^{-1} \AA^{-1} sr^{-1}`.
    """
    if not hasattr(lbda, 'unit'): # assumed Angstrom
        lbda = u.Quantity(lbda, u.AA)

    bb_nu = blackbody_nu(lbda, temperature) * u.sr  # Remove sr for conversion
    flux = bb_nu.to(_flam_units, u.spectral_density(lbda)) / u.sr  # Add per steradian to output flux unit
    
    if normed:
        lbda_max = get_wein_lbdamax(temperature)
        f_max = blackbody_lambda(lbda_max, temperature, normed=False)
        flux = (flux/f_max).value
    
    return flux 

def get_wein_lbdamax(temperature):
    r"""Return the wavelength of maximum emission of a blackbody (Wien's law).

    .. math::

        \lambda_{\mathrm{max}} = \frac{h c}{4.96511423174 \, k_B T}

    Parameters
    ----------
    temperature : float or astropy.units.Quantity
        Blackbody temperature. If not a Quantity, it is assumed to be in Kelvin.

    Returns
    -------
    lbda : astropy.units.Quantity
        Wavelength of maximum emission, in Angstrom.
    """
    if not hasattr(temperature, 'unit'): # assumed Kelvin
        temperature = u.Quantity(temperature, u.Kelvin)

    lbda = constants.h*constants.c/(4.96511423174*constants.k_B * temperature)

    return lbda.to(u.Angstrom)