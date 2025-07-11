""" module containing black-body related functions """

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
    """ get an evolving black-body sncosmo TimeSeriesSource


    phase: array
        phase.

    temperature: float, array
        temperature of the black body as a function of phase 
        (in Kelvin)

    amplitude: float, array
        amplitude of the black body peak amplitude (Wein's lbda_max).
        (in flux units)

    lbda: str, array
        wavelength in Angstrom. 
        If str, this is assumed to by a np.r_ format.

    zero_before: bool
        should pre-phase magnitude be 0 (true) or the first flux (False)
        
    name: str
        source name.

    """
    from sncosmo import TimeSeriesSource
    if type(lbda) == str: # assumed r_ input
        lbda = eval(f"np.r_[{lbda}]")
    
    fluxes = get_blackbody_transient_flux(lbda, temperature=temperature, amplitude=amplitude)
    bb_source = TimeSeriesSource(phase=phase, wave=lbda, flux=fluxes, zero_before=zero_before, 
                                name=name)
    return bb_source 


def get_blackbody_transient_flux(lbda, temperature, amplitude, normed=True):
    """ provides a 2d surface assuming black body temperature and amplitude evolution.
    
    Parameters
    ----------
    lbda:  number, array-like, or `~astropy.units.Quantity`
        wavelength.
        If not a Quantity, it is assumed to be in Angstrom.

    temperature : number, array-like, or `~astropy.units.Quantity`
        Blackbody temperature.
        If not a Quantity, it is assumed to be in Kelvin.


    amplitude: number, array-like, or `~astropy.units.Quantity`
        amplitude of the black body.
        If normed=True: amplitude on the peak-magnitude.

    normed: bool
        set peak luminosity to 1. 
        Estimated by Wein's lambda_max.

    Returns
    -------
    2d-array
    """
    normed_blackbody = blackbody_lambda(lbda, temperature=np.atleast_1d(temperature)[:,None],
                                       normed=normed)
    return normed_blackbody*amplitude[:,None]




def blackbody_nu(freq, temperature):
    """ Calculate blackbody flux per steradian, :math:`B_{\\nu}(T)`.

    .. note::

        Use `numpy.errstate` to suppress Numpy warnings, if desired.

    .. warning::

        Output values might contain ``nan`` and ``inf``.

    Parameters
    ----------
    f : number, array-like, or `~astropy.units.Quantity`
        Frequency, wavelength, or wave number.
        If not a Quantity, it is assumed to be in Hz.

    temperature : number or `~astropy.units.Quantity`
        Blackbody temperature.
        If not a Quantity, it is assumed to be in Kelvin.

    Returns
    -------
    flux : `~astropy.units.Quantity`
        Blackbody monochromatic flux in
        :math:`erg \\; cm^{-2} s^{-1} Hz^{-1} sr^{-1}`.

    Raises
    ------
    ValueError
        Invalid temperature.

    ZeroDivisionError
        Wavelength is zero (when converting to frequency).

    """
    # Convert to units for calculations | float64 required by astropy units
    with u.add_enabled_equivalencies(u.spectral() + u.temperature()):
        freq = u.Quantity(freq, u.Hz, dtype="float64")
        temp = u.Quantity(temperature, u.K, dtype="float64")

    # Check if input values are physically possible        
    if np.any(freq <= 0):
        warnings.warn('freq contains invalid values (<0)')

    # Calculate blackbody flux
    bb_nu = (2.0 * constants.h * freq ** 3 /
             (constants.c ** 2 * np.expm1(constants.h * freq / (constants.k_B * temp))))
    flux = bb_nu.to(_fnu_units, u.spectral_density(freq))

    return flux / u.sr  # Add per steradian to output flux unit


def blackbody_lambda(lbda, temperature, normed=True):
    """Like :func:`blackbody_nu` but for :math:`B_{\\lambda}(T)`.

    Parameters
    ----------
    lbda: number, array-like, or `~astropy.units.Quantity`
        wavelength.
        If not a Quantity, it is assumed to be in Angstrom.

    temperature: number or `~astropy.units.Quantity`
        Blackbody temperature.
        If not a Quantity, it is assumed to be in Kelvin.

    normed: bool
        set peak luminosity to 1. 
        Estimated by Wein's lambda_max.
        
    Returns
    -------
    flux : `~astropy.units.Quantity`
        Blackbody monochromatic flux in
        :math:`erg \\; cm^{-2} s^{-1} \\AA^{-1} sr^{-1}`.

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
    """ lambda max temperature
    {\displaystyle \lambda _{m}={\frac {hc}{4.96511423174\,\mathrm {kT} }}}
    """
    if not hasattr(temperature, 'unit'): # assumed Angstrom
        temperature = u.Quantity(temperature, u.Kelvin)

    lbda = constants.h*constants.c/(4.96511423174*constants.k_B * temperature)
    return lbda.to(u.Angstrom)
