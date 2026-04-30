"""
This module defines `AngularTimeSeriesSource`, a `sncosmo` source class for spectral time series models.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline as Spline2d
import sncosmo


class AngularTimeSeriesSource(sncosmo.Source):
    r"""
    A single-component spectral time series model.
    
    The spectral flux density of this model is given by
    
    .. math::

        F(t, \lambda) = A \times M(t, \lambda, \cos\theta)

    where :math:`M` is the flux defined on a grid in phase and wavelength
    and :math:`A` (amplitude) is the single free parameter of the model. The
    amplitude :math:`A` is a simple unitless scaling factor applied to
    whatever flux values are used to initialize the
    ``TimeSeriesSource``. Therefore, the :math:`A` parameter has no
    intrinsic meaning. It can only be interpreted in conjunction with
    the model values. Thus, it is meaningless to compare the :math:`A`
    parameter between two different ``TimeSeriesSource`` instances with
    different model data.

    Parameters
    ----------
    phase : `numpy.ndarray`
        Phases in days.

    wave : `numpy.ndarray`
        Wavelengths in Angstroms.

    flux : `numpy.ndarray`
        Model spectral flux density in arbitrary units.
        Must have shape `(num_phases)`.

    zero_before : bool, optional
        If True, flux at phases before minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the minimum phase (``flux[0, :]`` in the input array).

    zero_after : bool, optional
        If True, flux at phases after minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the maximum phase (``flux[-1, :]`` in the input array).

    cos_theta : `numpy.ndarray`
        cosine of viewing angle

    name : str, optional
        Name of the model. Default is None.

    version : str, optional
        Version of the model. Default is None.
    """

    _param_names = ['amplitude', 'theta']
    param_names_latex = ['A', r'\theta']

    def __init__(self, phase, wave, cos_theta, flux,
                     zero_before=False, zero_after=False, name=None,
                     version=None):
        """ Initialize the AngularTimeSeriesSource class."""
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._cos_theta = cos_theta
        self._flux_array = flux
        self._parameters = np.array([1., 0.])
        self._current_theta = 0.
        self._zero_before = zero_before
        self._zero_after = zero_after
        self._set_theta()

    def _set_theta(self):
        """ Update the internal 2D interpolation grid based on the current theta parameter. """
        logflux_ = np.zeros(self._flux_array.shape[:2])
        
        for k in range(len(self._phase)):
            adding = 1e-10 # Here we add 1e-10 to avoid problems with null values
            f_tmp = Spline2d(self._wave, self._cos_theta, np.log(self._flux_array[k]+adding),
                             kx=1, ky=1)
            logflux_[k] = f_tmp(self._wave, np.cos(self._parameters[1]*np.pi/180)).T

        self._model_flux = Spline2d(self._phase, self._wave, logflux_, kx=1, ky=1)
        self._current_theta = self._parameters[1]
        
    def _flux(self, phase, wave):
        """
        Compute the spectral flux density at given phase and wavelength.

        Parameters
        ----------
        phase : `numpy.ndarray`
            Array of phases.

        wave : `numpy.ndarray`
            Array of wavelengths.

        Returns
        -------
        flux : `numpy.ndarray`
            The interpolated flux density, scaled by amplitude.
        """
        if self._current_theta != self._parameters[1]:
            self._set_theta()
            
        f = self._parameters[0] * (np.exp(self._model_flux(phase, wave)))
        
        if self._zero_before:
            mask = np.atleast_1d(phase) < self.minphase()
            f[mask, :] = 0.
            
        if self._zero_after:
            mask = np.atleast_1d(phase) > self.maxphase()
            f[mask, :] = 0.
            
        return f