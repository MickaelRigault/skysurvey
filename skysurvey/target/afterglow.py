
import numpy as np
from scipy import stats

import sncosmo
try:
    import afterglowpy
except:
    raise ImportError("could not import afterglowpy ; run pip install afterglowpy")


from .core import Transient


__all__ = ["Afterglow"]


phases = np.linspace(1.0e3, 1.0e7, 300)
wave = np.linspace(3600,6600,300)
nu = np.empty(phases.shape)
nu[:] = 1.0e18

grb_params = {'jetType':     afterglowpy.jet.TopHat,     # Top-Hat jet
              'specType':    0,                          # Basic Synchrotron Emission Spectrum

              'thetaObs':    0.05,   # Viewing angle in radians
              'E0':          1.0e53, # Isotropic-equivalent energy in erg
              'thetaCore':   0.1,    # Half-opening angle in radians
              'n0':          1.0,    # circumburst density in cm^{-3}
              'p':           2.2,    # electron energy distribution index
              'epsilon_e':   0.1,    # epsilon_e
              'epsilon_B':   0.01,   # epsilon_B
              'xi_N':        1.0,    # Fraction of electrons accelerated
              'd_L':         1.0e28, # Luminosity distance in cm
              'z':           0.55}   # redshift
     
# explicitly case E0 and d_L as float as they like to be an int
grb_params["E0"] = float(grb_params["E0"])
grb_params["d_L"] = float(grb_params["d_L"])

flux = []
for phase in phases:
    t = phase * np.ones(nu.shape)
    mJys = afterglowpy.fluxDensity(t, nu, **grb_params)
    Jys = 1e-3 * mJys
    # convert to erg/s/cm^2/A
    flux.append(Jys * 2.99792458e-05 / (wave**2))

template = sncosmo.Model(sncosmo.TimeSeriesSource(phases, wave, np.array(flux)))

class Afterglow( Transient ):
    """A class to model afterglows.

    Parameters
    ----------
    _KIND : str, optional
        The kind of transient. The default is "afterglow".
    _TEMPLATE : sncosmo.Model, optional
        The template to use. The default is a `sncosmo.Model` with a
        `sncosmo.TimeSeriesSource` source.
    _RATE : int, optional
        The rate of afterglows. The default is 20.
    _MODEL : dict, optional
        The model to use. The default is a dictionary with the following
        keys:

        - `redshift`: The redshift of the afterglow.
        - `t0`: The time of maximum of the afterglow.
    """

    _KIND = "afterglow"
    _TEMPLATE = template
    _RATE = 20
    _MODEL = dict( redshift = {"kwargs":{"zmax":0.2},
                                  "as":"z"},

                   t0 = {"func": np.random.uniform,
                         "kwargs": {"low":56_000, "high":56_200} },

                   )

