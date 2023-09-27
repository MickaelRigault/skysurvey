""" This is the code to model kilonova """

import numpy as np

import sncosmo
from .core import Transient
from ..tools.utils import random_radec
from ..effects import dust


__all__ = ["Kilonova"]


def read_possis_file(filename):
    """Read in a spectral model created by POSSIS (1906.04205), as appropriate
       for injestion as a skysurvey.source.angular.AngularTimeSeriesSource.
       Model grids can be found here: https://github.com/mbulla/kilonova_models.

    Parameters
    ----------
    filename : str
        Path to the POSSIS file
    Returns
    -------
    phase : `~numpy.ndarray`
        Phases in days.
    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
    flux : `~numpy.ndarray`
        Model spectral flux density in arbitrary units.
        Must have shape `(num_phases)`.
    cos_theta : `~numpy.ndarray`
        cosine of viewing angle
    """

    f = open(filename)
    lines = f.readlines()

    nobs = int(lines[0])
    nwave = float(lines[1])
    line3 = (lines[2]).split(' ')
    ntime = int(line3[0])
    t_i = float(line3[1])
    t_f = float(line3[2])

    cos_theta = np.linspace(0, 1, nobs)  # 11 viewing angles
    phase = np.linspace(t_i, t_f, ntime)  # epochs

    file_ = np.genfromtxt(filename, skip_header=3)

    wave = file_[0:int(nwave),0]
    flux = []
    for i in range(int(nobs)):
        flux.append(file_[i*int(nwave):i*int(nwave)+int(nwave),1:])
    flux = np.array(flux).T

    phase = np.linspace(t_i, t_f, len(flux.T[0][0]))  # epochs

    return phase, wave, cos_theta, flux

def get_kilonova_model(filename=None):
    """ """
    from ..source.angular import AngularTimeSeriesSource
    if filename is None:
        import os
        from .. import _PACKAGE_PATH
        filename = os.path.join(_PACKAGE_PATH, "data", "nsns_nph1.0e+06_mejdyn0.020_mejwind0.130_phi30.txt")
    
    phase, wave, cos_theta, flux = read_possis_file(filename)
    source = AngularTimeSeriesSource(phase=phase, wave=wave, flux=flux, cos_theta=cos_theta,
                                         name="kilonova")
    model = sncosmo.Model(source)
    return model

# =============== #
#                 #
#  Kilonova       #
#                 #
# =============== #

_KILONOVA_MODEL = get_kilonova_model()
class Kilonova( Transient ):

    _KIND = "kilonova"
    _TEMPLATE = _KILONOVA_MODEL
    _RATE = 1e3 # event per Gyr**3
    _MODEL = dict( # when
                   t0 = {"func": np.random.uniform,
                         "kwargs": {"low":56_000, "high":56_200}
                        },
                         
                   # what
                   redshift = {"kwargs":{"zmax":0.2}, "as":"z"},
                                  
                   magabs = {"func": np.random.normal,
                             "kwargs": {"loc": -18, "scale": 1}
                            },
                             
                   magobs = {"func": "magabs_to_magobs",
                             "kwargs": {"z":"@z", "magabs": "@magabs"}
                            },
                               
                   amplitude = {"func": "magobs_to_amplitude",
                                "kwargs": {"magobs": "@magobs"}
                            },

                   theta = {"func": np.random.uniform,
                            "kwargs": {"low":0., "high":90.}
                            },
                   # where
                   radec = {"func": random_radec,
                            "kwargs": {},
                            "as": ["ra","dec"]
                           },
                        
                    mwebv = {"func": dust.get_mwebv, "kwargs":{"ra":"@ra", "dec":"@dec"}}
                   )

    
