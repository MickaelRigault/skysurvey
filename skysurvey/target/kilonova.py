from .core import Transient
#This is the code to model kilonova
import sncosmo as sc
from simsurvey.utils import model_tools
from simsurvey.models import AngularTimeSeriesSource
import numpy as np
optional_injection_parameters={}
phase, wave, cos_theta, flux = model_tools.read_possis_file(
                "nsns_nph1.0e+06_mejdyn0.020_mejwind0.130_phi30.txt")
model= sc.Model(AngularTimeSeriesSource(phase=phase, wave=wave, flux=flux, cos_theta=cos_theta
                    )
                )
            
template = "AngularTimeSeriesSource"

__all__ = ["Kilonova"]


class Kilonova( Transient ):

    _KIND = "kilonova"
    _TEMPLATE = model
    _RATE = 1
    _MODEL = dict( redshift = {"kwargs":{"zmax":0.2},
                                  "as":"z"},

                   t0 = {"func": np.random.uniform,
                         "kwargs": {"low":56_000, "high":56_200} },

                   )

    
