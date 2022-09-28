
from .core import Transient

__all__ = ["SNeIa"]





class SNeIa( Transient ):

    _KIND = "SNIa"
    _TEMPLATE = "salt2"
    _VOLUME_RATE = 2.35 * 10**4 # Perley 2020

    # {'model': func, 'prop': dict, 'input':, 'as':}
    _MODEL = dict( redshift ={"param":{"zmax":0.2}, "as":"z"},
                              
                   x1={"model":"nicolas2021"}, 
                   
                   c={"model":"intrinsic_and_dust"},

                   t0={"model":"uniform", 
                       "param":{"mjd_range":[59000, 59000+365*4]} },
                       
                   magabs={"model":"tripp1998",
                           "input":["x1","c"],
                           "param":{"mabs":-19.3, "sigmaint":0.10}
                          },
                           
                   magobs={"model":"magabs_to_magobs",
                         "input":["z", "magabs"]},

                   x0={"model":"magobs_to_amplitude",
                       "input":["magobs"],
                       "param":{"param_name":"x0"}}, #because it needs to call sncosmo_model.get(param_name)
                       
                   radec={"model":"random",
                          "param":dict(ra_range=[0, 360], dec_range=[-30, 90]),
                          "as":["ra","dec"]}
                    )
