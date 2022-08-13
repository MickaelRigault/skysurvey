
from .core import Transient

__all__ = ["SNeIa"]

class SNeIa( Transient ):
    _KIND = "SNIa"
    _TEMPLATE_SOURCE = "salt2"
    _VOLUME_RATE = 2.35 * 10**4 # Perley 2020
    _MODEL = dict( redshift ={"prop":{"zmax":0.2}, "as":"z"},
                   x1={"model":"nicolas2021"},
                   c={"model":"intrinsic_and_dust"},
                   t0={"model":"uniform", 
                         "prop":{"mjd_range":[59000, 59000+365*4]}
                        },
                    x0={"model":"tripp1998", "input":["x1","c"], "prop":{"size":None}},
                    radec={"model":"random", 
                            "prop":dict(ra_range=[0, 360], dec_range=[-30, 90]),
                            "as":["ra","dec"]}
                    )
