
from .core import Transient

__all__ = ["SNeIa"]





class SNeIa( Transient ):

    _KIND = "SNIa"
    _TEMPLATE_SOURCE = "salt2"
    _VOLUME_RATE = 2.35 * 10**4 # Perley 2020

    # {'model': func, 'prop': dict, 'input':, 'as':}
    _MODEL = dict( redshift ={"param":{"zmax":0.2}, "as":"z"},
                              
                   x1={"model":"nicolas2021"}, 
                   
                   c={"model":"intrinsic_and_dust"},
                  
                   host_mass={"model":"asymetric_gaussian"},

                   t0={"model":"uniform", 
                       "param":{"mjd_range":[59000, 59000+365*4]} },
                       
                   magabs={"model":"massstep",
                           "input":["x1","c", "host_mass"],
                           "param":{"mabs":-19.3, "sigmaint":0.10}
                          },
                           
                   magobs={"model":"magabs_to_magobs",
                         "input":["z", "magabs"]},

                   x0={"model":"magobs_to_x0",
                       "input":["magobs"]},
                       
                   radec={"model":"random",
                          "param":dict(ra_range=[0, 360], dec_range=[-30, 90]),
                          "as":["ra","dec"]}
                    )


    # ============== #
    #  Methods       #
    # ============== #    
    def magobs_to_x0(self, magobs, band="bessellb",zpsys="ab"):
        """ """
        template = self.get_template()
        m_current = template._source.peakmag("bessellb","ab")
        return 10.**(0.4 * (m_current - magobs)) * template.get("x0")
    