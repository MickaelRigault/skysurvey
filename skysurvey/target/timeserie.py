import numpy as np
from .core import Transient
from ..tools.utils import random_radec
__all__ = ["TSTransient"]

class TSTransient( Transient ):
    """ TimeSerie Transient.

    This model will generate a Transient object from
    any TimeSerieSource model from sncosmo.
    [see list](https://sncosmo.readthedocs.io/en/stable/source-list.html)

    Example
    -------
    >>> snii = TSTransient.from_draw("snana-2004fe", 4000)
    >>> _ = snii.show_lightcurve(["ztfg","ztfr"], index=10, in_mag=True)
    """
    _RATE = 0.001
    _MODEL = dict( redshift = {"kwargs": {"zmax": 0.05}, 
                               "as": "z"},
                   t0 = {"func": np.random.uniform,
                         "kwargs": {"low": 56_000, "high": 56_200}
                        },
                         
                   magabs = {"func": np.random.normal,
                             "kwargs": {"loc": -18, "scale": 1}
                            },
                             
                   magobs = {"func": "magabs_to_magobs",
                             "kwargs": {"z":"@z", "magabs": "@magabs"}
                            },
                               
                   amplitude = {"func": "magobs_to_amplitude",
                                "kwargs": {"magobs": "@magobs"}
                            },
                   # This you need to match with the survey
                   radec = {"func": random_radec,
                            "as": ["ra","dec"]
                            }
                 )


    def __init__(self, source_or_template=None, *args, **kwargs):
        """ loads a TimeSerie Transient 

        Parameters
        ----------
        source_or_template: str, `sncosmo.Source`, `sncosmo.Model`, skysurvey.Template
            the sncosmo TimeSeriesSource, you can provide:
            - str: the name, e.g. "v19-2013ge-corr"
            - sncosmo.Source: a loaded sncosmo.Source
            - sncosmo.Model: a  loaded sncosmo.Model
            this is eventually converted into a generic skysurvey.Template.

        """
        if source_or_template is not None:
            self.set_template(source_or_template)

        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_sncosmo(cls, source_or_template,
                         rate=None,
                         model=None,
                         magabs=None, magscatter=None):
        """ loads an instance from a sncosmo TimeSeriesSource source
        (see https://sncosmo.readthedocs.io/en/stable/source-list.html#list-of-built-in-sources) 

        Parameters
        ----------
        source: str, `sncosmo.Source`, `sncosmo.Model`, skysurvey.Template
            the sncosmo TimeSeriesSource, you can provide:
            - str: the name, e.g. "v19-2013ge-corr"
            - sncosmo.Source: a loaded sncosmo.Source
            - sncosmo.Model: a  loaded sncosmo.Model
            this is eventually converted into a generic skysurvey.Template.
            
        rate: float, func
            the transient rate
            - float: assumed volumetric rate
            - func: function of redshift rate(z) 
                    that provides the rate as a function of z
        
        model: dict
            provide the model graph structure on how transient 
            parameters are drawn. 

        magabs: float
            Absolute magnitude. See cls._MODEL for default value (e.g. -18)
            (the absolute magnitude will randomly draw from magabs and magscatter
            see _MODEL or input model)
            
        magscatter: float
            abslute magnitude scatter. See cls._MODEL for default value (e.g. 1)
            (the absolute magnitude will randomly draw from magabs and magscatter
            see _MODEL or input model)

        Returns
        -------
        instance
            loaded instance. 

        See also
        --------
        from_draw: load an instance and draw the transient parameters
        """
        this = cls()

        if rate is not None:
            this.set_rate(rate)
            
        if template is not None:
            this.set_template(template)
            
        if model is not None:
            this.update_model(**model) # will update any model entry.

        # short cut to update the model
        if magabs is not None:
            this.update_model_parameter(magabs={"loc":magabs})
            
        if magscatter is not None:
            this.update_model_parameter(magabs={"scale":magscatter})
            
        return this
