import numpy as np
from .core import Transient

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
    
    _MODEL = dict( redshift = {"kwargs": {"zmax": 0.05}, 
                               "as": "z"},
                   t0 = {"func": np.random.uniform,
                         "kwargs": {"low": 56000, "high": 56000+4*365}
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
                   radec = {"func": "random",
                            "as": ["ra","dec"]
                            }
                 )
    
    @classmethod
    def from_sncosmo(cls, source, rate=1e3, model=None,
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
        this.set_template(source)
        this._rate = rate
        
        if model is not None:
            self._model = model

        if magabs is not None:
            this.change_model_parameter(magabs={"loc":magabs})
            
        if magscatter is not None:
            this.change_model_parameter(magabs={"scale":magscatter})
            
        return this
    
    @classmethod
    def from_draw(cls, source, size, rate=1e-3, model=None, 
                      magabs=None, magscatter=None,
                      zmin=0, zmax=None,
                      tstart=None, tstop=None,
                      **kwargs):
        """ loads an instance from a sncosmo TimeSeriesSource source and draws trasient parameters
        (see https://sncosmo.readthedocs.io/en/stable/source-list.html#list-of-built-in-sources) 

        = this is loads using cls.from_sncosmo = 


        Parameters
        ----------
        source: str, `sncosmo.Source`, `sncosmo.Model`, skysurvey.Template
            the sncosmo TimeSeriesSource, you can provide:
            - str: the name, e.g. "v19-2013ge-corr"
            - sncosmo.Source: a loaded sncosmo.Source
            - sncosmo.Model: a  loaded sncosmo.Model
            this is eventually converted into a generic skysurvey.Template.
            
        
        size: int
            number of transient to draw

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

        zmin: float
            minimum redshift to be simulated.

        zmax: float
            maximum redshift to be simulated.

        tstart: float
            starting time of the simulation
            
        tstop: float
            ending time of the simulation
            (if tstart and nyears are both given, tstop will be
            overwritten by ``tstart+365.25*nyears``

        *kwargs goes to self.draw() e.g. nyears

        Returns
        -------
        instance
            loaded instance with a self.data loaded.

        See also
        --------
        from_draw: load an instance and draw the transient parameters
        """
        this = cls.from_sncosmo(source, rate=rate, model=model,
                                      magabs=magabs, magscatter=magscatter)
        this.draw(size=size,
                      zmin=zmin, zmax=zmax,
                      tstart=tstart, tstop=tstop,
                      **kwargs)
        return this
