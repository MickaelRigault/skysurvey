import numpy as np
from .core import Transient

__all__ = ["TSTransient"]

class TSTransient( Transient ):
    """ TimeSerie Transient.

    This model will generate a Transient object from
    any TimeSerieSource model from sncosmo.
    [see](https://sncosmo.readthedocs.io/en/stable/source-list.html)

    Example
    -------
    >>> snii = TSTransient.from_draw("snana-2004fe", 4000)
    >>> _ = snii.show_lightcurve(["ztfg","ztfr"], index=10, in_mag=True)
    """
    
    _MODEL = dict( redshift = {"param": {"zmax": 0.05}, 
                               "as": "z"},
                   t0 = {"model": np.random.uniform,
                         "param": {"low": 56000, "high": 56000+4*365}},
                         
                   magabs = {"model": np.random.normal,
                             "param": {"loc": -18, "scale": 1}},
                             
                   magobs = {"model": "magabs_to_magobs",
                               "input": ["z", "magabs"]},
                               
                   amplitude = {"model": "magobs_to_amplitude",
                                "input": ["magobs"]},
                   # This you need to match with the survey
                   radec={"model": "random",
                          "as": ["ra","dec"]}
                 )
    
    @classmethod
    def from_sncosmo_source(cls, source, rate=1e3, model=None,
                                magabs=None, magscatter=None):
        """ """
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
    def from_draw(cls, source, size, model=None, rate=1e-3,
                      magabs=None, magscatter=None,
                      **kwargs):
        """ """
        this = cls.from_sncosmo_source(source, rate=rate, model=model,
                                      magabs=magabs, magscatter=magscatter)
        this.draw(size=size, **kwargs)
        return this
