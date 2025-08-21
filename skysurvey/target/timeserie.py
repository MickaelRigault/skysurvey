import numpy as np
from scipy import stats
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
    _MAGABS = None # None to be ignored.
    # Format:
    # - Gaussian: (loc, scatter)
    # - skewed Gaussian: (loc, scatter_low, scatter_high)
    _MODEL = dict( redshift = {"kwargs": {"zmax": 0.05}, 
                               "as": "z"},
                   t0 = {"func": np.random.uniform,
                         "kwargs": {"low": 56_000, "high": 56_200}
                        },
                         
                   magabs = {"func": np.random.normal,
                             "kwargs": {"loc": np.nan, "scale": 1} # forcing loc to be given
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
        if self._MAGABS is not None: #
            self.set_magabs(self._MAGABS)
    
    @classmethod
    def from_sncosmo(cls, source_or_template,
                         rate=None,
                         model=None,
                         magabs=None, magscatter=None):
        """ loads an instance from a sncosmo TimeSeriesSource source
        (see https://sncosmo.readthedocs.io/en/stable/source-list.html#list-of-built-in-sources) 

        Parameters
        ----------
        source_or_template: str, `sncosmo.Source`, `sncosmo.Model`, skysurvey.Template
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
            Absolute magnitude central value.
            This bypasses cls._MAGABS.
            
        magscatter: float, list
            = ignored if magabs is None =
            absolute magnitude scatter. 
            If float a gaussian scale is assumed. If list, an assymetric gaussian
            is assumed such that  scatter_low, scatter_high = magscatter.
            This bypasses cls._MAGABS.

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
            
        if source_or_template is not None:
            this.set_template(source_or_template)
            
        if model is not None:
            this.update_model(**model) # will update any model entry.

        # short cut to update the model
        if magabs is not None: # This overwrites with is inside _MAGABS.
            if np.ndim(magscatter) == 0: #float
                magabs_ = magabs, magscatter
            else:
                magabs_ = [magabs] + list(magscatter)
                
            this.set_magabs(magabs_) #

        return this

    def set_magabs(self, magabs):
        """ update the model for the loc *and* scale of the absolute magnitude distribution """
        if magabs is not None:
            loc, *scale = magabs
            if len(scale) == 1: # gaussian
                model_magabs = {"func": np.random.normal, "kwargs": {"loc": loc, "scale": scale[0]}}

            elif len(scale) == 2: # skewed gaussian
                from ..tools.stats import skewed_gaussian_pdf
                model_magabs = {"func": skewed_gaussian_pdf,
                                "kwargs": {"xx": f"{loc-scale[0]*10}:{loc+scale[1]*10}:10000j",
                                           "loc": loc, "scale_low": scale[0], "scale_high": scale[1]}
                                }
                
                
            self.update_model(magabs=model_magabs)
        


class MultiTemplateTSTransient( TSTransient ):
            
    def as_targets(self):
        """ convert the collection in a list of same-template targets """
        if "template" not in self.data:
            raise AttributeError("self.data has no 'template' column")
        
        gtemplates = self.data.groupby("template")
        return [TSTransient.from_data(self.data.loc[indices], template=template_)
                for template_, indices in gtemplates.groups.items()]
    
    def set_template(self, template, force_uniquetype=True):
        """ """
        from ..template import TemplateCollection
        template = np.atleast_1d(template)
        templatecol = TemplateCollection.from_list(template)
        if force_uniquetype and not templatecol.is_uniquetype:
            raise ValueError("input templates are of multiple class. This is not allowed (force_uniquetype set to True)")
            
        self._template = templatecol

    def set_rate(self, rate):
        """ set the transient rate

        Parameters
        ----------
        rate: float, func or list of
            func: a function that takes as input an array or redshift "z"
            float: number of targets per Gpc3, then skysurvey.target.rates.get_volumetric_rate() is used.
            could be a lsit of these
        """
        rate = np.atleast_1d(rate)
        if len(rate) == 1: # as usual
            rate = rate[0]
            if not callable(rate): # func or float
                rate = float(rate)
                
        else: # as list
            rate = np.asarray([float(rate_) if not callable(rate_) else rate_
                               for rate_ in rate])
            
            # does it broadcast with existing templates ?
            if hasattr(self, "_template") and self._template is not None:
                rate = np.broadcast_to(rate, (self.template.ntemplates,))[:,None]
                
        # set it
        self._rate = rate

    # =========== #
    # 2D drawing  #
    # =========== #
    def draw_redshift(self, zmax, zmin=0, 
                      zstep=1e-4,
                      size=None, rate=None, **kwargs):
        """ based on the rate (see get_rate()) """
        from .rates import draw_redshift
        if rate is None:
            rate = self.rate

        return draw_redshift(size=size, rate=rate, zmax=zmax, zmin=zmin, zstep=zstep, flatten_ndim=True, **kwargs)

    def get_template(self, index=None, which="default", as_model=False, **kwargs):
        """ """
        if index is None and which is None:
            raise ValueError("either index of which must be given to know which template you are requesting")
            
        if index is not None:
            prop = self.get_template_parameters(index).to_dict()
            kwargs = prop | kwargs
            if which is None or which == "default":
                which = self.data["template"].loc[index]

        if which == "default": # not been through index
            which = 0
            
        templateindex = self.template.nameorindex_to_index(which)
        sncosmo_model = self.template.get(ref_index=templateindex, **kwargs)
        if not as_model:
            from ..template import Template
            return Template.from_sncosmo(sncosmo_model)
        
        return sncosmo_model

    # =========== #
    #  Modeling   #
    # =========== #
    def draw_template(self, size=None, redshift=None):
        """ """
        size = len(redshift)
        if not self.has_multirates and redshift is not None:
            return np.random.choice(self.template.names, size=size)

        # flatted all shapes
        fullnames = np.full( (self.template.ntemplates, size), np.asarray(self.template.names)[:,None]).reshape(-1)
        
        # convert rates into weight of being drawn | so rate=0 templates are never drawn.
        weights = self.get_rate(redshift).reshape(-1)
        
        return np.random.choice(fullnames, size=size, p=weights/weights.sum())
        
    @property
    def model(self):
        """ """
        if not hasattr(self, "_model") or self._model is None:
            from copy import deepcopy
            basicmodel = deepcopy(self._MODEL) if self._MODEL is not None else {}
            basicmodel |= {"template": {"func": "draw_template", "kwargs": {"redshift":"@z"}} }
            self.set_model( basicmodel )
            
        return self._model

    @property
    def has_multirates(self):
        """ """
        # only 1 float or all the same
        return (np.ndim(self.rate) == 2) and not (len(np.unique(self.rate))==1)
