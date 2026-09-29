"""
This module defines `TSTransient` and `MultiTemplateTSTransient` classes, enabling simulation of transients from any sncosmo 
time-series source, including multi-template populations.
"""

import numpy as np

from .core import Transient
from ..tools.utils import random_radec


RNG = np.random.default_rng()

class TSTransient( Transient ):
    """
    TimeSerie Transient.

    This model will generate a Transient object from
    any TimeSerieSource model from `sncosmo`.
    [see list](https://sncosmo.readthedocs.io/en/stable/source-list.html)

    Parameters
    ----------
    template: str, `sncosmo.Source`, `sncosmo.Model`, ``skysurvey.Template``
        the `sncosmo` TimeSeriesSource, you can provide:

        - str: the name, e.g. "v19-2013ge-corr"
        - `sncosmo.Source`: a loaded `sncosmo.Source`
        - `sncosmo.Model`: a  loaded `sncosmo.Model`
            this is eventually converted into a generic ``skysurvey.Template``.
        
        Default is None.

    magabs: list
        define the absolute magnitude parameters. Could be 2 or 3 values:
            
        - len(magabs)==2 => drawn from normal distribution: 
            loc, scale = magabs
        - len(magabs)==3 => drawn from asymetric normal distribution: 
            loc, scale_low, scale_high = magabs
        
            Default is None.

    _RATE : float, optional
        The rate of the TimeSerie Transient. The default is 0.001.
    _MAGABS : float, optional
        The absolute magnitude. The default is None.
    _MODEL : dict, optional
        The model to use. The default is a dictionary with the following
        keys:

        - `redshift`: The redshift of the TimeSerie Transient.
        - `t0`: The time of maximum of the TimeSerie Transient.
        - `magabs`: The absolute magnitude of the TimeSerie Transient.
        - `magobs`: The observed magnitude of the TimeSerie Transient.
        - `radec`: The ra and dec of the TimeSerie Transient.

    Examples
    --------
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
                   t0 = {"func": RNG.uniform,
                         "kwargs": {"low": 56_000, "high": 56_200}
                        },
                         
                   magabs = {"func": RNG.normal,
                             "kwargs": {"loc": np.nan, "scale": 1} # forcing loc to be given
                            },
                             
                   magobs = {"func": "magabs_to_magobs",
                             "kwargs": {"z":"@z", "magabs": "@magabs"}
                            },
                               
                   # This you need to match with the survey
                   radec = {"func": random_radec,
                            "as": ["ra","dec"]
                            }
                 )


    def __init__(self, template=None, magabs=None, *args, **kwargs):
        """ Initialize the TimeSerie Transient. """
        if template is not None:
            self.set_template(template)

        super().__init__(*args, **kwargs)
        
        if magabs is not None:
            self.set_magabs(magabs)
        elif self._MAGABS is not None: #
            self.set_magabs(self._MAGABS)

    @classmethod
    def _parse_init_kwargs_(cls, **kwargs):
        """ Trick to add specific subclass kwargs into the init. """
        # remove any magabs option from input kkwargs => init_kwargs
        init_kwargs = {"magabs": kwargs.pop("magabs", None)}
        # first => init_kwargs
        # second => generic kwargs
        return init_kwargs, kwargs

    @classmethod
    def from_sncosmo(cls, template,
                         rate=None,
                         model=None,
                         magabs=None, **kwargs):
        """ Loads an instance from a sncosmo TimeSeriesSource source.
        (see https://sncosmo.readthedocs.io/en/stable/source-list.html#list-of-built-in-sources) 

        Parameters
        ----------
        template: str, `sncosmo.Source`, `sncosmo.Model`, ``skysurvey.Template``
            the `sncosmo` TimeSeriesSource, you can provide:

            - str: the name, e.g. "v19-2013ge-corr"
            - `sncosmo.Source`: a loaded `sncosmo.Source`
            - `sncosmo.Model`: a  loaded `sncosmo.Model`

            This is eventually converted into a generic ``skysurvey.Template``.

        rate: float, func
            the transient rate, can be either:
            - float: assumed volumetric rate
            - func: function of redshift rate(z) 

        model: dict
            provide the model graph structure on how transient parameters are drawn. 

        magabs: list
            define the absolute magnitude parameters. Could be 2 or 3 values:

            - len(magabs)==2 => drawn from normal distribution: 
                loc, scale = magabs
            - len(magabs)==3 => drawn from asymetric normal distribution: 
                loc, scale_low, scale_high = magabs   

        Returns
        -------
        instance
            loaded instance. 

        See also
        --------
        ``from_draw``: load an instance and draw the transient parameters
        """
        init_kwargs, kwargs = cls._parse_init_kwargs_(**kwargs)
        this = cls(**init_kwargs)

        if rate is not None:
            this.set_rate(rate)
            
        if template is not None:
            this.set_template(template)
            
        if model is not None:
            this.update_model(**model) # will update any model entry.

        # short cut to update the model
        if magabs is not None: # This overwrites with is inside _MAGABS.
            this.set_magabs(magabs) #

        return this

    def set_magabs(self, magabs):
        """ Update the model for the loc *and* scale of the absolute magnitude distribution.
        
        Parameters
        ----------
        magabs: list
            define the absolute magnitude parameters. Could be 2 or 3 values:

            - len(magabs)==2 => drawn from normal distribution: 
                loc, scale = magabs
            - len(magabs)==3 => drawn from asymetric normal distribution: 
                loc, scale_low, scale_high = magabs

        """
        if magabs is not None:
            loc, *scale = magabs
            if len(scale) == 1: # gaussian
                model_magabs = {"func": RNG.normal, "kwargs": {"loc": loc, "scale": scale[0]}}

            elif len(scale) == 2: # skewed gaussian
                from ..tools.stats import skewed_gaussian_pdf
                model_magabs = {"func": skewed_gaussian_pdf,
                                "kwargs": {"xx": f"{loc-scale[0]*10}:{loc+scale[1]*10}:10000j",
                                           "loc": loc, "scale_low": scale[0], "scale_high": scale[1]}
                                }
                
            self.update_model(magabs=model_magabs)
        


class MultiTemplateTSTransient( TSTransient ):
    """
    A class to model time-series transient drawn from multiple templates simultaneously.
    
    Parameters
    ----------
    template: str, `sncosmo.Source`, `sncosmo.Model`, ``skysurvey.Template``
        the `sncosmo` TimeSeriesSource, you can provide:

        - str: the name, e.g. "v19-2013ge-corr"
        - `sncosmo.Source`: a loaded `sncosmo.Source`
        - `sncosmo.Model`: a  loaded `sncosmo.Model`
            this is eventually converted into a generic ``skysurvey.Template``.
        
        Default is None.

    magabs: list
        define the absolute magnitude parameters. Could be 2 or 3 values:
            
        - len(magabs)==2 => drawn from normal distribution: 
            loc, scale = magabs
        - len(magabs)==3 => drawn from asymetric normal distribution: 
            loc, scale_low, scale_high = magabs
        
            Default is None.
    """

    def as_targets(self):
        """ Convert the collection in a list of same-template targets. """
        if "template" not in self.data:
            raise AttributeError("self.data has no 'template' column")
        
        gtemplates = self.data.groupby("template")
        targets = []
        for template_name, indices in gtemplates.groups.items():
            template_index = self.template.nameorindex_to_index(template_name)
            template = self.template.get(template_index)
            target = TSTransient.from_data(
                data=self.data.loc[indices],
                template=template
            )
            targets.append(target)
        return targets
    
    def set_template(self, template, force_uniquetype=True):
        """ Set a collection of templates.

        Parameters
        ----------
        template : str, list of str, or list of `sncosmo` sources
            One or more `sncosmo` TimeSeriesSource templates.

        force_uniquetype : bool, optional
            If True, raise an error if templates are of different types.
            Default is True.
        """
        from ..template import TemplateCollection
        template = np.atleast_1d(template)
        templatecol = TemplateCollection.from_list(template)
        if force_uniquetype and not templatecol.is_uniquetype:
            raise ValueError("input templates are of multiple class. This is not allowed (force_uniquetype set to True)")
            
        self._template = templatecol

    def set_rate(self, rate):
        """ Set the transient rate.

        Parameters
        ----------
        rate: float, func or list of
            func: a function that takes as input an array or redshift "z"
            float: number of targets per Gpc3. could be a list.
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
        """ Draw redshifts based on the rate (see ``get_rate()``).

        Parameters
        ----------
        zmax : float
            Maximum redshift.

        zmin : float, optional
            Minimum redshift. Default is 0.

        zstep : float, optional
            Redshift step. Default is 1e-4.

        size : int, optional
            Number of redshifts to draw. Default is None.

        rate : float or callable, optional
            The transient rate. If None, ``self.rate`` is used. Default is None.

        Returns
        -------
        array
            The drawn redshifts.
        """
        from .rates import draw_redshift
        if rate is None:
            rate = self.rate

        return draw_redshift(size=size, rate=rate, zmax=zmax, zmin=zmin, zstep=zstep, flatten_ndim=True, **kwargs)

    def get_template(self, index=None, as_model=False, data=None, set_magabs=False, **kwargs):
        """Get a template (`sncosmo.Model`).

        Parameters
        ----------
        index : int, optional
            Index of a target (see ``self.data.index``) to set the template
            parameters to that of the target. If None, the default
            `sncosmo.Model` parameters will be used. By default None.

        as_model : bool, optional
            should this return the `sncosmo.Model` (True) or the 
            ``skysurvey.Template`` (for info `sncosmo.Model` => ``skysurvey.Template.sncosmo_model``)

        data: `pandas.DataFrame`, None, optional
            which data should be used to set the parameter of the template. Ignored if index is None.

        set_magabs: bool, optional
            should the peal magnitude of the template be set to magabs ?

        **kwargs
            Goes to ``self.template.get()`` and passed to `sncosmo.Model`.

        Returns
        -------
        ``skysurvey.Template`` or `sncosmo.Model`
            An instance of the template (or its associated `sncosmo.Model`).
            (see ``as_model``)
        """

        if data is None:
            data = self.data

        if index is None:
            index = 0

        prop = self.get_template_parameters(index, data=data).to_dict()
        kwargs = prop | kwargs
        _ = kwargs.pop(self.amplitude_name, None)
        which = data["template"].loc[index]

        templateindex = self.template.nameorindex_to_index(which)
        sncosmo_model = self.template.get(ref_index=templateindex, **kwargs)

        if set_magabs:
            peak_absmag = data.loc[index, "magabs"]
            peak_absmag_band = self.peak_absmag_band
            peak_absmag_magsys = self.magsys

            sncosmo_model.set_source_peakabsmag(
                absmag=peak_absmag,
                band=peak_absmag_band,
                magsys=peak_absmag_magsys,
                cosmo=self.cosmology
                )

        if not as_model:
            from ..template import Template
            return Template.from_sncosmo(sncosmo_model)
        
        return sncosmo_model

    # =========== #
    #  Modeling   #
    # =========== #
    def draw_template(self, size=None, redshift=None, rng=None):
        """Draw a template name for each transient, weighted by their rates.

        If all templates share the same rate, templates are drawn uniformly.
        Otherwise, templates are drawn proportionally to their rates at the
        given redshifts.

        Parameters
        ----------
        size : int, optional
            Number of templates to draw. Overridden by `len(redshift)` if
            redshift is provided. Default is None.

        redshift : array, optional
            Redshifts at which to evaluate the rates. Default is None.

        rng : None, int, or `(Bit)Generator`, optional
            Seed for the random number generator. Default is None.

        Returns
        -------
        array
            Template names, one per transient.
        """
        size = len(redshift)
        rng = np.random.default_rng(rng)
        if not self.has_multirates and redshift is not None:
            return rng.choice(self.template.names, size=size)

        # flatted all shapes
        fullnames = np.full( (self.template.ntemplates, size), np.asarray(self.template.names)[:,None]).reshape(-1)
        
        # convert rates into weight of being drawn | so rate=0 templates are never drawn.
        weights = self.get_rate(redshift).reshape(-1)
        
        return rng.choice(fullnames, size=size, p=weights/weights.sum())
        
    @property
    def model(self):
        """The model of the transient"""
        if not hasattr(self, "_model") or self._model is None:
            from copy import deepcopy
            basicmodel = deepcopy(self._MODEL) if self._MODEL is not None else {}
            basicmodel |= {"template": {"func": "draw_template", "kwargs": {"redshift":"@z"}} }
            self.set_model( basicmodel )
            
        return self._model

    @property
    def has_multirates(self):
        """Whether templates have different rates from one another."""
        # only 1 float or all the same
        return (np.ndim(self.rate) == 2) and not (len(np.unique(self.rate))==1)
