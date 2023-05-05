
import numpy as np
import pandas

from .core import Target, Transient
from .timeserie import TSTransient

__all__ = ["TargetCollection"]


def reshape_values(values, shape):
    """ """
    values = np.atleast_1d(values)
    if len(values) == 1:
        values = np.resize(values, shape)
    assert len(values) == shape
    # success
    return values


class TargetCollection( object ):
    _COLLECTION_OF = Target
    _TEMPLATES = []
    
    def __init__(self, targets=None):
        """ """
        self.set_targets(targets)

    def as_targets(self):
        """ convert the collection in a list of same-template targets """
        if "template" not in self.data:
            raise AttributeError("self.data has no 'template' column")
        
        gtemplates = self.data.groupby("template")
        return [self._COLLECTION_OF.from_data(self.data.loc[indices],
                                              template=template_)
                for template_, indices in gtemplates.groups.items()]
        
    # ============= #
    #  Collection   #
    # ============= #            
    def call_down(self, which, margs=None, **kwargs):
        """ """
        if margs is not None:
            margs = reshape_values(margs, self.ntargets)
            return [getattr(t, which)(marg_, **kwargs) for marg_, t in zip(margs, self.targets)]
            
        return [attr if not callable(attr:=getattr(t, which)) else\
                attr(**kwargs) 
                for t in self.targets]
    
    # ============= #
    #  Methods      #
    # ============= #    
    def set_targets(self, targets):
        """ """
        self._targets = np.atleast_1d(targets) if targets is not None else []
            
    def get_model_parameters(self, entry, key, default=None):
        """ """
        return self.call_down("get_model_parameter", 
                              entry=entry, key=key, default=default)

    def get_data(self, keys="_KIND", colname="kind"):
        """ get a concat 
        
        """
        if keys is not None and type(keys) is str: 
            keys = self.call_down(keys)

        list_of_data = self.call_down("data")
        data = pandas.concat(list_of_data, keys=keys)
        if keys is not None:
            if colname is None:
                colname = keys
            data = data.reset_index(names=[colname,"subindex"])
            
        return data


    def get_target_template(self, index):
        """ """
        from ..template import Template
        data_index = self.data.loc[index]
        this_template = Template.from_sncosmo( data_index["template"] )
        target_params = data_index[np.in1d(data_index.index, this_template.parameters)].to_dict()
        this_template.sncosmo_model.set(**target_params)
        return this_template

        
    def show_lightcurve(self, band, index, params=None,
                            ax=None, fig=None, colors=None,
                            time_range=[-20,50], npoints=500,
                            zp=25, zpsys="ab",
                            format_time=True, t0_format="mjd", 
                            in_mag=False, invert_mag=True, **kwargs):
        """ 
        params: None or dict
        """

        if params is None:
            params = {}
        # get the template
        template = self.get_target_template(index, **params)
        return template.show_lightcurve(band, params=params,
                                             ax=ax, fig=fig, colors=colors,
                                             time_range=time_range, npoints=npoints,
                                             zp=zp, zpsys=zpsys,
                                             format_time=format_time,
                                             t0_format=t0_format, 
                                             in_mag=in_mag, invert_mag=invert_mag,
                                             **kwargs)

    
    def to_transient(self, keys=None, **kwargs):
        """ """
        data = self.get_data(keys=keys)
        return Transient.from_data(data, **kwargs)
        
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def targets(self):
        """ list of transients """
        return self._targets

    @property
    def data(self):
        """ """
        if not hasattr(self,"_data"):
            self._data = self.get_data()
        return self._data
    
    @property
    def ntargets(self):
        """ number of targets """
        return len(self.targets)
    
    @property
    def target_ids(self):
        """ targets id """
        return np.arange(self.ntargets)
    
    @property
    def models(self):
        """ list of the target models """
        return self.call_down("model")


    @property
    def template(self):
        """ shortcut to self.templates for self-consistency """
        return self.templates
    
    @property
    def templates(self):
        """ """
        if not hasattr(self,"_templates") or self._templates is None:
            self._templates = self._TEMPLATES
            
        return self._templates

    
class TransientCollection( TargetCollection ):
    _COLLECTION_OF = Transient    
    # ============= #
    #  Methods      #
    # ============= #
    def get_rates(self, z, relative=False, **kwargs):
        """ """
        rates = self.call_down("get_rate", margs=z, **kwargs)
        if relative:
            rates /= np.nansum(rates)
        return rates
    
    def draw(self, size=None,
                 zmin=None, zmax=None,
                 tstart=None, tstop=None,
                 nyears=None,
                 inplace=True, shuffle=True, **kwargs):
        """ """
        if size is not None:
            relat_rate = np.asarray(self.get_rates(0.1, relative=True))
            templates = np.random.choice(np.arange( self.ntargets), size=size,
                                        p=relat_rate/relat_rate.sum())
            # using pandas to convert that into sizes.
            # Most likely, there is a nuympy way, but it's fast enough?
            templates = pandas.Series(templates)
            # count entries and force 0 and none exist.
            sizes = templates.value_counts().reindex( np.arange(self.ntargets)
                                                     ).fillna(0).astype(int)
            # and simply get the values
            size = sizes.values # numpy
            
            
        draws = self.call_down("draw", margs=size,
                              zmin=zmin, zmax=zmax,     
                              tstart=tstart, tstop=tstop,
                              nyears=nyears, inplace=False, 
                              **kwargs)
        
        data = pandas.concat(draws, keys=self.templates, axis=0)
        data = data.reset_index(level=0).rename({"level_0":"template"}, axis=1)
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
            
        if inplace:
            self._data = data

        return data
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def rates(self):
        """ list of transients """
        return self.call_down("rate")


class CompositeTransient( TransientCollection ):
    _COLLECTION_OF = Transient

    _KIND = "unknown"    
    _RATE = 1e5

    _MAGABS = (-18, 1) # 
    # ============= #
    #  Methods      #
    # ============= #
    @classmethod
    def from_draw( cls, size=None, templates=None,
                   zmax=None, tstart=None, tstop=None,
                   nyears=None, **kwargs):
        """ loads the instance from a random draw of targets given the model 

        Parameters
        ----------
        size: int, None
            number of target you want to sample
            size=None as in numpy. Usually means 1.
            = ignored if nyears given =

        templates: str, None
            list of template names (sncosmo.Model(source)). 
            = leave to None if unsure, cls._TEMPLATES used as default =

        zmax: float
            maximum redshift to be simulated.

        tstart: float
            starting time of the simulation
            
        tstop: float
            ending time of the simulation
            (if tstart and nyears are both given, tstop will be
            overwritten by ``tstart+365.25*nyears``

        nyears: float
            if given, nyears will set:
            - size: it will be the number of target expected up to zmax 
            in the given  number of years. 
            This uses get_rate(zmax).
            - tstop: tstart+365.25*nyears

        **kwargs goes to self.draw()

        Returns
        -------
        class instance
            self.data, self.model and self.template will be loaded.

        See also
        --------
        from_setting:  loads an instance given model parameters (dict)            
        """
        this = cls()
        if templates is not None:
            this._templates = templates
            
        _ = this.draw(size=size, zmax=zmax, tstart=tstart, tstop=tstop,
                      nyears=nyears, **kwargs)
        return this

    # ============= #
    #  Properties   #
    # ============= #    
    @property
    def targets(self):
        """ list of targets forming the composite transients """
        if not hasattr(self,"_targets") or self._targets is None or len(self._targets) == 0:
            prop = {"rate": self.rate}
            prop["magabs"], prop["magscatter"] = self.magabs
            self._targets = [self._COLLECTION_OF.from_sncosmo(source_, **prop)
                                 for source_ in self.templates]
        return self._targets

    @property
    def magabs(self):
        """ """
        if not hasattr(self,"_magabs") or self._magabs is None:
            self._magabs = self._MAGABS
            
        return self._magabs
        
    @property
    def rate(self):
        """ rate.
        (If float, assumed to be volumetric rate in Gpc-3 / yr-1.)
        """
        if not hasattr(self,"_rate"):
            self._rate = self._RATE # default
            
        return self._rate
    
    
class TSTransientCollection( TransientCollection ):
    _COLLECTION_OF = TSTransient
        
    @classmethod
    def from_draw(cls, sources, size=None, nyears=None, 
                      rates=1e3, magabs=None, magscatter=None,
                      **kwargs):
        """ """
        this = cls.from_sncosmo(sources, rates=rates,
                                        magabs=magabs, 
                                        magscatter=magscatter)
        _ = this.draw(size=size, nyears=nyears, inplace=True,
                      **kwargs)
        return this
        
    @classmethod
    def from_sncosmo(cls, sources, rates=1e3, 
                             magabs=None, magscatter=None):
        """ loads the instance from a list of sources
        (and relative rates)
        """
        # make sure the sizes match
        rates = reshape_values(rates, len(sources))
        transients = [cls._COLLECTION_OF.from_sncosmo(source_, rate_)
                     for source_, rate_ in zip(sources, rates)]
        
        # Change the model.
        if magabs is not None:
            magabs = reshape_values(magabs, len(sources))
            _ = [t.change_model_parameter(magabs={"loc":magabs_}) 
                 for t, magabs_ in zip(transients, magabs)]
            
        if magscatter is not None:
            magscatter = reshape_values(magscatter, len(sources))
            _ = [t.change_model_parameter(magabs={"scale":magscatter_}) 
                 for t, magscatter_ in zip(transients, magscatter)]
            
        # and loads it
        return cls(transients)
