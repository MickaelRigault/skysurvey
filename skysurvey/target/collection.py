
import numpy as np
import pandas

from .core import Target, Transient
from .timeserie import TSTransient

__all__ = ["TargetCollection"]


def targets_from_collection(transientcollection):
    """Get targets from a transient collection."""
    # which targets

    return targets
    

def broadcast_mapping(value, ntargets):
    """Broadcast a value to a given number of targets."""
    value = np.atleast_1d(value)
    if np.ndim(value)>1:
        # squeeze drop useless dimensions.
        broadcasted_values = np.broadcast_to(value, (ntargets, value.shape[-1]) )
    else:
        broadcasted_values = np.broadcast_to(value, ntargets)
        
    return broadcasted_values


    
class TargetCollection( object ):
    """A collection of targets.

    Parameters
    ----------
    _COLLECTION_OF : type, optional
        The type of target in the collection. The default is `Target`.
    _TEMPLATES : list, optional
        A list of templates. The default is [].
    """
    _COLLECTION_OF = Target
    _TEMPLATES = []
    
    def __init__(self, targets=None):
        """Initialize the TargetCollection.

        Parameters
        ----------
        targets : list, optional
            A list of targets. The default is None.
        """
        self.set_targets(targets)

    def as_targets(self):
        """Convert the collection into a list of same-template targets."""
        if "template" not in self.data:
            raise AttributeError("self.data has no 'template' column")
        
        gtemplates = self.data.groupby("template")
        return [self._COLLECTION_OF.from_data(self.data.loc[indices],
                                              template=template_)
                for template_, indices in gtemplates.groups.items()]
        
    # ============= #
    #  Collection   #
    # ============= #            
    def call_down(self, which, margs=None, allow_call=True, **kwargs):
        """Call a method on each target in the collection."""
        if margs is not None:
            margs = broadcast_mapping(margs, self.ntargets)
            return [getattr(t, which)(marg_, **kwargs)
                        for marg_, t in zip(margs, self.targets)]
            
        return [attr if not (callable(attr:=getattr(t, which)) and allow_call) else\
                attr(**kwargs) 
                for t in self.targets]
    
    # ============= #
    #  Methods      #
    # ============= #    
    def set_targets(self, targets):
        """Set the targets in the collection."""
        self._targets = np.atleast_1d(targets) if targets is not None else []

    def get_model_parameters(self, entry, key, default=None):
        """Get the model parameters for each target in the collection."""
        return self.call_down("get_model_parameter", 
                              entry=entry, key=key, default=default)

    def get_data(self, keys="_KIND", colname="kind"):
        """Get a concatenated dataframe of the data from each target."""
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
        """Get the template for a given target."""
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
        """Show the lightcurve of a given target.

        Parameters
        ----------
        band : str
            The band to show.
        index : int
            The index of the target.
        params : dict, optional
            Parameters to pass to `get_target_template`. The default is {}.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. The default is None.
        fig : matplotlib.figure.Figure, optional
            The figure to plot on. The default is None.
        colors : list, optional
            A list of colors to use. The default is None.
        time_range : list, optional
            The time range to plot. The default is [-20, 50].
        npoints : int, optional
            The number of points to plot. The default is 500.
        zp : float, optional
            The zero point to use. The default is 25.
        zpsys : str, optional
            The zero point system to use. The default is "ab".
        format_time : bool, optional
            Whether to format the time axis. The default is True.
        t0_format : str, optional
            The format of the time axis. The default is "mjd".
        in_mag : bool, optional
            Whether to plot in magnitudes. The default is False.
        invert_mag : bool, optional
            Whether to invert the magnitude axis. The default is True.
        **kwargs
            Additional keyword arguments to pass to `template.show_lightcurve`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
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
        """Convert the collection to a `Transient` object."""
        data = self.get_data(keys=keys)
        return Transient.from_data(data, **kwargs)
        
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def targets(self):
        """The list of targets in the collection."""
        return self._targets

    @property
    def data(self):
        """The data of the collection."""
        if not hasattr(self,"_data"):
            self._data = self.get_data()
        return self._data
    
    @property
    def ntargets(self):
        """The number of targets in the collection."""
        return len(self.templates)
    
    @property
    def target_ids(self):
        """The IDs of the targets in the collection."""
        return np.arange(self.ntargets)
    
    @property
    def models(self):
        """The models of the targets in the collection."""
        return self.call_down("model")


    @property
    def template(self):
        """A shortcut to `self.templates` for self-consistency."""
        return self.templates
    
    @property
    def templates(self):
        """The templates of the targets in the collection."""
        if not hasattr(self,"_templates") or self._templates is None:
            self._templates = self._TEMPLATES
            
        return self._templates

    
class TransientCollection( TargetCollection ):
    """A collection of transients.

    Parameters
    ----------
    _COLLECTION_OF : type, optional
        The type of transient in the collection. The default is `Transient`.
    """
    _COLLECTION_OF = Transient    
    # ============= #
    #  Methods      #
    # ============= #
    def set_rates(self, float_or_func):
        """Call `set_rate` for each target in the collection."""
        _ = self.call_down("set_rate", float_or_func)

    def update_model(self, rate_update=True, **kwargs):
        """Call `update_model` for each target in the collection."""
        _ = self.call_down("update_model", rate_update=True, **kwargs)
        
    def get_rates(self, z, relative=False, **kwargs):
        """Get the rates for each target in the collection."""
        rates = self.call_down("get_rate", margs=z, **kwargs)
        if relative:
            rates /= np.nansum(rates)
        return rates
    
    def draw(self, size=None,
                 zmin=None, zmax=None,
                 tstart=None, tstop=None,
                 nyears=None,
                 inplace=True, shuffle=True,
                 **kwargs):
        """Draw the transients in the collection."""
        if size is not None:
            relat_rate = np.asarray( self.get_rates(0.1, relative=True) ).reshape(self.ntargets)
            templates = np.random.choice( np.arange( self.ntargets ), size=size,
                                          p=relat_rate/relat_rate.sum() )
            
            # using pandas to convert that into sizes.
            # Most likely, there is a nuympy way, but it's fast enough.
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

class CompositeTransient( TransientCollection ):
    """A composite transient.

    Parameters
    ----------
    _COLLECTION_OF : type, optional
        The type of transient in the collection. The default is `Transient`.
    _KIND : str, optional
        The kind of transient. The default is "unknown".
    _RATE : float, optional
        The rate of the transient. The default is 1e5.
    _MAGABS : tuple, optional
        The absolute magnitude of the transient. The default is (-18, 1).
    """
    _COLLECTION_OF = Transient

    _KIND = "unknown"    
    _RATE = 1e5
    _MAGABS = (-18, 1) #
    
    # ============= #
    #  Methods      #
    # ============= #
    @classmethod
    def from_draw( cls,
                   size=None, model=None, templates=None,
                   zmax=None, tstart=None, tstop=None,
                   zmin=0, nyears=None,
                   skyarea=None,
                   rate=None, effect=None,
                   **kwargs):
        """Load the instance from a random draw of targets given the model.

        Parameters
        ----------
        size : int, optional
            Number of target you want to sample. If None, 1 is assumed.
            Ignored if `nyears` is given. By default None.
        model : dict, optional
            Defines how template parameters are drawn and how they are
            connected. The model will update the default `cls._MODEL` if any.
            If None, `cls._MODEL` is used as default. By default None.
        templates : str, optional
            Name of the template (`sncosmo.Model(source)`). If None,
            `cls._TEMPLATE` is used as default. By default None.
        zmax : float, optional
            Maximum redshift to be simulated. By default None.
        tstart : float, str, optional
            Starting time of the simulation. If a string is given, it is
            converted to mjd. By default None.
        tstop : float, str, optional
            Ending time of the simulation. If a string is given, it is
            converted to mjd. If `tstart` and `nyears` are both given,
            `tstop` will be overwritten by `tstart + 365.25 * nyears`.
            By default None.
        zmin : float, optional
            Minimum redshift to be simulated. By default 0.
        nyears : float, optional
            If given, `nyears` will set:

            - `size`: it will be the number of target expected up to `zmax`
              in the given number of years. This uses `get_rate(zmax)`.
            - `tstop`: `tstart + 365.25 * nyears`

            By default None.
        skyarea : None, str, geometry, optional
            Sky area to be considered.

            - str: 'full' (equivalent to None), ['extra-galactic', not implemented yet]
            - geometry: shapely.Geometry
            - None: full sky

            By default None.
        rate : float, callable, optional
            The transient rate.

            - float: assumed volumetric rate
            - callable: function of redshift `rate(z)` that provides the rate
              as a function of z.

            By default None.
        effect : [type], optional
            [description]. By default None.
        **kwargs
            Goes to `self.draw()`.

        Returns
        -------
        CompositeTransient
            The loaded instance.

        See Also
        --------
        from_setting: loads an instance given model parameters (dict)
        """
        this = cls()
    
        if rate is not None:
            this.set_rates(rate) # this uses call_down('set_rate')
        
        if templates is not None:
            this._templates = templates
    
        if model is not None:
            this.call_down("update_model", **model, rate_update=False) # will update any model entry.
        
        if effect is not None:
            this.call_down("add_effect", effect) # will update any model entry.
    
        if kwargs:
            this.update_model_parameter(**kwargs, rate_update=False)
            
        # cleaning rate automatic feeding in model
        #this._update_rate_in_model_()
        _ = this.draw( size=size,
                       zmin=zmin, zmax=zmax,
                       tstart=tstart, tstop=tstop,
                       nyears=nyears,
                       skyarea=skyarea,
                       inplace=True, # creates self.data
                       )
        return this

    # ============= #
    #  Properties   #
    # ============= #
    @property
    def targets(self):
        """The list of targets forming the composite transients."""
        if not hasattr(self,"_targets") or self._targets is None or len(self._targets) == 0:
            # build targets
            self._targets = [self._COLLECTION_OF.from_sncosmo(source_)
                             for source_ in self.templates]
            self.set_rates( self._RATE ) # default
            self.call_down("set_magabs", np.atleast_2d(self._MAGABS) ) # default
            
        return self._targets

    @property
    def magabs(self):
        """The absolute magnitudes of the transients in the collection."""
        return self.call_down("magabs")
        
    @property
    def rate(self):
        """The rate of the transients in the collection.

        If float, it is assumed to be the volumetric rate in Gpc-3 / yr.
        """
        return self.call_down("rate", allow_call=False)

    @property
    def ntargets(self):
        """The number of templates in the collection."""
        return len(self.templates)
    
    
class TSTransientCollection( TransientCollection ):
    """A collection of time-series transients.

    Parameters
    ----------
    _COLLECTION_OF : type, optional
        The type of transient in the collection. The default is `TSTransient`.
    """
    _COLLECTION_OF = TSTransient
        
    @classmethod
    def from_draw(cls, sources, size=None, nyears=None, 
                      rates=1e3, magabs=None, magscatter=None,
                      **kwargs):
        """Load the instance from a random draw of targets given the model."""
        this = cls.from_sncosmo(sources, rates=rates,
                                        magabs=magabs, 
                                        magscatter=magscatter)
        _ = this.draw(size=size, nyears=nyears, inplace=True,
                      **kwargs)
        return this
        
    @classmethod
    def from_sncosmo(cls, sources, rates=1e3, 
                        magabs=None, magscatter=None):
        """Load the instance from a list of sources (and relative rates)."""
        # make sure the sizes match
        rates = broadcast_mapping(rates, len(sources))
        transients = [cls._COLLECTION_OF.from_sncosmo(source_, rate_)
                     for source_, rate_ in zip(sources, rates)]
        
        # Change the model.
        if magabs is not None:
            magabs = broadcast_mapping(magabs, len(sources))
            _ = [t.change_model_parameter(magabs={"loc":magabs_}) 
                 for t, magabs_ in zip(transients, magabs)]
            
        if magscatter is not None:
            magscatter = broadcast_mapping(magscatter, len(sources))
            _ = [t.change_model_parameter(magabs={"scale":magscatter_}) 
                 for t, magscatter_ in zip(transients, magscatter)]
            
        # and loads it
        return cls(transients)
