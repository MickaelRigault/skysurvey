import warnings
import numpy as np
import pandas
from astropy import cosmology, time
from astropy.utils.decorators import classproperty


from ..template import Template
from ..tools.utils import parse_skyarea, surface_of_skyarea


__all__ = ["Target", "Transient"]

class Target( object ):

    _KIND = "unknow"
    _TEMPLATE = None
    _MODEL = None # dict config
    
    # - Cosmo
    _COSMOLOGY = cosmology.Planck18

    def __init__(self):
        """ 
        See also
        --------
        from_setting: loads an instance given model parameters (dict)
        """
        pass
        
    def __repr__(self):
        """ """
        
        return self.__str__()
    
    def __str__(self):
        """ """
        import pprint
        return pprint.pformat(self.model.model, sort_dicts=False)

    @classmethod
    def from_setting(cls, setting, **kwargs):
        """ loads the target from a setting dictionary
        
        = Not Implemented Yet = 

        Parameters
        ----------
        setting: dict
            dictionary containing the model parameters

        **kwargs ....


        Returns
        -------
        class instance
            not implemented yet
        """
        raise NotImplementedError("from_setting is not Implemented ")


    @classmethod
    def from_data(cls, data, template=None, model=None):
        """ loads the instance given existing data. 
        
        This means that the model will be ignored as 
        data will not be generated but input.

        Parameters
        ----------
        data: pandas.DataFrame
            dataframe containing (at least) the template
            parameters

        template: str, `sncosmo.Source`, `sncosmo.Model` or skysurvey.Template
            the template source.
            - str: any sncosmo model name

        model: dict, None
            defines how  template parameters to draw and how they are connected
            model will update the default cls._MODEL if any
            = leave to None if unsure, cls._MODEL used as default = 

        Returns
        -------
        instance
        
        See also
        --------
        from_draw: loads the instance from a random draw of targets given the model
        """
        this = cls()

        if template is not None:
            this.set_template(template, rate_update=False)

        if model is not None:
            this.update_model(**model, rate_update=False) # will update any model entry.

        if template is not None and model is not None:
            this._update_rate_in_model_()

            
        this.set_data(data)
        return this
        
    @classmethod
    def from_draw(cls, size=None, model=None, template=None,
                      zmax=None, tstart=None, tstop=None,
                      zmin=0, nyears=None,
                      skyarea=None, rate=None,
                      effect=None,
                      **kwargs):
        """ loads the instance from a random draw of targets given the model 

        Parameters
        ----------
        size: int, None
            number of target you want to sample
            size=None as in numpy. Usually means 1.
            = ignored if nyears given =

        model: dict, None
            defines how  template parameters to draw and how they are connected
            model will update the default cls._MODEL if any
            = leave to None if unsure, cls._MODEL used as default = 

        template: str, None
            name of the template (sncosmo.Model(source)). 
            = leave to None if unsure, cls._TEMPLATE used as default =

        zmax: float
            maximum redshift to be simulated.

        zmin: float
            minimum redshift to be simulated.

        tstart: float, str
            starting time of the simulation
            - str, this enters Astropy.Time, e.g. '2020-08-24'
               and got converted into mjd
            - float: date mjd

        tstop: float, str
            ending time of the simulation
            (if tstart and nyears are both given, tstop will be
            overwritten by ``tstart+365.25*nyears``
            - str, this enters Astropy.Time, e.g. '2020-08-24'
               and got converted into mjd
            - float: date mjd

        nyears: float
            if given, nyears will set:
            - size: it will be the number of target expected up to zmax 
            in the given  number of years. 
            This uses get_rate(zmax).
            - tstop: tstart+365.25*nyears

        skyarea: None, string, geometry
            sky area to be considered.
            - str: full (equivalent to None), ['extra-galactic', not implemented yet]
            - geometry: shapely.Geometry
            - None: full sky 

        rate: float, func
            the transient rate
            - float: assumed volumetric rate
            - func: function of redshift rate(z) 
                    that provides the rate as a function of z

        **kwargs goes to self.update_model_parameter()

        Returns
        -------
        class instance
            self.data, self.model and self.template will be loaded.

        See also
        --------
        from_setting:  loads an instance given model parameters (dict)            
        """
        this = cls()

        # backward compatibility
        if template is None:
            if "source" in kwargs:
                warnings.warn("Deprecation warning: source option is now called template")
                template = kwargs.pop("source")
                
            if "source_or_template" in kwargs:
                template = kwargs.pop("source_or_template")
                    
        if template is not None:
            this.set_template(template)
            
        if rate is not None:
            this.set_rate(rate)
            
        if model is not None:
            this.update_model(**model, rate_update=False) # will update any model entry.

        if effect is not None:
            this.add_effect(effect) # may update the model entry.

        if kwargs:
            this.update_model_parameter(**kwargs, rate_update=False)

        # cleaning rate automatic feeding in model
        this._update_rate_in_model_()
        
        _ = this.draw( size=size,
                       zmin=zmin, zmax=zmax,
                       tstart=tstart, tstop=tstop,
                       nyears=nyears,
                       skyarea=skyarea,
                       inplace=True, # creates self.data
                       )
        return this
        
    # ------------- #
    #   Template    #
    # ------------- #
    def set_template(self, template, rate_update=False):
        """ set the template 

        = unlikely you want to set this directly =

        Parameters
        ----------
        template: str, `sncosmo.Source`, `sncosmo.Model` or skysurvey.Template
            This will reset self.template to the new template source.

        See also
        --------
        from_draw: load the instance by a random draw generation.
        from_setting: loads an instance given model parameters
        """
        from ..template import parse_template
        self._template = parse_template(template)
        if rate_update:
            warnings.warn("rate_update in set_template is not implemented. If you see this message, contact Mickael")

        
    def get_template(self, index=None, as_model=False, **kwargs):
        """ get a template (sncosmo.Model) 

        Parameters
        ----------
        index: int, None
            index of a target (see self.data.index) to set the 
            template parameters to that of the target.
            If None, the default sncosmo.Model parameters will be used.
            
        *kwargs goes to seld.template.get() and passed to sncosmo.Model

        Returns
        -------
        sncosmo.Model
            an instance of the template (a sncosmo.Model)

        See also
        --------
        get_target_template: get a template set to the target parameters.
        get_template_parameters: get the template parameters for the given target
        """
        if index is not None:
            prop = self.get_template_parameters(index).to_dict()
            kwargs = prop | kwargs

        sncosmo_model = self.template.get(**kwargs)
        if not as_model:
            from ..template import Template
            return Template.from_sncosmo(sncosmo_model)
        
        return sncosmo_model

    def get_target_template(self, index, **kwargs):
        """ get a template set to the target parameters.

        This is a shortcut to 
        ``get_template(index=index, **kwargs)``

        Parameters
        ----------
        index: int
            index of a target (see self.data.index) to set the 
            template parameters to that of the target.
            
        *kwargs goes to seld.template.get() and passed to sncosmo.Model

        Returns
        -------
        sncosmo.Model
            an instance of the template (a sncosmo.Model)

        See also
        --------
        get_template: get a template instance (sncosmo.Model)
        get_template_parameters: get the template parameters for the given target

        """
        return self.get_template(index=index, **kwargs)
    
    def get_target_flux(self, index, band, phase, zp=None, zpsys=None, restframe=True):
        """ Flux through the given bandpass(es) at the given time(s).

        Default return value is flux in photons / s / cm^2. If zp and zpsys
        are given, flux(es) are scaled to the requested zeropoints.

        Parameters
        ----------
        index:
            index of a target (see self.data.index) to set the 
            template parameters to that of the target.

        band : str or list_like
            Name(s) of Bandpass(es) in registry.

        phase : float or list_like
            phase in day

        zp : float or list_like, optional
            If given, zeropoint to scale flux to (must also supply ``zpsys``).
            If not given, flux is not scaled.

        zpsys : str or list_like, optional
            Name of a magnitude system in the registry, specifying the system
            that ``zp`` is in.

        restframe: bool
            is phase given in restframe ?

        Returns
        -------
        bandflux : float or `~numpy.ndarray`
            Flux in photons / s /cm^2, unless `zp` and `zpsys` are
            given, in which case flux is scaled so that it corresponds
            to the requested zeropoint. Return value is `float` if all
            input parameters are scalars, `~numpy.ndarray` otherwise.
            = sncosmo doc = 
        """
        sncosmo_model = self.get_target_template(index).sncosmo_model
        phase_obs = phase if not restframe else phase*(1+self.data.loc[index]["z"])
        return sncosmo_model.bandflux(band, sncosmo_model.get('t0')+phase_obs, zp=zp, zpsys=zpsys)

    def get_target_mag(self, index, band, phase, magsys="ab", restframe=True):
        """ magnitude through the given bandpass(es) at the given time(s).

        Default return value is flux in photons / s / cm^2. If zp and zpsys
        are given, flux(es) are scaled to the requested zeropoints.

        Parameters
        ----------
        index:
            index of a target (see self.data.index) to set the 
            template parameters to that of the target.

        band : str or list_like
            Name(s) of Bandpass(es) in registry.

        phase : float or list_like
            phase in day

        magsys : str or list_like
            Name(s) of `~sncosmo.MagSystem` in registry.
            
        restframe: bool
            is phase given in restframe ?
            
        Returns
        -------
        mag : float or `~numpy.ndarray`
            Magnitude for each item in time, band, magsys.
            The return value is a float if all parameters are not interables.
            The return value is an `~numpy.ndarray` if any are interable.
            = sncosmo doc = 
        """
        sncosmo_model = self.get_target_template(index).sncosmo_model
        phase_obs = phase if not restframe else phase*(1+self.data.loc[index]["z"])
        return sncosmo_model.bandmag(band=band, time=sncosmo_model.get('t0')+phase_obs, magsys=magsys)
        
    def clone_target_change_entry(self, index, name, values, as_dataframe=False):
        """ get a clone of the given target at the given redshifts.
        This: 
        (1) copies the index entries, 
        (2) sets the `name` to the input `values`
        (3) redraw the model starting from `name` (creating a new dataframe)
        (4, optional) sets a new instance with the updated dataframe
        

        Parameters
        ----------
        index: 
            index of a target (see self.data.index)
            
        name: str
            name of the entry to change

        values: list, array
            new values for this entry.

        as_dataframe: bool
            should this return the created new dataframe (True)
            or a new instance (False)

        Returns
        -------
        instance or DataFrame
        """
        dd = self.data.loc[index].to_frame().T
        dd.loc[index, name] = np.atleast_1d(values)
        dd = dd.explode(name)
#        dd[name] = dd[name].convert_dtypes()
        data = self.model.redraw_from(name, dd, incl_name=False)
        if as_dataframe:
            return data
        
        return self.__class__.from_data(data, model=self.model.model, template=self.template)
    
    # -------------- #
    #   Getter       #
    # -------------- #
    def get_template_parameters(self, index=None):
        """ get the template parameters for the given target 
        
        This method selects from self.data the parameters that actually
        are parameters of the template (and disregards the rest).


        Parameters
        ----------
        index: int, None
            index of a target (see self.data.index) to get the 
            template parameters from that target only.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            depending of index

        See also
        --------
        template_parameter: parameters of the template (sncosmo.Model) | argument
        get_template: get a template instance (sncosmo.Model)
        """
        known = self.get_template_columns()
        prop = self.data[known]
        if index is not None:
            return prop.loc[index]
        
        return prop

    def get_template_columns(self):
        """ get the data columns that are template parameters 
        
        Returns
        -------
        pandas.columns
        """
        return self.data.columns[np.in1d(self.data.columns, self.template_parameters)]


    # -------------- #
    #   Apply        #
    # -------------- #
    def apply_gaussian_noise(self, errmodel, data=None):
        """ apply gaussian noise to current entries.

        Parameters
        ----------
        errmodel: dict
            dict that will feed a ModelDAG. 
            format: {x: {func:, kwargs:{}}}. 
            this will draw x_err following the this formula and will update 
            x assuming x_true for the original x and x_err for the given x drawn here.
            you can refeer to the original x using '@x_true' in the func kwargs.

        data: None
            original dataframe to be noisified. If None self.data is used.

        Returns
        -------
        self or dataframe
            - self if data is None
            - dataframe otherwise.

        Example
        -------
        >>>python
        import skysurvey
        from scipy import stats
        errmodel = {"x1": {"func": stats.lognorm.rvs, "kwargs":{"s":0.6, "loc":0.001, "scale":0.15}},
                    "c": {"func": stats.lognorm.rvs, "kwargs":{"s":0.7, "loc":0.03, "scale":0.01}},
                    "magobs": {"func": stats.lognorm.rvs, "kwargs":{"s":0.9, "loc":0.03, "scale":0.01}},
                    }
        snia = skysurvey.SNeIa.from_draw(1000)
        noisy_data = snia.apply_gaussian_noise(errmodel, data=snia.data)
        >>>
        """
        from modeldag.tools import apply_gaussian_noise
        
        if data is None:
            data = self.data
            as_dataframe = False
        else:
            as_dataframe = True
            
        new_data = apply_gaussian_noise(errmodel, data=data)
        if as_dataframe:
            return new_data
        
        return self.__class__.from_data(new_data, model=self.model.model, template=self.template)
    
    # -------------- #
    #   Converts     #
    # -------------- #
    def magabs_to_magobs(self, z, magabs, cosmology=None):
        """ converts absolute magnitude into observed magnitude 
        given the (cosmological) redshift and a cosmology 

        Parameters
        ----------
        z: float, array
            cosmological redshift
            
        magabs: float, array
            absolute magnitude

        cosmology: astropy.Cosmology, None
            cosmology to use. If None given, this will use
            the cosmology from self.cosmology (Planck18 by default)

        Returns
        -------
        array
            array of observed magnitude (``distmod(z)+magabs``)
        """
        if cosmology is None:
            cosmology = self.cosmology

        return self._magabs_to_magobs(z, magabs, cosmology=cosmology)
    
    @staticmethod
    def _magabs_to_magobs(z, magabs, cosmology):
        """ converts absolute magnitude into observed magnitude 
        given the (cosmological) redshift and a cosmology 

        = internal method =

        Parameters
        ----------
        z: float, array
            cosmological redshift
            
        magabs: float, array
            absolute magnitude

        cosmology: astropy.Cosmology
            cosmology to use. If None given, this will use
            the cosmology from self.cosmology (Planck18 by default)

        Returns
        -------
        array
            array of observed magnitude (``distmod(z)+magabs``)
        
        """
        return cosmology.distmod(np.asarray(z, dtype="float32")).value + magabs

    # -------------- #
    #   Model        #
    # -------------- #
    def set_model(self, model, rate_update=True):
        """ set the target model 

        what template parameters to draw and how they are connected 

        = It is unlikely you need to use that directly. =

        Parameters
        ----------
        model: dict or ModelDAG,
            model that will be used to draw the Target parameter

        rate_update: bool
            should this check for rate options and feedin rate=self.rate ?

        Returns
        -------
        None

        See also
        --------
        from_setting: loads an instance given model parameters (dict)
        from_draw: loads and draw random data.
        """
        from modeldag import ModelDAG
        if type( model ) is dict:
            model = ModelDAG(model, self)
            
        self._model = model
        
        if rate_update:
            self._update_rate_in_model_()

    def set_data(self, data, incl_template=True):
        """ attach data  to this instance. 

        Parameters
        ----------
        data: pandas.DataFrame
            dataframe containing (at least) the template
            parameters

        incl_template: bool
            if data does not contain the template column
            should this add it ?

        Return
        ------
        None
        """
        if "template" not in data and incl_template:
            if self.template is None:
                templatename = "unknown"
            else:
                templatename = self.template_source.name
            data["template"] = templatename

        self._data = data
        
    def get_model(self, **kwargs):
        """ get a copy of the model (dict) 

        You can change the model you get (not the current model)
        using the kwargs. 

        Parameters
        ----------

        **kwargs can change the model entry parameters
            for istance, t0: {"low":0, "high":10}
            will update model["t0"]["param"] = ...

        Returns
        -------
        dict
           a copy of the model (with param potentially updated)
           
           
        See also
        --------
        update_model: change the current model (not just the one you get)
        get_model_parameter: access the model parameters.
        """
        return self.model.get_model(**kwargs)

    def get_model_parameter(self, entry, key, default=None, model=None):
        """ access a parameter of the model.

        Parameters
        ----------
        entry: str
            name of the variable as given by the model dict

        key: str
            name of the parameters

        default: 
            value returned if the parameter is not found.

        model: modelDAG
            get the parameter of this model instead of self.model
            = use with caution =

        Returns
        -------
        value of the entry parameter
        
        Example
        -------
        >>> self.get_model_parameter('redshift', 'zmax', None)

        """
        if model is None:
            model = self.model
            
        return model.model[entry]["kwargs"].get(key, default)

    def update_model_parameter(self, rate_update=True, **kwargs):
        """ change the kwargs entry of a model. """
        # use copy to avoid classmethod issues        
        for k, v in kwargs.items():
            self.model.model[k]["kwargs"] = self.model.model[k].get("kwargs",{}) | v

        if rate_update:
            self._update_rate_in_model_()
            
    def update_model(self, rate_update=True, **kwargs):
        """ Change the given entries of the model.

        **kwargs will update any model entry (or create a new one at the end).

        Example
        -------
        Changing the b entry function and make it depends on "a"
        >>> self.update_model( b={"func":np.random.normal, "kwargs":{"loc":"@a", "scale":1}})
        """
        new_model = self.model.model | kwargs
        _ = self.set_model(new_model, rate_update=rate_update)

    def _update_rate_in_model_(self, warn_if_more=1):
        """ """
        keys = self.model.get_func_with_args("rate")
        if len(keys)>warn_if_more:
            warnings.warn(f"more than {warn_if_more} entries have 'rate' in their options ({keys=})")
            
        self.update_model_parameter(**{k: {"rate": self.rate} for k in keys},
                                        rate_update=False)

    def add_effect(self, effect, model=None, data=None, overwrite=False):
        """ add an effect to the target affecting how spectra or lightcurve are generated
        
        This changes the template, using self.template.add_effect(), and changes the target's model
        if effect.model is set.

        Parameters
        ----------
        effect: dict, skysurvey.effect.Effect
            Effect that should be used to change the target.
            e.g. mw_ebv = skysurvey.effect.Effect.from_name('mw')
            these format are accepted:
            - dict: {effect: sncosmo.Effect, "name": str, "frame": str, (model: optionel)}
            - skysurvey.effect.Effect
            
        model: dict, None
            defines how the data will be drawn.
            this updates self.model

        data: pandas.DataFrame, None
            value that will be added to the data to capture the effect (if any).
            If data and model are given, model is not used.
            
        Returns
        -------
        None
        """
        if type(effect) is dict:
            from .. import Effect
            effect = Effect(**effect)

        # update the model
        if model is not None:
            effect._model = model
            
        if effect.model is not None:
            self.update_model(**effect.model, rate_update=False)

        # update the data
        if data is not None:
            if self.data is None:
                warnings.warn("no current data. cannot merge. input effect 'data' is ignored")
            else:
                new_data = self.data.merge(data, **kwargs)
                self.set_data(new_data)
            
        elif effect.model is not None and self.data is not None:
            # if not self.data, this will be drawn along with the data on time.
            keys_to_draw = list(effect.model.keys())
            if not overwrite and np.any([k in self.data for k in keys_to_draw]):
                warnings.warn(f"some or all of {keys_to_draw} are already in self.data. Set overwrite to True to overwrite them. Data unchanged.")
            else:
                new_data = self.model.redraw_from(keys_to_draw, self.data)
                self.set_data(new_data)

        # update the template from this effect
        _ = self.template.add_effect(effect)
        
    # -------------- #
    #   Plotter      #
    # -------------- #
    def show_scatter(self, xkey, ykey, ckey=None, ax=None, fig=None, 
                         index=None, data=None, colorbar=True,
                         bins=None, bcolor="0.6", err_suffix="_err",
                         **kwargs):
        """ """
        import matplotlib.pyplot as plt

        # ------- #
        #  Data   #
        # ------- #
        if data is None:
            data = self.data if index is None else self.data.loc[index]
            
        xvalue = data[xkey]
        yvalue = data[ykey]
        cvalue = None if ckey is None else data[ckey]

        # ------- #
        #  axis   #
        # ------- #
        if ax is None:
            if fig is None:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=[7,4])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        # scatter
        prop = {**dict(zorder=3), **kwargs}
        sc = ax.scatter(xvalue, yvalue, c=cvalue, **prop)
        # errorbar        
        if f"{xkey}{err_suffix}" in data or f"{ykey}{err_suffix}" in data:
            xerr = data.get(f"{xkey}{err_suffix}")
            yerr = data.get(f"{ykey}{err_suffix}")
            zorder = prop.pop("zorder") - 1
            _ = ax.errorbar(xvalue, yvalue, xerr=xerr, yerr=yerr,
                                ls="None", marker="None",
                                zorder=zorder, ecolor="0.7")
           
        if cvalue is not None and colorbar:
            fig.colorbar(sc, ax=ax)

        if bins is not None:
            from matplotlib.colors import to_rgba
            binned = pandas.cut(xvalue, bins) # defines the bins
            # Add them to a copy of the dataframe along with the y-data
            data_tmp = data[[ykey]].copy()
            data_tmp["xbins"] = binned
            # compute the binned mean, std and size (err_mean = std/sqrt(size-1)
            gbins = data_tmp.groupby("xbins")[ykey].agg(["mean","std", "size"]).reset_index()
            # get the bin centroid
            bincentroid = gbins["xbins"].apply(lambda x: x.mid)
            # and show the bins
            ax.errorbar(bincentroid.values, gbins["mean"], yerr=gbins["std"]/np.sqrt(gbins["size"]-1), 
                        ls="None", marker="s", mfc=to_rgba(bcolor, 0.8),
                        mec=bcolor, zorder=9, ms=7, ecolor=bcolor)
            
        return fig
            
    # =============== #
    #   Draw Methods  #
    # =============== #
    def draw(self, size=None,
                 z=None,
                 zmax=None, zmin=0,
                 tstart=None, tstop=None, nyears=None,
                 skyarea=None,
                 inplace=False,
                 model=None,
                 **kwargs):
        """ draws the parameter model (using self.model.draw()) 

        Parameters
        ----------
        size: int
            = ignored is nyears is not None =
            number of target you want to draw.

        zmax: float
            maximum redshift to be simulated.

        zmin: float
            minimum redshift to be simulated.

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
            This uses ``get_rate(zmax)``.
            - tstop: ``tstart+365.25*nyears``

        skyarea: None, string, geometry
            sky area to be considered.
            - str: full (equivalent to None), 'extra-galactic'
            - geometry: shapely.Geometry
            - None: full sky

        inplace: bool
            sets self.data to the newly drawn dataframe

        Returns
        -------
        DataFrame
            the simulated dataframe.

        """
        #
        # Drawn model
        # 
        if model is None:
            drawn_model = self.model # a modelDAG
        else:
            from modeldag import ModelDAG
            current_model_dict = self.model.model
            drawn_model = ModelDAG( current_model_dict | model, obj=self)
            
        # => tstart, tstop format
        if type(tstart) is str:
            tstart = time.Time(tstart).mjd
        elif type(tstart) is time.Time:
            tstart = tstart.mjd

        if type(tstop) is str:
            tstop = time.Time(tstop).mjd
        elif type(tstop) is time.Time:
            tstop = tstop.mjd
        
        # => nyears and times    
        if nyears is None and (tstart is not None and tstop is not None):
            nyears = (tstop-tstart)/365.25
                
        if nyears is not None and (tstart is not None and tstop is None):
            tstop = tstart + nyears*365.25

        if nyears is not None and (tstart is  None and tstop is not None):
            tstart = tstop - nyears*365.25
                
        if nyears is None and size is None:
            raise ValueError(" You must provide either nyears or size")
        
        if nyears is not None and size is not None:
            nyears = None # its job is done.

        #
        # Redshift
        #
        key_redshift = drawn_model.get_func_with_args("zmax")
        
        # zmax
        for zkey in key_redshift:
            if zmax is not None:
                kwargs.setdefault(zkey, {}).update({"zmax": zmax})
            
            elif nyears is not None:
                zmax = self.get_model_parameter(zkey, "zmax", None, model=drawn_model)
            
        # zmin
        key_redshift = drawn_model.get_func_with_args("zmin")
        for zkey in key_redshift:
            if zmin is not None and "redshift" in self.model.model:
                kwargs.setdefault(zkey, {}).update({"zmin": zmin})
            
            elif nyears is not None:
                zmin = self.get_model_parameter(zkey, "zmin", None, model=drawn_model)

        if tstop is not None:
            if type( tstop ) is str:
                tstop = time.Time(tstop).mjd

            kwargs.setdefault("t0", {}).update({"high": tstop})

        #
        # time range
        #
        if tstart is not None:
            if type( tstart ) is str:
                tstart = time.Time(tstart).mjd
                
            kwargs.setdefault("t0",{}).update({"low": tstart})
            if tstop is None and nyears is None: # do 1 year by default
                kwargs.setdefault("t0",{}).update({"high": tstart+365.25})
                
        # tstart is None, then what ?
        elif tstop is not None and nyears is not None:
            tstart = tstop - 365.25*nyears # fixed later
            
        elif nyears is not None:
            tstart = self.get_model_parameter("t0", "low", None, model=drawn_model)

        #
        # Sky area
        #
        skyarea = parse_skyarea(skyarea) # shapely.geometry or skyarea
        if skyarea is not None:
            param_affected = drawn_model.get_func_with_args("skyarea")
            if "radec" in drawn_model.model.keys() and "radec" not in param_affected:
                warnings.warn("radec in model, skyarea given, but the radec func does not accept skyarea.")
            if len(param_affected) ==0:
                warnings.warn("skyarea given but no model have skyarea as parameters. This is ignored.")
            
            for k in param_affected:
                kwargs.setdefault(k,{}).update({"skyarea": skyarea})
                

        #
        # Size
        #
        # skyarea affect get_rate
        if nyears is not None:
            rate_min = self.get_rate(zmin, skyarea=skyarea) if (zmin is not None and zmin >0) else 0
            kwargs.setdefault("t0",{}).update({"low": tstart, "high": tstart + 365.25*nyears})
            size = int( (self.get_rate(zmax, skyarea=skyarea)-rate_min) * nyears)
        
        # actually draw the data
        data = drawn_model.draw(size=size, **kwargs)

        if z is not None:
            if len(z) != size:
                raise ValueError('Length of redshift vector must be same as size')
            data['z'] = z

        # shall data be attached to the object?
        if inplace:
            # lower precision
            data = data.astype( {k: str(v).replace("64","32") for k, v in data.dtypes.to_dict().items()})
            self.set_data(data)
            # since this is inplace, let's update stored model kwargs
            self.update_model_parameter(**kwargs)
            
        return data

    # ============== #
    #   Properties   #
    # ============== #  
    @classproperty
    def kind(self):
        """ """
        if not hasattr(self,"_kind"):
            self._kind = self._KIND
            
        return self._kind
            
    @classproperty
    def cosmology(self):
        """ """
        return self._COSMOLOGY

    # model
    @property
    def model(self):
        """ modeling who the transient is generated """
        if not hasattr(self, "_model") or self._model is None:
            from copy import deepcopy
            self.set_model( deepcopy(self._MODEL) if self._MODEL is not None else {} )
            
        return self._model
    
    @property
    def data(self):
        """ data """
        if not hasattr(self,"_data"):
            return None
        return self._data

    # template
    @property
    def template(self):
        """ """
        if not hasattr(self,"_template") or self._template is None:
            self.set_template(self._TEMPLATE)
        return self._template

    @property
    def template_source(self):
        """ """
        return self.template.source

    @property
    def template_parameters(self):
        """ """
        return self.template.parameters
    
    @property
    def template_effect_parameters(self):
        """ """
        return self.template.effect_parameters  


    
class Transient( Target ):
    # - Transient
    _RATE = None    
    
    # ============== #
    #  Methods       #
    # ============== #
    # Rates    
    def set_rate(self, float_or_func):
        """ set the transient rate

        Parameters
        ----------
        float_or_func: float, func
            func: a function that takes as input an array or redshift "z"
            float: number of targets per Gpc3, then skysurvey.target.rates.get_volumetric_rate() is used.

        """
        if callable(float_or_func):
            self._rate = float_or_func
        else:
            self._rate = float(float_or_func)

    def draw_redshift(self, zmax, zmin=0, zstep=1e-4, size=None, rate=None, **kwargs):
        """ based on the rate (see get_rate()) """
        from .rates import draw_redshift
        if rate is None:
            rate = self.rate
            
        return draw_redshift(size=size, rate=rate, zmax=zmax, zmin=zmin, zstep=zstep, **kwargs)
    
    # ------- #
    #  GETTER #
    # ------- #
    def get_rate(self, z, skyarea=None, rate=None, **kwargs):
        """number of target (per year) up to the given redshift

        Parameters
        ----------
        z: float
            redshift
        
        skyarea: None, str, float, geometry
            sky area (in deg**2).
            - None or 'full': 4pi
            - "extra-galactic": 4pi - (milky-way b<5)
            - float: area in deg**2
            - geometry: shapely.geometry.area is used (assumed in deg**2)

        rate: float, func or None
            if None, self.rate is used.
            if float: assumed volumetric rate (target/Gpc3)
            if func: function as a function of rate ( rate(z, **kwargs) )

        **kwargs goes to the rate function (if a function, not a number)

        Returns
        -------
        int
        
        See also
        --------
        draw_redshift: draws redshifts from rate distribution.
        """
        from .rates import get_rate
        if rate is None:
            rate = self.rate
            
        return get_rate(z, skyarea=skyarea, rate=rate, **kwargs)
    
    def get_lightcurve(self, band, times,
                           sncosmo_model=None, index=None,
                           in_mag=False, zp=25, zpsys="ab",
                           **kwargs):
        """ the transient lightcurve 

        Parameters
        ----------
        band: str, list
            name of the band (should be known by sncosmo) or list of.

        times: float, list
            time of the observations
            
        Returns
        -------
        nd-array
            1 lightcurve per band.
        """
        # get the template            
        if index is not None:
            if sncosmo_model is None:
                sncosmo_model = self.get_template(index=index, as_model=True)
            else:
                prop = self.get_template_parameters(index).to_dict()
                kwargs = prop | kwargs
            
        return self.template.get_lightcurve(band, times,
                                            sncosmo_model=sncosmo_model,
                                            in_mag=in_mag, zp=zp, zpsys=zpsys,
                                            **kwargs)

    def get_spectrum(self, time, lbdas, as_phase=True,
                           sncosmo_model=None, index=None,
                           **kwargs):
        """ the transient spectrum at the given phase (time) 

        Parameters
        ----------
        time : float or list_like
            Time(s) in days. If `None` (default), the times corresponding
            to the native phases of the model are used.

        lbdas : float or list_like
            Wavelength(s) in Angstroms. If `None` (default), the native
            wavelengths of the model are used.
            
        as_phase: bool
            Is the given time a phase ? (as_phase=True) or a actual time (False)

        Returns
        -------
        flux : float or `~numpy.ndarray`
            Spectral flux density values in ergs / s / cm^2 / Angstrom.
        
        See also
        --------
        get_lightcurve: get the transient lightcurve 
        """
        prop = {}
        # get the template            
        if index is not None:
            if sncosmo_model is None:
                sncosmo_model = self.get_template(index=index, as_model=True)
            else:
                prop = self.get_template_parameters(index).to_dict()

        kwargs = prop | kwargs                
        return self.template.get_spectrum(time, lbdas,
                                          sncosmo_model=sncosmo_model,
                                          as_phase=as_phase,
                                          **kwargs)

    # ------------ #
    #  Model       #
    # ------------ #    
    def magobs_to_amplitude(self, magobs, band="bessellb", zpsys="ab", param_name="amplitude"):
        """ """
        template = self.get_template(as_model=True)
        m_current = template._source.peakmag(band, zpsys)
        return 10.**(0.4 * (m_current - magobs)) * template.get(param_name)

            
    # ------------ #
    #  Show LC     #
    # ------------ #
    def show_lightcurve(self, band, index, params=None,
                            ax=None, fig=None, colors=None,
                            phase_range=None, npoints=500,
                            zp=25, zpsys="ab",
                            format_time=True, t0_format="mjd", 
                            in_mag=False, invert_mag=True, **kwargs):
        """ 
        params: None or dict
        """
        # get the template
        if params is None:
            params = {}
            
        template = self.get_target_template(index, **params)
        return template.show_lightcurve(band, params=params,
                                             ax=ax, fig=fig, colors=colors,
                                             phase_range=phase_range, npoints=npoints,
                                             zp=zp, zpsys=zpsys,
                                             format_time=format_time,
                                             t0_format=t0_format, 
                                             in_mag=in_mag, invert_mag=invert_mag,
                                             **kwargs)
            
    # ============== #
    #   Properties   #
    # ============== #  
    # Rate
    @property
    def rate(self):
        """ rate.
        (If float, assumed to be volumetric rate in Gpc-3 / yr-1.)
        """
        if not hasattr(self,"_rate"):
            self.set_rate( self._RATE ) # default
            
        return self._rate
