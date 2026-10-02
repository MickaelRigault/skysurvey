"""Basic `Target` and `Transient` classes.

Pre-defined transients (like SNIa) inherit from `Transient`, and so should
any new transient you want to define.
"""

import warnings
import numpy as np
import pandas

from copy import deepcopy
from tqdm import tqdm
from astropy import cosmology, time
from astropy.utils.decorators import classproperty

from ..tools.utils import parse_skyarea


class Target( object ):
    """Base class for targets.

    This class provides a framework for representing astronomical targets,
    including their models, templates, and cosmological parameters.

    Attributes
    ----------
    _KIND : str
        The kind of target. The default is "unknow".

    _TEMPLATE : str, sncosmo.Source, sncosmo.Model or skysurvey.Template
        The template for the target. The default is None.

    _MODEL : dict
        The model for the target. The default is None.

    _MAGSYS : str
        The photometric magnitude system. The default is "ab".

    _PEAK_ABSMAG_BAND : str
        The bandpass in which the peak absolute magnitude is defined.
        The default is "bessellb".

    _AMPLITUDE_NAME : str
        The name of the parameter used to scale the model flux.
        The default is "amplitude".

    _COSMOLOGY : astropy.cosmology.Cosmology
        The cosmology to use. The default is `astropy.cosmology.Planck18`.

    See Also
    --------
    from_draw : Load an instance from a random draw of targets given the model.
    from_data : Load an instance given existing data.
    """

    _KIND = "unknow"
    _TEMPLATE = None
    _MODEL = None # dict config

    # Params to set peak amplitude
    _MAGSYS = "ab"
    _PEAK_ABSMAG_BAND = "bessellb"
    _AMPLITUDE_NAME = "amplitude"

    # - Cosmo
    _COSMOLOGY = cosmology.Planck18

    def __init__(self):
        pass

#    def __repr__(self):
#        """ String representation of the instance. """
#        return self.__str__()

#    def __str__(self):
#        """ String representation of the instance. """
#        return self.__class__

    @classmethod
    def from_setting(cls, setting, **kwargs):
        """Load the target from a setting dictionary.

        .. note::

            Not implemented yet.

        Parameters
        ----------
        setting : dict
            Dictionary containing the model parameters.

        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Target
            The loaded target.

        Raises
        ------
        NotImplementedError
            Always, as this method is not implemented yet.
        """
        raise NotImplementedError("from_setting is not Implemented ")


    @classmethod
    def from_data(cls, data, template=None, model=None, **kwargs):
        """Load the instance given existing data.

        This means that the model will be ignored as data will not be generated
        but input.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing (at least) the template parameters.

        template : str, sncosmo.Source, sncosmo.Model or skysurvey.Template, optional
            The template source. If a string is given, it is assumed to be a
            `sncosmo` model name. If None, `cls._TEMPLATE` is used.
            The default is None.

        model : dict, optional
            Defines how template parameters are drawn and how they are
            connected. The model will update the default `cls._MODEL` if any.
            If None, `cls._MODEL` is used. The default is None.

        **kwargs
            Subclass-specific options parsed by :meth:`_parse_init_kwargs_` and
            passed to the class constructor.

        Returns
        -------
        Target
            The loaded target.

        See Also
        --------
        from_draw : Load the instance from a random draw of targets given the model.
        """
        init_kwargs, kwargs = cls._parse_init_kwargs_(**kwargs)
        this = cls(**init_kwargs)

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
                      skyarea=None,
                      rate=None, rate_H0=None,
                      effect=None,
                      cosmology=None,
                      verbose=False,
                      set_amplitude=False,
                      **kwargs):
        """Load the instance from a random draw of targets given the model.

        Parameters
        ----------
        size : int, optional
            Number of targets to sample. Either `size` or `nyears` must be given.
            If both are given, `size` sets the number of targets.
            The default is None.

        model : dict, optional
            Defines how template parameters are drawn and how they are
            connected. The model will update the default `cls._MODEL` if any.
            If None, `cls._MODEL` is used. The default is None.

        template : str, sncosmo.Source, sncosmo.Model or skysurvey.Template, optional
            The template source. If a string is given, it is assumed to be a
            `sncosmo` model name. If None, `cls._TEMPLATE` is used.
            The default is None.

        zmax : float, optional
            Maximum redshift to be simulated. The default is None.

        tstart : float, str or astropy.time.Time, optional
            Starting time of the simulation. If a string or a Time is given, it is
            converted to mjd. The default is None.

        tstop : float, str or astropy.time.Time, optional
            Ending time of the simulation. If a string or a Time is given, it is
            converted to mjd. If `tstart` and `nyears` are both given,
            `tstop` will be overwritten by `tstart + 365.25 * nyears`.
            The default is None.

        zmin : float, optional
            Minimum redshift to be simulated. The default is 0.

        nyears : float, optional
            If given, `nyears` will set:

            - `size`: the number of targets expected up to `zmax` in the given
              number of years (ignored if `size` is given). This uses the rate.
            - `tstop`: `tstart + 365.25 * nyears`.

            The default is None.

        skyarea : None, str or shapely.geometry.Polygon, optional
            Sky area to be considered.

            - str: 'full' (equivalent to None); 'extra-galactic' is not
              implemented yet.
            - geometry: a shapely geometry.
            - None: full sky.

            The default is None.

        rate : float or callable, optional
            The transient rate. If a float is given, it is assumed to be the
            number of targets per Gpc3 per year. If a callable is given, it is
            supposed to be a function of z that returns the volumetric rate.
            If None, the class default rate is used. The default is None.

        rate_H0 : float, optional
            Hubble constant (in km/s/Mpc) assumed when deriving `rate`.
            The rate is rescaled by (H0 / rate_H0)**3 to match the H0 of
            the simulation cosmology. Ignored if `rate` is None.
            If None, `cls._RATE_H0` is used. The default is None.

        effect : dict or skysurvey.effect.Effect, optional
            Effect to add to the target (see :meth:`add_effect`).
            The default is None.

        cosmology : astropy.cosmology.Cosmology, optional
            The cosmology to be used. If None, `cls._COSMOLOGY` is used.
            The default is None.

        verbose : bool, optional
            If True, show a progress bar while setting the amplitudes.
            The default is False.

        set_amplitude : bool, optional
            Should the amplitude of the template be set at this stage?
            The default is False.

        **kwargs
            Subclass-specific options (e.g. `magabs` for `TSTransient`, see the
            class constructor) are parsed by :meth:`_parse_init_kwargs_` and passed
            to the constructor. The rest goes to :meth:`update_model_parameter`.

        Returns
        -------
        Target
            The loaded target with data, model and template loaded.

        See Also
        --------
        from_data : Load an instance given existing data.
        draw : Draw the parameter model.
        """
        init_kwargs, kwargs = cls._parse_init_kwargs_(**kwargs)
        this = cls(**init_kwargs)

        # backward compatibility
        if cosmology is not None:
            this.set_cosmology(cosmology)

        if template is not None:
            this.set_template(template)

        if rate is not None:
            this.set_rate(rate, H0=rate_H0)

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
                       verbose=verbose,
                       set_amplitude=set_amplitude
                       )
        return this

    @classmethod
    def _parse_init_kwargs_(cls, **kwargs):
        """Split kwargs between the constructor and the rest.

        This is a hook for subclasses to add specific kwargs to the constructor.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be split.

        Returns
        -------
        init_kwargs : dict
            Keyword arguments passed to the constructor (empty here).

        kwargs : dict
            Remaining keyword arguments.
        """
        # first kwargs/dict => init
        # second kwargs => rest (template)
        return {}, kwargs


    def set_cosmology(self, cosmology):
        """Set the cosmology to be used across the target.

        Parameters
        ----------
        cosmology : astropy.cosmology.Cosmology
            The cosmology to be used.
        """
        self._cosmology = cosmology

    # ------------- #
    #   Template    #
    # ------------- #
    def set_template(self, template, rate_update=False):
        """Set the template.

        .. note::

            It is unlikely you want to set this directly.

        Parameters
        ----------
        template : str, sncosmo.Source, sncosmo.Model or skysurvey.Template
            This will reset ``self.template`` to the new template source.

        rate_update : bool, optional
            Not implemented; a warning is raised if True. The default is False.

        See Also
        --------
        from_draw : Load the instance by a random draw generation.
        from_data : Load an instance given existing data.
        """
        from ..template import parse_template
        self._template = parse_template(template)
        if rate_update:
            warnings.warn("rate_update in set_template is not implemented. If you see this message, contact Mickael")


    def get_template(self, index=None, as_model=False, data=None, set_magabs=False, **kwargs):
        """Get a template.

        Parameters
        ----------
        index : int, optional
            Index of a target (see `self.data.index`) to set the template
            parameters to that of the target. If None, the default
            `sncosmo.Model` parameters are used. The default is None.

        as_model : bool, optional
            Should this return the `sncosmo.Model` (True) or the
            `skysurvey.Template` (False)? Note that
            `skysurvey.Template.sncosmo_model` gives the `sncosmo.Model`.
            The default is False.

        data : pandas.DataFrame, optional
            Data used to set the template parameters. Ignored if `index` is None.
            If None, `self.data` is used. The default is None.

        set_magabs : bool, optional
            Should the peak magnitude of the template be set to the target's
            `magabs`? Ignored if `index` is None. The default is False.

        **kwargs
            Goes to ``self.template.get()`` and passed to `sncosmo.Model`.

        Returns
        -------
        skysurvey.Template or sncosmo.Model
            An instance of the template (or its associated `sncosmo.Model`,
            see `as_model`).

        See Also
        --------
        get_target_template : Get a template set to the target parameters.
        get_template_parameters : Get the template parameters for the given target.
        """

        if data is None:
            data = self.data

        if index is not None:
            prop = self.get_template_parameters(index, data=data).to_dict()
            kwargs = prop | kwargs
            _ = kwargs.pop(self.amplitude_name, None)

        sncosmo_model = self.template.get(**kwargs)

        if index is not None and set_magabs:
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

    def get_target_template(self, index, as_model=False, **kwargs):
        """Get a template set to the target parameters.

        This is a shortcut to `get_template(index=index, **kwargs)`.

        Parameters
        ----------
        index : int
            Index of a target (see `self.data.index`) to set the template
            parameters to that of the target.

        as_model : bool, optional
            Should this return the `sncosmo.Model` (True) or the
            `skysurvey.Template` (False)? Note that
            `skysurvey.Template.sncosmo_model` gives the `sncosmo.Model`.
            The default is False.

        **kwargs
            Goes to :meth:`get_template`.

        Returns
        -------
        skysurvey.Template or sncosmo.Model
            An instance of the template (or its associated `sncosmo.Model`,
            see `as_model`).

        See Also
        --------
        get_template : Get a template instance.
        get_template_parameters : Get the template parameters for the given target.
        """
        return self.get_template(index=index, as_model=as_model, **kwargs)

    def get_target_flux(self, index, band, phase, zp=None, zpsys=None, restframe=True):
        """Flux through the given bandpass(es) at the given phase(s).

        Default return value is flux in photons / s / cm^2. If `zp` and `zpsys`
        are given, flux(es) are scaled to the requested zeropoints.

        Parameters
        ----------
        index : int
            Index of a target (see `self.data.index`) to set the template
            parameters to that of the target.

        band : str or array_like
            Name(s) of bandpass(es) in the registry.

        phase : float or array_like
            Phase in days.

        zp : float or array_like, optional
            If given, zeropoint to scale flux to (must also supply `zpsys`).
            If None, flux is not scaled. The default is None.

        zpsys : str or array_like, optional
            Name of a magnitude system in the registry, specifying the system
            that `zp` is in. The default is None.

        restframe : bool, optional
            Is phase given in restframe? The default is True.

        Returns
        -------
        float or numpy.ndarray
            Flux in photons / s / cm^2, unless `zp` and `zpsys` are given, in
            which case flux is scaled so that it corresponds to the requested
            zeropoint. Return value is a float if all input parameters are
            scalars, a numpy.ndarray otherwise.
        """
        sncosmo_model = self.get_target_template(index).sncosmo_model
        phase_obs = phase if not restframe else phase*(1+self.data.loc[index]["z"])
        return sncosmo_model.bandflux(band, sncosmo_model.get('t0')+phase_obs, zp=zp, zpsys=zpsys)

    def get_target_peakmag(self, index, band, magsys="ab"):
        """Peak magnitude through the given bandpass(es).

        Parameters
        ----------
        index : int
            Index of a target (see `self.data.index`) to set the template
            parameters to that of the target.

        band : str or array_like
            Name(s) of bandpass(es) in the registry.

        magsys : str or array_like, optional
            Name(s) of `sncosmo.MagSystem` in the registry. The default is "ab".

        Returns
        -------
        float or numpy.ndarray
            Magnitude at peak for the given band(s).
        """
        sncosmo_model = self.get_template(index=index, set_magabs=True, as_model=True)
        return sncosmo_model.bandmag(band, magsys, sncosmo_model.get("t0") + sncosmo_model._source.peakphase(band))

    def get_target_mag(self, index, band, phase, magsys="ab", restframe=True):
        """Magnitude through the given bandpass(es) at the given phase(s).

        Parameters
        ----------
        index : int
            Index of a target (see `self.data.index`) to set the template
            parameters to that of the target.

        band : str or array_like
            Name(s) of bandpass(es) in the registry.

        phase : float or array_like
            Phase in days.

        magsys : str or array_like, optional
            Name(s) of `sncosmo.MagSystem` in the registry. The default is "ab".

        restframe : bool, optional
            Is phase given in restframe? The default is True.

        Returns
        -------
        float or numpy.ndarray
            Magnitude for each item in phase, band, magsys. The return value is
            a float if no parameter is iterable, a numpy.ndarray otherwise.
        """
        sncosmo_model = self.get_template(index=index, set_magabs=True, as_model=True)
        phase_obs = phase if not restframe else phase*(1+self.data.loc[index]["z"])
        return sncosmo_model.bandmag(band=band, time=sncosmo_model.get('t0')+phase_obs, magsys=magsys)

    def clone_target_change_entry(self, index, name, values, as_dataframe=False):
        """Get a clone of the given target with new values for one entry.

        This:

        1. copies the index entries,
        2. sets the `name` to the input `values`,
        3. redraws the model starting from `name` (creating a new dataframe),
        4. (optional) sets a new instance with the updated dataframe.

        Parameters
        ----------
        index : int
            Index of a target (see `self.data.index`).

        name : str
            Name of the entry to change.

        values : array_like
            New values for this entry.

        as_dataframe : bool, optional
            Should this return the created new dataframe (True) or a new
            instance (False)? The default is False.

        Returns
        -------
        Target or pandas.DataFrame
            The cloned target or the new dataframe.
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
    def get_template_parameters(self, index=None, data=None):
        """Get the template parameters for the given target.

        This method selects from the data the parameters that actually are
        parameters of the template (and disregards the rest).

        Parameters
        ----------
        index : int, optional
            Index of a target (see ``self.data.index``) to get the template
            parameters from that target only. If None, all targets are returned.
            The default is None.

        data : pandas.DataFrame, optional
            Data to select the parameters from. If None, `self.data` is used.
            The default is None.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            The template parameters (a Series if `index` is given).

        See Also
        --------
        template_parameters : Parameters of the template.
        get_template : Get a template instance.
        """
        if data is None:
            data = self.data

        known = self.get_template_columns(data=data)
        prop = data[known]
        if index is not None:
            return prop.loc[index]

        return prop

    def get_template_columns(self, data=None):
        """Get the data columns that are template parameters.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to get the columns from. If None, `self.data` is used.
            The default is None.

        Returns
        -------
        pandas.Index
            The template columns.
        """
        if data is None:
            data = self.data
        return data.columns[np.isin(data.columns, self.template_parameters)]


    # -------------- #
    #   Apply        #
    # -------------- #
    def apply_gaussian_noise(self, errmodel, data=None):
        """Apply gaussian noise to current entries.

        Parameters
        ----------
        errmodel : dict
            Dict that will feed a `ModelDAG`. The format is
            `{x: {func:, kwargs:{}}}`. This will draw `x_err` following
            this formula and will update `x` assuming `x_true` for the
            original `x` and `x_err` for the given `x` drawn here. You can
            refer to the original `x` using `'@x_true'` in the func kwargs.

        data : pandas.DataFrame, optional
            Original dataframe to be noisified. If None, `self.data` is used.
            The default is None.

        Returns
        -------
        Target or pandas.DataFrame
            A new instance with the noisified data if `data` is None, the
            noisified dataframe otherwise.

        Examples
        --------
        >>> import skysurvey
        >>> from scipy import stats
        >>> errmodel = {"x1": {"func": stats.lognorm.rvs, "kwargs":{"s":0.6, "loc":0.001, "scale":0.15}},
        ...             "c": {"func": stats.lognorm.rvs, "kwargs":{"s":0.7, "loc":0.03, "scale":0.01}},
        ...             "magobs": {"func": stats.lognorm.rvs, "kwargs":{"s":0.9, "loc":0.03, "scale":0.01}},
        ...             }
        >>> snia = skysurvey.SNeIa.from_draw(1000)
        >>> snia = snia.apply_gaussian_noise(errmodel, data=snia.data)
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
        """Convert absolute magnitude into observed magnitude.

        This is done given the (cosmological) redshift and a cosmology.

        Parameters
        ----------
        z : float or array_like
            Cosmological redshift.

        magabs : float or array_like
            Absolute magnitude.

        cosmology : astropy.cosmology.Cosmology, optional
            The cosmology used to convert absolute to observed magnitude.
            If None, `self.cosmology` is used. *Careful* when specifying the
            cosmology: it should be self-consistent. The default is None.

        Returns
        -------
        float or numpy.ndarray
            Observed magnitude (`distmod(z) + magabs`).
        """
        if cosmology is None:
            cosmology = self.cosmology

        return self._magabs_to_magobs(z, magabs, cosmology=cosmology)

    @staticmethod
    def _magabs_to_magobs(z, magabs, cosmology):
        """Convert absolute magnitude into observed magnitude.

        This is an internal method.

        Parameters
        ----------
        z : float or array_like
            Cosmological redshift.

        magabs : float or array_like
            Absolute magnitude.

        cosmology : astropy.cosmology.Cosmology
            Cosmology to use.

        Returns
        -------
        float or numpy.ndarray
            Observed magnitude (`distmod(z) + magabs`).
        """
        return cosmology.distmod(np.asarray(z, dtype="float32")).value + magabs

    # -------------- #
    #   Model        #
    # -------------- #
    def set_model(self, model, rate_update=True):
        """Set the target model.

        The model defines what template parameters to draw and how they are
        connected.

        .. note::

            It is unlikely you need to use that directly.

        Parameters
        ----------
        model : dict or modeldag.ModelDAG
            Model that will be used to draw the `Target` parameters.

        rate_update : bool, optional
            Should this check for rate options and feed in `rate=self.rate`?
            The default is True.

        See Also
        --------
        from_draw : Load and draw random data.
        update_model : Change the given entries of the model.
        """
        from modeldag import ModelDAG
        if type( model ) is dict:
            model = ModelDAG(model, self)

        self._model = model

        if rate_update:
            self._update_rate_in_model_()

    def set_data(self, data, incl_template=True):
        """Attach data to this instance.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing (at least) the template parameters.

        incl_template : bool, optional
            If data does not contain the template column, should this add it?
            The default is True.
        """
        if "template" not in data and incl_template:
            if self.template is None:
                templatename = "unknown"
            else:
                templatename = self.template_source.name
            data["template"] = templatename

        self._data = data

    def get_model(self, **kwargs):
        """Get a copy of the model (dict).

        You can change the model you get (not the current model) using the
        kwargs.

        Parameters
        ----------
        **kwargs
            Change the model entry parameters. For instance,
            `t0={"low": 0, "high": 10}` will update
            `model["t0"]["param"] = ...`.

        Returns
        -------
        dict
            A copy of the model (with parameters potentially updated).

        See Also
        --------
        update_model : Change the current model (not just the one you get).
        get_model_parameter : Access the model parameters.
        """
        return self.model.get_model(**kwargs)

    def get_model_parameter(self, entry, key, default=None, model=None):
        """Access a parameter of the model.

        Parameters
        ----------
        entry : str
            Name of the variable as given by the model dict.

        key : str
            Name of the parameter.

        default : any, optional
            Value returned if the parameter is not found. The default is None.

        model : modeldag.ModelDAG, optional
            Get the parameter of this model instead of `self.model`.
            Use with caution. The default is None.

        Returns
        -------
        any
            Value of the entry parameter.

        Examples
        --------
        >>> self.get_model_parameter('redshift', 'zmax', None)
        """
        if model is None:
            model = self.model

        return model.model[entry]["kwargs"].get(key, default)

    def update_model_parameter(self, rate_update=True, **kwargs):
        """Change the kwargs entry of a model.

        Parameters
        ----------
        rate_update : bool, optional
            Should this check for rate options and feed in `rate=self.rate`?
            The default is True.

        **kwargs
            Model entry names and the dict of kwargs to update them with, e.g.
            `t0={"low": 0}`.
        """
        # use copy to avoid classmethod issues
        for k, v in kwargs.items():
            self.model.model[k]["kwargs"] = self.model.model[k].get("kwargs",{}) | v

        if rate_update:
            self._update_rate_in_model_()

    def update_model(self, rate_update=True, **kwargs):
        """Change the given entries of the model.

        Parameters
        ----------
        rate_update : bool, optional
            Should this check for rate options and feed in `rate=self.rate`?
            The default is True.

        **kwargs
            Will update any model entry (or create a new one at the end).

        Examples
        --------
        Changing the `b` entry function and make it depend on `a`:

        >>> rng = np.random.default_rng()
        >>> self.update_model(b={"func":rng.normal, "kwargs":{"loc":"@a", "scale":1}})
        """
        new_model = self.model.model | kwargs
        _ = self.set_model(new_model, rate_update=rate_update)

    def _update_rate_in_model_(self, warn_if_more=1):
        """Update the rate in the model.

        Parameters
        ----------
        warn_if_more : int, optional
            Warn if more than this number of model entries accept a `rate`
            argument. The default is 1.
        """
        keys = self.model.get_func_with_args("rate")
        if len(keys)>warn_if_more:
            warnings.warn(f"more than {warn_if_more} entries have 'rate' in their options ({keys=})")

        self.update_model_parameter(**{k: {"rate": self.rate} for k in keys},
                                        rate_update=False)

    def add_effect(self, effect, model=None, data=None, overwrite=False, **kwargs):
        """Add an effect affecting how spectra or lightcurves are generated.

        This changes the template, using ``self.template.add_effect()``, and
        changes the target's model if ``effect.model`` is set.

        Parameters
        ----------
        effect : dict or skysurvey.effect.Effect
            Effect that should be used to change the target,
            e.g. ``mw_ebv = skysurvey.effect.Effect.from_name('mw')``.
            These formats are accepted:

            - dict: ``{effect: sncosmo.Effect, "name": str, "frame": str, (model: optional)}``
            - ``skysurvey.effect.Effect``

        model : dict, optional
            Defines how the data will be drawn. This overrides the effect model
            and updates ``self.model``. The default is None.

        data : pandas.DataFrame, optional
            Values that will be merged to the data to capture the effect (if any).
            If given, the data are not redrawn from the model. The default is None.

        overwrite : bool, optional
            If the effect parameters are already in the data, should they be
            redrawn? Ignored if `data` is given. The default is False.

        **kwargs
            Goes to ``self.data.merge(data, **kwargs)`` if `data` is given.
            Ignored otherwise.
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
        """Show a scatter plot of the data.

        Parameters
        ----------
        xkey : str
            The key for the x-axis data.

        ykey : str
            The key for the y-axis data.

        ckey : str, optional
            The key for the color-axis data. The default is None.

        ax : matplotlib.axes.Axes, optional
            The axes on which to plot. If None, a new one is created.
            The default is None.

        fig : matplotlib.figure.Figure, optional
            The figure on which to plot. Ignored if `ax` is given. If None, a new
            one is created. The default is None.

        index : array_like, optional
            The index of the data to plot. Ignored if `data` is given.
            The default is None.

        data : pandas.DataFrame, optional
            The data to plot. If None, `self.data` is used. The default is None.

        colorbar : bool, optional
            Whether to show a colorbar (if `ckey` is given). The default is True.

        bins : int or array_like, optional
            The x-axis bins (see :func:`pandas.cut`) used to show the binned
            mean of the y-axis data. If None, no binning is shown.
            The default is None.

        bcolor : str, optional
            The color of the binned points. The default is "0.6".

        err_suffix : str, optional
            The suffix for the error columns. The default is "_err".

        **kwargs
            Additional keyword arguments to pass to `ax.scatter`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
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
                 zmax=None, zmin=0,
                 tstart=None, tstop=None, nyears=None,
                 skyarea=None,
                 inplace=False,
                 model=None,
                 verbose=False,
                 set_amplitude=False,
                 **kwargs):
        """Draw the parameter model (using ``self.model.draw()``).

        Parameters
        ----------
        size : int, optional
            Number of targets to draw. Either `size` or `nyears` must be given.
            If both are given, `size` sets the number of targets.
            The default is None.

        zmax : float, optional
            Maximum redshift to be simulated. The default is None.

        zmin : float, optional
            Minimum redshift to be simulated. The default is 0.

        tstart : float, str or astropy.time.Time, optional
            Starting time of the simulation. If a string or a Time is given, it is
            converted to mjd. The default is None.

        tstop : float, str or astropy.time.Time, optional
            Ending time of the simulation. If `tstart` and `nyears` are both
            given, `tstop` will be overwritten by `tstart + 365.25 * nyears`.
            The default is None.

        nyears : float, optional
            If given, `nyears` will set:

            - `size`: the number of targets expected up to `zmax` in the given
              number of years (ignored if `size` is given). This uses the rate.
            - `tstop`: `tstart + 365.25 * nyears`.

            The default is None.

        skyarea : None, str or shapely.geometry.Polygon, optional
            Sky area to be considered.

            - str: 'full' (equivalent to None); 'extra-galactic' is not
              implemented yet.
            - geometry: a shapely geometry.
            - None: full sky.

            The default is None.

        inplace : bool, optional
            Sets `self.data` to the newly drawn dataframe. The default is False.

        model : dict, optional
            Model entries that update (a copy of) the current model for this
            draw only. The default is None.

        verbose : bool, optional
            If True, show a progress bar while setting the amplitudes.
            The default is False.

        set_amplitude : bool, optional
            Should the template amplitude be computed and stored in the data?
            The default is False.

        **kwargs
            Model entry names and the dict of kwargs to update them with for this
            draw, passed to ``self.model.draw()``.

        Returns
        -------
        pandas.DataFrame
            The simulated dataframe.

        Raises
        ------
        ValueError
            If neither `size` nor `nyears` (or `tstart` and `tstop`) is given.
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
        # zmax
        # -> get forward entries that have 'zmax' as parameters
        key_redshift = drawn_model.get_func_with_args("zmax")
        for zkey in key_redshift:
            if zmax is not None:
                kwargs.setdefault(zkey, {}).update({"zmax": zmax})

            elif nyears is not None:
                zmax = self.get_model_parameter(zkey, "zmax", None, model=drawn_model)

        # zmin
        # -> get forward entries that have 'zmin' as parameters
        key_redshift = drawn_model.get_func_with_args("zmin")
        for zkey in key_redshift:
            # note: Why condition "on redshift" ?
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
                kwargs.setdefault(k, {}).update({"skyarea": skyarea})

        #
        # Size
        #
        # skyarea affect get_rate
        if nyears is not None:
            from .rates import get_ntargets
            from ..tools.projection import radecmodel_to_skysurface
            if "radec" in drawn_model.model.keys():
                # radec model
                radec_model = deepcopy(drawn_model.model["radec"])
                # as updated by requested kwargs
                radec_model["kwargs"] |= kwargs.get("radec", {})
                f_area = radecmodel_to_skysurface( radec_model )
            else:
                if skyarea is not None:
                    warnings.warn("skyarea given, but no radec not in model | *nyears* will not account for skyarea.")

                f_area = 1

            # redefine timing given nyears
            kwargs.setdefault("t0", {}).update({"low": tstart, "high": tstart + 365.25*nyears})

            if zmin is None:
                zmin = 0

            # get_ntargets is full sky. f_area corrects that.
            ntarget_per_year = get_ntargets(zmax, rate=self.rate,
                                            rate_H0=self._rateh0,
                                            zmin=zmin, zstep=1e-4, astype="float",
                                            cosmology=self.cosmology)
            size = int(ntarget_per_year * nyears * f_area)

        # actually draw the data
        data = drawn_model.draw(size=size, **kwargs)

        # patch the missing `amplitude` back to .data
        if set_amplitude:
            amplitudes = np.zeros( len(data) )
            for i in tqdm(range(size)) if verbose else range(size):
                sncosmo_model_i = self.get_template(index=i, as_model=True, data=data, set_magabs=True)
                amplitude = sncosmo_model_i.get(self.amplitude_name)
                amplitudes[i] = amplitude

            data[self.amplitude_name] = amplitudes

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
    def amplitude_name(self):
        """Name of the amplitude parameter."""
        if not hasattr(self, "_amplitude_name"):
            self._amplitude_name = self._AMPLITUDE_NAME
        return self._amplitude_name

    @classproperty
    def peak_absmag_band(self):
        """Band used to set the peak absolute magnitude."""
        if not hasattr(self, "_peak_absmag_band"):
            self._peak_absmag_band = self._PEAK_ABSMAG_BAND
        return self._peak_absmag_band

    @classproperty
    def magsys(self):
        """Magnitude system used for the peak absolute magnitude."""
        if not hasattr(self, "_magsys"):
            self._magsys = self._MAGSYS
        return self._magsys

    @classproperty
    def kind(self):
        """Kind of target."""
        if not hasattr(self,"_kind"):
            self._kind = self._KIND

        return self._kind

    @property
    def cosmology(self):
        """Cosmology used by the target."""
        if not hasattr(self, "_cosmology") or self._cosmology is None:
            self.set_cosmology( self._COSMOLOGY )

        return self._cosmology

    # model
    @property
    def model(self):
        """Model of the target."""
        if not hasattr(self, "_model") or self._model is None:
            from copy import deepcopy
            self.set_model( deepcopy(self._MODEL) if self._MODEL is not None else {} )

        return self._model

    @property
    def data(self):
        """Data of the target."""
        if not hasattr(self,"_data"):
            return None
        return self._data

    # template
    @property
    def template(self):
        """Template of the target."""
        if not hasattr(self,"_template") or self._template is None:
            self.set_template(self._TEMPLATE)
        return self._template

    @property
    def template_source(self):
        """Source of the template."""
        return self.template.source

    @property
    def template_parameters(self):
        """Parameters of the template."""
        return self.template.parameters

    @property
    def template_effect_parameters(self):
        """Effect parameters of the template."""
        return self.template.effect_parameters



class Transient( Target ):
    """A transient target.

    This class inherits from `Target` and adds a rate.

    Attributes
    ----------
    _RATE : float or callable
        The rate of the transient. The default is None.

    _RATE_H0 : float
        Hubble constant (in km/s/Mpc) assumed when deriving `_RATE`.
        The default is 70.
    """
    # - Transient
    _RATE = None
    _RATE_H0 = 70 # default H0 for rate definition included in skysurvey

    # ============== #
    #  Methods       #
    # ============== #
    # Rates
    def set_rate(self, float_or_func, H0=None):
        """Set the transient rate.

        Parameters
        ----------
        float_or_func : float or callable
            If a float is given, it is assumed to be the number of targets per
            Gpc3 per year. If a callable is given, it is supposed to be a function
            of z that returns the volumetric rate.

        H0 : float, optional
            Hubble constant (in km/s/Mpc) assumed when deriving the rate.
            If None, `self._RATE_H0` is used. The default is None.
        """
        if callable(float_or_func):
            self._rate = float_or_func
        else:
            self._rate = float(float_or_func)

        self._hrateh0 = float(H0) if H0 is not None else H0

    def draw_redshift(self, zmax, zmin=0, zstep=1e-4, size=None, **kwargs):
        """Draw redshifts based on the rate (see :meth:`get_rate`).

        This uses `self.rate`, rescaled from `self._rateh0` to the H0 of
        `self.cosmology`.

        Parameters
        ----------
        zmax : float
            Maximum redshift.

        zmin : float, optional
            Minimum redshift. The default is 0.

        zstep : float, optional
            Redshift step. The default is 1e-4.

        size : int, optional
            Number of redshifts to draw. The default is None.

        **kwargs
            Additional keyword arguments to pass to
            :func:`skysurvey.target.rates.draw_redshift`.

        Returns
        -------
        numpy.ndarray
            The drawn redshifts.
        """
        from .rates import draw_redshift
        return draw_redshift(size=size, rate=self.rate,
                            zmax=zmax, zmin=zmin, zstep=zstep,
                            rate_H0=self._rateh0,
                            cosmology=self.cosmology, **kwargs)

    # ------- #
    #  GETTER #
    # ------- #
    def get_rate(self, z, **kwargs):
        """Get the volumetric rate (per Gpc3 per year) at the given redshift.

        This uses `self.rate`, rescaled from `self._rateh0` to the H0 of
        `self.cosmology`.

        Parameters
        ----------
        z : float or array_like
            Redshift.

        **kwargs
            Goes to the rate function (if a function, not a number).

        Returns
        -------
        float or numpy.ndarray
            The volumetric rate.

        See Also
        --------
        draw_redshift : Draw redshifts from the rate distribution.
        """
        from .rates import get_rate
        return get_rate(z, rate=self.rate,
                        rate_H0=self._rateh0, H0=self.cosmology.H0.value,
                        **kwargs)

    def get_lightcurve(self, band, times,
                           sncosmo_model=None, index=None,
                           in_mag=False, zp=25, zpsys="ab",
                           **kwargs):
        """Get the transient lightcurve.

        Parameters
        ----------
        band : str or list of str
            Name of the band (should be known by sncosmo) or list of.

        times : float or array_like
            Time of the observations.

        sncosmo_model : sncosmo.Model, optional
            The sncosmo model to use. If None and `index` is given, the model is
            set to the target parameters. The default is None.

        index : int, optional
            The index of the target. If given together with `sncosmo_model`, the
            target template parameters are passed as kwargs. The default is None.

        in_mag : bool, optional
            If True, the lightcurve is returned in magnitude. The default is False.

        zp : float, optional
            The zeropoint to use. The default is 25.

        zpsys : str, optional
            The zeropoint system to use. The default is "ab".

        **kwargs
            Additional keyword arguments to pass to
            `self.template.get_lightcurve`.

        Returns
        -------
        numpy.ndarray
            One lightcurve per band.
        """
        # get the template
        if index is not None:
            if sncosmo_model is None:
                sncosmo_model = self.get_template(index=index, as_model=True, set_magabs=True)
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
        """Get the transient spectrum at the given phase (time).

        Parameters
        ----------
        time : float or array_like
            Time(s) in days. If None, the times corresponding to the native
            phases of the model are used.

        lbdas : float or array_like
            Wavelength(s) in Angstroms. If None, the native wavelengths of the
            model are used.

        as_phase : bool, optional
            Is the given time a phase (True) or an actual time (False)?
            The default is True.

        sncosmo_model : sncosmo.Model, optional
            The sncosmo model to use. If None and `index` is given, the model is
            set to the target parameters. The default is None.

        index : int, optional
            The index of the target. If given together with `sncosmo_model`, the
            target template parameters are passed as kwargs. The default is None.

        **kwargs
            Additional keyword arguments to pass to
            `self.template.get_spectrum`.

        Returns
        -------
        flux : float or numpy.ndarray
            Spectral flux density values in ergs / s / cm^2 / Angstrom.

        See Also
        --------
        get_lightcurve : Get the transient lightcurve.
        """
        prop = {}
        # get the template
        if index is not None:
            if sncosmo_model is None:
                sncosmo_model = self.get_template(index=index, as_model=True, set_magabs=True)
            else:
                prop = self.get_template_parameters(index).to_dict()

        kwargs = prop | kwargs
        return self.template.get_spectrum(time, lbdas,
                                          sncosmo_model=sncosmo_model,
                                          as_phase=as_phase,
                                          **kwargs)

    # ------------ #
    #  Show LC     #
    # ------------ #
    def show_lightcurve(self, band, index, params=None,
                            ax=None, fig=None, colors=None,
                            phase_range=None, npoints=500,
                            zp=25, zpsys="ab",
                            format_time=True, t0_format="mjd",
                            in_mag=False, invert_mag=True, **kwargs):
        """Show the lightcurve.

        Parameters
        ----------
        band : str or list of str
            The band(s) to show.

        index : int
            The index of the target.

        params : dict, optional
            Parameters passed to :meth:`get_target_template` and to
            ``template.show_lightcurve``. The default is None.

        ax : matplotlib.axes.Axes, optional
            The axes to show the lightcurve on. The default is None.

        fig : matplotlib.figure.Figure, optional
            The figure to show the lightcurve on. The default is None.

        colors : list, optional
            The colors to use for the lightcurve. The default is None.

        phase_range : list, optional
            The phase range to show. The default is None.

        npoints : int, optional
            The number of points to show. The default is 500.

        zp : float, optional
            The zero point to use. The default is 25.

        zpsys : str, optional
            The zero point system to use. The default is "ab".

        format_time : bool, optional
            Whether to format the time. The default is True.

        t0_format : str, optional
            The format of the time. The default is "mjd".

        in_mag : bool, optional
            Whether to show the magnitude. The default is False.

        invert_mag : bool, optional
            Whether to invert the magnitude axis. The default is True.

        **kwargs
            Additional keyword arguments to pass to ``template.show_lightcurve``.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        # get the template
        if params is None:
            params = {}

        template = self.get_target_template(index, set_magabs=True, **params)
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
        """Rate of the transient.

        If float, it is assumed to be the volumetric rate in Gpc-3 yr-1.
        If callable, it is supposed to be a function of z that returns the
        volumetric rate.
        """
        if not hasattr(self,"_rate"):
            self.set_rate( self._RATE, H0=self._RATE_H0 ) # default

        return self._rate

    @property
    def _rateh0(self,):
        """Hubble constant (in km/s/Mpc) assumed when deriving the rate.

        If not set by `set_rate()`, `self._RATE_H0` is used.
        """
        if not hasattr(self,"_hrateh0") or self._hrateh0 is None:
            if hasattr(self, "_RATE_H0"):
                self._hrateh0 = self._RATE_H0
            else:
                warnings.warn("No _RATE_H0 attribute found. Using 70 km/s/Mpc as default.")
                self._hrateh0 = 70.0

        return self._hrateh0
