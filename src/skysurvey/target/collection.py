"""Collection objects to group and operate on multiple targets at once."""

import pandas
import warnings
import numpy as np

from ..template import Template
from .timeserie import TSTransient
from .core import Target, Transient


def targets_from_collection(transientcollection):
    """Get targets from a transient collection.

    Parameters
    ----------
    transientcollection : TransientCollection
        Collection of transients.

    Raises
    ------
    NotImplementedError
        Always; this function is not implemented yet.
    """
    raise NotImplementedError


def broadcast_mapping(value, ntargets):
    """Broadcast a value to a given number of targets.

    Parameters
    ----------
    value : array_like or scalar
        Input value to broadcast. If the input has more than one
        dimension, broadcasting is applied along the first axis.

    ntargets : int
        Number of targets to broadcast the value to.

    Returns
    -------
    numpy.ndarray
        Broadcasted array of shape:

        - (ntargets,) if `value` is 1D or scalar.
        - (ntargets, N) if `value` is 2D or higher, where N is the
          size of the last dimension of `value`.
    """
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
    targets : list, optional
        A list of targets. The default is None.

    Attributes
    ----------
    _COLLECTION_OF : type
        The type of target in the collection. The default is `Target`.

    _TEMPLATES : list
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
        """Convert the collection into a list of same-template targets.

        Returns
        -------
        list
            One target (of type `_COLLECTION_OF`) per template, built from
            the corresponding subset of `data`.

        Raises
        ------
        AttributeError
            If `data` has no 'template' column.
        """
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
        """Call a method (or get an attribute) on each target in the collection.

        Parameters
        ----------
        which : str
            Name of the method or attribute to access on each target.

        margs : array_like, optional
            Per-target positional argument, broadcast to `ntargets` (see
            :func:`broadcast_mapping`). If given, ``target.which(marg, **kwargs)``
            is called for each target. The default is None.

        allow_call : bool, optional
            If True, callable attributes are called with `kwargs`; otherwise
            the attribute itself is returned. Ignored if `margs` is given.
            The default is True.

        **kwargs
            Passed to the called method.

        Returns
        -------
        list
            The result for each target.
        """
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
        """Set the targets in the collection.

        Parameters
        ----------
        targets : list or None
            A list of targets. If None, an empty list is set.
        """
        self._targets = np.atleast_1d(targets) if targets is not None else []

    def get_model_parameters(self, entry, key, default=None):
        """Get the model parameters for each target in the collection.

        Parameters
        ----------
        entry : str
            Name of the model entry.

        key : str
            Name of the parameter within the model entry.

        default : optional
            Value returned if the entry or key does not exist.
            The default is None.

        Returns
        -------
        list
            The model parameter of each target.
        """
        return self.call_down("get_model_parameter",
                              entry=entry, key=key, default=default)

    def get_data(self, keys="_KIND", colname="kind"):
        """Get a concatenated dataframe of the data from each target.

        Parameters
        ----------
        keys : str or list, optional
            Keys used to label each target's data in the concatenation. If a
            str, it is the name of the attribute fetched on each target (see
            :meth:`call_down`). If None, no labelling is applied.
            The default is "_KIND".

        colname : str, optional
            Name of the column storing the keys. If None, `keys` is used.
            Ignored if `keys` is None. The default is "kind".

        Returns
        -------
        pandas.DataFrame
            The concatenated data.
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

    def get_target_template(self, index, as_model=False, set_magabs=False):
        """Get the template for a given target.

        Parameters
        ----------
        index : int
            Index of a target (see `self.data.index`) to set the template
            parameters to that of the target.

        as_model : bool, optional
            Whether to return the `sncosmo.Model` (True) or the
            `skysurvey.Template` (False). For info, the `sncosmo.Model` is
            ``skysurvey.Template.sncosmo_model``. The default is False.

        set_magabs : bool, optional
            Whether to set the peak magnitude of the template to the
            target's `magabs`. The default is False.

        Returns
        -------
        skysurvey.Template or sncosmo.Model
            An instance of the template (or its associated `sncosmo.Model`,
            see `as_model`).
        """

        data_index = self.data.loc[index]
        template_name = data_index["template"]
        template_index = self.template_names.index(template_name)

        try:
            target = self.targets[template_index]
            target_template = target.template
            # TODO: Generalize. Currently not handling the edge case where we have a
            # collection of targets with the same template but different peak
            # magsys / rest-frame band.
            peak_absmag_magsys = target.magsys
            peak_absmag_band = target.peak_absmag_band
            amplitude_name = target.amplitude_name
            cosmology = target.cosmology

        except Exception as e:
            warning_string = (
                    f"Failed getting target template for index {index} with " +
                    f"name {template_name} and template index {template_index}. " +
                     "Exception on failure was: \n" +
                    f"{e}\n" +
                    "Attempting to load template from SNCosmo registry. " +
                    "THIS WILL IGNORE ANY MODEL EFFECTS YOU HAVE SET!"
                )
            warnings.warn(warning_string)
            target_template = Template.from_sncosmo(template_name)
            peak_absmag_magsys = "ab"
            peak_absmag_band = "bessellb"
            amplitude_name = "amplitude"
            cosmology = cosmology.Planck18


        param_mask = np.isin(data_index.index, target_template.parameters)
        target_params = data_index[param_mask].to_dict()
        _ = target_params.pop(amplitude_name, None)
        target_template.sncosmo_model.set(**target_params)

        if set_magabs:
            target_template.sncosmo_model.set_source_peakabsmag(
                absmag=data_index['magabs'],
                band=peak_absmag_band,
                magsys=peak_absmag_magsys,
                cosmo=cosmology
                )

        if as_model:
            output_template = target_template.sncosmo_model
        else:
            output_template = target_template

        return output_template


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
            Parameters to pass to :meth:`get_target_template` and to the
            template's ``show_lightcurve``. If None, ``{}`` is used.
            The default is None.

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
            Passed to the template's ``show_lightcurve``.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """

        if params is None:
            params = {}
        # get the template
        template = self.get_target_template(index, set_magabs=True, **params)
        return template.show_lightcurve(band, params=params,
                                             ax=ax, fig=fig, colors=colors,
                                             time_range=time_range, npoints=npoints,
                                             zp=zp, zpsys=zpsys,
                                             format_time=format_time,
                                             t0_format=t0_format,
                                             in_mag=in_mag, invert_mag=invert_mag,
                                             **kwargs)


    def to_transient(self, keys=None, **kwargs):
        """Convert the collection to a `Transient` object.

        Parameters
        ----------
        keys : str or list, optional
            Keys used to label each target's data (see :meth:`get_data`).
            The default is None.

        **kwargs
            Passed to :meth:`Transient.from_data`.

        Returns
        -------
        skysurvey.Transient
            A transient containing the concatenated data.
        """
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

    # @property
    # def magsys_targets(self):
    #     if not hasattr(self, "_magsys"):
    #         self._magsys = self.call_down("magsys")
    #     return self._magsys

    # @property
    # def peak_absmag_band(self):
    #     if not hasattr(self, "_peak_absmag_band"):
    #         self._peak_absmag_band = self.call_down("peak_absmag_band")
    #     return self._peak_absmag_band

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

    @property
    def template_names(self):
        """The source names of the targets' templates."""
        if not hasattr(self, "_template_names") or self._template_names is None:
            self._template_names = [
                target.template.source.name for target in self.targets
            ]
        return self._template_names

class TransientCollection( TargetCollection ):
    """A collection of transients.

    Parameters
    ----------
    targets : list, optional
        A list of targets. The default is None.

    Attributes
    ----------
    _COLLECTION_OF : type
        The type of transient in the collection. The default is `Transient`.
    """
    _COLLECTION_OF = Transient
    # ============= #
    #  Methods      #
    # ============= #
    def set_rates(self, float_or_func, H0=None):
        """Call `set_rate` for each target in the collection.

        Parameters
        ----------
        float_or_func : float or callable
            If a float is given, it is assumed to be the number of targets per
            Gpc3. If a callable is given, it is supposed to be a function of z
            that returns the volumetric rate as a function of redshift.

        H0 : float, optional
            Hubble constant (in km/s/Mpc) assumed when deriving the rate.
            If None, each target's `_RATE_H0` is used. The default is None.
        """
        _ = self.call_down("set_rate", float_or_func, H0=H0)

    def update_model(self, rate_update=True, **kwargs):
        """Call `update_model` for each target in the collection.

        Parameters
        ----------
        rate_update : bool, optional
            Whether to update the rate entry of each model.
            The default is True.

        **kwargs
            Passed to each target's `update_model`.
        """
        _ = self.call_down("update_model", rate_update=rate_update, **kwargs)

    def get_rates(self, z, relative=False, **kwargs):
        """Get the rates for each target in the collection.

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which the rates are evaluated; broadcast to the
            number of targets (see :func:`broadcast_mapping`).

        relative : bool, optional
            If True, rates are normalized to sum to one. The default is False.

        **kwargs
            Passed to each target's `get_rate`.

        Returns
        -------
        list or numpy.ndarray
            The rate of each target.
        """
        rates = self.call_down("get_rate", margs=z, **kwargs)
        if relative:
            rates /= np.nansum(rates)

        return rates

    def draw(self, size=None,
                 zmin=None, zmax=None,
                 tstart=None, tstop=None,
                 nyears=None,
                 inplace=True, shuffle=True,
                 rng=None,
                 **kwargs):
        """Draw the transients in the collection.

        Parameters
        ----------
        size : int, optional
            Total number of targets to draw. If given, the number of targets
            per template is randomly drawn following the relative rates
            (evaluated at z=0.1). If None, each target's `draw` default is
            used. The default is None.

        zmin, zmax : float, optional
            Minimum and maximum redshift to be simulated. The default is None.

        tstart, tstop : float or str, optional
            Starting and ending time of the simulation. The default is None.

        nyears : float, optional
            Number of years of simulation (see each target's `draw`).
            The default is None.

        inplace : bool, optional
            Whether to store the drawn data as the collection's `data`.
            The default is True.

        shuffle : bool, optional
            Whether to shuffle the rows of the output data.
            The default is True.

        rng : None, int, or numpy.random.Generator, optional
            Seed for the random number generator used to split `size` among
            templates; ignored if `size` is None. (Doc adapted from
            :func:`numpy.random.default_rng`.) If None, an unpredictable
            entropy will be pulled from the OS. If an int (>0), it will set
            the initial `BitGenerator` state. If a `(Bit)Generator`, it will
            be returned as a `Generator` unaltered. The default is None.

        **kwargs
            Passed to each target's `draw`.

        Returns
        -------
        pandas.DataFrame
            The drawn data, with a 'template' column.
        """
        if size is not None:
            relat_rate = np.asarray( self.get_rates(0.1, relative=True) ).reshape(self.ntargets)
            rng = np.random.default_rng(rng)
            templates = rng.choice( np.arange( self.ntargets ), size=size,
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
    targets : list, optional
        A list of targets. The default is None.

    Attributes
    ----------
    _COLLECTION_OF : type
        The type of transient in the collection. The default is `Transient`.

    _KIND : str
        The kind of transient. The default is "unknown".

    _RATE : float
        The rate of the transient. The default is 1e5.

    _RATE_H0 : float
        Hubble constant (in km/s/Mpc) assumed when deriving `_RATE`.
        The default is 70.

    _MAGABS : tuple
        The absolute magnitude of the transient. The default is (-18, 1).
    """
    _COLLECTION_OF = Transient

    _KIND = "unknown"
    _RATE = 1e5 # this assumes H0=70 | see Transient._RATE_H0
    _RATE_H0 = 70
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
                   rate=None, rate_H0=None, effect=None,
                   **kwargs):
        """Load the instance from a random draw of targets given the model.

        Parameters
        ----------
        size : int, optional
            Number of targets you want to sample. If None, 1 is assumed.
            Ignored if `nyears` is given. The default is None.

        model : dict, optional
            Defines how template parameters are drawn and how they are
            connected. It updates the model of each target (see
            `update_model`). The default is None.

        templates : list of str, optional
            Names of the templates (`sncosmo.Model(source)`). If None,
            `cls._TEMPLATES` is used. The default is None.

        zmax : float, optional
            Maximum redshift to be simulated. The default is None.

        tstart : float or str, optional
            Starting time of the simulation. If a string is given, it is
            converted to mjd. The default is None.

        tstop : float or str, optional
            Ending time of the simulation. If a string is given, it is
            converted to mjd. If `tstart` and `nyears` are both given,
            `tstop` will be overwritten by ``tstart + 365.25 * nyears``.
            The default is None.

        zmin : float, optional
            Minimum redshift to be simulated. The default is 0.

        nyears : float, optional
            If given, `nyears` will set:

            - `size`: it will be the number of targets expected up to `zmax`
              in the given number of years. This uses `get_rate(zmax)`.
            - `tstop`: ``tstart + 365.25 * nyears``.

            The default is None.

        skyarea : None, str, or shapely.geometry.Polygon, optional
            Sky area to be considered.

            - str: 'full' (equivalent to None), ['extra-galactic', not
              implemented yet].
            - geometry: shapely geometry.
            - None: full sky.

            The default is None.

        rate : float or callable, optional
            If a float is given, it is assumed to be the number of targets per
            Gpc3. If a callable is given, it is supposed to be a function of z
            that returns the volumetric rate as a function of redshift.
            The default is None.

        rate_H0 : float, optional
            Hubble constant (in km/s/Mpc) assumed when deriving `rate`.
            Ignored if `rate` is None. If None, each target's `_RATE_H0`
            is used. The default is None.

        effect : skysurvey.Effect, optional
            Effect added to each target (see `add_effect`).
            The default is None.

        **kwargs
            Passed to `update_model_parameter` (with ``rate_update=False``).

        Returns
        -------
        CompositeTransient
            The loaded instance.

        See Also
        --------
        from_setting : Load an instance given model parameters (dict).
        """
        this = cls()

        if rate is not None:
            this.set_rates(rate, H0=rate_H0) # this uses call_down('set_rate')

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
            self.set_rates( self._RATE, H0=self._RATE_H0) # default
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
    targets : list, optional
        A list of targets. The default is None.

    Attributes
    ----------
    _COLLECTION_OF : type
        The type of transient in the collection. The default is `TSTransient`.
    """
    _COLLECTION_OF = TSTransient

    @classmethod
    def from_draw(cls, sources, size=None, nyears=None,
                      rates=1e3, magabs=None, magscatter=None,
                      **kwargs):
        """Load the instance from a random draw of targets given the model.

        Parameters
        ----------
        sources : list
            List of sncosmo sources (or source names).

        size : int, optional
            Total number of targets to draw (see :meth:`draw`).
            The default is None.

        nyears : float, optional
            Number of years of simulation (see :meth:`draw`).
            The default is None.

        rates : float or array_like, optional
            Volumetric rate(s) of the transients, broadcast to the number of
            sources. The default is 1e3.

        magabs : float or array_like, optional
            Mean absolute magnitude(s) (``magabs`` ``loc``), broadcast to
            the number of sources. The default is None.

        magscatter : float or array_like, optional
            Absolute magnitude scatter(s) (``magabs`` ``scale``), broadcast
            to the number of sources. The default is None.

        **kwargs
            Passed to :meth:`draw`.

        Returns
        -------
        TSTransientCollection
            The loaded instance.
        """
        this = cls.from_sncosmo(sources, rates=rates,
                                        magabs=magabs,
                                        magscatter=magscatter)
        _ = this.draw(size=size, nyears=nyears, inplace=True,
                      **kwargs)
        return this

    @classmethod
    def from_sncosmo(cls, sources, rates=1e3,
                        magabs=None, magscatter=None):
        """Load the instance from a list of sources (and relative rates).

        Parameters
        ----------
        sources : list
            List of sncosmo sources (or source names).

        rates : float or array_like, optional
            Volumetric rate(s) of the transients, broadcast to the number of
            sources. The default is 1e3.

        magabs : float or array_like, optional
            Mean absolute magnitude(s) (``magabs`` ``loc``), broadcast to
            the number of sources. The default is None.

        magscatter : float or array_like, optional
            Absolute magnitude scatter(s) (``magabs`` ``scale``), broadcast
            to the number of sources. The default is None.

        Returns
        -------
        TSTransientCollection
            The loaded instance.
        """
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
