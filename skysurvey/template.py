""" core template tools """

import os
import numpy as np
import pandas
import sncosmo

from astropy.utils.decorators import classproperty

__all__ = ["get_sncosmo_model", "Template"]

def get_sncosmo_model(source="salt2", zero_before=True,
                      **params):
    """ get the template (sncosmo.Model)

    Parameters
    ----------
    source : `~sncosmo.Source` or str
        The model for the spectral evolution of the source. If a string
        is given, it is used to retrieve a `~sncosmo.Source` from
        the registry.

    **kwargs goes to model.set() to change the parameter's model

    Returns
    -------
    `sncosmo.Model`
        the sncosmo.Model template
    """
    modelprop = dict(source=source)
    model = sncosmo.Model(**modelprop)
    model.set(**params)
    if zero_before: # strange sncosmo feature. Hard to change afterward.
        model.source._zero_before = True
    
    return model

def sncosmoresult_to_pandas(result):
    """ takes a sncosmo.Results (lc fit output) and converts it in pandas's objects.

    Parameters
    ----------
    result: sncosmo.Result
        output of sncosmo's lightcurve fit function.

    Returns
    -------
    pandas.DataFrame, pandas.Series
        results (value, errors, covariances) and metadata (chi2, etc.)
    """
    error = pandas.Series( dict(result.get("errors") ), name="error")
    values = pandas.Series( result.get("parameters"),
                                index=result.get("param_names"),
                                name="value")

    cov = pandas.DataFrame( result.get("covariance"),
                                index=error.index, columns="cov_"+error.index)

    fit_res = pandas.concat( [values,error, cov], axis=1)
    fit_meta = pandas.Series( {k:result.get(k) for k in ["success", "ncall", "chisq", "ndof"]} )
    fit_meta["chi2dof"] = fit_meta["chisq"]/fit_meta["ndof"]
    return fit_res, fit_meta

def parse_template(template):
    """ read template or source """
    template = np.atleast_1d(template)
    if len(template)>1:
        return [parse_template(template_) for template_ in template]

    # ok, let's go then.
    template = template[0]
    import sncosmo
    # you provided a sncosmo.model.
    if type(template) is sncosmo.models.Model: 
        template = Template.from_sncosmo(template) # let's build a skysurvey.Template

     # you provided a source | do the same
    elif sncosmo.Source in template.__class__.__mro__ or type(template) in [str, np.str_]:
        template = Template.from_sncosmo(template) # let's build a skysurvey.Template

    # you provided a skysurvey.Template
    else:
        pass
    
    return template
# =============== #
#                 #
#  Template       #
#                 #
# =============== #
class Template( object ):

    def __init__(self, sncosmo_model):
        """Initialize the Template class.

        Parameters
        ----------
        sncosmo_model: sncosmo.Model
            The sncosmo model.
        """
        self._sncosmo_model = sncosmo_model

    @classmethod
    def from_sncosmo(cls, source, **kwargs):
        """
        loads the instance given the source name.

        Parameters
        ----------
        source : `~sncosmo.Source` or str
            The model for the spectral evolution of the source. If a string
            is given, it is used to retrieve a `~sncosmo.Source` from
            the registry.


        **kwargs goes to ``get_sncosmo_model(source, **kwargs)``

        Returns
        -------
        instance
        """
        if type(source) != sncosmo.Model:
            sncosmo_model = get_sncosmo_model(source, **kwargs)
        else:
            sncosmo_model = source

        return cls(sncosmo_model)

    # ============== #
    #   Methods      #
    # ============== #
    # -------- #
    #  Effect  #
    # -------- #
    def add_effect(self, effects):
        """ add effects to the sncosmo model

        Parameters
        ----------
        effects: `skysurvey.effect`, list
            effect to be applied to the sncosmo model
            It could be a list of effects.
            (see skysurvey.effect.from_sncosmo, if you have a sncosmo.PropagationEffect, name and frame)

        Returns
        -------
        None
        """
        for eff_ in np.atleast_1d(effects):
            self.sncosmo_model.add_effect(eff_.effect, eff_.name, eff_.frame)

    # -------- #
    #  GETTER  #
    # -------- #
    def get(self, **kwargs):
        """ return a copy of the model.
        You can set new parameter to this copy using kwargs

        Returns
        -------
        sncosmo.Model
        """
        from copy import deepcopy
        model = deepcopy(self.sncosmo_model)
        if kwargs:
            model.set(**kwargs)

        return model

    def get_lightcurve(self, band, times,
                           sncosmo_model=None, in_mag=False, zp=25, zpsys="ab",
                           **kwargs):
        """Get the lightcurves (flux or mag).

        Parameters
        ----------
        band: str or list
            Band or list of bands.
        times: array
            Array of times.
        sncosmo_model: sncosmo.Model, optional
            The sncosmo model to use. If None, the instance's model is used.
        in_mag: bool, optional
            If True, return the lightcurve in magnitude.
        zp: float, optional
            Zero point for the flux.
        zpsys: str, optional
            Zero point system.
        **kwargs
            Goes to self.get() to set the model parameters.

        Returns
        -------
        array
            The lightcurve values.
        """

        if sncosmo_model is None:
            sncosmo_model = self.get(**kwargs)

        # patch for odd sncosmo behavior (see https://github.com/sncosmo/sncosmo/issues/346)
        squeeze = type(band) in [str, np.str_] # for the output format

        # make sure all are arrays
        band_ = np.atleast_1d(band)
        times_ = np.atleast_1d(times)
        # flatten for bandflux
        band_ = np.hstack([np.resize(band_, len(times)) for band_ in band])
        times_ = np.resize(times, len(band)*len(times))

        # in flux
        if not in_mag:
            values = sncosmo_model.bandflux(band_, times_, zp=zp, zpsys=zpsys
                                            ).reshape( len(band),len(times) )
        # in mag
        else:
            values = sncosmo_model.bandmag(band_, zpsys, times_).reshape( len(band), len(times) )

        return np.squeeze(values) if squeeze else values

    def get_spectrum(self, time, lbdas, sncosmo_model=None, as_phase=True, **kwargs):
        """ get the spectrum at phase (time) for the given wavelength

        (based in sncosmo_model.flux(time, lbda)

        Parameters
        ----------
        time: float or list_like
            Time(s) in days. If `None` (default), the times corresponding
            to the native phases of the model are used.

        lbdas: float or list_like
            Wavelength(s) in Angstroms. If `None` (default), the native
            wavelengths of the model are used.

        as_phase: bool
            Is the given time a phase ? (as_phase=True) or a actual time (False)
            If phase, it is multiplied by (1+z) to be in restframe

        Returns
        -------
        flux : float or `~numpy.ndarray`
            Spectral flux density values in ergs / s / cm^2 / Angstrom.

        See also
        --------
        get_lightcurve: get the transient lightcurve
        """
        if sncosmo_model is None:
            sncosmo_model = self.get(**kwargs)


        wmin, wmax = sncosmo_model.minwave(), sncosmo_model.maxwave()
        lbdas = np.atleast_1d(lbdas)
        sel = (lbdas > wmin) & (lbdas < wmax)
        if not np.any(sel):
            warnings.warn("no wavelength matched [def range {wmin}, {wmax}], given. {lbdas} ")

        flux = np.zeros_like(lbdas)
        if as_phase:
            time *= (1+sncosmo_model.get("z"))
            time += sncosmo_model.get("t0")


        flux[sel] = sncosmo_model.flux(time, lbdas[sel])
        return flux

    # -------- #
    # Plotter  #
    # -------- #
    def show_spectum(self, time, lbdas, params={},
                         ax=None, fig=None, **kwargs):
        """Show the spectrum at a given time.

        Parameters
        ----------
        time: float
            Time (in phase).
        lbdas: array
            Wavelengths.
        params: dict, optional
            Parameters for the model.
        ax: matplotlib.axes, optional
            The axes to plot on.
        fig: matplotlib.figure, optional
            The figure to plot on.
        **kwargs
            Goes to ax.plot().

        Returns
        -------
        matplotlib.figure
        """
        spec = self.get_spectrum(time, lbdas, **params)

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

        ax.plot(lbdas, spec, **kwargs)

    def show_lightcurve(self, band, params=None,
                            ax=None, fig=None, colors=None,
                            phase_range=None, npoints=500,
                            zp=25, zpsys="ab",
                            format_time=True, t0_format="mjd",
                            in_mag=False, invert_mag=True, **kwargs):
        """Show the lightcurve.

        Parameters
        ----------
        band: str or list
            Band or list of bands.
        params: dict, optional
            Parameters for the model.
        ax: matplotlib.axes, optional
            The axes to plot on.
        fig: matplotlib.figure, optional
            The figure to plot on.
        colors: list, optional
            List of colors for the bands.
        phase_range: list, optional
            Phase range to plot.
        npoints: int, optional
            Number of points to plot.
        zp: float, optional
            Zero point for the flux.
        zpsys: str, optional
            Zero point system.
        format_time: bool, optional
            If True, format the time axis.
        t0_format: str, optional
            Format of the t0.
        in_mag: bool, optional
            If True, plot in magnitude.
        invert_mag: bool, optional
            If True, invert the magnitude axis.
        **kwargs
            Goes to ax.plot().

        Returns
        -------
        matplotlib.figure
        """
        from .config import get_band_color
        # get the sncosmo_model
        if params is None:
            params = {}

        sncosmo_model = self.get(**params)

        # ------- #
        #  x-data #
        # ------- #
        # time range
        t0 = sncosmo_model.get("t0")
        if phase_range is not None:
            phase_range = np.asarray(phase_range)
        else:
            phase_range = sncosmo_model.mintime()-t0, np.min([sncosmo_model.maxtime()-t0, 200])

        times = np.linspace(*np.asarray(phase_range)+t0, npoints)

        # ------- #
        #  y-data #
        # ------- #
        # flux
        band = np.atleast_1d(band)
        values = self.get_lightcurve(band,
                                     times, in_mag=in_mag,
                                     zp=zp, zpsys=zpsys,
                                     sncosmo_model=sncosmo_model)

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

        # ------- #
        #  Plot   #
        # ------- #
        # The plot
        if format_time:
            from astropy.time import Time
            times = Time(times, format=t0_format).datetime

        colors = np.resize(colors, len(values))
        for band_, value_, color_ in zip(band, values, colors):
            if color_ is None: # default back to config color
                color_ = get_band_color(band_)

            ax.plot(times, value_, color=color_, **kwargs)

        # ------- #
        #  Format #
        # ------- #
        # mag upside down
        if in_mag and invert_mag:
            ax.invert_yaxis()
        # time format
        if format_time:
            from matplotlib import dates as mdates
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        else:
            ax.set_xlabel("time [in day]", fontsize="large")

        if in_mag:
            ax.set_ylabel(f"Magnitude", fontsize="large")
        elif zp is None:
            ax.set_ylabel(f"Flux [erg/s/cm^2/A]", fontsize="large")
        else:
            ax.set_ylabel(f"Flux [zp={zp}]", fontsize="large")

        return fig

    # -------- #
    #  FITTER  #
    # -------- #
    def fit_data(self, data,
                     guessparams=None,
                     fixedparams=None,
                     vparam_names=None,
                     bounds=None,
                     **kwargs):
        """Fit the data with the template.

        Parameters
        ----------
        data: pandas.DataFrame
            The data to fit.
        guessparams: dict, optional
            Guess parameters for the fit.
        fixedparams: dict, optional
            Fixed parameters for the fit.
        vparam_names: list, optional
            List of parameters to vary.
        bounds: dict, optional
            Bounds for the parameters.
        **kwargs
            Goes to sncosmo.fit_lc().

        Returns
        -------
        pandas.DataFrame, pandas.Series
            The results of the fit.
        """

        if vparam_names is None:
            vparam_names = self.parameters.copy()

        if fixedparams is None:
            fixedparams = {}
        else:
            vparam_names = [k for k in vparam_names if k not in fixedparams.keys()]

        if guessparams is None:
            guessparams = {}

        if bounds is None:
            bounds = {}

        sncosmo_model = self.get(**{**guessparams,**fixedparams}) # sets the default parameters fixed win
        return self._fit_data(data, sncosmo_model,
                              vparam_names=vparam_names,
                              bounds=bounds, **kwargs)

    @staticmethod
    def _fit_data(data, sncosmo_model, *args, **kwargs):
        """Fit the data with the template.

        Parameters
        ----------
        data: pandas.DataFrame or astropy.table.Table
            The data to fit.
        sncosmo_model: sncosmo.Model
            The model to use for the fit.
        *args
            Goes to sncosmo.fit_lc().
        **kwargs
            Goes to sncosmo.fit_lc().

        Returns
        -------
        pandas.DataFrame, pandas.Series
            The results of the fit.
        """
        if type(data) is pandas.DataFrame: # sncosmo format
            from astropy.table import Table
            data = Table.from_pandas(data)

        result, fitted_model = sncosmo.fit_lc( data, sncosmo_model, *args, **kwargs)
        return sncosmoresult_to_pandas(result)

    # ============== #
    #   Properties   #
    # ============== #

    # sncosmo_model
    @property
    def source(self):
        """The sncosmo source."""
        return self.sncosmo_model.source

    @property
    def sncosmo_model(self):
        """ hiden sncosmo_model model to check what's inside.
        """
        return self._sncosmo_model

    @property
    def parameters(self):
        """The model parameters."""
        return self.sncosmo_model.param_names

    @property
    def effect_parameters(self):
        """The model effect parameters."""
        return self.sncosmo_model.effect_names

    @property
    def core_parameters(self):
        """The model core parameters."""
        return self.sncosmo_model.source.param_names



class GridTemplate( Template ):

    _GRID_OF = None

    @staticmethod
    def _read_single_file(filename, sncosmo_source):
        source = sncosmo_source.from_filename(filename)
        return Template.from_sncosmo(source)

    @classmethod
    def from_filenames(cls, filenames, refindex=0, grid_of=None):
        """Load the instance from a list of filenames.

        Parameters
        ----------
        filenames: list
            List of filenames.
        refindex: int, optional
            Reference index for the grid.
        grid_of: sncosmo.Source, optional
            The source to use for the grid.

        Returns
        -------
        GridTemplate
        """
        if grid_of is None:
            grid_of = cls.grid_of

        datafile = pandas.Series(filenames, name="filepath").to_frame()
        datafile["basename"] = datafile["filepath"].apply(os.path.basename)
        datafile["template"] = datafile["filepath"].apply(lambda x: cls._read_single_file(x, grid_of))

        source = grid_of.from_filename(filenames[refindex]) # first
        this = cls(source)
        this.set_grid_datafile(datafile)
        return this

    # ============== #
    #   Methods      #
    # ============== #
    def set_grid_datafile(self, datafile):
        """Set the grid datafile.

        Parameters
        ----------
        datafile: pandas.DataFrame
            The grid datafile.
        """
        self._grid_datafile = datafile
        self._grid = None

    def set_grid_data(self, data):
        """Set the grid data.

        Parameters
        ----------
        data: pandas.DataFrame
            The grid data.
        """
        self._grid_data = data
        self._grid

    # ================= #
    #  handle Elements  #
    # ================= #
    def get(self, grid_element,  **kwargs):
        """Return a sncosmo model for the template's source name (self.source).

        Parameters
        ----------
        grid_element: tuple
            The grid element to get.
        **kwargs
            Goes to Template.get().

        Returns
        -------
        sncosmo.Model
        """
        return self.grid.loc[grid_element]["template"].get(**kwargs)

    def get_lightcurve(self, grid_element, band, times,
                           sncosmo_model=None, params=None,
                           in_mag=False, zp=25, zpsys="ab"):
        """Get the lightcurve for a given grid element.

        Parameters
        ----------
        grid_element: tuple
            The grid element to get.
        band: str or list
            Band or list of bands.
        times: array
            Array of times.
        sncosmo_model: sncosmo.Model, optional
            The sncosmo model to use. If None, the instance's model is used.
        params: dict, optional
            Parameters for the model.
        in_mag: bool, optional
            If True, return the lightcurve in magnitude.
        zp: float, optional
            Zero point for the flux.
        zpsys: str, optional
            Zero point system.

        Returns
        -------
        array
            The lightcurve values.
        """
        if params is None:
            params = {}

        params["grid_element"]= grid_element
        props = {k:v for k,v in locals().items()
                     if k not in ["self", "grid_element"]}
        print(props)
        return super().get_lightcurve(**props)

    def fit_data(self, data, grid_element,
                     guessparams=None,
                     fixedparams=None,
                     vparam_names=None,
                     bounds=None,
                     **kwargs):
        """Fit the data with the template for a given grid element.

        Parameters
        ----------
        data: pandas.DataFrame
            The data to fit.
        grid_element: tuple
            The grid element to use.
        guessparams: dict, optional
            Guess parameters for the fit.
        fixedparams: dict, optional
            Fixed parameters for the fit.
        vparam_names: list, optional
            List of parameters to vary.
        bounds: dict, optional
            Bounds for the parameters.
        **kwargs
            Goes to sncosmo.fit_lc().

        Returns
        -------
        pandas.DataFrame, pandas.Series
            The results of the fit.
        """
        # let's put it inside guesses to goes to self.get()
        guessparams["grid_element"]= grid_element
        props = locals()
        _ = props.pop("self")
        _ = props.pop("grid_element")
        return super().fit_data(**props)


    def show_lightcurve(self, band, grid_element,
                            params=None,
                            ax=None, fig=None, colors=None,
                            phase_range=None, npoints=500,
                            zp=25, zpsys="ab",
                            format_time=True, t0_format="mjd",
                            in_mag=False, invert_mag=True, **kwargs):
        """Show the lightcurve for a given grid element.

        Parameters
        ----------
        band: str or list
            Band or list of bands.
        grid_element: tuple
            The grid element to use.
        params: dict, optional
            Parameters for the model.
        ax: matplotlib.axes, optional
            The axes to plot on.
        fig: matplotlib.figure, optional
            The figure to plot on.
        colors: list, optional
            List of colors for the bands.
        phase_range: list, optional
            Phase range to plot.
        npoints: int, optional
            Number of points to plot.
        zp: float, optional
            Zero point for the flux.
        zpsys: str, optional
            Zero point system.
        format_time: bool, optional
            If True, format the time axis.
        t0_format: str, optional
            Format of the t0.
        in_mag: bool, optional
            If True, plot in magnitude.
        invert_mag: bool, optional
            If True, invert the magnitude axis.
        **kwargs
            Goes to ax.plot().

        Returns
        -------
        matplotlib.figure
        """
        params["grid_element"]= grid_element
        props = locals()
        _ = props.pop("self")
        _ = props.pop("grid_element")

        return super().show_lightcurve(**props)

    # ============== #
    #   Grid         #
    # ============== #
    # grid
    @property
    def grid(self):
        """The grid of templates."""
        if self._grid is None:
            self._grid = self.grid_datafile.join(self.grid_data).set_index(self.grid_parameters)

        return self._grid

    @property
    def grid_datafile(self):
        """The grid datafile."""
        return self._grid_datafile

    @property
    def grid_data(self):
        """The grid data."""
        return self._grid_data

    @property
    def grid_parameters(self):
        """The grid parameters."""
        return list(self._grid_data.columns)

    @property
    def full_parameters(self):
        """Template parameters plus grid parameters."""
        return self.parameters + self.grid_parameters


    @classproperty
    def grid_of(cls):
        if not hasattr(cls,"_grid_of"):
            return cls._GRID_OF
        return cls._grid_of

    
class TemplateCollection( object ):
    def __init__(self, templates):
        """Initialize the TemplateCollection class.

        Parameters
        ----------
        templates: list
            List of templates.
        """
        self._templates = templates

    def __iter__(self):
        """Iterate over the templates."""
        return self.templates
        
    @classmethod
    def from_sncosmo(cls, templates):
        """Load the instance from a list of sncosmo sources.

        Parameters
        ----------
        templates: list
            List of sncosmo sources.

        Returns
        -------
        TemplateCollection
        """
        templates = [Template.from_sncosmo(template) for template in np.atleast_1d(templates)]
        return cls(templates)

    @classmethod
    def from_list(cls, templates):
        """Load the instance from a list of templates.

        Parameters
        ----------
        templates: list
            List of templates.

        Returns
        -------
        TemplateCollection
        """
        templates = parse_template(templates)
        return cls(templates)
        
    def call_down(self, which, margs=None, allow_call=True, **kwargs):
        """Call a method on all templates.

        Parameters
        ----------
        which: str
            The method to call.
        margs: list, optional
            List of arguments for the method.
        allow_call: bool, optional
            If True, call the method.
        **kwargs
            Goes to the method.

        Returns
        -------
        list
            List of results.
        """
        if margs is not None:
            from .target.collection import broadcast_mapping
            margs = broadcast_mapping(margs, self.templates)
            return [getattr(t, which)(marg_, **kwargs)
                        for marg_, t in zip(margs, self.templates)]
            
        return [attr if not (callable(attr:=getattr(t, which)) and allow_call) else\
                attr(**kwargs) 
                for t in self.templates]

    def call_down_source(self, which, margs=None, allow_call=True, **kwargs):
        """Call a method on all templates sources.

        Parameters
        ----------
        which: str
            The method to call.
        margs: list, optional
            List of arguments for the method.
        allow_call: bool, optional
            If True, call the method.
        **kwargs
            Goes to the method.

        Returns
        -------
        list
            List of results.
        """
        if margs is not None:
            from .target.collection import broadcast_mapping
            margs = broadcast_mapping(margs, self.templates)
            return [getattr(t.source, which)(marg_, **kwargs)
                        for marg_, t in zip(margs, self.templates)]
            
        return [attr if not (callable(attr:=getattr(t.source, which)) and allow_call) else\
                attr(**kwargs) 
                for t in self.templates]

    def add_effect(self, effects):
        """Add an effect to all templates.

        Parameters
        ----------
        effects: list
            List of effects.
        """
        return self.call_down("add_effect", effects=effects)

    def nameorindex_to_index(self, name_or_index):
        """Convert a name or index to an index.

        Parameters
        ----------
        name_or_index: str or int
            The name or index to convert.

        Returns
        -------
        int
            The index.
        """
        if type(name_or_index) in [str, np.str_]: # is name
            name_or_index = np.argwhere( np.asarray(self.names) == name_or_index).squeeze()
        
        return name_or_index

    # ---------- #
    #  GETTER    #
    # ---------- #
    def get(self, ref_index=0, **kwargs):
        """Get a template.

        Parameters
        ----------
        ref_index: int, optional
            The index of the template to get.
        **kwargs
            Goes to the template's get method.

        Returns
        -------
        Template
        """
        if self.is_uniquetype:
            return self.templates[ref_index].get(**kwargs)
            
        raise NotImplementedError("get() is not implemented for non uniquetyep templates")        

    def get_lightcurve(self, band, times, 
                       index=None, sncosmo_model=None, 
                       in_mag=False, zp=25, zpsys="ab",
                        **kwargs):
        """Get the lightcurve for a given template.

        Parameters
        ----------
        band: str or list
            Band or list of bands.
        times: array
            Array of times.
        index: int, optional
            The index of the template to use.
        sncosmo_model: sncosmo.Model, optional
            The sncosmo model to use. If None, the instance's model is used.
        in_mag: bool, optional
            If True, return the lightcurve in magnitude.
        zp: float, optional
            Zero point for the flux.
        zpsys: str, optional
            Zero point system.
        **kwargs
            Goes to the template's get_lightcurve method.

        Returns
        -------
        array
            The lightcurve values.
        """
        if index is None and sncosmo_model is None:
            raise ValueError("index or sncosmo_model must be given.")
            
        if sncosmo_model is None:
            sncosmo_model = self.get(ref_index=index, as_model=True)
        elif index is not None:
            warnings.warn(f"{index=} is ignored as sncosmo_model is given.")

        return self.templates[0].get_lightcurve(band, times, 
                                                   sncosmo_model=sncosmo_model, 
                                                   in_mag=in_mag, zp=zp, zpsys=zpsys,
                                                    **kwargs)
            
    # ============ #
    #  Properties  #
    # ============ #
    @property
    def templates(self):
        """The list of templates."""
        return self._templates

    @property
    def ntemplates(self):
        """The number of templates."""
        return len(self.templates)

    @property
    def names(self):
        """The names of the templates."""
        return self.call_down_source("name")

    @property
    def is_uniquetype(self):
        """Whether the templates are of a unique type."""
        ntypes = len(np.unique([str(c) for c in self.call_down_source("__class__", allow_call=False)]))
        return ntypes == 1

    # = unique of not
    @property
    def effect_parameters(self):
        """The effect parameters of the templates."""

    @property
    def template_parameters(self):
        """The template parameters of the templates."""
        return self.parameters

    @property
    def parameters(self):
        """The parameters of the templates."""
