""" core template tools """

import os
import numpy as np
import pandas
import sncosmo
from sncosmo.models import _SOURCES

from astropy.utils.decorators import classproperty

__all__ = ["get_sncosmo_model", "Template",
           "get_sncosmo_sourcenames"]



SNCOSMO_SOURCES_DF = pandas.DataFrame(_SOURCES.get_loaders_metadata())
def get_sncosmo_sourcenames(of_type=None, startswith=None, endswith=None):
    """ get the list of available sncosmo source names

    Parameters
    ----------
    of_type: str, list
        type name (of list of). e.g SN II

    startswith: str
        the source name should start by this (e.g. v19)

    endswith: str
        the source name should end by this

    Returns
    -------
    list
        list of names

    """
    sources = SNCOSMO_SOURCES_DF.copy()
    if of_type is not None:
        typenames = sources[sources["type"].isin(np.atleast_1d(of_type))]["name"]
    else:
        typenames = sources["name"]
        
    if endswith is not None:
        typenames = typenames[typenames.str.startswith(startswith)]
    
    if endswith is not None:
        typenames = typenames[typenames.str.endswith(endswith)]

    return list(typenames)


def get_sncosmo_model(source="salt2",
                             incl_dust=True, 
                             **params):
    """ get the template (sncosmo.Model)

    Parameters
    ----------
    source : `~sncosmo.Source` or str
        The model for the spectral evolution of the source. If a string
        is given, it is used to retrieve a `~sncosmo.Source` from
        the registry.

    incl_dust: bool
        shall this add the dust modeling offset ? 
        (CCM89Dust)

    **kwargs goes to model.set() to change the parameter's model

    Returns
    -------
    `sncosmo.Model`
        the sncosmo.Model template
    """
    modelprop = dict(source=source)
    if incl_dust:
        dust = sncosmo.CCM89Dust()
        modelprop["effects"] = [dust]
        modelprop["effect_names"]=['mw']
        modelprop["effect_frames"]=['obs']
        
    model = sncosmo.Model(**modelprop)
    model.set(**params)
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

# =============== #
#                 #
#  Template       #
#                 #
# =============== #
class Template( object ):

    def __init__(self, sncosmo_model):
        """ """
        self._sncosmo_model = sncosmo_model
        
    @classmethod
    def from_sncosmo(cls, source, incl_dust=True, **kwargs):
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
            # useless for now but I may change things in init.
            sncosmo_model = get_sncosmo_model(source, incl_dust=incl_dust,
                                          **kwargs)
        else:
            sncosmo_model = source
            
        return cls(sncosmo_model)
        
    # ============== #
    #   Methods      #
    # ============== #
    # -------- #
    #  GETTER  #
    # -------- #
    def get(self,**kwargs):
        """ return a copy of the model you can set new parameter with the options """
        from copy import deepcopy
        model = deepcopy(self.sncosmo_model)
        if kwargs:
            model.set(**kwargs)
        return model
    
    def get_lightcurve(self, band, times,
                           sncosmo_model=None, in_mag=False, zp=25, zpsys="ab",
                           **kwargs):
        """ """

        if sncosmo_model is None:
            sncosmo_model = self.get(**kwargs)

        # patch for odd sncosmo behavior (see https://github.com/sncosmo/sncosmo/issues/346)
        squeeze = type(band) in [str, np.str_] # for the output format

        # make sure all are array
        # make sure all are array
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
            values = sncosmo_model.bandmag(band_, zpsys, times_).reshape( len(band),len(times) )

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
            time += sncosmo_model.get("t0")
        
        flux[sel] = sncosmo_model.flux(time, lbdas[sel])  
        return flux

    
    # -------- #
    # Plotter  #
    # -------- #
    def show_spectum(self, time, lbdas, params={},
                         ax=None, fig=None, **kwargs):
        """ """
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
                            time_range=[-20,50], npoints=500,
                            zp=25, zpsys="ab",
                            format_time=True, t0_format="mjd", 
                            in_mag=False, invert_mag=True, **kwargs):
        """ """
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
        times = np.linspace(*np.asarray(time_range)+t0, npoints)

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
        """ """

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
        """ """
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
        """ """
        return self.sncosmo_model.source
    
    @property
    def sncosmo_model(self):
        """ hiden sncosmo_model model to check what's inside. 
        """
        return self._sncosmo_model
        
    @property
    def parameters(self):
        """ """
        return self.sncosmo_model.param_names
    
    @property
    def effect_parameters(self):
        """ """
        return self.sncosmo_model.effect_names        

    @property
    def core_parameters(self):
        """ """
        return self.sncosmo_model.source.param_names



class GridTemplate( Template ):

    _GRID_OF = None
    
    @staticmethod
    def _read_single_file(filename, sncosmo_source):
        source = sncosmo_source.from_filename(filename)
        return Template.from_sncosmo(source)
    
    @classmethod
    def from_filenames(cls, filenames, refindex=0, grid_of=None):
        """ """
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
        """ """
        self._grid_datafile = datafile
        self._grid = None
        
    def set_grid_data(self, data):
        """ """
        self._grid_data = data
        self._grid
        
    # ================= #
    #  handle Elements  #
    # ================= #
    def get(self, grid_element, incl_dust=True, **kwargs):
        """ return a sncosmo model for the template's source name (self.source) """
        return self.grid.loc[grid_element]["template"].get(incl_dust=True, **kwargs)

    def get_lightcurve(self, grid_element, band, times,
                           sncosmo_model=None, params=None,
                           in_mag=False, zp=25, zpsys="ab"):
        """ """
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
        """ """
        # let's put it inside guesses to goes to self.get()
        guessparams["grid_element"]= grid_element
        props = locals()
        _ = props.pop("self")
        _ = props.pop("grid_element")
        return super().fit_data(**props)
    
    
    def show_lightcurve(self, band, grid_element,
                            params=None,
                            ax=None, fig=None, colors=None,
                            time_range=[-20,50], npoints=500,
                            zp=25, zpsys="ab",
                            format_time=True, t0_format="mjd", 
                            in_mag=False, invert_mag=True, **kwargs):
        """ """
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
        """ """
        if self._grid is None:
            self._grid = self.grid_datafile.join(self.grid_data).set_index(self.grid_parameters)
            
        return self._grid
    
    @property
    def grid_datafile(self):
        """ """
        return self._grid_datafile
    
    @property
    def grid_data(self):
        """ """
        return self._grid_data
    
    @property
    def grid_parameters(self):
        """ """
        return list(self._grid_data.columns)
    
    @property
    def full_parameters(self):
        """ template parameters plus grid parameters """
        return self.parameters + self.grid_parameters

    
    @classproperty
    def grid_of(cls):
        if not hasattr(cls,"_grid_of"):
            return cls._GRID_OF
        return cls._grid_of
        
