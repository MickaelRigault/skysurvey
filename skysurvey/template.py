""" core template tools """

import numpy as np
import pandas
import sncosmo


__all__ = ["get_sncosmo_template", "Template"]

def get_sncosmo_template(source="salt2", 
                      incl_dust=True, 
                      **params):
    """ """
    modelprop = dict(source=source)
    if incl_dust:
        dust  = sncosmo.CCM89Dust()
        modelprop["effects"] = [dust]
        modelprop["effect_names"]=['mw']
        modelprop["effect_frames"]=['obs']
        
    model = sncosmo.Model(**modelprop)
    model.set(**params)
    return model      

def sncosmoresult_to_pandas(result):
    """ """
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

    def __init__(self, source):
        """ """
        self._source = source

    @classmethod
    def from_sncosmo_model(cls, model):
        """ """
        this = cls(model.source.name)
        this._hsncosmo_model = model
        return this
    
    # ============== #
    #   Methods      #
    # ============== #
    # -------- #
    #  GETTER  #
    # -------- #
    def get(self, incl_dust=True, **kwargs):
        """ return a sncosmo model for the template's source name (self.source) """
        return self._get(self.source, incl_dust=incl_dust, **kwargs)
    
    @staticmethod
    def _get(source, incl_dust=True, **kwargs):
        return get_sncosmo_template(source=source, 
                                    incl_dust=incl_dust, 
                                   **kwargs)

    def get_lightcurve(cls, band, times,
                           sncosmo_model=None, params=None,
                           in_mag=False, zp=25, zpsys="ab"):
        """ """

        if sncosmo_model is None:
            if params is None:
                params = {}
            
            sncosmo_model = cls.get(**params)

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
            values = sncosmo_model.bandflux(band_, times_, zp=zp, zpsys=zpsys).reshape( len(band),len(times) )
        # in mag
        else:                
            values = sncosmo_model.bandmag(band_, zpsys, times_).reshape( len(band),len(times) )

        return np.squeeze(values) if squeeze else values
    
    # -------- #
    # Plotter  #
    # -------- #
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
        return self._source
    
    @property
    def _sncosmo_model(self):
        """ hiden sncosmo_model model to check what's inside. 
        """
        if not hasattr(self,"_hsncosmo_model"):
            self._hsncosmo_model = self.get()
        return self._hsncosmo_model
        
    @property
    def parameters(self):
        """ """
        return self._sncosmo_model.param_names
    
    @property
    def effect_parameters(self):
        """ """
        return self._sncosmo_model.effect_names        

    @property
    def core_parameters(self):
        """ """
        return self._sncosmo_model.source.param_names