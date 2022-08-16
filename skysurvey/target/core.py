import warnings
import numpy as np
import pandas

from astropy import cosmology

__all__ = ["get_sncosmo_template",
           "Target", "Transient"]


from .. import sampling

    
def get_sncosmo_template(source="salt2", 
                      incl_dust=True, 
                      **params):
    """ """
    import sncosmo
    modelprop = dict(source=source)
    if incl_dust:
        dust  = sncosmo.CCM89Dust()
        modelprop["effects"] = [dust]
        modelprop["effect_names"]=['mw']
        modelprop["effect_frames"]=['obs']
        
    model = sncosmo.Model(**modelprop)
    model.set(**params)
    return model



class Target( object ):
    """ """
    _KIND = "unknow"
    _TEMPLATE_SOURCE = None
    _MODEL = None # dict config 
    
    # - Cosmo
    _COSMOLOGY = cosmology.Planck18


    @classmethod
    def from_setting(cls, setting, **kwargs):
        """ """
        raise NotImplementedError("from_setting is not Implemented ")
        
    # ------------- # 
    #   Template    #
    # ------------- #
    def get_template(self, index=None, incl_dust=True, **kwargs):
        """ template """
        if index is not None:
            prop = self.get_template_parameters(index).to_dict()
            kwargs = {**prop, **kwargs}
            
        template = self._get_template(self.template_source, incl_dust=True, **kwargs)
        return template

    def get_target_template(self, index, incl_dust=True, **kwargs):
        """ shortcut to get_template(index=index, incl_dust=incl_dust, **kwargs) """
        return self.get_template(index=index, incl_dust=incl_dust, **kwargs)
        
    @staticmethod
    def _get_template(source, incl_dust=True, **kwargs):
        """ """
        return get_sncosmo_template(source=source, 
                                    incl_dust=True, 
                                   **kwargs)
    # -------------- #
    #   Getter       #
    # -------------- #
    def get_template_parameters(self, index=None):
        """ """
        known = self.data.columns[np.in1d(self.data.columns, self.template_parameters)]
        prop = self.data[known]
        if index is not None:
            return prop.loc[index]
        
        return prop

    # -------------- #
    #   Converts     #
    # -------------- #
    def magabs_to_magobs(self, z, magabs, cosmology=None):
        """ converts the absolute magnitude into a cosmology """
        if cosmology is None:
            cosmology = self.cosmology

        return self._magabs_to_magobs(z, magabs, cosmology=cosmology)
    
    @staticmethod
    def _magabs_to_magobs(z, magabs, cosmology):
        """ distmod(z) + Mabs """
        return cosmology.distmod(np.asarray(z)).value + magabs

    
    # -------------- #
    #   Model        #
    # -------------- #
    def set_model(self, **kwargs):
        """ """
        self._model = kwargs
        
    def get_model(self, **kwargs):
        """ """
        return {**self.model, **kwargs}

    # -------------- #
    #   Plotter      #
    # -------------- #
    def show_scatter(self, xkey, ykey, ckey=None, ax=None, fig=None, 
                     index=None, data=None):
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


        ax.scatter(xvalue, yvalue, c=cvalue)
        return fig
        
    # ============== #
    #   Properties   #
    # ============== #        
    @property
    def model(self):
        """ modeling who the transient is generated """
        if not hasattr(self, "_model") or self._model is None:
            self._model = {}
        return self._model
    


    # =============== #
    #   Draw Methods  #
    # =============== #
    def draw(self, size=None,  **kwargs):
        """ core method based on _draw and draw_param """
        model = self.get_model(**kwargs)
        self._data = self._draw(model, size=size)
        return self._data

    def _draw(self, model, size=None):
        """ core method converting model into a DataFrame (interp) """
        interp = pandas.DataFrame()
        params = dict(size=size)
        
        for param_name, param_model in model.items():
            # Default kwargs given
            if (inprop := param_model.get("prop", {})) is None:
                inprop = {}

            # set the model ; this overwrite prop['model'] but that make sense.
            inprop["model"] = param_model.get("model", None)

            # read the parameters
            if (inparam := param_model.get("input", None)) is not None:
                for k in inparam:
                    if k in interp:
                        inprop[k] = interp[k].values
                    elif hasattr(self,k):
                        inprop[k] = getattr(self, k)
                    else:
                        raise ValueError(f"cannot get the input parameter {k}")

            # update the general properties for that of this parameters
            prop = {**params, **inprop}
            # 
            # Draw it
            samples = np.asarray(self.draw_param(param_name, **prop))
            
            # and feed
            output_name = param_model.get("as", param_name)
            interp[output_name] = samples.T
            
        return interp

    
    def draw_param(self, name, model=None, size=None, xx=None, **kwargs):
        """ """
        if model is not None and hasattr(self, model):
            return getattr(self, model)(**kwargs)
        
        if hasattr(self, f"draw_{name}"):
            if model is not None:
                kwargs["model"] = model # handle draw_{name} with no model
            return getattr(self,f"draw_{name}")(size=size, **kwargs)
        
        return eval(f"sampling.Sampling_{name}").draw(model, size=size, xx=xx, **kwargs)

    # ============== #
    #   Properties   #
    # ============== #  
    @property
    def kind(self):
        """ """
        return self._KIND
            
    @property
    def cosmology(self):
        """ """
        return self._COSMOLOGY

    # Template
    @property
    def template_source(self):
        """ """
        return self._TEMPLATE_SOURCE
    
    @property
    def _testtemplate(self):
        """ hiden test model to check what's inside. 
        """
        if not hasattr(self,"_htesttemplate"):
            self._htesttemplate = self.get_template()
        return self._htesttemplate
        
    @property
    def template_parameters(self):
        """ """
        return self._testtemplate.param_names
    
    @property
    def template_effect_parameters(self):
        """ """
        return self._testtemplate.effect_names        

    # model
    @property
    def model(self):
        """ modeling who the transient is generated """
        if not hasattr(self, "_model") or self._model is None:
            self._model = self._MODEL if self._MODEL is not None else {}
            
        return self._model
    
    @property
    def data(self):
        """ data """
        if not hasattr(self,"_data"):
            return None
        return self._data


    
class Transient( Target ):
    # - Transient    
    _VOLUME_RATE = None    
    
    # ============== #
    #  Methods       #
    # ============== #
    # Rates
    def getpdf_redshift(self, z, **kwargs):
        """ """
        rates = np.diff(self.get_rate(z, **kwargs))
        return rates/np.nansum(rates)
    
    def get_rate(self, z, **kwargs):
        """ """
        if self.volume_rate is not None:
            volume = self._COSMOLOGY.comoving_volume(z).to("Gpc**3").value 
            z_rate = volume * self.volume_rate
            return z_rate
        
        raise NotImplementedError("you must implement get_rate() or provide self._VOLUME_RATE for your transient.")

    def get_lightcurve(self, band, times,
                           template=None, index=None, params=None,
                           in_mag=False, zp=25, zpsys="ab"):
        """ """

        if template is None:
            if params is None:
                params = {}
            template = self.get_template(index=index, **params)

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
            values = template.bandflux(band_, times_, zp=zp, zpsys=zpsys).reshape( len(band),len(times) )
        # in mag
        else:                
            values = template.bandmag(band_, zpsys, times_).reshape( len(band),len(times) )

        return np.squeeze(values) if squeeze else values
    
    # ------------ #
    #   Draw       #
    # ------------ #
    def draw_redshift(self, zmax, zmin=0, zstep=1e-3, size=None):
        """ based on the rate (see get_rate()) """
        xx = np.arange(zmin, zmax, zstep)
        pdf = self.getpdf_redshift(xx)
        return np.random.choice(np.mean([xx[1:],xx[:-1]], axis=0), 
                      size=size, p=pdf/pdf.sum())
            
    def draw_mjd(self, model="uniform", size=None, mjd_range=[59000, 59000]):
        """ """
        if model == "uniform":
            return np.random.uniform(*mjd_range, size=size)
        else:
            raise NotImplementedError("model '{model}' not implemented in draw_radec")
    
    def draw_t0(self, **kwargs):
        """ shortcut to draw_mjd(**kwargs) """
        return self.draw_mjd(**kwargs)

    # ------------ #
    #  Show LC     #
    # ------------ #
    def show_lightcurve(self, band, index=None, params=None,
                            ax=None, fig=None, colors=None,
                            time_range=[-20,50], npoints=500,
                            zp=25, zpsys="ab",
                            format_time=True, t0_format="mjd", 
                            in_mag=False, invert_mag=True, **kwargs):
        """ """
        from ..config import get_band_color
        # get the template
        if params is None:
            params = {}

        template = self.get_template(index=index, **params)
        
        # ------- #
        #  x-data #
        # ------- #
        # time range
        t0 = template.parameters[template.param_names.index("t0")]
        times = np.linspace(*np.asarray(time_range)+t0, npoints)

        # ------- #
        #  y-data #
        # ------- #        
        # flux
        band = np.atleast_1d(band)
        values = self.get_lightcurve(band,
                                     times, in_mag=in_mag,
                                     zp=zp, zpsys=zpsys,
                                     template=template)

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
            
    # ============== #
    #   Properties   #
    # ============== #  
    # Rate
    @property
    def volume_rate(self):
        """ volumetric rate in Gpc-3 / yr-1 """
        return self._VOLUME_RATE
