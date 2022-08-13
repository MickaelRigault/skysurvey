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
    def get_template(self, incl_dust=True, **kwargs):
        """ template """
        template = self._get_template(self.template_source, incl_dust=True, **kwargs)
        return template

    def get_target_template(self, index, incl_dust=True, **kwargs):
        """ """
        known = self.data.columns[np.in1d(self.data.columns, self.template_parameters)]
        prop = self.data[known].loc[index].to_dict()
        model = self.get_template(**prop)
        return model
        
    @staticmethod
    def _get_template(source, incl_dust=True, **kwargs):
        """ """
        return get_sncosmo_template(source=source, 
                                    incl_dust=True, 
                                   **kwargs)

    # -------------- #
    #   Model        #
    # -------------- #
    def set_model(self, **kwargs):
        """ """
        self._model = kwargs
        
    def get_model(self, **kwargs):
        """ """
        return {**self.model, **kwargs}
        
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
    def _draw(self, model, size=None):
        """ core method converting model into a DataFrame (interp) """
        interp = pandas.DataFrame()
        params = dict(size=size)
        
        for param_name, param_model in model.items():
            # Default kwargs given
            if (inprop := param_model.get("prop", {})) is None:
                inprop = {}

            # set the model ; this overwrite prop['model'] but that make sense.
            inprop["model"] = param_model.get("model",None)

            # read the parameters
            if (inparam := param_model.get("input", None)) is not None:
                for k in inparam:
                    inprop[k] = interp[k].values

            # update the general properties for that of this parameters
            prop = {**params, **inprop}
            # 
            # Draw it
            samples = np.asarray(self.draw_param(param_name, **prop))
            
            # and feed
            output_name = param_model.get("as", param_name)
            interp[output_name] = samples.T
            
        return interp

    def draw(self, size=None,  **kwargs):
        """ """
        model = self.get_model(**kwargs)
        self._data = self._draw(model, size=size)
        return self._data
    
    def draw_param(self, name, model=None, size=None, xx=None, **kwargs):
        """ """
        if hasattr(self, f"draw_{name}"):
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
    
    # ============== #
    #   Properties   #
    # ============== #  
    # Rate
    @property
    def volume_rate(self):
        """ volumetric rate in Gpc-3 / yr-1 """
        return self._VOLUME_RATE
