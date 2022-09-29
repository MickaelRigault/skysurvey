
import numpy as np
from .core import Transient

__all__ = ["SNeIa"]


# ================== #
#                    #
# Pre-Defined models #
#                    #
# ================== #
class SNeIaColor( object ):
    
    @staticmethod
    def intrinsic_and_dust(xx="-0.3:1:0.01", cint=-0.05, sigmaint=0.05, tau=0.1):
        """ exponential decay convolved with and intrinsic gaussian color distribution.

        Parameters
        ----------
        
        Returns
        -------
        2d-array:
           xx, pdf
        """
        if type(xx) == str: # assumed r_ input
            xx = eval(f"np.r_[{xx}]")

        from scipy import stats
        from scipy.ndimage import gaussian_filter1d
        # exponential decay center on cint            
        expon = stats.expon.pdf(xx, loc=-0.05, scale=0.1)
        # applying gaussian filtering        
        #  - which require sigmaint in pixel.
        sigmaint_inpix = sigmaint/(xx[1]-xx[0]) # assuming constant step
        pdf = gaussian_filter1d(expon, sigmaint_inpix)
        
        return xx, pdf

    
class SNeIaStretch( object ):

    @staticmethod
    def nicolas2021(xx="-4:4:0.05", 
                    mu1=0.33, sigma1=0.64, 
                    mu2=-1.50, sigma2=0.58, a=0.45,
                    redshift=None, fprompt=0.5):
        """ pdf of the Nicolas (2021) model
        
        Parameters
        ----------

        Returns
        -------
        2d-array:
           xx, pdf
        """
        from scipy.stats import norm
        if type(xx) == str: # assumed r_ input
            xx = eval(f"np.r_[{xx}]")

        mode1 = norm.pdf(xx, loc=mu1, scale=sigma1)
        mode2 = norm.pdf(xx, loc=mu2, scale=sigma2)
        if redshift is not None:
            raise NotImplementedError("estimating fprompt from redshift is not implemented.")
            
        pdf = fprompt*mode1 + (1-fprompt)*(a*mode1 + (1-a)*mode2)
        
        return xx, pdf
    

class SNeIaMagnitude( object ):

    @staticmethod
    def tripp1998(x1, c,
                    mabs=-19.3, sigmaint=0.10,
                    alpha=-0.14, beta=3.15):
        """ """
        mabs = np.random.normal(loc=mabs, scale=sigmaint, size=None)
        mabs_notstandard = mabs + (x1*alpha + c*beta)
        return mabs_notstandard

    
# ================== #
#                    #
# SNeIa Target       #
#                    #
# ================== #


class SNeIa( Transient ):

    _KIND = "SNIa"
    _TEMPLATE = "salt2"
    _RATE = 2.35 * 10**4 # Perley 2020

    # {'model': func, 'prop': dict, 'input':, 'as':}
    _MODEL = dict( redshift = {"param":{"zmax":0.2}, "as":"z"},
                              
                   x1 = {"model": SNeIaStretch.nicolas2021}, 
                   
                   c = {"model": SNeIaColor.intrinsic_and_dust},

                   t0 = {"model": np.random.uniform, 
                         "param": {"low":56000, "high":57000} },
                       
                   magabs = {"model": SNeIaMagnitude.tripp1998,
                             "input": ["x1","c"],
                             "param": {"mabs":-19.3, "sigmaint":0.10}
                            },
                           
                   magobs = {"model": "magabs_to_magobs", # defined in Target (mother of Transients)
                             "input": ["z", "magabs"]},

                   x0 = {"model": "magobs_to_amplitude", # defined in Transients
                         "input": ["magobs"],
                         "param": {"param_name":"x0"}}, #because it needs to call sncosmo_model.get(param_name)
                       
                   radec = {"model": "random",
                            "param": dict(ra_range=[0, 360], dec_range=[-30, 90]),
                            "as": ["ra","dec"]}
                    )




