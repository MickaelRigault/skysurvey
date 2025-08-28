import pandas
import numpy as np


from .timeserie import MultiTemplateTSTransient
from ..source import get_sncosmo_sourcenames


__all__ = ["SNeII", "SNeIIn", "SNeIIb",
           "SNeIb", "SNeIc", "SNeIcBL"]


CC_RATE = 1.0e5 # Perley+2020
    
# https://sncosmo.readthedocs.io/en/stable/source-list.html

class VincenziModels( object ):
    """Default parametrization for the TimeSeriesSources based on Vincenzi et al. 2019.

    These are stored in sncosmo.

    Parameters
    ----------
    object : class
        The parent class of the VincenziModels class.

    Reference: https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.5802V
    """
    _KIND = None
    # takes the v19-*-corr corresponding to the given _KIND
    _TEMPLATES = "complex"
    # Default rate from Perley 2020 ; this is wrong but close.
    
    _RATE = np.nan # CC 1e5 * (0.75 *0.72) for Type II.
    # Perley+2020: CC-rate (all combined) is 1e5 Gyr-3/yr-1
    
    # For Vincenzi model, we considered average absolute magnitude
    # as defined in their Table 1. We favor the right-most column.

    @property
    def template(self):
        """ # TO DO
        """
        if not hasattr(self,"_template") or self._template is None:
            template_list = get_sncosmo_sourcenames(self._KIND,
                                                        startswith="v19",
                                                        endswith="corr") # all -corr models
            self.set_template(template_list)
            
        return self._template

class SnanaModels( VincenziModels ):
    """ Same as VincenziModels but matching different naming convention in sncosmo

    Parameters
    ----------
    VincenziModels : class
        The parent class of the SnanaModels class.
    """
    
    @property
    def template(self):
        """ # TO DO
        """
        if not hasattr(self,"_template") or self._template is None:
            template_list = get_sncosmo_sourcenames(self._KIND,
                                                        startswith="snana",
                                                        endswith="") # all -corr models
            self.set_template(template_list)
            
        return self._template
    
# =============== #
#                 #
#   Type II       #
#                 #
# =============== #
## Info on _MAGABS
## Format:
## - Gaussian: (loc, scatter)
## - skewed Gaussian: (loc, scatter_low, scatter_high)


class SNeII( VincenziModels, MultiTemplateTSTransient ):
    """ SNe II model from Vincenzi et al. 2019.

    Parameters
    ----------
    VincenziModels : class
        The parent class of the SNeII class.
    MultiTemplateTSTransient : class
        The parent class of the SNeII class.
    """
    _KIND = "SN II"
    # change the absolute magnitude parameters
    # Perley+2020 total cc-rate * relative rate from Vincenzi+2019
    # This is consistant with Perley+2020 for type II being 75% * 72%
    _RATE = CC_RATE * 0.649  # this combines IIL & IIP
    # _MAGABS = (-16.0, 1.3) # Table 1 of Vincenzi19
    _MAGABS = (-17.48, 0.7) # MR from BTS z<0.05
    
class SNeIIn( VincenziModels, MultiTemplateTSTransient ):
    """ SNe IIn model from Vincenzi et al. 2019.

    Parameters
    ----------
    VincenziModels : class
        The parent class of the SNeIIn class.
    MultiTemplateTSTransient : class
        The parent class of the SNeIIn class.
    """    
    _KIND = "SN IIn"
    _RATE = CC_RATE * 0.047
#    _MAGABS = (-17.7, 1.1) # Table 1 of Vincenzi19
    _MAGABS = (-18.0, 0.8) # MR from BTS z<0.05
        
class SNeIIb( VincenziModels, MultiTemplateTSTransient ):
    """ SNe IIb model from Vincenzi et al. 2019.

    Parameters
    ----------
    VincenziModels : class
        The parent class of the SNeIIb class.
    MultiTemplateTSTransient : class
        The parent class of the SNeIIb class.
    """    
    _KIND = "SN IIb"
    _RATE = CC_RATE * 0.109
#    _MAGABS = (-16.7, 2.0) # Table 1 of Vincenzi19
    _MAGABS = (-17.45, 0.6) # MR from BTS z<0.05
    
# =============== #
#                 #
#   Type I        #
#                 #
# =============== #
class SNeIb( VincenziModels, MultiTemplateTSTransient ):
    """ SNe Ib model from Vincenzi et al. 2019.

    Parameters
    ----------
    VincenziModels : class
        The parent class of the SNeIb class.
    MultiTemplateTSTransient : class
        The parent class of the SNeIb class.
    """    
    _KIND = "SN Ib"
    _RATE = CC_RATE * 0.108
    # changing the errors averaging with R14
    #_MAGABS = (-18.3, 0.5) # Table 1 of Vincenzi19
    _MAGABS = (-17.35, 0.53) # MR from BTS z<0.05
    
class SNeIc( VincenziModels, MultiTemplateTSTransient ):
    """ SNe Ic model from Vincenzi et al. 2019.

    Parameters
    ----------
    VincenziModels : class
        The parent class of the SNeIc class.
    MultiTemplateTSTransient : class
        The parent class of the SNeIc class.
    """    
    _KIND = "SN Ic"
    _RATE = CC_RATE * 0.075
    # _MAGABS = (-17.4, 0.7) # Table 1 of Vincenzi19
    _MAGABS = (-17.50, 0.7) # MR from BTS z<0.05

class SNeIcBL( VincenziModels, MultiTemplateTSTransient ):
    """ SNe Ic-BL model from Vincenzi et al. 2019.

    Parameters
    ----------
    VincenziModels : class
        The parent class of the SNeIcBL class.
    MultiTemplateTSTransient : class
        The parent class of the SNeIcBL class.
    """    
    _KIND = "SN Ic-BL"
    _RATE = CC_RATE * 0.097 # joining Ic-BL & Ic-pec    
    # _MAGABS = (-17.7, 1.2) # Table 1 of Vincenzi19 
    _MAGABS = (-18.12, 0.9) # MR from BTS z<0.05
