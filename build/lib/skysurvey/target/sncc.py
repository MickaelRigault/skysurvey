import pandas
import numpy as np


from .timeserie import MultiTemplateTSTransient
from ..source import get_sncosmo_sourcenames


__all__ = ["SNeII", "SNeIIn", "SNeIIb",
           "SNeIb", "SNeIc", "SNeIcBL"]


# https://sncosmo.readthedocs.io/en/stable/source-list.html

class VincenziModels( object ):
    """ Default parametrization for the TimeSeriesSources 
    based on Vincenzi et al. 2019 stored in sncosmo.
    """
    _KIND = None
    # takes the v19-*-corr corresponding to the given _KIND
    _TEMPLATES = "complex"
    # Default rate from Perley 2020 ; this is wrong but close.
    _RATE = 5.4e4 # CC 1e5 * (0.75 *0.72) for Type II. 

    # For Vincenzi model, we considered average absolute magnitude
    # as defined in their Table 1. We favor the right-most column.

    @property
    def template(self):
        """ """
        if not hasattr(self,"_template") or self._template is None:
            template_list = get_sncosmo_sourcenames(self._KIND,
                                                        startswith="v19",
                                                        endswith="corr") # all -corr models
            self.set_template(template_list)
            
        return self._template
    
# =============== #
#                 #
#   Type II       #
#                 #
# =============== #

class SNeII( VincenziModels, MultiTemplateTSTransient ):
    _KIND = "SN II"
    # change the absolute magnitude parameters
    # This is from (Perley 2020)
    _MAGABS = (-16.0, 1.3) # Table 1 of Vincenzi19
    
class SNeIIn( VincenziModels, MultiTemplateTSTransient ):
    _KIND = "SN IIn"
    _MAGABS = (-17.7, 1.1) # Table 1 of Vincenzi19
    
class SNeIIb( VincenziModels, MultiTemplateTSTransient ):
    _KIND = "SN IIb"
    _MAGABS = (-16.7, 2.0) # Table 1 of Vincenzi19

# =============== #
#                 #
#   Type I        #
#                 #
# =============== #
class SNeIb( VincenziModels, MultiTemplateTSTransient ):
    _KIND = "SN Ib"
    # changing the errors averaging with R14
    _MAGABS = (-18.3, 0.5) # Table 1 of Vincenzi19
    
class SNeIc( VincenziModels, MultiTemplateTSTransient ):
    _KIND = "SN Ic"
    _MAGABS = (-17.4, 0.7) # Table 1 of Vincenzi19 

class SNeIcBL( VincenziModels, MultiTemplateTSTransient ):
    _KIND = "SN Ic-BL"
    _MAGABS = (-17.7, 1.2) # Table 1 of Vincenzi19 
