"""Core-collapse (CC) supernova classes.

See the corresponding documentation page in "List of transient classes" for
more detail on those Transient classes.
"""

import numpy as np

from .timeserie import MultiTemplateTSTransient
from ..source import get_sncosmo_sourcenames

CC_RATE = 1.0e5 # Perley+2020 | this assumes H0=70, see Transient._RATE_H0.

# https://sncosmo.readthedocs.io/en/stable/source-list.html

class VincenziModels( object ):
    """Default parametrization for the Vincenzi et al. 2019 `TimeSeriesSources`.

    These templates are stored in `sncosmo` (reference:
    https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.5802V).

    Attributes
    ----------
    _KIND : str or None, optional
        The specific supernova type (e.g., 'II', 'IIP', 'Ib'). Used to
        identify the corresponding ``v19-*-corr`` template in sncosmo.
        The default is None.

    _TEMPLATES : str, optional
        The template selection mode. For Vincenzi models, this is set to
        'complex' to handle the variety of templates associated with
        each supernova sub-type. The default is 'complex'.

    _RATE : float, optional
        The volumetric rate of the transient. Users should provide a rate
        (typically in units of :math:`Gpc^{-3} yr^{-1}`). The default is
        `np.nan`.
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
        """List of `sncosmo` source names for the Vincenzi 2019 models."""
        if not hasattr(self,"_template") or self._template is None:
            template_list = get_sncosmo_sourcenames(self._KIND,
                                                        startswith="v19",
                                                        endswith="corr") # all -corr models
            self.set_template(template_list)

        return self._template

class SnanaModels( VincenziModels ):
    """Same as `VincenziModels` but matching the SNANA naming in `sncosmo`."""

    @property
    def template(self):
        """List of `sncosmo` source names using the SNANA naming convention."""
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
    r"""SNe II model from Vincenzi et al. 2019.

    This model combines IIL and IIP types.

    Attributes
    ----------
    _KIND : str, optional
        The transient type identifier. The default is ``"SN II"``.

    _RATE : float, optional
        Volumetric rate. Calculated as :math:`CC\_RATE \times 0.649`,
        consistent with the Type II fraction from Perley et al. 2020.

    _MAGABS : tuple, optional
        The peak absolute magnitude distribution ``(mean, sigma)``.
        The default is ``(-17.48, 0.7)``, based on BTS z < 0.05.
    """
    _KIND = "SN II"
    # change the absolute magnitude parameters
    # Perley+2020 total cc-rate * relative rate from Vincenzi+2019
    # This is consistant with Perley+2020 for type II being 75% * 72%
    _RATE = CC_RATE * 0.649  # this combines IIL & IIP
    # _MAGABS = (-16.0, 1.3) # Table 1 of Vincenzi19
    _MAGABS = (-17.48, 0.7) # MR from BTS z<0.05

class SNeIIn( VincenziModels, MultiTemplateTSTransient ):
    r"""SNe IIn model from Vincenzi et al. 2019.

    Attributes
    ----------
    _KIND : str, optional
        The transient type identifier. The default is ``"SN IIn"``.

    _RATE : float, optional
        Volumetric rate. Calculated as :math:`CC\_RATE \times 0.047`.

    _MAGABS : tuple, optional
        The peak absolute magnitude distribution ``(mean, sigma)``.
        The default is ``(-18.0, 0.8)``, based on BTS z < 0.05.
    """
    _KIND = "SN IIn"
    _RATE = CC_RATE * 0.047
    #_MAGABS = (-17.7, 1.1) # Table 1 of Vincenzi19
    _MAGABS = (-18.0, 0.8) # MR from BTS z<0.05

class SNeIIb( VincenziModels, MultiTemplateTSTransient ):
    r"""SNe IIb model from Vincenzi et al. 2019.

    Attributes
    ----------
    _KIND : str, optional
        The transient type identifier. The default is ``"SN IIb"``.

    _RATE : float, optional
        Volumetric rate. Calculated as :math:`CC\_RATE \times 0.109`.

    _MAGABS : tuple, optional
        The peak absolute magnitude distribution ``(mean, sigma)``.
        The default is ``(-17.45, 0.6)``, based on BTS z < 0.05.
    """
    _KIND = "SN IIb"
    _RATE = CC_RATE * 0.109
    #_MAGABS = (-16.7, 2.0) # Table 1 of Vincenzi19
    _MAGABS = (-17.45, 0.6) # MR from BTS z<0.05

# =============== #
#                 #
#   Type I        #
#                 #
# =============== #
class SNeIb( VincenziModels, MultiTemplateTSTransient ):
    r"""SNe Ib model from Vincenzi et al. 2019.

    Attributes
    ----------
    _KIND : str, optional
        The transient type identifier. The default is ``"SN Ib"``.

    _RATE : float, optional
        Volumetric rate. Calculated as :math:`CC\_RATE \times 0.108`.

    _MAGABS : tuple, optional
        The peak absolute magnitude distribution ``(mean, sigma)``.
        The default is ``(-17.35, 0.53)``, based on BTS z < 0.05.
    """
    _KIND = "SN Ib"
    _RATE = CC_RATE * 0.108
    # changing the errors averaging with R14
    #_MAGABS = (-18.3, 0.5) # Table 1 of Vincenzi19
    _MAGABS = (-17.35, 0.53) # MR from BTS z<0.05

class SNeIc( VincenziModels, MultiTemplateTSTransient ):
    r"""SNe Ic model from Vincenzi et al. 2019.

    Attributes
    ----------
    _KIND : str, optional
        The transient type identifier. The default is ``"SN Ic"``.

    _RATE : float, optional
        Volumetric rate. Calculated as :math:`CC\_RATE \times 0.075`.

    _MAGABS : tuple, optional
        The peak absolute magnitude distribution ``(mean, sigma)``.
        The default is ``(-17.50, 0.7)``, based on BTS z < 0.05.
    """
    _KIND = "SN Ic"
    _RATE = CC_RATE * 0.075
    # _MAGABS = (-17.4, 0.7) # Table 1 of Vincenzi19
    _MAGABS = (-17.50, 0.7) # MR from BTS z<0.05

class SNeIcBL( VincenziModels, MultiTemplateTSTransient ):
    r"""SNe Ic-BL model from Vincenzi et al. 2019.

    Attributes
    ----------
    _KIND : str, optional
        The transient type identifier. The default is ``"SN Ic-BL"``.

    _RATE : float, optional
        Volumetric rate. Calculated as :math:`CC\_RATE \times 0.097`,
        joining rates from SNe Ic-BL and SNe Ic-pec.

    _MAGABS : tuple, optional
        The peak absolute magnitude distribution ``(mean, sigma)``.
        The default is ``(-18.12, 0.9)``, based on BTS z < 0.05.
    """
    _KIND = "SN Ic-BL"
    _RATE = CC_RATE * 0.097 # joining Ic-BL & Ic-pec
    # _MAGABS = (-17.7, 1.2) # Table 1 of Vincenzi19
    _MAGABS = (-18.12, 0.9) # MR from BTS z<0.05
