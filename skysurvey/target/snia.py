
import numpy as np
from scipy import stats
from .core import Transient

from .environments import get_hostmass_rvs
from ..tools.utils import random_radec

__all__ = ["SNeIa"]
# ================== #
#                    #
# Pre-Defined models #
#                    #
# ================== #
class SNeIaColor( object ):
    """A class to model the color of SNe Ia."""

    @staticmethod
    def color_rvs(size, a=3.63, loc=-0.416, scale=1.62):
        """Draw random variates from an alpha function.

        This is used to model the color of SNe Ia.

        Parameters
        ----------
        size : int
            The number of random variates to draw.
        a : float, optional
            The alpha parameter of the alpha function. The default is 3.63.
        loc : float, optional
            The location parameter of the alpha function. The default is -0.416.
        scale : float, optional
            The scale parameter of the alpha function. The default is 1.62.

        Returns
        -------
        ndarray
            The drawn random variates.
        """
        return stats.alpha.rvs(size=size, a=a, loc=loc, scale=scale)

    @staticmethod
    def asymetric_gaussian(xx="-0.3:1:0.001", cint=-0.05, sigmalow=0.03, sigmahigh=0.1):
        """Get an asymetric gaussian distribution.

        As in Scolnic and Kessler 2016 (https://arxiv.org/pdf/1603.01559).

        Parameters
        ----------
        xx : str, optional
            The x-axis of the color distribution. It should be a string with
            the format "min:max:step". The default is "-0.3:1:0.01".
            This is evaluated as `np.r_[xx]`.
        cint : float, optional
            The mean of the intrinsic color distribution. The default is -0.05.
        sigmalow : float, optional
            The standard deviation for the bluer tails. The default is 0.03.
        sigmahigh : float, optional
            The standard deviation for the redder tails. The default is 0.1.

        Returns
        -------
        tuple
            A tuple containing the x-axis and the pdf.
        """
        if type(xx) == str: # assumed r_ input
            xx = eval(f"np.r_[{xx}]")
        
        from scipy import stats
        # full blue
        pdf = stats.norm.pdf(xx, loc=cint, scale=sigmalow)
        redder = (xx>=cint)
        pdf[redder] = stats.norm.pdf(xx[redder], loc=cint, scale=sigmahigh)
        return xx, pdf
        
    @staticmethod
    def intrinsic_and_dust(xx="-0.3:1:0.001", cint=-0.075, sigmaint=0.05, tau=0.14):
        """Get an exponential decay convolved with and intrinsic gaussian color distribution.

        As in Ginolin et al. 2024 (https://arxiv.org/pdf/2406.02072).

        Parameters
        ----------
        xx : str, optional
            The x-axis of the color distribution. It should be a string with
            the format "min:max:step". The default is "-0.3:1:0.01".
            This is evaluated as `np.r_[xx]`.
        cint : float, optional
            The mean of the intrinsic color distribution. The default is -0.075.
        sigmaint : float, optional
            The standard deviation of the intrinsic color distribution.
            The default is 0.05.
        tau : float, optional
            The decay constant of the dust distribution. The default is 0.14.

        Returns
        -------
        tuple
            A tuple containing the x-axis and the pdf.
        """
        if type(xx) == str: # assumed r_ input
            xx = eval(f"np.r_[{xx}]")

        from scipy import stats
        from scipy.ndimage import gaussian_filter1d
        # exponential decay center on cint
        expon = stats.expon.pdf(xx, loc=cint, scale=tau)
        # applying gaussian filtering
        #  - which require sigmaint in pixel.
        sigmaint_inpix = sigmaint/(xx[1]-xx[0]) # assuming constant step
        pdf = gaussian_filter1d(expon, sigmaint_inpix)

        return xx, pdf


class SNeIaStretch( object ):
    """A class to model the stretch of SNe Ia."""

    @staticmethod
    def nicolas2021( xx="-4:4:0.005", 
                     mu1=0.33, sigma1=0.64, 
                     mu2=-1.50, sigma2=0.58, a=0.45,
                     fprompt=0.5, redshift=None):
        """Get the pdf of the Nicolas (2021) model.

        Parameters
        ----------
        xx : str or array, optional
            Definition range for the parameters. Draws will be done from this
            array given the pdf that will be estimated for it. If a string is
            given, it is evaluated as `np.r_[xx]`. The default is
            "-4:4:0.005".
        mu1 : float, optional
            The mean of the first gaussian. The default is 0.33.
        sigma1 : float, optional
            The standard deviation of the first gaussian. The default is 0.64.
        mu2 : float, optional
            The mean of the second gaussian. The default is -1.50.
        sigma2 : float, optional
            The standard deviation of the second gaussian. The default is 0.58.
        a : float, optional
            The relative influence of both modes (1 or 2) in the delayed
            environment. `a>0.5` means more mode 1. The default is 0.45.
        fprompt : float, optional
            The fraction of prompt SNe Ia. This is ignored if `redshift` is
            given. The default is 0.5.
        redshift : array, optional
            The redshift of the target. This defines `fprompt`. The default is
            None.

        Returns
        -------
        tuple
            A tuple containing the x-axis and the pdf.
        """
        from scipy.stats import norm
        if type(xx) == str: # assumed r_ input
            xx = eval(f"np.r_[{xx}]")

        if redshift is not None:
            fprompt = Rigault_AgePop.deltaz(redshift)
            
        mode1 = norm.pdf(xx, loc=mu1, scale=sigma1)
        mode2 = norm.pdf(xx, loc=mu2, scale=sigma2)
        if type(fprompt) is not float: 
            fprompt = np.asarray(fprompt)[:,None]
            
        pdf = fprompt*mode1 + (1-fprompt)*(a*mode1 + (1-a)*mode2)
        return xx, pdf
    

class SNeIaMagnitude( object ):
    """A class to model the magnitude of SNe Ia."""

    @staticmethod
    def tripp1998( x1, c,
                   mabs=-19.3, sigmaint=0.10,
                   alpha=-0.14, beta=3.15):
        """Get the 2-parameter absolute (natural) SNe Ia magnitude.

        Parameters
        ----------
        x1 : array
            The lightcurve stretch. `x1` and `c` must have the same size.
        c : array
            The lightcurve color. `x1` and `c` must have the same size.
        mabs : float, optional
            The average absolute magnitude at `c=0` and `x1=0`. The default is
            -19.3.
        sigmaint : float, optional
            The scale of the normal grey scatter (on `mabs`). The default is
            0.10.
        alpha : float, optional
            The stretch linear law coefficient. The default is -0.14.
        beta : float, optional
            The color linear law coeeficient. The default is 3.15.

        Returns
        -------
        array
           The absolute magnitude, with the same format as `x1` and `c`.
        """
        mabs = np.random.normal(loc=mabs, scale=sigmaint, size=len(x1))
        mabs_notstandard = mabs + (x1*alpha + c*beta)
        return mabs_notstandard


    @classmethod
    def tripp_and_step( cls, x1, c, isup,
                        mabs=-19.3, sigmaint=0.10,
                        alpha=-0.14, beta=3.15, gamma=0.1):
        """Get the 2-parameter and step absolute (natural) SNe Ia magnitude.

        Parameters
        ----------
        x1 : array
            The lightcurve stretch. `x1` and `c` must have the same size.
        c : array
            The lightcurve color. `x1` and `c` must have the same size.
        isup : array
            An array of 0 or 1, flagging which target has `+gamma/2` (1) or
            `-gamma/2` (0).
        mabs : float, optional
            The average absolute magnitude at `c=0` and `x1=0`. The default is
            -19.3.
        sigmaint : float, optional
            The scale of the normal grey scatter (on `mabs`). The default is
            0.10.
        alpha : float, optional
            The stretch linear law coefficient. The default is -0.14.
        beta : float, optional
            The color linear law coeeficient. The default is 3.15.
        gamma : float, optional
            The step's amplitude. The default is 0.1.

        Returns
        -------
        array
           The absolute magnitude, with the same format as `x1`, `c` and `isup`.
        """
        tripp_mabs = cls.tripp1998( x1, c,
                                    mabs=mabs, sigmaint=sigmaint,
                                    alpha=alpha, beta=beta)
        return tripp_mabs + (isup-0.5)*gamma # 0 gets -gamma/2 and 1 get +gamma/2

    @classmethod
    def tripp_and_massstep( cls, x1, c, hostmass,
                            mabs=-19.3, sigmaint=0.10,
                            alpha=-0.14, beta=3.15,
                            gamma=0.1, split=10):
        """Get the 2-parameter and mass step absolute (natural) SNe Ia magnitude.

        Parameters
        ----------
        x1 : array
            The lightcurve stretch. `x1` and `c` must have the same size.
        c : array
            The lightcurve color. `x1` and `c` must have the same size.
        hostmass : array
            The host stellar mass.
        mabs : float, optional
            The average absolute magnitude at `c=0` and `x1=0`. The default is
            -19.3.
        sigmaint : float, optional
            The scale of the normal grey scatter (on `mabs`). The default is
            0.10.
        alpha : float, optional
            The stretch linear law coefficient. The default is -0.14.
        beta : float, optional
            The color linear law coeeficient. The default is 3.15.
        gamma : float, optional
            The step's amplitude. The default is 0.1.
        split : float, optional
            The host mass boundary between low-mass and high-mass hosts.
            The default is 10.

        Returns
        -------
        array
           The absolute magnitude, with the same format as `x1`, `c` and `isup`.
        """
        isup = np.asarray( hostmass>split, dtype=float)
        return cls.tripp_and_step( x1, c, isup,
                                   mabs=mabs, sigmaint=sigmaint,
                                   alpha=alpha, beta=beta, gamma=gamma)
# ================== #
#                    #
# SNeIa Target       #
#                    #
# ================== #


class SNeIa( Transient ):
    """A class to model SNe Ia.

    Parameters
    ----------
    _KIND : str, optional
        The kind of transient. The default is "SNIa".
    _TEMPLATE : str, optional
        The template to use. The default is "salt2".
    _RATE : float, optional
        The rate of SNe Ia. The default is 2.35 * 10**4.
    _MODEL : dict, optional
        The model to use. The default is a dictionary with the following
        keys:

        - `redshift`: The redshift of the SNe Ia.
        - `x1`: The stretch of the SNe Ia.
        - `c`: The color of the SNe Ia.
        - `t0`: The time of maximum of the SNe Ia.
        - `magabs`: The absolute magnitude of the SNe Ia.
        - `magobs`: The observed magnitude of the SNe Ia.
        - `x0`: The amplitude of the SNe Ia.
        - `radec`: The ra and dec of the SNe Ia.
    """

    _KIND = "SNIa"
    _TEMPLATE = "salt2"
    _RATE = 2.35 * 10**4 # Perley 2020

    # {'name': {func: ,'kwargs': {}, 'as': str_or_list }}
    _MODEL = dict( redshift = {"func": "draw_redshift", # implicit
                                "kwargs": {"zmax":0.2},
                                "as":"z"},
                              
                   x1 = {"func": SNeIaStretch.nicolas2021}, 
                   
                   c = {"func": SNeIaColor.intrinsic_and_dust},

                   t0 = {"func": np.random.uniform, 
                         "kwargs": {"low":56_000, "high":56_200} },
                       
                   magabs = {"func": SNeIaMagnitude.tripp1998,
                             "kwargs": {"x1": "@x1", "c": "@c",
                                        "mabs":-19.3, "sigmaint":0.10}
                            },
                           
                   magobs = {"func": "magabs_to_magobs", # str-> method of the class
                             "kwargs": {"z":"@z", "magabs":"@magabs"},
                            },

                   x0 = {"func": "magobs_to_amplitude", # str-> method of the class
                         "kwargs": {"magobs":"@magobs", "param_name": "x0"},
                        }, #because it needs to call sncosmo_model.get(param_name)
                       
                   radec = {"func": random_radec,
                            "kwargs": {},
                            "as": ["ra","dec"]
                           },
                    )
