
import numpy as np
from .core import Transient

from .environments import getpdf_asymetric_gaussian
from ..tools.utils import random_radec

__all__ = ["SNeIa"]



# ================== #
#                    #
#  2 Pop model       #
#                    #
# ================== #
class Rigault_AgePop( object ):

    @classmethod
    def deltaz(cls, redshift, k=0.87, phi=2.8):
        """ fraction of prompt SNeIa as a function of redshift 
        following Rigault et al. 2020's model
        deltaz = (1-psiz)
        """
        return 1-cls.psiz(redshift, k=k, phi=phi)

    @staticmethod    
    def psiz(redshift, k=0.87, phi=2.8):
        """ fraction of delayed SNeIa as a function of redshift 
        following Rigault et al. 2020's model
        """
        return (k * (1+redshift)**(phi) + 1)**(-1)

    @classmethod    
    def get_promptpdf(cls, redshift, xx=[0,1]):
        """ get the pdf of the probability to be prompt or delayed as a function of redshift.
        (see self.psiz)
        """
        xx = np.asarray(xx)
        delayed_cut = cls.psiz(np.atleast_1d(redshift))
        return xx, np.asarray( [delayed_cut, 1-delayed_cut]).T

    @staticmethod    
    def get_massdistribution(redshift, fprompt, xx="6:13:300j", normed=True):
        """ get the host mass distribution as a function of redshift and fraction
        of prompt in your sample. 

        This is based on stellar mass function (SMF) from the literature (see get_stellarmassfunction).
        
        As prompt SNeIa rate follows the star formation, the mass-distribution of prompt SNeIa is
        - smf_bluegalaxies * SFR_of_SF_galaxies.
        Here SFR_of_SF_galaxies is defined by the host mass (See Wiseman et al. 2021).
        
        As delayed SNeIa rate follows that of stellar mass, the mass-distribution of delayed SNeIa is:
        - smf_allgalaxies * stellar_mass

        
        Parameters
        ----------
        redshift: float, array
            redshift of the mass distribution. 
            If array, same size (N) as fprompt.

        fprompt: float, array
            fraction of prompt in the sample
            If array, same size (N) as redshift.

        xx: str, array
            binning of the pdf (size M)

        normed: bool
            should the returned pdf be normalized to 1 ?
        
        Returns
        -------
        array, array
            - xx (M)
            - pdf (M,) or (N, M) see redshift and fprompt.
        
        """
        from . import host
        
        if type(xx) == str: # assumed r_ input
            xx = eval(f"np.r_[{xx}]")

        redshift = np.atleast_1d(redshift)
        # xx here is the log(stellar_mass)
        _, smfall = host.get_stellarmassfunction(redshift, which="all", xx=xx)
        _, smfsf = host.get_stellarmassfunction(redshift, which="blue", xx=xx)
        sfr = host.get_sfr_as_function_of_mass_and_redshift(mass=xx, redshift=redshift[:,None])

        mass_pdf_prompt = smfsf * sfr
        if normed:
            mass_pdf_prompt /= np.sum(mass_pdf_prompt, axis=1)[:,None] # norm
            
        mass_pdf_delayed = smfall * 10**(xx)/1e10
        if normed:
            mass_pdf_delayed /= np.sum(mass_pdf_delayed, axis=1)[:,None] # norm

        # broadcasting
        if len(np.atleast_1d(fprompt)) > 1:
            fprompt = np.asarray(fprompt)[:,None]

        masspdf = fprompt*mass_pdf_prompt + (1-fprompt) * mass_pdf_delayed
        return xx, np.squeeze(masspdf)


# ================== #
#                    #
# Pre-Defined models #
#                    #
# ================== #
class SNeIaColor( object ):

    @staticmethod
    def intrinsic_and_dust(xx="-0.3:1:0.001", cint=-0.05, sigmaint=0.05, tau=0.1):
        """ exponential decay convolved with and intrinsic gaussian color distribution.

        Parameters
        ----------
        xx: str
            the x-axis of the color distribution. It should be a string with the
            format "min:max:step". The default is "-0.3:1:0.01".
            inputs np.r_[xx]

        cint: float
            the mean of the intrinsic color distribution. The default is -0.05.

        sigmaint: float
            the standard deviation of the intrinsic color distribution. The default is 0.05.

        tau: float
            the decay constant of the dust distribution. The default is 0.1.

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
    def nicolas2021( xx="-4:4:0.005", 
                     mu1=0.33, sigma1=0.64, 
                     mu2=-1.50, sigma2=0.58, a=0.45,
                     fprompt=0.5, redshift=None):
        
        """ pdf of the Nicolas (2021) model
        
        Parameters
        ----------
        xx: str or array
            definition range for the parameters.
            draws will be done from this array given the 
            pdf that will be estimated for it.
            If string this inputs np.r_[xx]

        mu1, mu2: float
            gaussian centroid (loc)

        sigma1, sigma2: float
            scale of the gaussian distribution.

        a: float
            relative influence of both modes (1 or 2) 
            in the delayed environment. 
            a>0.5 means more mode 1.

        fprompt: float
            fraction of prompt SNeIa.
            = ignored if redshift given = 

        redshift: 
            target's redshift. This defined fprompt.

        Returns
        -------
        array, array
            - xx (M)
            - pdf (M,) or (N, M) see redshift and fprompt.

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

    @staticmethod
    def tripp1998( x1, c,
                   mabs=-19.3, sigmaint=0.10,
                   alpha=-0.14, beta=3.15):
        """ 2-parameter absolute (natural) SNeIa magnitude
        
        Parameters
        ----------
        x1: array
            lightcurve stretch (x1 and c must have the same size)

        c: array
            lightcurve color (x1 and c must have the same size)

        mabs: float
            average absolute magnitude at c=0 and x1=0

        sigmaint: float
            scale of the normal grey scatter (on mabs) 

        alpha: float
            stretch linear law coefficient

        beta: float
            color linear law coeeficient

        Returns
        -------
        array
           absolute magnitude same format as x1 and c.
        """
        mabs = np.random.normal(loc=mabs, scale=sigmaint, size=len(x1))
        mabs_notstandard = mabs + (x1*alpha + c*beta)
        return mabs_notstandard


    @classmethod
    def tripp_and_step( cls, x1, c, isup,
                        mabs=-19.3, sigmaint=0.10,
                        alpha=-0.14, beta=3.15, gamma=0.1):
        """ 2-parameter and step absolute (natural) SNeIa magnitude
        
        Parameters
        ----------
        x1: array
            lightcurve stretch (x1 and c must have the same size)

        c: array
            lightcurve color (x1 and c must have the same size)

        isup: array of 0 or 1
            flag saying which target has +gamma/2 (1) or -gamma/2 (0)

        mabs: float
            average absolute magnitude at c=0 and x1=0

        sigmaint: float
            scale of the normal grey scatter (on mabs) 

        alpha: float
            stretch linear law coefficient

        beta: float
            color linear law coeeficient

        gamma: float
            the step's amplitude.

        Returns
        -------
        array
           absolute magnitude same format as x1,c and isup
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
        """ 2-parameter and mass step absolute (natural) SNeIa magnitude
        
        Parameters
        ----------
        x1: array
            lightcurve stretch (x1 and c must have the same size)

        c: array
            lightcurve color (x1 and c must have the same size)

        hostmass: array
            host stellar mass

        mabs: float
            average absolute magnitude at c=0 and x1=0

        sigmaint: float
            scale of the normal grey scatter (on mabs) 

        alpha: float
            stretch linear law coefficient

        beta: float
            color linear law coeeficient

        gamma: float
            the step's amplitude.

        split: float
            host mass boundary between low-mass and high-mass hosts.

        Returns
        -------
        array
           absolute magnitude same format as x1,c and isup
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

    _KIND = "SNIa"
    _TEMPLATE = "salt2"
    _RATE = 2.35 * 10**4 # Perley 2020

    # {'model': func, 'prop': dict, 'input':, 'as':}
    _MODEL = dict( redshift = {"kwargs": {"zmax":0.2}, "as":"z"},
                              
                   x1 = {"func": SNeIaStretch.nicolas2021}, 
                   
                   c = {"func": SNeIaColor.intrinsic_and_dust},

                   t0 = {"func": np.random.uniform, 
                         "kwargs": {"low":56_000, "high":56_200} },
                       
                   magabs = {"func": SNeIaMagnitude.tripp1998,
                             "kwargs": {"x1": "@x1", "c": "@c",
                                        "mabs":-19.3, "sigmaint":0.10}
                            },
                           
                   magobs = {"func": "magabs_to_magobs", # defined in Target (mother of Transients)
                             "kwargs": {"z":"@z", "magabs":"@magabs"},
                            },

                   x0 = {"func": "magobs_to_amplitude", # defined in Transients
                         "kwargs": {"magobs":"@magobs", "param_name": "x0"},
                        }, #because it needs to call sncosmo_model.get(param_name)
                       
                   radec = {"func": random_radec,
                            "kwargs": {},
                            "as": ["ra","dec"]
                           }
                    )


class SNeIaHostMass( Transient ):
    
    _KIND = "SNIa"
    _TEMPLATE = "salt2"
    _RATE = 2.35 * 10**4 # Perley 2020

    # {'func': func, 'prop': dict, 'input':, 'as':}
    _MODEL = dict( redshift = {"kwargs":{"zmax":0.2},
                                  "as":"z"},
                              
                   x1 = {"func": SNeIaStretch.nicolas2021}, 
                   
                   c = {"func": SNeIaColor.intrinsic_and_dust},

                   hostmass = {"func": getpdf_asymetric_gaussian},

                   t0 = {"func": np.random.uniform, 
                         "kwargs": {"low":56_000, "high":56_200} },
                       
                   magabs = {"func": SNeIaMagnitude.tripp_and_massstep,
                             "kwargs": { "x1": "@x1", "c": "@c", "hostmass": "@hostmass",
                                        "mabs":-19.3, "sigmaint":0.10, "split":10}
                            },
                           
                   magobs = {"func": "magabs_to_magobs", # defined in Target (mother of Transients)
                             "kwargs": {"z":"@z", "magabs":"@magabs"},
                             },

                   x0 = {"func": "magobs_to_amplitude", # defined in Transients
                         "kwargs": {"magobs": "@magobs", "param_name":"x0"}
                        }, #because it needs to call sncosmo_model.get(param_name)
                       
                   radec = {"func": random_radec,
                            "kwargs": {},
                            "as": ["ra","dec"]}
                    )


class SNeIa_AgePop( Transient ):
    
    _KIND = "SNIa"
    _TEMPLATE = "salt2"
    _RATE = 2.35 * 10**4 # Perley 2020

    # {'func': func, 'prop': dict, 'input':, 'as':}
    _MODEL = dict( redshift = {"kwargs":{"zmax":0.4}, "as":"z"},
                  
                   prompt = {"func": Rigault_AgePop.get_promptpdf,
                             "kwargs":{"redshift":"@z"}
                            }, 
                  
                   x1 = {"func": SNeIaStretch.nicolas2021,
                        "kwargs":{"fprompt":"@prompt"}
                        }, 
                  
                   mass = {"func": Rigault_AgePop.get_massdistribution,
                          "kwargs": {"redshift":"@z", "fprompt":"@prompt"}
                          },
                  
                   c = {"func": SNeIaColor.intrinsic_and_dust},
                  
                   t0 = {"func": np.random.uniform, 
                         "kwargs": {"low":56_000, "high":56_200} 
                        },
                       
                   magabs = {"func": SNeIaMagnitude.tripp_and_step,
                             "kwargs": {"x1": "@x1", "c": "@c", "isup":'@prompt',
                                        "mabs":-19.3, "sigmaint":0.10, "gamma":0.2}
                            },
                           
                   magobs = {"func": "magabs_to_magobs", # defined in Target()
                             "kwargs": {"z":"@z", "magabs":"@magabs"},
                            },

                   x0 = {"func": "magobs_to_amplitude", # defined in Transients()
                         "kwargs": {"magobs":"@magobs", "param_name": "x0"},
                        }, #because it needs to call sncosmo_model.get(param_name)
                       
                   radec = {"func": random_radec,
                            "kwargs": {},
                            "as": ["ra","dec"]
                           }

                 )

class SNeIa_MassStepAgeStretch( Transient ):
    
    _KIND = "SNIa"
    _TEMPLATE = "salt2"
    _RATE = 2.35 * 10**4 # Perley 2020

    # {'func': func, 'prop': dict, 'input':, 'as':}
    _MODEL = dict( redshift = {"kwargs":{"zmax":0.4}, "as":"z"},
                  
                   prompt = {"func": Rigault_AgePop.get_promptpdf,
                             "kwargs":{"redshift":"@z"}
                            }, 
                  
                   x1 = {"func": SNeIaStretch.nicolas2021,
                        "kwargs":{"fprompt":"@prompt"}
                        }, 
                  
                   mass = {"func": Rigault_AgePop.get_massdistribution,
                          "kwargs": {"redshift":"@z", "fprompt":"@prompt"}
                          },
                  
                   c = {"func": SNeIaColor.intrinsic_and_dust},
                  
                   t0 = {"func": np.random.uniform, 
                         "kwargs": {"low":56_000, "high":56_200} 
                        },
                       
                   magabs = {"func": SNeIaMagnitude.tripp_and_massstep,
                             "kwargs": {"x1": "@x1", "c": "@c", "hostmass":'@mass',
                                        "mabs":-19.3, "sigmaint":0.10, "gamma":0.2}
                            },
                           
                   magobs = {"func": "magabs_to_magobs", # defined in Target()
                             "kwargs": {"z":"@z", "magabs":"@magabs"},
                            },

                   x0 = {"func": "magobs_to_amplitude", # defined in Transients()
                         "kwargs": {"magobs":"@magobs", "param_name": "x0"},
                        }, #because it needs to call sncosmo_model.get(param_name)
                       
                   radec = {"func": random_radec,
                            "kwargs": {},
                            "as": ["ra","dec"]
                           }

                 )


