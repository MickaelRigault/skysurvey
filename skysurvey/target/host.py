# Host 

import numpy as np

# =============== #
#                 #
#  Mass & SFR     #
#                 #
# =============== #
def get_sfr_as_function_of_mass_and_redshift(mass, redshift):
    """Get the star formation rate as a function of mass and redshift.

    This is based on equation A.7 from Childress et al. 2014, used in
    Wiseman et al. 2021.

    Parameters
    ----------
    mass : float or array
        The mass of the host galaxy.
    redshift : float or array
        The redshift of the host galaxy.

    Returns
    -------
    float or array
        The star formation rate.
    """
    ampl = (10**(mass)/10**10)**0.7
    fraction = np.exp(1.9*redshift)/ (np.exp(1.7* (redshift-2)) + np.exp(0.2* (redshift-2)))
    return  ampl * fraction

def get_schechterpdf(mass, mstar, alpha, phi, alpha2=None, phi2=None):
    """Get the Schechter probability density function."""
    delta_logmass = mass-mstar
    # single schechter
    if alpha2 is None or phi2 is None: 
        return np.log(10)* np.exp(-10**(delta_logmass)) *  phi*(10**delta_logmass)**(1+alpha) 
    # double schechter
    return np.log(10)* np.exp(-10**(delta_logmass)) * 10**(delta_logmass) * (phi*10**(delta_logmass*alpha) + 
                                                                    phi2*10**(delta_logmass*alpha2))
def get_stellarmassfunction(redshift, which="all", xx="6:13:100j"):
    """Get the stellar mass function.

    Parameters
    ----------
    redshift : float or array
        The redshift of the host galaxy.
    which : str, optional
        Which stellar mass function to use. Can be "all", "blue", or "red".
        The default is "all".
    xx : str, optional
        The mass range to use. The default is "6:13:100j".

    Returns
    -------
    tuple
        A tuple containing the mass array and the pdf.
    """
    if type(xx) == str: # assumed r_ input
        xx = eval(f"np.r_[{xx}]")
                
    prop = {#(0, 0.3): # Driver et al. 2022
            #{"blue":{"mstar": 10.70, "phi":0.855*1e-3, "alpha":-1.39}, # Disc | Moffett2016
            # "red": {"mstar": 10.74, "phi":3.67*1e-3, "alpha":-0.525}, # Spherical | Moffett2016
            # "all": {"mstar": 10.745, "phi":10**(-2.437), "alpha":-0.466, "phi2":10**(-3.201), "alpha2":-1.530}, # Driver et al. 2022
            #},
            (0., 0.5): # Mortlock et al. 2015
               {"blue":{"mstar": 10.83, "phi":10**(-3.31), "alpha":-1.41},
                "red": {"mstar": 10.90, "phi":10**(-4.87), "alpha":-1.74, "phi2":10**(-2.80), "alpha2":-0.42},
                "all": {"mstar": 10.90, "phi":10**(-3.51), "alpha":-1.59, "phi2":10**(-2.59), "alpha2":-0.71},
                },
            (0.5, 1.0):# Mortlock et al. 2015
               {"blue":{"mstar": 10.77, "phi":10**(-3.28), "alpha":-1.45},
                "red": {"mstar": 10.77, "phi":10**(-4.11), "alpha":-1.37, "phi2":10**(-2.75), "alpha2":-0.27},
                "all": {"mstar": 10.90, "phi":10**(-3.21), "alpha":-1.42, "phi2":10**(-2.93), "alpha2":-0.49},
                },
            (1.0, 1.5):# Mortlock et al. 2015
               {"blue":{"mstar": 10.64, "phi":10**(-3.14), "alpha":-1.37},
                "red": {"mstar": 10.78, "phi":10**(-2.96), "alpha":-0.35},
                "all": {"mstar": 11.04, "phi":10**(-3.21), "alpha":-1.31},
                },
            (1.5, 2.0):# Mortlock et al. 2015
               {"blue":{"mstar": 11.01, "phi":10**(-4.05), "alpha":-1.74},
                "red": {"mstar": 10.71, "phi":10**(-3.31), "alpha":-0.24},
                "all": {"mstar": 11.00, "phi":10**(-3.50), "alpha":-1.40}, # guess, missing line
                }
           }
    
    redshift = np.atleast_1d(redshift)
    if len(redshift)==1:
        redshift_ = redshift[0]
        for k,prop_ in prop.items():
            if redshift_>k[0] and redshift_<k[1]:
                break
        pdf = [get_schechterpdf(xx, **prop_[which])]
    else:
        pdf = []
        for redshift_ in redshift:
            for k,prop_ in prop.items():
                if redshift_>k[0] and redshift_<k[1]:
                    break
            pdf.append(get_schechterpdf(xx, **prop_[which]))
            
        
    return xx, np.asarray(pdf)
