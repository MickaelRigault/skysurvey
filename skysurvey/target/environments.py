import numpy as np
from scipy import stats

def getpdf_asymetric_gaussian(xx="6:13:0.01",
                                ksi=10, omega=1, alpha=-3):
    """ asymetric gaussian 

    """
    if type(xx) == str: # assumed r_ input
        xx = eval(f"np.r_[{xx}]")
         
    pdf = (2/omega) * stats.norm.pdf(xx, ksi, omega) * stats.norm.cdf(alpha*(xx-ksi)/omega, 0, 1)
    return xx, pdf
