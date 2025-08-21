import numpy as np

def skewed_gaussian_pdf(xx, loc, scale_low, scale_high):
    """ Compute the probability density function (PDF) of a skewed Gaussian distribution.

    The skewed Gaussian distribution is defined by different scale parameters
    for values below and above the location parameter

    Parameters
    ----------
    xx : array_like or str
        Input values at which to evaluate the PDF.
        If a string, it is assumed to be in NumPy's `r_` format (e.g., "1:10").
    loc : float
        Location parameter (mean) of the distribution.
    scale_low : float
        Scale parameter (standard deviation) for values less than `loc`.
    scale_high : float
        Scale parameter (standard deviation) for values greater than `loc`.

    Returns
    -------
    xx : ndarray
        Input values as a NumPy array.
    pdf : ndarray
        Probability density function values for the input `xx`.

    Notes
    -----
    The normalization factor ensures the PDF integrates to 1.
    """
    if type(xx) == str: # assumed r_ input
        xx = eval(f"np.r_[{xx}]")
        
    # doing it first symetric assuming sigma-low
    pdf_unnormed = np.exp(-0.5 * ((xx-loc)/(scale_low))**2 )

    # doing the sigma_high now
    pdf_unnormed[xx>loc] = np.exp(-0.5 * ((xx[xx>loc]-loc)/(scale_high))**2 )

    # normalisation
    sigma_eff = 0.5*(scale_high+scale_low) # mean sigma
    norm = 1/(sigma_eff * np.sqrt(2*np.pi))
    
    return xx, norm * pdf_unnormed
