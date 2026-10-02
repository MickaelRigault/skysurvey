"""Default photometric band colors and a utility function to retrieve them."""

import numpy as np

BAND_COLORS  = {"ztfr":"tab:red",
                "ztfg":"tab:green",
                "ztfi":"tab:orange",
                "desg":"forestgreen",
                "desr":"crimson",
                "desi":"darkgoldenrod",
                "desz":"0.4",
                    }

def get_band_color(bands, fill_value=None):
    """Get the color of the given bands.

    Parameters
    ----------
    bands : str or list of str
        Band or list of bands.

    fill_value : str or None, optional
        Value to return if the band is not found. The default is None.

    Returns
    -------
    str or list of str
        Color (if `bands` is a str) or list of colors.
    """
    squeeze = isinstance(bands, (str, np.str_))
    colors = [BAND_COLORS.get(band_, fill_value) for band_ in np.atleast_1d(bands)]
    return colors if not squeeze else colors[0]
