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
    bands: str or list
        Band or list of bands.
    fill_value: str or None, optional
        Value to fill if the band is not found.

    Returns
    -------
    str or list
        Color or list of colors.
    """
    squeeze = type(bands) in [str, np.str_]
    colors = [BAND_COLORS.get(band_, fill_value) for band_ in np.atleast_1d(bands)]
    return colors if not squeeze else colors[0]
