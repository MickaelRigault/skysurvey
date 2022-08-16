""" General configuration imput """

import numpy as np

BAND_COLORS  = {"ztfr":"tab:red",
                "ztfg":"tab:green",
                "ztfi":"tab:orange"}

def get_band_color(bands, fill_value=None):
    """ """
    squeeze = type(bands) in [str, np.str_]
    colors = [BAND_COLORS.get(band_, fill_value) for band_ in np.atleast_1d(bands)]
    return colors if not squeeze else colors[0]

