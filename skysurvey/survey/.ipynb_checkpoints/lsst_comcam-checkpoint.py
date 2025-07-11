#skysurvey/survey/lsst_comcam.py

import pandas as pd
from shapely.geometry import box
from skysurvey import Survey
from skysurvey.tools.utils import get_skynoise_from_maglimit

def from_dp1_parquet(filepath, zp=30.0, exclude_yband=True, fov_arcmin=40.0):
    """
    Construct a skysurvey.Survey from a DP1 visit-level parquet file.

    Parameters
    ----------
    filepath : str
        Path to a `dp1_visits.parquet` file (or filtered version).
    zp : float
        Zeropoint to use in sky noise conversion (default: 30).
    exclude_yband : bool
        If True, excludes 'y' band visits.
    fov_arcmin : float
        Field of view in arcminutes (default: 40 for LSST ComCam).

    Returns
    -------
    survey : skysurvey.Survey
        Survey object usable with skysurvey.
    """
    df = pd.read_parquet(filepath)

    if exclude_yband:
        df = df[df["band"] != "y"]

    simdata = pd.DataFrame({
        "skynoise": df["limitingMagnitude"].apply(get_skynoise_from_maglimit, zp=zp).values,
        "mjd": df["mjd"].values,
        "band": "lsst" + df["band"].values,
        "gain": 1.0,
        "zp": zp,
        "ra": df["ra"].values,
        "dec": df["dec"].values
    }, index=df.index)

    # Construct square footprint (centered on 0,0 in local projection)
    fov_deg = fov_arcmin / 60.0
    half_size = fov_deg / 2.0
    footprint = box(-half_size, -half_size, half_size, half_size)

    return Survey.from_pointings(simdata, footprint=footprint)

