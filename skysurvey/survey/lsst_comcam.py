#skysurvey/survey/lsst_comcam.py

import pandas as pd
import numpy as np
from shapely.geometry import box
from skysurvey import Survey
from skysurvey.tools.utils import get_skynoise_from_maglimit
from skysurvey.target import SNeIa
from shapely.ops import unary_union
from skysurvey.effects import mw_extinction
import matplotlib.pyplot as plt


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

def estimate_snia_population_from_parquet(
    parquet_path,
    zp=30.0,
    exclude_yband=True,
    fov_arcmin=40.0,
    zmin=0.0,
    zmax=0.2,
    include_effects=True,
):
    """
    Load a Survey from DP1 parquet and estimate SNeIa population given z-range and footprint.

    Parameters
    ----------
    parquet_path : str
        Path to visit-level parquet.
    zp : float
        Zeropoint for skynoise conversion.
    exclude_yband : bool
        Whether to exclude 'y' band observations.
    fov_arcmin : float
        Field of view size for footprint box.
    zmin, zmax : float
        Redshift bounds for population simulation.
    include_effects : bool
        Whether to apply mw_extinction.

    Returns
    -------
    snia : SNeIa target object
        The simulated population overlapping the footprint.
    survey : Survey object
        The loaded survey.
    """
    # Load survey from parquet
    survey = from_dp1_parquet(
        parquet_path,
        zp=zp,
        exclude_yband=exclude_yband,
        fov_arcmin=fov_arcmin,
    )

    # Reconstruct shapely.MultiPolygon skyarea from survey visit centers
    visit_df = pd.read_parquet(parquet_path)
    if exclude_yband:
        visit_df = visit_df[visit_df["band"] != "y"]

    fov_deg = fov_arcmin / 60.0
    half_fov = fov_deg / 2.0
    tiles = []
    for _, row in visit_df.iterrows():
        ra = row["ra"]
        dec = row["dec"]
        tile = box(ra - half_fov, dec - half_fov, ra + half_fov, dec + half_fov)
        tiles.append(tile)
    skyarea = unary_union(tiles)

    # Time range of the survey
    tstart, tstop = survey.get_timerange()

    # Build population with or without extinction
    effects = mw_extinction if include_effects else None
    
    snia = SNeIa.from_draw(
        tstart=tstart,
        tstop=tstop,
        skyarea=skyarea,
        zmin=zmin,
        zmax=zmax,
        effect=effects,
    )

    return snia, survey


