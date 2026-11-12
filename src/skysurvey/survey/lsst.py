"""LSST (Rubin Observatory) camera layout and survey class.

This module defines the LSST camera footprint (189 science CCDs, or the
camera outline), the conversion between the focal plane and the sky given the
camera rotation of each visit (OpSim ``rotSkyPos``), and the `LSST` survey,
built from an OpSim database (see :meth:`LSST.from_opsim`).

The camera layout comes from the Rubin camera description used by the Rubin
pipelines and imSim, `lsst/obs_lsst <https://github.com/lsst/obs_lsst>`_
(``policy/``), and the photometric zero points from
`SMTN-002 <https://smtn-002.lsst.io/>`_.
"""

import warnings
import pandas
import numpy as np

import geopandas
from shapely import geometry

from .basesurvey import GridSurvey
from ..tools.projection import camera_to_sky, sky_to_camera

__all__ = ["LSST", "get_lsst_ccd_footprints", "get_lsst_footprint", "get_lsst_fields",
           "parse_opsim", "read_opsim"]


# ================= #
#   Camera layout   #
# ================= #
# Rubin camera description, lsst/obs_lsst (policy/):
# - plate scale: cameraTransforms.yaml
# - raft centers: rafts.yaml (5x5 grid, 127 mm pitch; the 4 corner rafts only have
#   wavefront and guide sensors)
# - CCD centers within a raft: cameraHeader.yaml (RAFT_ITL, RAFT_E2V; 3x3, 42.25 mm pitch)
# - CCD imaging area: cameraHeader.yaml (CCD_ITL, CCD_E2V; 10 micron pixels)
# - vendor of each raft: lsstCam/R*.yaml
# Raft Rrc (row r, column c) is centered at x=(c-2) x 127 mm, y=(r-2) x 127 mm,
# and its CCD Src at (c-1) x 42.25 mm, (r-1) x 42.25 mm from the raft center.
LSST_PLATESCALE = 20.005867576692737 # [arcsec/mm]
LSST_RAFT_PITCH = 127. # [mm]
LSST_CCD_PITCH = 42.25 # [mm]
LSST_CCD_SIZE = {"ITL": (4072 * 0.010, 4000 * 0.010), # [mm] (x, y)
                 "E2V": (4096 * 0.010, 4004 * 0.010)}

# science rafts, in detector id order, and their CCD vendor
LSST_RAFT_VENDORS = {"R01": "ITL", "R02": "ITL", "R03": "ITL",
                     "R10": "ITL", "R11": "E2V", "R12": "E2V", "R13": "E2V", "R14": "E2V",
                     "R20": "ITL", "R21": "E2V", "R22": "E2V", "R23": "E2V", "R24": "E2V",
                     "R30": "E2V", "R31": "E2V", "R32": "E2V", "R33": "E2V", "R34": "E2V",
                     "R41": "ITL", "R42": "ITL", "R43": "ITL"}

# SMTN-002 (version of 2026-08-12): magnitude giving 1 electron per second
LSST_ZP_INSTRUMENTAL = {"u": 26.52, "g": 28.51, "r": 28.36, "i": 28.17, "z": 27.78, "y": 26.82}


# Geopandas of the individual ccd footprint within the camera
def get_lsst_ccd_footprints(rotation=0, exclude_ccds=None, as_dataframe=False):
    """Get the footprint of each LSST science CCD in the focal plane.

    The 189 science CCDs are in 21 rafts of 3x3 CCDs, on a 5x5 grid without
    the 4 corners (see `LSST_RAFT_VENDORS`).

    Parameters
    ----------
    rotation : float, optional
        Rotation of the camera in deg (counter-clockwise). The default is 0.

    exclude_ccds : list or None, optional
        CCD ids (detector ids) to remove. The default is None.

    as_dataframe : bool, optional
        If True, return a GeoDataFrame with the CCD ``name`` (e.g. ``R22_S11``)
        and ``vendor`` (``E2V`` or ``ITL``). The default is False.

    Returns
    -------
    geopandas.GeoSeries or geopandas.GeoDataFrame
        One box per CCD, in deg, centered on the camera center (y is 'up'),
        indexed by ``ccdid`` (detector id, 0 to 188).
    """
    scale = LSST_PLATESCALE / 3600 # [deg/mm]
    ccds, names, vendors = {}, {}, {}
    for iraft, (raft_, vendor_) in enumerate(LSST_RAFT_VENDORS.items()):
        raft_row, raft_col = int(raft_[1]), int(raft_[2])
        width, height = LSST_CCD_SIZE[vendor_]
        for ccd_row in range(3):
            for ccd_col in range(3):
                ccdid = 9 * iraft + 3 * ccd_row + ccd_col # raft id0 + S index (obs_lsst)
                x = (raft_col - 2) * LSST_RAFT_PITCH + (ccd_col - 1) * LSST_CCD_PITCH # [mm]
                y = (raft_row - 2) * LSST_RAFT_PITCH + (ccd_row - 1) * LSST_CCD_PITCH # [mm]
                ccds[ccdid] = geometry.box((x - width / 2) * scale, (y - height / 2) * scale,
                                           (x + width / 2) * scale, (y + height / 2) * scale)
                names[ccdid] = f"{raft_}_S{ccd_row}{ccd_col}"
                vendors[ccdid] = vendor_

    ccds = geopandas.GeoDataFrame({"name": names, "vendor": vendors}, geometry=geopandas.GeoSeries(ccds))
    ccds.index.name = "ccdid"
    if rotation != 0:
        ccds = ccds.set_geometry(ccds.rotate(rotation, origin=(0, 0)))

    if exclude_ccds is not None:
        ccds = ccds.drop(exclude_ccds)

    if as_dataframe:
        return ccds

    return ccds.geometry

# The camera footprint, with or without the ccd details.
def get_lsst_footprint(per_ccd=True, **kwargs):
    """Get the LSST camera footprint.

    The 189 CCDs cover 9.55 deg2 of silicon (the official field of view is
    9.6 deg2); their outline encloses 10.38 deg2, gaps included.

    Parameters
    ----------
    per_ccd : bool, optional
        If True, the union of the science CCDs (a MultiPolygon, with the gaps
        between CCDs and rafts). If False, the outline of the camera as a
        single polygon (gaps filled). The default is True.

    **kwargs
        Goes to :func:`get_lsst_ccd_footprints`.

    Returns
    -------
    shapely.geometry.MultiPolygon or shapely.geometry.Polygon
        The camera footprint in deg, centered on 0.
    """
    ccd_footprint = get_lsst_ccd_footprints(**kwargs).union_all()
    if per_ccd:
        return ccd_footprint

    # closing the gaps: growing the CCDs by more than half the largest gap
    # (~45 arcsec, between CCD rows) merges them, shrinking them back restores
    # the outer edges. (mitre: keeps the square corners of the CCDs)
    gap = 50 / 3600 # [deg]
    return ccd_footprint.buffer(gap, join_style="mitre").buffer(-gap, join_style="mitre")


def get_lsst_ccdid(x, y, ccd_footprints=None):
    """Get the CCD containing focal plane positions.

    The CCDs are boxes centered on a regular 15x15 grid (raft and CCD
    pitches): each position is compared to its nearest CCD center.

    Parameters
    ----------
    x, y : array_like
        Focal plane positions in deg (see :func:`radec_to_camera`).

    ccd_footprints : geopandas.GeoSeries, optional
        CCD footprints (see :func:`get_lsst_ccd_footprints`, without
        rotation). If None, the default layout. The default is None.

    Returns
    -------
    numpy.ndarray
        Detector id of each position; -1 if in a gap or outside the CCDs.
    """
    if ccd_footprints is None:
        ccd_footprints = get_lsst_ccd_footprints()

    # grid of CCD centers | 15 x 15 positions, -1 for the corner rafts (and excluded CCDs)
    centers = ccd_footprints.centroid
    xcenters, ycenters = np.unique(np.round(centers.x, 9)), np.unique(np.round(centers.y, 9))
    grid = np.full((len(ycenters), len(xcenters)), -1)
    grid[np.searchsorted(ycenters, np.round(centers.y, 9)),
         np.searchsorted(xcenters, np.round(centers.x, 9))] = ccd_footprints.index
    bounds = ccd_footprints.bounds
    halfwidth = ((bounds["maxx"] - bounds["minx"]) / 2).to_numpy() # CCD sizes depend on the vendor
    halfheight = ((bounds["maxy"] - bounds["miny"]) / 2).to_numpy()

    def get_nearest(values, centers_):
        """ index of the nearest center """
        i = np.clip(np.searchsorted(centers_, values), 1, len(centers_) - 1)
        return np.where(np.abs(values - centers_[i - 1]) < np.abs(values - centers_[i]), i - 1, i)

    x, y = np.atleast_1d(x), np.atleast_1d(y)
    ix, iy = get_nearest(x, xcenters), get_nearest(y, ycenters)
    ccdid = grid[iy, ix]

    # within the box of that CCD
    on_grid = np.flatnonzero(ccdid >= 0)
    row = np.searchsorted(ccd_footprints.index.to_numpy(), ccdid[on_grid])
    inside = (np.abs(x[on_grid] - xcenters[ix[on_grid]]) <= halfwidth[row]) & \
             (np.abs(y[on_grid] - ycenters[iy[on_grid]]) <= halfheight[row])

    out = np.full(len(x), -1)
    out[on_grid[inside]] = ccdid[on_grid[inside]]
    return out


# ======================= #
#   Camera on the sky     #
# ======================= #
# OpSim rotSkyPos: "the angle between ICRF North on the sky and the 'up' direction in
# the focal plane" (rubin_scheduler output schema). SITCOMTN-003 (section 10): at zero
# rotation North is up and East is left, and an increase of rotSkyPos makes North move
# toward East relative to the camera.
# This is the convention of skysurvey.tools.projection.camera_to_sky and sky_to_camera.
# N.B. this convention follows the documentation, it has not been validated against the
# WCS of LSSTCam images. It does not change coverage statistics, only which CCD is where.

def project_lsst_footprint(footprint, ra, dec, rotskypos=0):
    """Project a focal plane footprint on the sky, one polygon per visit.

    This is :func:`skysurvey.tools.projection.project_to_radec` with a
    camera rotation per visit.

    Parameters
    ----------
    footprint : shapely.geometry.Polygon
        Footprint in the focal plane, in deg (e.g. the camera outline,
        ``get_lsst_footprint(per_ccd=False)``).

    ra, dec : float or array_like
        Boresight of each visit in deg.

    rotskypos : float or array_like, optional
        Camera rotation of each visit in deg. The default is 0.

    Returns
    -------
    list of shapely.geometry.Polygon
        One polygon (ra, dec in deg) per visit.
    """
    from ..tools.projection import rot_xz_sph

    x, y = np.asarray(footprint.exterior.xy) # [nvertices]
    ra, dec, rotskypos = np.broadcast_arrays(*[np.atleast_1d(v_).astype(float) for v_ in [ra, dec, rotskypos]])
    east, north = camera_to_sky(x[:, None], y[:, None], rotskypos) # [nvertices, nvisits]
    # as in project_to_radec | shape (2, 1, nvertices, nvisits)
    ra_, dec_ = np.asarray(rot_xz_sph(east / np.cos(np.radians(north)), north, dec))[:, 0]
    ra_ = ra_ + ra
    return [geometry.Polygon(np.stack([ra__, dec__], axis=-1)) for ra__, dec__ in zip(ra_.T, dec_.T)]


def radec_to_camera(ra, dec, ra_boresight, dec_boresight, rotskypos=0):
    """Get the focal plane position of sky coordinates.

    This is the inverse of :func:`project_lsst_footprint`, for points.

    Parameters
    ----------
    ra, dec : array_like
        Sky coordinates in deg.

    ra_boresight, dec_boresight : array_like
        Boresight of the visit, one per coordinate (or a single one), in deg.

    rotskypos : float or array_like, optional
        Camera rotation of the visit in deg. The default is 0.

    Returns
    -------
    x, y : numpy.ndarray
        Focal plane positions in deg.
    """
    from ..tools.projection import rot_xz_sph

    ra, dec, ra_boresight, dec_boresight = np.broadcast_arrays(
        *[np.atleast_1d(v_).astype(float) for v_ in [ra, dec, ra_boresight, dec_boresight]])
    l, b = np.asarray(rot_xz_sph(ra - ra_boresight, dec, -dec_boresight)).reshape(2, -1)
    l = (l + 180) % 360 - 180 # ra offset in [-180, 180[
    return sky_to_camera(l * np.cos(np.radians(b)), b, rotskypos)


# ================== #
#   Visits (OpSim)   #
# ================== #
def read_opsim(filepath, columns = ["fieldRA", "fieldDec", "observationStartMJD",
                                    "visitExposureTime", "filter", "skyBrightness",
                                    "fiveSigmaDepth", "night", "numExposures",
                                    "observationId"],
              sql_where=None):
    """Parse an input OpSim database and return a dataframe.

    Parameters
    ----------
    filepath : str
        Path to the OpSim database.

    columns : list of str or None, optional
        List of columns to load from the OBSERVATIONS table. If None, all
        columns are loaded. If 'note' is requested but absent, 'scheduler_note'
        is used instead if available, otherwise it is dropped. The default is
        ["fieldRA", "fieldDec", "observationStartMJD", "visitExposureTime",
        "filter", "skyBrightness", "fiveSigmaDepth", "night", "numExposures",
        "observationId"].

    sql_where : str, optional
        SQL condition to select the rows to load (e.g. 'night<365').
        If None, all rows are loaded. The default is None.

    Returns
    -------
    pandas.DataFrame
        The loaded observations.
    """
    import sqlite3
    connect = sqlite3.connect(filepath)

    # Detect which note column name this db uses, if any
    cursor = connect.execute("PRAGMA table_info(OBSERVATIONS)")
    available_cols = {row[1] for row in cursor.fetchall()}

    if columns is None:
        sql_columns = "*"
    else:
        cols = list(columns)
        # Handle note column rename between opsim versions
        if "note" in cols:
            if "note" not in available_cols and "scheduler_note" in available_cols:
                cols[cols.index("note")] = "scheduler_note"
            elif "note" not in available_cols:
                cols.remove("note") # if neither exist, just drop it
        sql_columns = ", ".join(np.atleast_1d(cols))

    if sql_where is None:
        sql_where = ""
    else:
        sql_where = f"WHERE {sql_where}"

    df = pandas.read_sql_query(f'SELECT {sql_columns} FROM OBSERVATIONS {sql_where}', connect)
    connect.close()
    return df


def get_lsst_gain(band, exptime, zp=30, zp_instrumental=None):
    """Get the gain (electrons per flux unit) of LSST visits.

    skysurvey simulates the flux variance as ``skynoise**2 + flux/gain``,
    with the flux in units given by `zp`: the gain is the number of electrons
    per flux unit, ``10**(0.4 * (zp_band + 2.5 log10(exptime) - zp))``.

    Parameters
    ----------
    band : str or array_like
        LSST band(s): u, g, r, i, z or y.

    exptime : float or array_like
        Visit exposure time in s (OpSim ``visitExposureTime``).

    zp : float, optional
        Zero point of the flux unit. The default is 30.

    zp_instrumental : dict, optional
        Instrumental zero point per band (magnitude giving 1 e-/s). If None,
        `LSST_ZP_INSTRUMENTAL` (SMTN-002). The default is None.

    Returns
    -------
    numpy.ndarray
        Number of electrons per flux unit.
    """
    if zp_instrumental is None:
        zp_instrumental = LSST_ZP_INSTRUMENTAL

    zp_band = pandas.Series(np.atleast_1d(band)).map(zp_instrumental).to_numpy(dtype=float)
    return 10**(0.4 * (zp_band + 2.5 * np.log10(exptime) - zp))


# reads the opsim visits and provides the data that should be used by skysurvey
def parse_opsim(opsim, sql_where=None, zp=30, zp_instrumental=None, rotate=True):
    """Parse OpSim visits into skysurvey observing data.

    OpSim gives one row per visit (no CCD information): the boresight
    (``fieldRA``, ``fieldDec``), the camera rotation (``rotSkyPos``) and the
    5-sigma depth of the visit (``fiveSigmaDepth``, which includes the sky,
    seeing, airmass and read noise).

    Parameters
    ----------
    opsim : str or pandas.DataFrame
        Path to the OpSim database, or the visits themselves (e.g. from
        :func:`read_opsim`), with ``observationId``, ``fieldRA``,
        ``fieldDec``, ``rotSkyPos``, ``observationStartMJD``, ``filter``,
        ``fiveSigmaDepth`` and ``visitExposureTime``.

    sql_where : str, optional
        SQL condition to select the visits (e.g. ``"night < 365"``). Only
        used if `opsim` is a path. The default is None.

    zp : float, optional
        Zero point of the simulated fluxes. The default is 30.

    zp_instrumental : dict, optional
        Instrumental zero point per band (see :func:`get_lsst_gain`).
        The default is None.

    rotate : bool, optional
        Use the camera rotation of each visit. If False, all visits have
        ``rotskypos=0``. The default is True.

    Returns
    -------
    pandas.DataFrame
        One row per visit, with ``mjd``, ``band`` (``lsst`` + filter),
        ``skynoise``, ``gain``, ``zp``, ``fieldid`` (``observationId``) and the
        boresight ``ra``, ``dec`` and ``rotskypos`` (deg).

    Warns
    -----
    UserWarning
        If `sql_where` is given with visits already loaded (ignored).
    """
    from ..tools.utils import get_skynoise_from_maglimit

    if type(opsim) is str:
        columns = ["observationId", "fieldRA", "fieldDec", "rotSkyPos", "observationStartMJD",
                   "filter", "fiveSigmaDepth", "visitExposureTime"]
        opsim = read_opsim(opsim, columns=columns, sql_where=sql_where)
    elif sql_where is not None:
        warnings.warn("sql_where is ignored: the visits are already loaded.")

    rotskypos = opsim["rotSkyPos"].to_numpy() if rotate else 0.
    data = pandas.DataFrame({"mjd": opsim["observationStartMJD"].to_numpy(),
                             "band": ("lsst" + opsim["filter"]).to_numpy(),
                             "skynoise": get_skynoise_from_maglimit(opsim["fiveSigmaDepth"].to_numpy(), zp=zp),
                             "gain": get_lsst_gain(opsim["filter"], opsim["visitExposureTime"].to_numpy(),
                                                   zp=zp, zp_instrumental=zp_instrumental),
                             "zp": zp,
                             "fieldid": opsim["observationId"].to_numpy(),
                             "ra": opsim["fieldRA"].to_numpy(),
                             "dec": opsim["fieldDec"].to_numpy(),
                             "rotskypos": rotskypos})
    return data


def get_lsst_fields(ra, dec, rotskypos=0, fieldid=None, footprint=None):
    """Get the sky geometry of LSST visits.

    Parameters
    ----------
    ra, dec : array_like
        Boresight of each visit in deg.

    rotskypos : float or array_like, optional
        Camera rotation of each visit in deg. The default is 0.

    fieldid : array_like, optional
        Id of each visit. If None, ``0..nvisits-1``. The default is None.

    footprint : shapely.geometry.Polygon, optional
        Camera footprint in the focal plane. If None, the camera outline
        (``get_lsst_footprint(per_ccd=False)``). The default is None.

    Returns
    -------
    geopandas.GeoDataFrame
        One polygon per visit indexed by ``fieldid``, with the boresight
        (``ra``, ``dec``) and ``rotskypos``.
    """
    if footprint is None:
        footprint = get_lsst_footprint(per_ccd=False)

    ra, dec, rotskypos = np.broadcast_arrays(*[np.atleast_1d(v_).astype(float) for v_ in [ra, dec, rotskypos]])
    if fieldid is None:
        fieldid = np.arange(len(ra))

    return geopandas.GeoDataFrame({"ra": ra, "dec": dec, "rotskypos": rotskypos},
                                  geometry=project_lsst_footprint(footprint, ra, dec, rotskypos),
                                  index=pandas.Index(fieldid, name="fieldid"))

# ============== #
#                #
#  LSST Camera   #
#                #
# ============== #
class LSST( GridSurvey ):
    """A class to model the LSST survey from OpSim visits.

    Each visit is a field: the camera outline projected with the boresight and
    camera rotation of the visit (``rotSkyPos``), indexed by ``fieldid`` (the
    OpSim ``observationId``). With `per_ccd` (default), coordinates matched to
    a visit (:meth:`radec_to_fieldid`, used by :class:`skysurvey.DataSet`)
    must also fall on a CCD: the gaps between CCDs (8% of the outline) are not
    observed, and the ``ccdid`` is given. See :meth:`from_opsim`, the preferred
    way to build an instance.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Observing data, with a ``fieldid`` column. The default is None.

    fields : geopandas.GeoDataFrame, optional
        Visit geometries, with the boresight (``ra``, ``dec``) and
        ``rotskypos`` (see :func:`get_lsst_fields`). The default is None.

    footprint : shapely.geometry.Polygon, optional
        Camera footprint (outline). If None, ``get_lsst_footprint(per_ccd=False)``.
        The default is None.

    per_ccd : bool, optional
        Should coordinates falling in CCD gaps be removed? The default is True.

    ccd_footprints : geopandas.GeoSeries, optional
        CCD footprints used with `per_ccd` (see :func:`get_lsst_ccd_footprints`).
        If None, all science CCDs. The default is None.

    **kwargs
        Goes to :class:`skysurvey.survey.basesurvey.GridSurvey`.
    """
    _FOOTPRINT = get_lsst_footprint(per_ccd=False)

    def __init__(self, data=None, fields=None, footprint=None, per_ccd=True,
                 ccd_footprints=None, **kwargs):
        """Initialize the LSST class. See the class docstring."""
        if footprint is None:
            footprint = self._FOOTPRINT

        self._per_ccd = per_ccd
        self._ccd_footprints = ccd_footprints
        super().__init__(data=data, fields=fields, footprint=footprint, **kwargs)

    @classmethod
    def from_opsim(cls, opsim, sql_where=None, per_ccd=True, rotate=True,
                   exclude_ccds=None, zp=30, zp_instrumental=None):
        """Load an LSST survey from an OpSim database.

        Parameters
        ----------
        opsim : str or pandas.DataFrame
            Path to the OpSim database, or the visits themselves
            (see :func:`parse_opsim`).

        sql_where : str, optional
            SQL condition to select the visits (e.g. ``"night < 365"``).
            The default is None.

        per_ccd : bool, optional
            Remove coordinates falling in CCD gaps (see the class docstring).
            The default is True.

        rotate : bool, optional
            Use the camera rotation of each visit (``rotSkyPos``). If False,
            all visits have rotskypos=0. The default is True.

        exclude_ccds : list or None, optional
            CCD ids (detector ids) considered as not observing, used with
            `per_ccd`. The default is None.

        zp : float, optional
            Zero point of the simulated fluxes. The default is 30.

        zp_instrumental : dict, optional
            Instrumental zero point per band (see :func:`get_lsst_gain`).
            The default is None.

        Returns
        -------
        LSST
            The LSST survey instance.

        Warns
        -----
        UserWarning
            If `exclude_ccds` is given with ``per_ccd=False`` (ignored).
        """
        data = parse_opsim(opsim, sql_where=sql_where, zp=zp,
                           zp_instrumental=zp_instrumental, rotate=rotate)

        # one field per visit: the camera outline, rotated.
        fields = get_lsst_fields(data["ra"], data["dec"], data["rotskypos"], fieldid=data["fieldid"])

        if exclude_ccds is not None and not per_ccd:
            warnings.warn("exclude_ccds is ignored with per_ccd=False.")

        ccd_footprints = get_lsst_ccd_footprints(exclude_ccds=exclude_ccds)
        return cls(data=data, fields=fields, per_ccd=per_ccd, ccd_footprints=ccd_footprints)

    # ================ #
    #   Methods        #
    # ================ #
    def radec_to_fieldid(self, radec, observed_fields=False):
        """Get the visits (fields) covering the given coordinates.

        Parameters
        ----------
        radec : pandas.DataFrame or array_like
            Coordinates in deg, either a DataFrame with 'ra' and 'dec' columns
            or an array of shape (N, 2).

        observed_fields : bool, optional
            Should this be limited to fields actually observed?
            The default is False.

        Returns
        -------
        pandas.DataFrame
            Fieldid(s) of each coordinate, indexed as `radec`; with the
            ``ccdid`` if `per_ccd` (coordinates in gaps removed).
        """
        if type(radec) in [np.ndarray, list, tuple]:
            radec = pandas.DataFrame(np.atleast_2d(radec), columns=["ra", "dec"])

        fieldids = super().radec_to_fieldid(radec, observed_fields=observed_fields)
        if not self.per_ccd:
            return fieldids

        # focal plane position of each (coordinate, visit) match
        radec_ = radec.loc[fieldids.index]
        visits_ = self.fields.loc[fieldids["fieldid"], ["ra", "dec", "rotskypos"]]
        x, y = radec_to_camera(radec_["ra"].to_numpy(), radec_["dec"].to_numpy(),
                               visits_["ra"].to_numpy(), visits_["dec"].to_numpy(),
                               visits_["rotskypos"].to_numpy())
        fieldids = fieldids.assign(ccdid=get_lsst_ccdid(x, y, ccd_footprints=self.ccd_footprints))
        return fieldids[fieldids["ccdid"] >= 0]

    def get_nvisits_healpix(self, nside=32, bands=None):
        """Get the number of visits covering the center of each healpix pixel.

        Parameters
        ----------
        nside : int, optional
            Healpix nside (32: 12288 pixels of 1.8 deg). The default is 32.

        bands : str, list or None, optional
            Only count the visits in these bands (e.g. ``"lsstr"``). If None,
            all bands. The default is None.

        Returns
        -------
        numpy.ndarray
            Number of visits per healpix pixel (ring ordering).

        Notes
        -----
        The visits are counted at the pixel centers (not averaged over the
        pixel): with `per_ccd`, a center falling in a CCD gap of a visit is
        not counted for that visit.
        """
        import healpy as hp

        npix = hp.nside2npix(nside)
        ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        fieldids = self.radec_to_fieldid(pandas.DataFrame({"ra": ra, "dec": dec}))
        if bands is not None:
            band_of_field = self.data.set_index("fieldid")["band"]
            fieldids = fieldids[band_of_field.loc[fieldids["fieldid"]].isin(np.atleast_1d(bands)).to_numpy()]

        return np.bincount(fieldids.index.to_numpy(), minlength=npix)

    # --------- #
    #   PLOT    #
    # --------- #
    def show_footprint(self, ax=None, incl_both=True): # pragma: no cover
        """Show the LSST camera footprint in the focal plane.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes where to draw. If None, a new figure is created.
            The default is None.

        incl_both : bool, optional
            Show both the CCDs (with their detector id) and the camera
            outline. If False, only the one used by this survey (see
            `per_ccd`). The default is True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        ccd_footprints = self.ccd_footprints if (incl_both or self.per_ccd) else None
        edge_footprint = self.footprint if (incl_both or not self.per_ccd) else None

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7.5))
        else:
            fig = ax.figure

        # loop over the details of the camera.
        if ccd_footprints is not None:
            for ccdid_, ccd_ in ccd_footprints.items():
                ax.fill(*ccd_.exterior.xy, color="0.8", ec="0.3", lw=0.5)
                ax.text(ccd_.centroid.x, ccd_.centroid.y, f"{ccdid_}", ha="center", va="center", fontsize=5)

        if edge_footprint is not None:
            ax.plot(*edge_footprint.exterior.xy, color="C1", lw=2, label="camera outline (per_ccd=False)")

        ax.add_patch( Circle((0, 0), 1.75, fill=False, ls="--", color="C0", label="3.5 deg diameter"))
        ax.set_aspect("equal")
        ax.set_xlabel("x [deg]", fontsize="large")
        ax.set_ylabel("y [deg]", fontsize="large")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        return fig

    def show_nvisits_map(self, ra, dec, size=4, step=0.02, bands=None, ax=None,
                         cmap="viridis", **kwargs): # pragma: no cover
        """Show the number of visits around a position (e.g. a deep drilling field).

        Parameters
        ----------
        ra, dec : float
            Center of the map in deg.

        size : float, optional
            Size of the map in deg. The default is 4.

        step : float, optional
            Grid step in deg. The default is 0.02.

        bands : str, list or None, optional
            Only count the visits in these bands. If None, all bands.
            The default is None.

        ax : matplotlib.axes.Axes, optional
            Axes where to draw. If None, a new figure is created.
            The default is None.

        cmap : str, optional
            Colormap. The default is "viridis".

        **kwargs
            Goes to :meth:`matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.

        See Also
        --------
        get_nvisits_per_regions : The number of visits on a grid.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure

        region = geometry.box(ra - size / 2, dec - size / 2, ra + size / 2, dec + size / 2)
        rabins, decbins, nvisits = self.get_nvisits_per_regions(region, step=step, bands=bands)
        prop = dict(vmin=0, cmap=cmap) | kwargs
        im = ax.pcolormesh(rabins, decbins, np.ma.masked_equal(nvisits, 0), **prop)
        fig.colorbar(im, ax=ax, label="number of visits", shrink=0.8)

        ax.set_xlim(rabins[-1], rabins[0]) # ra increasing to the left
        ax.set_aspect("equal")
        ax.set_xlabel("ra [deg]", fontsize="large")
        ax.set_ylabel("dec [deg]", fontsize="large")
        return fig

    def show_nvisits_healpix(self, nside=32, bands=None, title=None, **kwargs): # pragma: no cover
        """Show the number of visits on the sky (healpix map, Mollweide projection).

        Parameters
        ----------
        nside : int, optional
            Healpix nside. The default is 32.

        bands : str, list or None, optional
            Only count the visits in these bands. If None, all bands.
            The default is None.

        title : str, optional
            Title of the map. The default is None.

        **kwargs
            Goes to :func:`healpy.mollview` (e.g. ``min``, ``max``, ``cmap``).

        Returns
        -------
        matplotlib.figure.Figure
            The figure.

        See Also
        --------
        get_nvisits_healpix : The number of visits per healpix pixel.
        """
        import healpy as hp
        import matplotlib.pyplot as plt

        nvisits = self.get_nvisits_healpix(nside=nside, bands=bands).astype(float)
        nvisits[nvisits == 0] = hp.UNSEEN # not observed: grey
        prop = dict(title=title, unit="number of visits", cmap="viridis") | kwargs
        hp.mollview(nvisits, **prop)
        hp.graticule()
        return plt.gcf()

    # ============= #
    #   Properties  #
    # ============= #
    @property
    def per_ccd(self):
        """Whether coordinates falling in CCD gaps are removed."""
        return self._per_ccd

    @property
    def ccd_footprints(self):
        """CCD footprints in the focal plane (deg), used with `per_ccd`."""
        if not hasattr(self, "_ccd_footprints") or self._ccd_footprints is None:
            self._ccd_footprints = get_lsst_ccd_footprints()

        return self._ccd_footprints
