"""Hyper Suprime-Cam (HSC, Subaru telescope) camera layout and survey class.

This module defines the HSC camera footprint (104 science CCDs, or the
camera outline) and the `HSC` survey, built from observing logs given per
exposure or per CCD (see :meth:`HSC.from_logs`).
"""

import warnings
import pandas
import numpy as np


import geopandas
from shapely import geometry

from .basesurvey import GridSurvey

__all__ = ["HSC", "get_hsc_ccd_footprints", "get_hsc_footprint", "get_hsc_fields", "parse_hsc_logs"]


# Documentation for the HSC camera layout can be found at https://subarutelescope.org/Instruments/HSC.

# Science CCD ids (DET-ID) as shown on the CCD arrangement diagram,
# rows from top to bottom, CCDs from left to right. All upright CCDs are on the same
# column grid: rows of 15 span columns -7..7, rows of 13 -6..6 and rows of 7 -3..3.
HSC_CCD_ROWS = [[77, 69, 61, 53, 45, 37, 29],
                [95, 89, 83, 76, 68, 60, 52, 44, 36, 28, 21, 15, 9],
                [99, 94, 88, 82, 75, 67, 59, 51, 43, 35, 27, 20, 14, 8, 3],
                [98, 93, 87, 81, 74, 66, 58, 50, 42, 34, 26, 19, 13, 7, 2],
                [97, 92, 86, 80, 73, 65, 57, 49, 41, 33, 25, 18, 12, 6, 1],
                [96, 91, 85, 79, 72, 64, 56, 48, 40, 32, 24, 17, 11, 5, 0],
                [90, 84, 78, 71, 63, 55, 47, 39, 31, 23, 16, 10, 4],
                [70, 62, 54, 46, 38, 30, 22]]

# CCDs lying on their side next to the top and bottom rows:
# DET-ID: (column of their center, row they are attached to, alignment with that row)
HSC_ROTATED_CCDS = {103: (-4.5, 0, "bottom"), 101: (4.5, 0, "bottom"),
                    102: (-4.5, 7, "top"), 100: (4.5, 7, "top")}

# CCDs marked as disabled on the diagram (version of October 2026)
HSC_DISABLED_CCDS = [9, 21, 44]


# Geopandas of the individual ccd footprint within the camera
def get_hsc_ccd_footprints(pixel_scale=0.17, npix=(2048, 4096), gaps=(12, 53),
                           rotation=0, exclude_ccds=None):
    """Get the footprint of each HSC science CCD in the focal plane.

    The 104 science CCDs are arranged in 8 rows of 7, 13, 15, 15, 15, 15, 13
    and 7 upright CCDs, plus 4 CCDs lying on their side next to the top and
    bottom rows (see `HSC_CCD_ROWS` and `HSC_ROTATED_CCDS`).

    Parameters
    ----------
    pixel_scale : float, optional
        Pixel scale in arcsec/pixel. The default is 0.17.

    npix : tuple, optional
        Number of pixels of a CCD along its (short, long) axis.
        The default is (2048, 4096).

    gaps : tuple, optional
        Gap between neighboring CCDs within a row and between rows, in
        arcsec. The default is (12, 53).

    rotation : float, optional
        Rotation of the camera in deg (counter-clockwise). With 0, the rows
        are along the x-axis (R.A. once projected on the sky).
        The default is 0.

    exclude_ccds : list or None, optional
        CCD ids (DET-ID) to remove, e.g. `HSC_DISABLED_CCDS`.
        The default is None.

    Returns
    -------
    geopandas.GeoSeries
        One polygon per CCD, in deg, centered on the camera center, indexed
        by ``ccdid`` (DET-ID).
    """
    width, height = np.asarray(npix) * pixel_scale / 3600 # [deg]
    gap_x, gap_y = np.asarray(gaps) / 3600 # [deg]
    pitch_x, pitch_y = width + gap_x, height + gap_y
    nrows = len(HSC_CCD_ROWS)

    ccds = {}
    # upright CCDs | each row is centered on column 0, row 0 is the top one.
    for row_, ccdids_ in enumerate(HSC_CCD_ROWS):
        ycenter = ((nrows - 1) / 2 - row_) * pitch_y
        columns_ = np.arange(len(ccdids_)) - (len(ccdids_) - 1) / 2
        for ccdid_, column_ in zip(ccdids_, columns_):
            ccds[ccdid_] = geometry.box(column_ * pitch_x - width / 2, ycenter - height / 2,
                                        column_ * pitch_x + width / 2, ycenter + height / 2)

    # rotated CCDs | long side along x, aligned with the outer edge of their row
    for ccdid_, (column_, row_, align_) in HSC_ROTATED_CCDS.items():
        ycenter = ((nrows - 1) / 2 - row_) * pitch_y
        ylow = ycenter - height / 2 if align_ == "bottom" else ycenter + height / 2 - width
        ccds[ccdid_] = geometry.box(column_ * pitch_x - height / 2, ylow,
                                    column_ * pitch_x + height / 2, ylow + width)

    ccds = geopandas.GeoSeries(ccds).sort_index()
    ccds.index.name = "ccdid"
    if rotation != 0:
        ccds = ccds.rotate(rotation, origin=(0, 0))

    if exclude_ccds is not None:
        ccds = ccds.drop(exclude_ccds)

    return ccds

# The camera footprint, with or without the ccd details.
def get_hsc_footprint(per_ccd=True, **kwargs):
    """Get the HSC camera footprint.

    Parameters
    ----------
    per_ccd : bool, optional
        If True, the union of the science CCDs (a MultiPolygon, with the gaps
        between CCDs). If False, the outline of the camera as a single
        polygon (gaps filled). The default is True.

    **kwargs
        Goes to :func:`get_hsc_ccd_footprints`.

    Returns
    -------
    shapely.geometry.MultiPolygon or shapely.geometry.Polygon
        The camera footprint in deg, centered on 0.
    """
    ccd_footprint = get_hsc_ccd_footprints(**kwargs).union_all()
    if per_ccd:
        return ccd_footprint

    # closing the gaps: growing the CCDs by more than half the largest gap
    # merges them, shrinking them back restores the outer edges.
    # (mitre: keeps the square corners of the CCDs)
    gap = 53 / 3600 # [deg] largest gap, see get_hsc_ccd_footprints
    return ccd_footprint.buffer(gap, join_style="mitre").buffer(-gap, join_style="mitre")


# reads the input logs and provide that that should be used by skysurvey
# with information of the ccd details (or not)
def parse_hsc_logs(logs, fieldid_decimals=4, ccdkey="ccdid"):
    """Parse HSC observing logs and tell if they are given per CCD.

    Parameters
    ----------
    logs : str or pandas.DataFrame
        Path to a csv file, or the logs themselves, with (at least) ``ra``,
        ``dec`` (pointing center, deg), ``mjd``, ``band`` (sncosmo name),
        ``zp``, ``gain`` and ``skynoise`` (flux unit given ``zp``), and
        optionally a CCD id column (`ccdkey`).

    fieldid_decimals : int, optional
        Exposures with the same (ra, dec) once rounded to this number of
        decimals (in deg) get the same ``fieldid``. The default is 4 (~0.4 arcsec).

    ccdkey : str, optional
        Name of the CCD id (DET-ID) column, renamed ``ccdid``.
        The default is ``"ccdid"``.

    Returns
    -------
    logs : pandas.DataFrame
        A copy of the logs with an extra ``fieldid`` column.

    per_ccd : bool
        True if the logs are given per CCD (`ccdkey` column found).
    """
    if type(logs) is str:
        logs = pandas.read_csv(logs, index_col=0)
    else:
        logs = logs.copy()

    per_ccd = ccdkey in logs
    if per_ccd and ccdkey != "ccdid":
        logs = logs.rename(columns={ccdkey: "ccdid"})

    pointings = logs[["ra", "dec"]].round(fieldid_decimals)
    logs["fieldid"] = pointings.groupby(["ra", "dec"]).ngroup()
    return logs, per_ccd

def get_hsc_fields(ra, dec, footprint=None, fieldid=None):
    """Get the sky geometry of the given pointings.

    Parameters
    ----------
    ra, dec : array_like
        Pointing coordinates (camera center) in deg.

    footprint : geopandas.GeoSeries or shapely.geometry.Polygon, optional
        Footprint in the focal plane: one polygon per CCD (see
        :func:`get_hsc_ccd_footprints`) or a single polygon (e.g. the camera
        outline, see :func:`get_hsc_footprint`). If None, the CCD footprints.
        The default is None.

    fieldid : array_like, optional
        Id of each pointing. If None, ``0..npointings-1``. The default is None.

    Returns
    -------
    geopandas.GeoDataFrame
        One polygon per pointing and CCD, indexed by (``fieldid``, ``ccdid``),
        or one polygon per pointing, indexed by ``fieldid``.
    """
    from ..tools.projection import project_to_radec


    if footprint is None:
        footprint = get_hsc_ccd_footprints()

    ra, dec = np.atleast_1d(ra), np.atleast_1d(dec)
    if fieldid is None:
        fieldid = np.arange(len(ra))

    # single polygon: one field per pointing
    if isinstance(footprint, geometry.Polygon):
        index = pandas.Index(fieldid, name="fieldid")
        return geopandas.GeoDataFrame(geometry=project_to_radec(footprint, ra=ra, dec=dec), index=index)

    # per CCD: one projection per CCD, vectorised over the pointings
    geometries = {ccdid_: project_to_radec(ccd_, ra=ra, dec=dec) # npointings polygons
                  for ccdid_, ccd_ in footprint.items()}
    index = pandas.MultiIndex.from_product([list(geometries.keys()), fieldid],
                                           names=["ccdid", "fieldid"])
    fields = geopandas.GeoDataFrame(geometry=np.concatenate(list(geometries.values())), index=index)
    return fields.reorder_levels(["fieldid", "ccdid"]).sort_index()

# ============== #
#                #
#  HSC Camera    #
#                #
# ============== #
class HSC( GridSurvey ):
    """A class to model a Hyper Suprime-Cam (Subaru) survey.

    The fields are either the CCDs of each pointing, indexed by (``fieldid``,
    ``ccdid``) as for :class:`skysurvey.ZTF`, if the logs are given per CCD,
    or the camera outline of each pointing, indexed by ``fieldid``, otherwise
    (see :meth:`from_logs`, the preferred way to build an instance).

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Observing data, with a ``fieldid`` column (and ``ccdid`` if per CCD).
        The default is None.

    fields : geopandas.GeoDataFrame, optional
        Field geometries (see :func:`get_hsc_fields`). The default is None.

    footprint : shapely.geometry.MultiPolygon or shapely.geometry.Polygon, optional
        Camera footprint. If None, :func:`get_hsc_footprint`.
        The default is None.

    **kwargs
        Goes to :class:`skysurvey.survey.basesurvey.GridSurvey`.
    """
    _FOOTPRINT = get_hsc_footprint()

    def __init__(self, data=None, fields=None, footprint=None, **kwargs):
        """Initialize the HSC class. See the class docstring."""
        if footprint is None:
            footprint = self._FOOTPRINT

        super().__init__(data=data, fields=fields, footprint=footprint, **kwargs)

    @classmethod
    def from_logs(cls, logs, exclude_ccds=None, **kwargs):
        """Load an HSC survey from observing logs, given per exposure or per CCD.

        The logs are parsed with :func:`parse_hsc_logs`, which tells whether
        they are given per CCD (``ccdid`` column):

        - per CCD: the fields are the CCDs of each pointing, indexed by
          (``fieldid``, ``ccdid``).
        - per exposure: the fields are the camera outline at each pointing
          (see ``get_hsc_footprint(per_ccd=False)``), indexed by ``fieldid``.

        Parameters
        ----------
        logs : str or pandas.DataFrame
            Path to the logs (csv) or the logs themselves
            (see :func:`parse_hsc_logs`).

        exclude_ccds : list or None, optional
            CCD ids (DET-ID) to remove, e.g. `HSC_DISABLED_CCDS`. For logs
            given per exposure, they are removed from the camera outline
            (holes). The default is None.

        **kwargs
            Goes to :func:`get_hsc_ccd_footprints` (e.g. ``rotation``).

        Returns
        -------
        HSC
            The HSC survey instance.

        Warns
        -----
        UserWarning
            If logs given per CCD have CCD ids unknown to the camera layout
            (these rows are dropped).
        """
        logs, per_ccd = parse_hsc_logs(logs)

        # pointing to define the "fields". One per actual coordinates.
        # one row per exposure and CCD (CCD-level matching)
        pointings = logs.groupby("fieldid")[["ra", "dec"]].first()
        if per_ccd:
            # Get all the CCDs of the camera (except these explicitely removed)
            ccd_footprints = get_hsc_ccd_footprints(exclude_ccds=exclude_ccds, **kwargs)
            # Check if the logs contains any unknown ccd ids.
            # This should never happends, but if so, the is flagged.
            known = logs["ccdid"].isin(ccd_footprints.index)
            unknown = set(logs.loc[~known, "ccdid"]) - set(np.atleast_1d(exclude_ccds if exclude_ccds is not None else []))
            if len(unknown) > 0:
                warnings.warn(f"logs rows with ccdid not in the camera layout are dropped: {sorted(unknown)}")

            logs = logs[known]

            # Get the field geometry corresponding to the actuall pointings.
            fields = get_hsc_fields(pointings["ra"], pointings["dec"],
                                    footprint=ccd_footprints,
                                    fieldid=pointings.index)
            # and get the actual footprint in the per-ccd format.
            footprint = ccd_footprints.union_all()

        else:
            # No ccd details,
            footprint = get_hsc_footprint(per_ccd=False, exclude_ccds=exclude_ccds,
                                          **kwargs)
            # Get the field geometry corresponding to the actual pointings.
            fields = get_hsc_fields(pointings["ra"], pointings["dec"],
                                    footprint=footprint, fieldid=pointings.index)

        # create the instance.
        return cls(data=logs, fields=fields, footprint=footprint)

    # --------- #
    #   PLOT    #
    # --------- #
    def show_footprint(self, ax=None, incl_both=True): # pragma: no cover
        """Show the HSC camera footprint in the focal plane.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes where to draw. If None, a new figure is created.
            The default is None.

        incl_both : bool, optional
            Show both the CCDs (with their DET-ID) and the camera outline. If
            False, only the one used by this survey (see `per_ccd`).
            The default is True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.

        Notes
        -----
        The CCDs and the outline are drawn for the default camera layout
        (no rotation, no excluded CCDs).
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        ccd_footprints = get_hsc_ccd_footprints() if (incl_both or self.per_ccd) else None
        edge_footprint = get_hsc_footprint(per_ccd=False) if (incl_both or not self.per_ccd) else None

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6.5))
        else:
            fig = ax.figure

        # loop over the details of the camera.
        if ccd_footprints is not None:
            for ccdid_, ccd_ in ccd_footprints.items():
                ax.fill(*ccd_.exterior.xy, color="0.8", ec="0.3", lw=0.5)
                ax.text(ccd_.centroid.x, ccd_.centroid.y, f"{ccdid_:03d}", ha="center", va="center", fontsize=6)

        if edge_footprint is not None:
            ax.plot(*edge_footprint.exterior.xy, color="C1", lw=2, label="camera outline (per_ccd=False)")
        ax.add_patch( Circle((0, 0), 0.75, fill=False, ls="--", color="C0", label="1.5 deg diameter"))
        ax.set_aspect("equal")
        ax.set_xlabel("x [deg]", fontsize="large")
        ax.set_ylabel("y [deg]", fontsize="large")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        return fig

    def show_deepfields(self): # pragma: no cover
        """Show the observed sky area, one panel per separate region.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.

        See Also
        --------
        get_regions : The separate regions of the observed sky area.
        show_nvisits : Show the number of visits.
        """
        import matplotlib.pyplot as plt
        regions = self.get_regions()
        nregions = len(regions)

        fig, axes = plt.subplots(1, nregions, figsize=(5*nregions, 4.5), squeeze=False)

        for ax, region_ in zip(axes[0], regions):
            self._show_skyarea_(region_, ax, color="0.8")
            # define edges as this is not automated in matplotlib.
            ramin, decmin, ramax, decmax = region_.bounds
            ax.set_xlim(ramax + 0.2, ramin - 0.2) # ra increasing to the left
            ax.set_ylim(decmin - 0.2, decmax + 0.2)
            ax.set_aspect("equal")
            ax.set_xlabel("ra [deg]", fontsize="large")
            ax.set_ylabel("dec [deg]", fontsize="large")

        return fig

    # ============= #
    #   Properties  #
    # ============= #
    @property
    def per_ccd(self):
        """Whether the fields (and logs) are given per CCD (fields indexed by ``ccdid``)."""
        return "ccdid" in self.fields.index.names
