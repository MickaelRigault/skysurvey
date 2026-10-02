"""LSST survey class and OpSim database utilities.

This module defines the `LSST` survey class and utilities for loading and
parsing LSST OpSim observation databases.
"""

import numpy as np
from .basesurvey import Survey
import pandas

def get_lsst_footprint():
    """Get the LSST camera footprint.

    The footprint is a (3 5 5 5 3) CCD structure centered on 0 with a
    9.6 deg**2 area.

    Returns
    -------
    shapely.geometry.Polygon
        The LSST camera footprint.
    """
    from shapely import geometry
    lowleft = 0
    upright = 5
    corner_x = (1/5) * upright
    corner_y = (1/5) * upright

    footprint = np.asarray(
                [[corner_x, lowleft],
                 [upright-corner_x, lowleft],
                 [upright-corner_x, corner_y],
                 [upright, corner_y],
                 [upright, upright-corner_y],
                 [upright-corner_x, upright-corner_y],
                 [upright-corner_x, upright],
                 [corner_x, upright],
                 [corner_x, upright-corner_y],
                 [lowleft, upright-corner_y],
                 [lowleft, corner_y],
                 [corner_x, corner_y]
                ]) - upright/2.

    return geometry.Polygon(footprint * 0.675)
    

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
    return df


class LSST( Survey ):
    """A class to model the `LSST` survey.

    Parameters
    ----------
    footprint : shapely.geometry.Polygon, optional
        Footprint in the sky of the observing camera. The default is None.

    nside : int, optional
        HEALPix nside parameter. The default is 200.

    data : pandas.DataFrame, optional
        Observing data. The default is None.

    Attributes
    ----------
    _FOOTPRINT : shapely.geometry.Polygon
        The LSST camera footprint loaded via :func:`get_lsst_footprint`.
    """
    _FOOTPRINT = get_lsst_footprint()

    @classmethod
    def from_opsim(cls, filepath, sql_where=None, zp=30, backend="pandas", **kwargs):
        """Load an LSST survey object from an OpSim database path.

        Parameters
        ----------
        filepath : str
            Path to the OpSim database.

        sql_where : str, optional
            SQL condition to select the rows to load (e.g. 'night<365').
            If None, all rows are loaded. The default is None.

        zp : float, optional
            Zero point used to convert the limiting magnitude into skynoise
            and used for the light-curve flux definition. The default is 30.

        backend : {'pandas', 'polars', 'dask'}, optional
            Backend used to merge the data:

            - 'polars' (fastest): requires polars installed; converted to
              pandas at the end.
            - 'pandas' (classic): the normal way.
            - 'dask' (lazy): a persisted dask.dataframe is returned.

            The default is 'pandas'.

        **kwargs
            Passed to :meth:`Survey.from_pointings`.

        Returns
        -------
        LSST
            The LSST survey instance.
        """
        from ..tools.utils import get_skynoise_from_maglimit
        
        df = read_opsim(filepath, sql_where=sql_where)
        
        simdata = pandas.DataFrame(
            {"skynoise": df["fiveSigmaDepth"].apply(get_skynoise_from_maglimit, zp=zp).values,
             "mjd" : df["observationStartMJD"].values,
             "band": "lsst"+df["filter"].values, 
             "gain": 1,
             "zp": zp,
             "ra": df["fieldRA"].values, 
             "dec": df["fieldDec"].values, 
             "observationId": df["observationId"].values,
            },
            index=df.index)

        return cls.from_pointings(simdata, backend=backend, **kwargs)
        
