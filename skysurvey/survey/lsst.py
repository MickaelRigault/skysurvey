import numpy as np
from .basesurvey import Survey
import pandas

def get_lsst_footprint():
    """ A (3 5 5 5 3) ccd structure centered on 0 with a 9.6 deg2 area

    Returns
    -------
    shapely.geometry.Polygon
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
    """ parse input opsim database and returns a dataframe
    
    Parameters
    ----------
    filepath: str, path
        path to the opsim db.

    columns: list, None
        list of column to load from the db. Is 'None', all loaded.

    sql_where: str, None
        options to select rows to load. e.g. night<365.

    Returns
    -------
    pandas.DataFrame
    """
    import sqlite3
    connect = sqlite3.connect(filepath)

    if sql_where is None:
        sql_where = ""
    else:
        sql_where= f"WHERE {sql_where}"

    if columns is None:
        sql_columns = "*"
    else:
        sql_columns = ", ".join(np.atleast_1d(columns))
    
    df = pandas.read_sql_query(f'SELECT {sql_columns} FROM OBSERVATIONS {sql_where}', connect)
    return df


class LSST( Survey ):
    _FOOTPRINT = get_lsst_footprint()

    @classmethod
    def from_opsim(cls, filepath, sql_where=None, zp=30, backend="pandas", **kwargs):
        """ load a LSST survey object from an opsim db path.

        Parameters
        ----------
        filepath: str, path
            path to the opsim db.

        sql_where: str, None
            options to select rows to load. e.g. night<365.

        zp: float
            zp to convert maglimit into skynoise and used for LC flux definition

        backend: str
            backend used to merge the data:
            - polars (fastest): requires polars installed -> converted to pandas at the end
            - pandas (classic): the normal way
            - dask (lazy): as persisted dask.dataframe is returned

        **kwargs goes to read_opsim(): columns

        Returns
        -------
        LSST
        """
        from ..tools.utils import get_skynoise_from_maglimit
        
        df = read_opsim(filepath, sql_where=sql_where, **kwargs)
        
        simdata = pandas.DataFrame(
            {"skynoise": df["fiveSigmaDepth"].apply(get_skynoise_from_maglimit, zp=zp).values,
             "mjd" : df["observationStartMJD"].values,
             "band": "lsst"+df["filter"].values, 
             "gain": 1,
             "zp": zp,
             "ra": df["fieldRA"].values, 
             "dec": df["fieldDec"].values, 
            },
            index=df.index)

        return cls.from_pointings(simdata, backend=backend)
        
