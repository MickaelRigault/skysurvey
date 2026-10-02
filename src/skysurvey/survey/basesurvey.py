"""Generic `Survey` and `GridSurvey` classes.

These combine the healpix and polygon survey capabilities with a camera
footprint.
"""

import pandas
import numpy as np

from .healpix  import HealpixSurvey
from .polygon  import PolygonSurvey


class _FootPrintHandler_( object ):
    """Mixin class handling footprint geometries for survey objects.

    Attributes
    ----------
    _FOOTPRINT : shapely.geometry.Polygon, shapely.geometry.MultiPolygon or None
        Default camera footprint geometry, used if no footprint is provided
        during initialization.
    """

    _FOOTPRINT = None
    # ============== #
    #  Method        #
    # ============== #
    def show_footprint(self, ax=None, add_text=False, **kwargs):
        """Show the survey footprint.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot the footprint on. If None, a new figure and axes are
            created. The default is None.

        add_text : bool, optional
            If True, adds the footprint area as text on the plot.
            The default is False.

        **kwargs
            Passed to :class:`matplotlib.patches.Polygon` (or
            :class:`matplotlib.collections.PolyCollection` for a MultiPolygon
            footprint), e.g. facecolor, edgecolor.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
    
        if ax is None:
            fig = plt.figure(figsize=[4,4])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        prop = {**dict(facecolor=to_rgba("C0", 0.3), edgecolor="k", lw=1, zorder=3),
                **kwargs}

        # MultiPolygon footprint
        if "MultiPolygon" in str( type(self.footprint) ):
            from matplotlib.collections import PolyCollection            
            coll = PolyCollection([np.asarray(p_.exterior.xy).T for p_ in self.footprint.geoms],
                                 **prop)
            ax.add_collection(coll)
        # Polygon footprint            
        else:
            from matplotlib.patches import Polygon
            polygon = Polygon(np.asarray(self.footprint.exterior.xy).T, **prop)
            ax.add_patch(polygon)
            
        if add_text:
            ax.text(0,0, f"area {self.footprint.area:.1f} deg2", fontsize="large", 
                        color="k", zorder=8, va="center", ha="center")
            
        ax.autoscale_view()
        return fig

    def get_skyarea(self, as_multipolygon=True):
        """Get the multipolygon (or list) of field geometries.

        Parameters
        ----------
        as_multipolygon : bool, optional
            If True, returns a multipolygon. Otherwise, returns the array of
            polygons. The default is True.

        Returns
        -------
        shapely.geometry.MultiPolygon or numpy.ndarray
            The field geometries.
        """
        from shapely import geometry
        list_of_geoms = self.fields["geometry"].values
        if as_multipolygon:
            return geometry.MultiPolygon(list_of_geoms)
        return list_of_geoms
    
    # ============== #
    #  Properties    #
    # ============== #
    @property
    def footprint(self):
        """Camera footprint (geometry)."""
        if not hasattr(self,"_footprint") or self._footprint is None:
            if self._FOOTPRINT is None:
                return None
            
            self._footprint = self._FOOTPRINT
            
        return self._footprint


# ================= #
#                   #
#  Generic Survey   #
#                   #
# ================= #    
class Survey( HealpixSurvey, _FootPrintHandler_ ):
    # A healpixSurvey based on geometry, so contains a footprint
    """Generic healpix-based survey with a camera footprint.

    Parameters
    ----------
    footprint : shapely.geometry.Polygon or None, optional
        Footprint in the sky of the observing camera. The default is None.

    nside : int, optional
        Healpix nside parameter. The default is 200.

    data : pandas.DataFrame or None, optional
        Observing data. The default is None.
    """
    def __init__(self, footprint=None, nside=200, data=None):
        """Initialize the Survey class."""
        super().__init__(nside=nside, data=data)
        self._footprint = footprint
        
    # ============== #
    #  I/O           #
    # ============== #
    @classmethod
    def from_random(cls, *args, **kwargs):
        """Not implemented.

        Parameters
        ----------
        *args
            Ignored.

        **kwargs
            Ignored.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(" not implemented ")
    
    @classmethod
    def from_data(cls, data, footprint=None, nside=200):
        """Load an instance given survey data and healpix size (nside).

        Parameters
        ----------
        data : pandas.DataFrame
            Observing data.

        footprint : shapely.geometry.Polygon or None, optional
            Footprint in the sky of the observing camera. The default is None.

        nside : int, optional
            Healpix nside parameter. The default is 200.

        Returns
        -------
        Survey
            The loaded instance.
        """
        return cls(data=data, footprint=footprint, nside=nside)
        
    @classmethod
    def from_pointings(cls, data, footprint=None,
                          rakey="ra", deckey="dec",
                          nside=200,
                          backend="polars",
                          use_pyarrow_extension_array=True,
                          **kwargs):
        """Load an instance given observing pointings of a survey.

        This loads a :class:`~skysurvey.survey.polygon.PolygonSurvey` using its
        ``from_pointings`` method and converts it into a healpix survey using its
        ``to_healpix()`` method.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            Observing data, must contain the `rakey` and `deckey` columns.

        footprint : shapely.geometry.Polygon or None, optional
            Footprint in the sky of the observing camera. If None, the class
            default footprint (`_FOOTPRINT`) is used. The default is None.

        rakey : str, optional
            Name of the R.A. column (in deg). The default is 'ra'.

        deckey : str, optional
            Name of the declination column (in deg). The default is 'dec'.

        nside : int, optional
            Healpix nside parameter. The default is 200.

        backend : {'polars', 'pandas', 'dask'}, optional
            Which backend to use to merge the data (speed issue):

            - 'polars' (fastest): requires polars installed; converted to pandas
              at the end.
            - 'pandas' (classic): the normal way.
            - 'dask' (lazy): a persisted dask.dataframe is returned.

            The default is 'polars'.

        use_pyarrow_extension_array : bool, optional
            Ignored if backend is not 'polars'. Should the pandas dataframe be
            based on pyarrow arrays (like in polars; faster to load, but
            numpy.asarray will be used by pandas when needed, which will then slow
            things down) rather than numpy arrays (slow to load but faster then).
            The default is True.

        **kwargs
            Passed to
            :meth:`~skysurvey.survey.polygon.PolygonSurvey.from_pointings`.

        Returns
        -------
        Survey
            The loaded instance.
        """
        if footprint is None:
            footprint = cls._FOOTPRINT
            
        # super() calls HealpixSurvey.
        this = super().from_pointings(nside=nside, data=data, footprint=footprint,
                                    rakey=rakey, deckey=deckey,
                                    backend=backend,
                                    use_pyarrow_extension_array=use_pyarrow_extension_array,
                                    **kwargs)
        
        return cls.from_healpix(healpixsurvey=this, footprint=footprint)

    @classmethod
    def from_healpix(cls, healpixsurvey, footprint):
        """Create an instance given a healpix survey and a footprint.

        Parameters
        ----------
        healpixsurvey : skysurvey.HealpixSurvey
            Healpix survey instance.

        footprint : shapely.geometry.Polygon
            Footprint in the sky of the observing camera.

        Returns
        -------
        Survey
            The loaded instance.
        """
        return cls(data=healpixsurvey.data,
                       footprint=footprint,
                       nside=healpixsurvey.nside)

# ================= #
#                   #
#    Grid Survey    #
#                   #
# ================= #
class GridSurvey(PolygonSurvey, _FootPrintHandler_ ):
    """Polygon-based survey with fields on a grid and a camera footprint.

    Parameters
    ----------
    data : pandas.DataFrame or None, optional
        Observing data. The default is None.

    fields : geopandas.GeoDataFrame or None, optional
        Field definitions. If None, the class default fields are used.
        The default is None.

    footprint : shapely.geometry.Polygon or None, optional
        Footprint in the sky of the observing camera. The default is None.

    **kwargs
        Ignored.
    """
    def __init__(self, data=None, fields=None, footprint=None, **kwargs):
        """Initialize the GridSurvey class."""
        self._footprint = footprint
        super().__init__(data=data, fields=fields)
        
    @classmethod
    def from_pointings(cls, data, fields_or_coords=None, footprint=None, **kwargs):
        """Load an instance given the observing data and the field definitions.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            Observing data.

        fields_or_coords : geopandas.GeoDataFrame, pandas.DataFrame, dict or None, optional
            Field definitions, or field center coordinates (with 'ra' and 'dec'
            entries, in deg) that are projected using `footprint`. If None, the
            class default fields are used. The default is None.

        footprint : shapely.geometry.Polygon or None, optional
            Footprint in the sky of the observing camera. Required if
            `fields_or_coords` are coordinates. The default is None.

        **kwargs
            Passed to the class constructor.

        Returns
        -------
        GridSurvey
            The loaded instance.
        """
        if type(data) is dict:
            data = pandas.DataFrame.from_dict(data)

        fields = cls._parse_fields(fields_or_coords, footprint)    
        return cls(data=data, fields=fields, footprint=footprint, **kwargs)

    @classmethod
    def from_logs(cls, **kwargs):
        """Not implemented.

        Parameters
        ----------
        **kwargs
            Ignored.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("from_logs is not Implemented for this survey")

    # ============== #
    #   Internal     #
    # ============== #
    @classmethod
    def _parse_fields(cls, fields_or_coords, footprint=None):
        """Parse the fields from field definitions or coordinates.

        Parameters
        ----------
        fields_or_coords : geopandas.GeoDataFrame, pandas.DataFrame, dict or None
            Field definitions, or field center coordinates (with 'ra' and 'dec'
            entries, in deg). If None, the class `_DEFAULT_FIELDS` is returned (if
            any).

        footprint : shapely.geometry.Polygon or None, optional
            Footprint in the sky of the observing camera. Required if
            `fields_or_coords` are coordinates. The default is None.

        Returns
        -------
        geopandas.GeoDataFrame or None
            The parsed fields.

        Raises
        ------
        ValueError
            If coordinates are given but `footprint` is None.
        """
        if fields_or_coords is None:
            if hasattr(cls, "_DEFAULT_FIELDS"):
                return cls._DEFAULT_FIELDS
            return None
        
        # this is list of coords
        if type(fields_or_coords) is dict and "ra" in list(fields_or_coords.values())[0]: 
            fields_or_coords = pandas.DataFrame(fields_or_coords).T
            # this enters the new if. 

        if type(fields_or_coords) is pandas.DataFrame and "ra" in fields_or_coords:
            if footprint is None:
                raise ValueError("fields given as list of coordinates but no footprint given?")
            from ztffields.projection import project_to_radec
            import geopandas
            if fields_or_coords.index.name is None:
                fields_or_coords.index.name = "fieldid"
            # Now expected geopandas
            fields = geopandas.GeoDataFrame( geometry=project_to_radec(footprint,
                                                                           fields_or_coords["ra"],
                                                                           fields_or_coords["dec"]),
                                            index=fields_or_coords.index)
            fields = fields.join(fields_or_coords) # store input data
        else:
            fields = fields_or_coords

        return super()._parse_fields(fields)

    # ============== #
    #   Properties   #
    # ============== #
    @property
    def fields(self):
        """GeoDataFrame containing the fields coordinates."""
        if not hasattr(self,"_fields") or self._fields is None:
            if self._DEFAULT_FIELDS is None:
                return None
            self._fields = self._DEFAULT_FIELDS.copy()
        return self._fields
