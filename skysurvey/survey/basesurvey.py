import pandas
import numpy as np
import warnings
from .healpix  import HealpixSurvey
from .polygon  import PolygonSurvey


__all__ = ["Survey", "GridSurvey"]

class _FootPrintHandler_( object ):
    _FOOTPRINT = None
    # ============== #
    #  Method        #
    # ============== #
    def show_footprint(self, ax=None, add_text=False, **kwargs):
        """ shows the survey footprint.
        
        Parameters
        ----------
        ax: matplotlib.axes
            axes to plot the footprint on.
            
        add_text: bool
            if True, adds text to the plot.
            
        **kwargs goes to matplotlib (e.g. facecolor, edgecolor)
        
        Returns
        -------
        matplotlib.figure
        
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
        """ multipolygon (or list) of field geometries

        Parameters
        ----------
        as_multipolygon: bool
            if True, returns a multipolygon.
            Otherwise, returns a list of polygons.

        Returns
        -------
        shapely.geometry.MultiPolygon or list
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
        """ camera footprint (geometry) """
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

    def __init__(self, footprint=None, nside=200, data=None):
        """ Initialize the Survey class.

        Parameters
        ----------
        footprint: shapely.geometry
            footprint in the sky of the observing camera

        nside : int
            healpix nside parameter

        data: pandas.DataFrame
            observing data.
        """
        super().__init__(nside=nside, data=data)
        self._footprint = footprint
        
    # ============== #
    #  I/O           #
    # ============== #
    @classmethod
    def from_random(cls, *args, **kwargs):
        """ Not implemented """
        raise NotImplementedError(" not implemented ")
    
    @classmethod
    def from_data(cls, data, footprint=None, nside=200):
        """ load an instance given survey data and healpix size (nside) 
        
        Parameters
        ----------
        data: pandas.DataFrame
            observing data.

        footprint: shapely.geometry
            footprint in the sky of the observing camera

        nside : int
            healpix nside parameter

        Returns
        -------
        instance

        See also
        --------
        from_random: generate random observing data and loads the instance.
        
        """
        return cls(data=data, footprint=footprint, nside=nside)
        
    @classmethod
    def from_pointings(cls, data, footprint=None,
                          rakey="ra", deckey="dec",
                          nside=200,
                          backend="polars",
                          use_pyarrow_extension_array=True,
                          **kwargs):
        """ loads an instance given observing poitings of a survey
        
        This loads an polygon.PolygonSurvey using from_pointing and 
        converts that into an healpix using the to_healpix() method

        Parameters
        ----------
        data: pandas.DataFrame or dict
            observing data, must contain the rakey and deckey columns.

        footprint: shapely.geometry
            footprint in the sky of the observing camera

        rakey: str
            name of the R.A. column (in deg)

        deckey: str
            name of the Declination column (in deg)

        nside : int
            healpix nside parameter

        backend: str
            which backend to use to merge the data (speed issue):
            - polars (fastest): requires polars installed -> converted to pandas at the end
            - pandas (classic): the normal way
            - dask (lazy): as persisted dask.dataframe is returned

        use_pyarrow_extension_array: bool
            = ignored in backend != 'polars' or polars_to_pandas is not True = 
            should the pandas dataframe be based on numpy array (slow to load but faster then)
            or based on pyarrow array (like in polars) ; faster but numpy.asarray will be 
            used by pandas when need (which will then slow things down).

        **kwargs goes to polygon.PolygonSurvey.from_pointings

        Returns
        -------
        instance
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
        """ creates an instance given a heapixsurvey and a footprint

        Parameters
        ----------
        healpixsurvey: HealpixSurvey
            healpix survey instance

        footprint: shapely.geometry
            footprint in the sky of the observing camera

        Returns
        -------
        Survey
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

    def __init__(self, data=None, fields=None, footprint=None, **kwargs):
        """ Initialize the GridSurvey class
        
        Parameters
        ----------
        data: pandas.DataFrame
            observing data.

        fields: geodataframe
            field definitions.

        footprint: shapely.geometry
            footprint in the sky of the observing camera
        
        """
        self._footprint = footprint
        super().__init__(data=data, fields=fields)
        
    @classmethod
    def from_pointings(cls, data, fields_or_coords=None, footprint=None, **kwargs):
        """ loads an instance given observing poitings of a survey

        Parameters
        ----------
        data: pandas.DataFrame or dict
            observing data, must contain the rakey and deckey columns.

        fields_or_coords: geodataframe or dict
            field definitions or coordinates.

        footprint: shapely.geometry
            footprint in the sky of the observing camera

        **kwargs goes to super().__init__

        Returns
        -------
        GridSurvey
        """
        if type(data) is dict:
            data = pandas.DataFrame.from_dict(data)

        fields = cls._parse_fields(fields_or_coords, footprint)    
        return cls(data=data, fields=fields, footprint=footprint, **kwargs)

    @classmethod
    def from_logs(cls, **kwargs):
        """ Not implemented """
        raise NotImplementedError("from_logs is not Implemented for this survey")

    # ============== #
    #   Internal     #
    # ============== #
    @classmethod
    def _parse_fields(cls, fields_or_coords, footprint=None):
        """ Parse the fields from coordinates.

        Parameters
        ----------
        fields_or_coords: geodataframe or dict
            field definitions or coordinates.

        footprint: shapely.geometry
            footprint in the sky of the observing camera

        Returns
        -------
        geopandas.GeoDataFrame
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
        """ geodataframe containing the fields coordinates """
        if not hasattr(self,"_fields") or self._fields is None:
            if self._DEFAULT_FIELDS is None:
                return None
            self._fields = self._DEFAULT_FIELDS.copy()
        return self._fields
