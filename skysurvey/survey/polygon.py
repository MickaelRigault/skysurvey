""" Survey with Polygon vertices based on Shapely. """

import pandas
import numpy as np
import geopandas
from shapely import geometry
import warnings

from .core import BaseSurvey
from ztffields.projection import spatialjoin_radec_to_fields, parse_fields, project_to_radec

__all__ = ["PolygonSurvey"] 

    
# ================== #
#                    #
#    Polygon         #
#                    #
# ================== #
class PolygonSurvey( BaseSurvey ):
    _DEFAULT_FIELDS = None
    
    def __init__(self, data=None, fields=None):
        """ 
        Initialize the PolygonSurvey class.

        Parameters
        ----------
        data: pandas.DataFrame
            observing data.

        fields: geopandas.GeoDataFrame
            field definitions.
        """
        if fields is None:
            if self._DEFAULT_FIELDS is None:
                raise NotImplementedError("No default fields known for this class. No fields given")
            fields = self._DEFAULT_FIELDS
        
        self._fields = self._parse_fields(fields)        
        super().__init__(data)
        
    @classmethod
    def from_pointings(cls, data, footprint=None, moc=None, rakey="ra", deckey="dec"):
        """ 
        Load an instance from pointings.

        Parameters
        ----------
        data: pandas.DataFrame or dict
            observing data, must contain the rakey and deckey columns.

        footprint: shapely.geometry
            footprint in the sky of the observing camera

        moc: mocpy.MOC
            MOC representation of the observing camera

        rakey: str
            name of the R.A. column (in deg)

        deckey: str
            name of the Declination column (in deg)

        Returns
        -------
        PolygonSurvey
        """
        if type(data) is dict:
            data = pandas.DataFrame.from_dict(data).copy()

        if footprint is not None:
            field_pointings = project_to_radec(footprint, ra=data[rakey], dec=data[deckey])
        elif moc is not None:
            skycoords = moc.get_boundaries()
            ra, dec = [], []
            for skycoord in skycoords:
                ra.append(skycoord.ra.deg)
                dec.append(skycoord.dec.deg)
            ra = np.concatenate(ra)
            dec = np.concatenate(dec)
            footprint = np.vstack((ra, dec))

            pointings = project_to_radec(footprint, ra=data[rakey], dec=data[deckey])
            field_pointings = [geometry.Polygon(p) for p in pointings]
        fields = geopandas.GeoDataFrame(geometry=field_pointings, index=data.index.copy(),)
        fields.index.name = "fieldid"
        data["fieldid"] = fields.index.copy()
        return cls(data=data, fields=fields)

    @classmethod
    def from_random(cls, size, 
                    bands, mjd_range, skynoise_range,
                    fields=None, **kwargs):
        """ 
        Load an instance with random observing data.

        Parameters
        ----------
        size: int
            number of observations to draw

        bands: list of str
            list of bands that should be drawn.

        mjd_range: list or array
            min and max mjd for the random drawing.

        skynoise_range: list or array
            min and max skynoise for the random drawing.

        fields: geopandas.GeoDataFrame
            field definitions.

        **kwargs goes to the draw_random() method

        Returns
        -------
        PolygonSurvey
        """
        this = cls(fields=fields)
        this.draw_random(size,  bands,  
                        mjd_range, skynoise_range, 
                        inplace=True, **kwargs)
        return this
    
    # ============== #
    #   Methods      #
    # ============== #
    def get_fields(self, observed=True):
        """ 
        Get the fields.

        Parameters
        ----------
        observed: bool
            if True, return only the observed fields.

        Returns
        -------
        geopandas.GeoDataFrame
        """
        if observed:
            if len(self.fieldids.names)==1: # Index
                observed_fields = self.data[ self.fieldids.name ].unique()
                fields = self.fields.loc[observed_fields].copy()
            else: # MultiIndex
                observed_fields = self.data[self.fieldids.names].drop_duplicates(ignore_index=True)
                observed_fields = observed_fields.set_index(self.fieldids.names)
                fields = self.fields.loc[(observed_fields.index.levels[0], observed_fields.index.levels[1]),]
        else:
            fields = self.fields.copy()
        return fields
    
    def get_observed_area(self, nside=200):
        """Measure the observed area.

        This uses healpy for accuracy.

        Parameters
        ----------
        nside: int, optional
            Healpix nside.
        
        Returns
        -------
        float
            Area in deg2.
        """
        hsurvey = self.to_healpix(nside=nside, pass_data=False)
        return hsurvey.get_observed_area()# min_obs=min_obs) # not correct with pass_data=False yet.

    def to_healpix(self, nside, pass_data=True, backend="polars",
                       polars_to_pandas=True,
                       use_pyarrow_extension_array=False):
        """Convert the current polygon survey into a healpix survey.
        
        Parameters
        ----------
        nside: int
            Healpix nside.
        pass_data: bool, optional
            Should the returned survey have the full data of just the fieldid matching?
        backend: str, optional
            Which backend to use to merge the data (speed issue):
            - polars (fastest): requires polars installed -> converted to pandas at the end
            - pandas (classic): the normal way
            - dask (lazy): as persisted dask.dataframe is returned
        polars_to_pandas: bool, optional
            = ignored if backend != 'polars' =
            Should the dataframe be converted into a pandas.DataFrame or say a polars.DataFrame
            (using the to_pandas() option).
        use_pyarrow_extension_array: bool, optional
            = ignored in backend != 'polars' or polars_to_pandas is not True = 
            Should the pandas dataframe be based on numpy array (slow to load but faster then)
            or based on pyarrow array (like in polars) ; faster but numpy.asarray will be 
            used by pandas when need (which will then slow things down).
            
        Returns
        --------
        HealpixSurvey
        """
        from .healpix import HealpixSurvey
        hpsurvey = HealpixSurvey(nside)
        ra, dec  = hpsurvey.get_field_centroid()
        # make sure to typing follows
        dtype_fieldid = "int16" if nside < 50 else "int32"
        
        fields = self.radec_to_fieldid( np.asarray([ra,dec]).T
                                      ).reset_index(names=["fieldid_hp"]
                                      ).astype({**self.data.dtypes.loc[self.fieldids.names].to_dict(),
                                                **{"fieldid_hp":dtype_fieldid}}
                                              )

        # limit to what has been observed
        if self.data is not None and self.fieldids.names in self.data:
            fields = fields[fields[self.fieldids.names].isin(self.data[self.fieldids.names].unique())]

        # What kind of data are passed
        if not pass_data: # minimal
            datain = fields
        else: # all, but how ?
            # 
            # Checking what is possible
            #
            if backend == "polars":
                try:
                    import polars as pl
                except:
                    warnings.warn("You do not have polars installed. conda/pip install polars. falling back to pandas backend")
                    backend = "pandas"
                    
            if backend == "dask":
                try:
                    import dask.dataframe as dd
                except:
                    warnings.warn("You do not have dask installed. conda/pip install dask. falling back to pandas backend")
                    backend = "pandas"
            
            # now let's merge
            
            if backend == "pandas": # classic
                datain = self.data.merge(fields, on=self.fieldids.names)
                
            elif backend == "polars": # fast
                p_fields = pl.from_pandas(fields)
                p_data = pl.from_pandas(self.data)
                datain = p_data.join(p_fields, on=self.fieldids.names)
                if polars_to_pandas:
                    datain = datain.to_pandas(use_pyarrow_extension_array=use_pyarrow_extension_array)
                
            elif backend == "dask": # dask
                d_data = dd.from_pandas(self.data, chunksize=1_000_000)
                # match types
                d_fields = dd.from_pandas(fields, chunksize=1_000_000)
                datain = d_data.merge(d_fields, on=self.fieldids.names).persist()
                
            else:
                raise NotImplementedError(f"Input backend {backend} is not implemented .")
            
        # dask.dataframe.rename has no axis option but columns or index
        name_mapping = {"fieldid":"fieldid_survey", "fieldid_hp":"fieldid"}
        if "pandas" in str( type(datain) ): # pandas.DataFrame
            datain = datain.rename(name_mapping, axis=1)
            
        elif "polars" in str( type(datain) ): # polars.DataFrame
            datain = datain.rename(name_mapping)
            
        elif "dask" in str( type(datain) ): # dask.DataFrame
            datain = datain.rename(columns=name_mapping)
            
        else:
            raise ValueError("cannot parse the returned DataFrame (yo")
        
        hpsurvey.set_data(datain)
        return hpsurvey
        
    # ------- #
    #  core   #
    # ------- #
    def radec_to_fieldid(self, radec, observed_fields=False):
        """Get the fieldid associated to the given coordinates.

        Parameters
        ----------
        radec: pandas.DataFrame or 2d array
            Coordinates in degree.
        observed_fields: bool, optional
            Should this be limited to fields actually observed?
            This is ignored is self.data is None.

        Returns
        -------
        pandas.DataFrame
        """
        if type(radec) in [np.ndarray, list, tuple]:
            inshape = np.shape(radec)
            if inshape[-1] != 2:
                raise ValueError(f"shape of radec must be (N, 2), {inshape} given.")
        
            radec = pandas.DataFrame(radec, columns=["ra","dec"])


        fields = self.get_fields(observed=observed_fields)
        _keyindex = 'index_radec'
        projection = spatialjoin_radec_to_fields(radec, fields,
                                                 index_radec=_keyindex,
                                                 ).sort_values(_keyindex
                                                 ).set_index(_keyindex)[self.fields.index.names]
        return projection
    
    # ------- #
    #  draw   #
    # ------- #        
    def draw_random(self, size, 
                    bands, mjd_range, skynoise_range,
                    gain_range=1, zp_range=25,
                    inplace=False, fieldids=None,
                    **kwargs):
        """ 
        Draw random observations.

        Parameters
        ----------
        size: int
            number of observations to draw

        bands: list of str
            list of bands that should be drawn.

        mjd_range: list or array
            min and max mjd for the random drawing.

        skynoise_range: list or array
            min and max skynoise for the random drawing.

        gain_range: list or array
            min and max gain for the random drawing.

        zp_range: list or array
            min and max zp for the random drawing.

        inplace: bool
            if True, the data are stored in the instance.
            Otherwise, a new instance is returned.

        fieldids: list
            list of fieldids to draw from.

        **kwargs goes to _draw_random

        Returns
        -------
        PolygonSurvey or None
        """
        if fieldids is None:
            fieldids = self.fieldids
            
        data = self._draw_random(fieldids, size, bands, mjd_range, skynoise_range, 
                                 gain_range=gain_range, zp_range=zp_range,
                                 **kwargs)
        
        if not inplace:
            return self.__class__.from_data(data=data, fields=self.fields)

        self.set_data(data)
        
    # ----------- #
    #  PLOTTER    #
    # ----------- #
    def show(self, stat='size', column=None, title=None, data=None, origin = 180, 
             vmin=None, vmax=None, cmap="tab10",
             autoscale=False,
            grid=True, **kwargs):
        """Show the sky coverage.

        Parameters
        ----------
        stat: str, optional
            Statistic to plot.
        column: str, optional
            Column to use for the statistic.
        title: str, optional
            Title of the plot.
        data: pandas.DataFrame, optional
            Data to plot.
        origin: float, optional
            Origin of the ra coordinates.
        vmin, vmax: float, optional
            Min and max values for the colorbar.
        cmap: str, optional
            Colormap to use.
        autoscale: bool, optional
            If True, autoscale the plot.
        grid: bool, optional
            If True, show the grid.
        **kwargs
            Goes to matplotlib.collections.PolyCollection.

        Returns
        -------
        matplotlib.figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import to_rgba
        import cartopy.crs as ccrs

        if data is None and self.data is not None and len(self.data)>0:
            data = self.get_fieldstat(stat=stat, columns=column, incl_zeros=True,
                                          fillna=np.NaN, data=data)

        geodf = self.fields.copy()
        xy = np.stack(geodf["geometry"].apply(lambda x: ((np.asarray(x.exterior.xy)).T) ).values)
        # correct edge effects
        flag_egde = np.any(np.diff(xy, axis=1)>300, axis=1)[:,0]
        xy[flag_egde] = ((xy[flag_egde] + origin)%360 - origin)
        geodf["xy"] = list(xy)
        data = np.random.uniform(size=len(geodf))
        if vmin is None: vmin = np.nanmin(data)
        if vmax is None: vmax = np.nanmax(data)
        geodf["value"] = (data-vmin)/(vmax-vmin)

        # figure
        rect = [0.15,0.22,0.75,0.75]
        fig = plt.figure(figsize=[7,5])
        ax = fig.add_axes(rect, projection=ccrs.Mollweide())
        ax.set_global() # not sure why we need that

        prop = dict(edgecolor="0.7", lw=1, alpha=0.5,
                    transform = ccrs.PlateCarree(central_longitude=origin)
                   )

        # color
        cmap = plt.get_cmap(cmap)
        #
        coll = PolyCollection(geodf["xy"], array=geodf["value"], 
                          cmap=cmap, **prop)
        
        ax.add_collection(coll)
        if grid:
            ax.gridlines()
        if autoscale:
            ax.autoscale()
            
        return fig
        
    # ============== #
    # Static Methods #
    # ============== #
    @staticmethod
    def _parse_fields(fields):
        """ 
        Parse the fields.

        Parameters
        ----------
        fields: geopandas.GeoDataFrame
            field definitions.

        Returns
        -------
        geopandas.GeoDataFrame
        """
        return parse_fields(fields)
    
    @staticmethod
    def _draw_random(fieldids, size, 
                     bands,  
                     mjd_range, skynoise_range,
                     gain_range=1,
                     zp_range=[27,30]):
        """ 
        Draw random observations.

        Parameters
        ----------
        fieldids: list
            list of fieldids to draw from.

        size: int
            number of observations to draw

        bands: list of str
            list of bands that should be drawn.

        mjd_range: list or array
            min and max mjd for the random drawing.

        skynoise_range: list or array
            min and max skynoise for the random drawing.

        gain_range: list or array
            min and max gain for the random drawing.

        zp_range: list or array
            min and max zp for the random drawing.

        Returns
        -------
        pandas.DataFrame
        """
        # np.resize(1, 2) -> [1,1]
        mjd = np.random.uniform(*np.resize(mjd_range,2), size=size)
        band = np.random.choice(bands, size=size)
        skynoise = np.random.uniform(*np.resize(skynoise_range, 2), size=size)
        gain = np.random.uniform(*np.resize(gain_range, 2), size=size)
        zp = np.random.uniform(*np.resize(zp_range, 2), size=size)
        # = coords
        fieldid = np.random.choice(fieldids, size=size)
        # data sorted by mjd
        data = pandas.DataFrame(zip(mjd, band, skynoise, gain, zp, fieldid),
                               columns=["mjd","band","skynoise", "gain", "zp","fieldid"]
                               ).sort_values("mjd"
                               ).reset_index(drop=False) # don't need to know the creation order
        return data
    
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def fieldids(self):
        """List of fields id."""
        if self.fields is None:
            return None
        return self.fields.index
    
    @property
    def nfields(self):
        """Number of fields."""
        if self.fields is None:
            return None
        return len(self.fields)
