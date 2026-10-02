"""Polygon-based survey, where fields are shapely polygons projected on the sky.

This module defines the `PolygonSurvey` class.
"""

import pandas
import numpy as np
import geopandas
from shapely import geometry
import warnings

from .core import BaseSurvey
from ..tools.projection import spatialjoin_radec_to_fields, parse_fields, project_to_radec

    
# ================== #
#                    #
#    Polygon         #
#                    #
# ================== #
class PolygonSurvey( BaseSurvey ):
    """Survey whose fields are shapely polygons projected onto the sky.

    Parameters
    ----------
    data : pandas.DataFrame or None, optional
        Observing data. The default is None.

    fields : geopandas.GeoDataFrame or None, optional
        Field definitions. If None, the class `_DEFAULT_FIELDS` are used.
        The default is None.

    Attributes
    ----------
    _DEFAULT_FIELDS : geopandas.GeoDataFrame or None
        Default field definitions for this survey type. Subclasses should
        override it to provide specific survey grids.

    Raises
    ------
    NotImplementedError
        If `fields` is None and the class has no default fields.
    """
    _DEFAULT_FIELDS = None
    
    def __init__(self, data=None, fields=None):
        """Initialize the PolygonSurvey class."""
        if fields is None:
            if self._DEFAULT_FIELDS is None:
                raise NotImplementedError("No default fields known for this class. No fields given")
            fields = self._DEFAULT_FIELDS
        
        self._fields = self._parse_fields(fields)        
        super().__init__(data)
        
    @classmethod
    def from_pointings(cls, data, footprint=None, moc=None, rakey="ra", deckey="dec"):
        """Load an instance from pointings.

        Each pointing defines a field, obtained by projecting the camera
        footprint (or MOC) at the pointing coordinates.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            Observing data, must contain the `rakey` and `deckey` columns.

        footprint : shapely.geometry.Polygon or None, optional
            Footprint in the sky of the observing camera. The default is None.

        moc : mocpy.MOC or None, optional
            MOC representation of the observing camera (used if `footprint` is
            None). The default is None.

        rakey : str, optional
            Name of the R.A. column (in deg). The default is 'ra'.

        deckey : str, optional
            Name of the declination column (in deg). The default is 'dec'.

        Returns
        -------
        PolygonSurvey
            The loaded instance.
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
        """Load an instance with random observing data.

        Parameters
        ----------
        size : int
            Number of observations to draw.

        bands : list of str
            List of bands that should be drawn.

        mjd_range : array_like
            Min and max mjd for the random drawing.

        skynoise_range : array_like
            Min and max skynoise for the random drawing.

        fields : geopandas.GeoDataFrame or None, optional
            Field definitions. If None, the class default fields are used.
            The default is None.

        **kwargs
            Passed to :meth:`draw_random`.

        Returns
        -------
        PolygonSurvey
            The loaded instance.
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
        """Get the fields.

        Parameters
        ----------
        observed : bool, optional
            If True, return only the observed fields. The default is True.

        Returns
        -------
        geopandas.GeoDataFrame
            Copy of the (observed) fields.
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

        This converts the survey into a healpix survey for accuracy.

        Parameters
        ----------
        nside : int, optional
            Healpix nside. The default is 200.

        Returns
        -------
        float
            Area in deg**2.
        """
        hsurvey = self.to_healpix(nside=nside, pass_data=False)
        return hsurvey.get_observed_area()# min_obs=min_obs) # not correct with pass_data=False yet.

    def to_healpix(self, nside, pass_data=True, backend="polars",
                       polars_to_pandas=True,
                       use_pyarrow_extension_array=False):
        """Convert the current polygon survey into a healpix survey.

        Parameters
        ----------
        nside : int
            Healpix nside.

        pass_data : bool, optional
            Should the returned survey have the full data (True) or just the
            matching fieldids (False)? The default is True.

        backend : {'polars', 'pandas', 'dask'}, optional
            Which backend to use to merge the data (speed issue):

            - 'polars' (fastest): requires polars installed; converted to pandas
              at the end.
            - 'pandas' (classic): the normal way.
            - 'dask' (lazy): a persisted dask.dataframe is returned.

            The default is 'polars'.

        polars_to_pandas : bool, optional
            Ignored if backend is not 'polars'. Should the dataframe be converted
            into a pandas.DataFrame (using ``to_pandas()``) rather than kept as a
            polars.DataFrame? The default is True.

        use_pyarrow_extension_array : bool, optional
            Ignored if backend is not 'polars' or `polars_to_pandas` is not True.
            Should the pandas dataframe be based on pyarrow arrays (like in
            polars; faster to load, but numpy.asarray will be used by pandas when
            needed, which will then slow things down) rather than numpy arrays
            (slow to load but faster then). The default is False.

        Returns
        -------
        skysurvey.HealpixSurvey
            The healpix survey.

        Raises
        ------
        NotImplementedError
            If the `backend` is not implemented.
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
                except ImportError:
                    # warnings.warn("You do not have polars installed. conda/pip install polars. falling back to pandas backend")
                    backend = "pandas"
                    
            if backend == "dask":
                try:
                    import dask.dataframe as dd
                except ImportError:
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
        radec : pandas.DataFrame or array_like
            Coordinates in degree, either a DataFrame with 'ra' and 'dec' columns
            or an array of shape (N, 2).

        observed_fields : bool, optional
            Should this be limited to fields actually observed?
            The default is False.

        Returns
        -------
        pandas.DataFrame
            Fieldid(s) of each coordinate, indexed by 'index_radec'.

        Raises
        ------
        ValueError
            If `radec` is an array whose shape is not (N, 2).
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
        """Draw random observations.

        Parameters
        ----------
        size : int
            Number of observations to draw.

        bands : list of str
            List of bands that should be drawn.

        mjd_range : array_like
            Min and max mjd for the random drawing.

        skynoise_range : array_like
            Min and max skynoise for the random drawing.

        gain_range : array_like or float, optional
            Min and max gain for the random drawing. The default is 1.

        zp_range : array_like or float, optional
            Min and max zp for the random drawing. The default is 25.

        inplace : bool, optional
            If True, the data are stored in the instance. Otherwise, a new
            instance is returned. The default is False.

        fieldids : list or None, optional
            List of fieldids to draw from. If None, all fields are used.
            The default is None.

        **kwargs
            Passed to :meth:`_draw_random`.

        Returns
        -------
        PolygonSurvey or None
            New instance if `inplace` is False, None otherwise.
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
        stat : str, optional
            Statistic to plot. The default is 'size'.

        column : str or None, optional
            Column to use for the statistic. The default is None.

        title : str or None, optional
            Title of the plot. The default is None.

        data : pandas.DataFrame or None, optional
            Data to plot. The default is None.

        origin : float, optional
            Origin of the ra coordinates. The default is 180.

        vmin, vmax : float or None, optional
            Min and max values for the colorbar. If None, the min (max) of the
            data is used. The default is None.

        cmap : str, optional
            Colormap to use. The default is 'tab10'.

        autoscale : bool, optional
            If True, autoscale the plot. The default is False.

        grid : bool, optional
            If True, show the grid. The default is True.

        **kwargs
            Currently ignored.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        import cartopy.crs as ccrs

        if data is None and self.data is not None and len(self.data)>0:
            data = self.get_fieldstat(stat=stat, columns=column, incl_zeros=True,
                                          fillna=np.nan, data=data)

        geodf = self.fields.copy()
        xy = np.stack(geodf["geometry"].apply(lambda x: ((np.asarray(x.exterior.xy)).T) ).values)
        # correct edge effects
        flag_egde = np.any(np.diff(xy, axis=1)>300, axis=1)[:,0]
        xy[flag_egde] = ((xy[flag_egde] + origin)%360 - origin)
        geodf["xy"] = list(xy)

        rng = np.random.default_rng()
        data = rng.uniform(size=len(geodf))
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
            
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
        """Parse the fields.

        Parameters
        ----------
        fields : geopandas.GeoDataFrame
            Field definitions.

        Returns
        -------
        geopandas.GeoDataFrame
            The parsed fields.
        """
        return parse_fields(fields)
    
    @staticmethod
    def _draw_random(fieldids, size, 
                     bands,  
                     mjd_range, skynoise_range,
                     gain_range=1,
                     zp_range=[27,30],
                     rng=None):
        """Draw random observations (internal).

        Parameters
        ----------
        fieldids : list
            List of fieldids to draw from.

        size : int
            Number of observations to draw.

        bands : list of str
            List of bands that should be drawn.

        mjd_range : array_like
            Min and max mjd for the random drawing.

        skynoise_range : array_like
            Min and max skynoise for the random drawing.

        gain_range : array_like or float, optional
            Min and max gain for the random drawing. The default is 1.

        zp_range : array_like or float, optional
            Min and max zp for the random drawing. The default is [27, 30].

        rng : None, int, or numpy.random.Generator, optional
            Seed for the random number generator. Currently ignored (a new
            unseeded generator is always used). The default is None.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the drawn observations.
        """
        rng = np.random.default_rng()
        # np.resize(1, 2) -> [1,1]
        mjd = rng.uniform(*np.resize(mjd_range,2), size=size)
        band = rng.choice(bands, size=size)
        skynoise = rng.uniform(*np.resize(skynoise_range, 2), size=size)
        gain = rng.uniform(*np.resize(gain_range, 2), size=size)
        zp = rng.uniform(*np.resize(zp_range, 2), size=size)
        # = coords
        fieldid = rng.choice(fieldids, size=size)
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
        """List of field ids."""
        if self.fields is None:
            return None
        return self.fields.index
    
    @property
    def nfields(self):
        """Number of fields."""
        if self.fields is None:
            return None
        return len(self.fields)
