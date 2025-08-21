from .core import BaseSurvey

import pandas
import numpy as np
import healpy as hp
import warnings


__all__ = ["HealpixSurvey"]


def get_ipix_in_range(nside, ra_range=None, dec_range=None, in_rad=False):
    """Get the healpix pixel index (ipix) that are with a given ra and dec range.

    Parameters
    ----------
    nside : int
        Healpix nside.
    ra_range, dec_range: 2d-array, None, optional
        Min and max to define a coordinate range to be considered.
        None means no limit.
    in_rad: bool, optional
        Are the ra and dec coordinates in radian (True)
        or degree (False).

    Returns
    -------
    list
        List of healpix pixel index ipix.
    """
    npix = hp.nside2npix(nside)
    pixs = np.arange(npix) # list of all healpix pixels
    if ra_range is None and dec_range is None:
        return pixs
    
    ras,decs = hp.pix2ang(nside, pixs)
    ras = (np.pi/2-ras)
    # only dec range
    if ra_range is None:
        if not in_rad:
            dec_range = np.multiply(dec_range, np.pi/180) # works if list given
        return pixs[(decs>=dec_range[0]) & (decs<=dec_range[1])]
    
    # only ra range    
    if dec_range is None:
        if not in_rad:
            ra_range = np.multiply(ra_range, np.pi/180) # works if list given
        return pixs[(ras>=ra_range[0]) & (ras<=ra_range[1])]
    
    # both
    if not in_rad:
        ra_range = np.multiply(ra_range, np.pi/180) # works if list given
        dec_range = np.multiply(dec_range, np.pi/180) # works if list given
    return pixs[(ras>=ra_range[0]) & (ras<=ra_range[1]) & (decs>=dec_range[0]) & (decs<=dec_range[1])]


# ================== #
#                    #
#    Healpix         #
#                    #
# ================== #
class HealpixSurvey( BaseSurvey ):
    
    def __init__(self, nside, data=None):
        """ 
        Initialize the HealpixSurvey class.

        Parameters
        ----------
        nside : int
            healpix nside parameter

        data: pandas.DataFrame
            observing data.

        See also
        --------
        from_data: loads the instance given observing data.
        from_random: generate random observing data and loads the instance.
        """
        super().__init__(data)
        self._nside = nside
        
    @classmethod
    def from_data(cls, nside, data):
        """ load an instance given survey data and healpix size (nside) 
        
        Parameters
        ----------
        nside : int
            healpix nside parameter

        data: pandas.DataFrame
            observing data.

        Returns
        -------
        instance

        See also
        --------
        from_random: generate random observing data and loads the instance.
        
        """
        return cls(nside=nside, data=data)

    @classmethod
    def from_random(cls, nside, size, 
                    bands,  
                    mjd_range, skynoise_range,
                    ra_range=None, dec_range=None, **kwargs):
        """ 
        Load an instance with random observing data.

        Parameters
        ----------
        nside : int
            healpix nside parameter

        size: int
            number of observations to draw

        bands: list of str
            list of bands that should be drawn.

        mjd_range: list or array
            min and max mjd for the random drawing.

        skynoise_range: list or array
            min and max skynoise for the random drawing.

        ra_range, dec_range: 2d-array, None
            min and max to define a coordinate range to be considered.
            None means no limit.
        
        **kwargs goes to the draw_random() method

        Returns
        -------
        HealpixSurvey
        """
        this = cls(nside=nside)
        this.draw_random(size,  bands,  
                        mjd_range, skynoise_range, 
                        ra_range=ra_range, dec_range=dec_range,
                        inplace=True, **kwargs)
        return this

    @classmethod
    def from_pointings(cls, nside, data,
                       footprint=None,  moc=None,
                       rakey="ra", deckey="dec",
                       backend="polars",
                       use_pyarrow_extension_array=False,
                       **kwargs):
        """ loads an instance given observing poitings of a survey
        
        This loads an polygon.PolygonSurvey using from_pointing and 
        converts that into an healpix using the to_healpix() method

        Parameters
        ----------
        nside : int
            healpix nside parameter

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
        from .polygon import PolygonSurvey
        # Create a generic polygon survey
        polysurvey = PolygonSurvey.from_pointings(data, footprint=footprint,
                                                  moc=moc,
                                                  rakey=rakey, deckey=deckey,
                                                  **kwargs)
        # convert it to healpix
        return polysurvey.to_healpix(nside, backend=backend,
                                         pass_data=True,
                                         polars_to_pandas=True,
                                         use_pyarrow_extension_array=use_pyarrow_extension_array)
        
    
    # ============== #
    #   Methods      #
    # ============== #
    def get_field_area(self):
        """ area (deg**2) of a healpy pixel """
        return hp.nside2pixarea(self.nside, degrees = True)
    
    def get_observed_area(self, min_obs=1):
        """ get the observed area (in deg**2).
        A healpix is consider observed if present more than min_obs time.

        Parameters
        ----------
        min_obs: int
            minimum number of observations to consider a field as observed.

        Returns
        -------
        float
        """
        if min_obs <=1: # 0 or 1 the same
            nfields = self.data["fieldid"].nunique()
        else:
            nobs = self.data["fieldid"].value_counts()
            nfields = len(nobs[nobs>min_obs])
        
        return self.get_field_area() * nfields

    def get_polygons(self, observed_fields=False, as_vertices=False, origin=180):
        """Get a list of polygons.

        Parameters
        ----------
        observed_fields: bool, optional
            Should this be limited to observed fields?
        as_vertices: bool, optional
            Should this returns a list of shapely.geometry.Polygon (False)
            or its vertices (shape N [fields], 2 [ra, dec], 4[corners]).
        origin: float, optional
            Origin of the R.A. coordinate (center of image).

        Returns
        -------
        list
            (as_vertices)
            - list of polygon
            - list of vertices
        """
        if observed_fields:
            fieldid = self.data[self.fieldids.name].unique()
        else:
            fieldid = ultrasat.fieldids

        corners = hp.boundaries(nside=self.nside, pix=fieldid)
        corners = np.moveaxis(corners,1,2)
        ang = np.asarray([hp.vec2ang(corners_, lonlat=True) for corners_ in corners])
        ang[:,0,:] = (origin-ang[:,0,:])%360 # set back origin

        if as_vertices:
            return ang
        
        from shapely import geometry
        polygons = [geometry.Polygon(ang_.T) for ang_ in ang]
        return polygons

    def get_skyarea(self, as_multipolygon=True):
        """Get multipolygon (or list) of field geometries.

        Parameters
        ----------
        as_multipolygon: bool, optional
            If True, returns a multipolygon.
            Otherwise, returns a list of polygons.

        Returns
        -------
        shapely.geometry.MultiPolygon or list
        """
        from shapely import ops, geometry
        
        ps = self.get_polygons(observed_fields=True, as_vertices=False)
        if as_multipolygon:
            return geometry.MultiPolygon(ps)

        return ops.unary_union(ps)    
    # ------- #
    #  core   #
    # ------- #

    def radec_to_fieldid(self, radec, origin=180, observed_fields=False):
        """Get the fieldid associated to the given coordinates.

        Parameters
        ----------
        radec: pandas.DataFrame or 2d array
            Coordinates in degree.
        origin: float, optional
            Value of the central R.A.
        observed_fields: bool, optional
            Should this be limited to fields actually observed?
            This is ignored is self.data is None.

        Returns
        -------
        pandas.DataFrame
        """
        if type(radec) is pandas.DataFrame:
            ra = np.asarray(radec["ra"].values, dtype="float")
            dec = np.asarray(radec["dec"].values, dtype="float")
            index = radec.index.copy()
            if index.name is None:
                index.name = "index_radec"
            
        else:
            ra, dec = np.atleast_1d(radec)
            ra = np.atleast_1d(ra)
            dec = np.atleast_1d(dec)
            index = pandas.Index(np.arange( len(ra) ), name="index_radec")
            
        fields = hp.ang2pix(self.nside, (90 - dec) * np.pi/180, (origin-ra) * np.pi/180)
        df = pandas.DataFrame(fields, columns = [self.fieldids.name], index=index)
        
        if observed_fields:
            observed_fields = self.data[self.fieldids.name].unique()
            df = df[df[self.fieldids.name].isin(observed_fields)]
        
        return df

    def get_field_centroid(self, origin=180):
        """Get the centroid of the fields.

        Parameters
        ----------
        origin: float, optional
            Origin of the ra coordinates.

        Returns
        -------
        (array, array)
            ra, dec
        """
        dec, ra = np.asarray(hp.pix2ang(self.nside, self.fieldids))*180/np.pi
        dec = 90-dec
        ra = (origin-ra)%360
        return ra, dec

    # ------- #
    #  draw   #
    # ------- #    
    def draw_random(self, size, 
                    bands, mjd_range, skynoise_range,
                    gain_range=1, zp_range=25,
                    ra_range=None, dec_range=None,
                    inplace=False, nside=None, **kwargs):
        """ draw observations 

        Parameters
        ----------

        size: int
            number of observations to draw

        bands: list of str
            list of bands that should be drawn.

        ra_range, dec_range: 2d-array, None
            min and max to define a coordinate range to be considered.
            None means no limit.

        skynoise_range, gain_range, zp_range: 2d-array, float, int
            range to be considered.
            If float or int, this value will always be used.
            otherwise, uniform distribution between the range assumed.

        inplace: bool
            shall this method replace the current self.data or
            return a new instance of the class with the 
            generated observing data.
            
        nside: int
            = ignore if inplace is set to True =
            provide a new healpix nside parameters.

        Returns
        -------
        class instance or None
            see the inplace option.

        See also
        --------
        from_random: generate random observing data and loads the instance.
        set_data: set the observing data to the instance.
        """
        if nside is None: # don't change nside
            nside = self.nside
            
        elif inplace: # change nside
            warnings.warn("Cannot change nside with inplace=True, a copy (inplace=False) is returned.")
            inplace = False
            
        data = self._draw_random(nside, size, 
                                 bands, mjd_range, skynoise_range, 
                                 ra_range=ra_range, dec_range=dec_range,
                                 gain_range=gain_range, zp_range=zp_range,
                                 **kwargs)
        
        if not inplace:
            return self.__class__.from_data(nside=nside, data=data)

        self.set_data(data)
        
    # ----------- #
    #  PLOTTER    #
    # ----------- #
    def show(self, stat='size', column=None, title=None, data=None, vmin=None, vmax=None, **kwargs):
        """ shows the sky coverage using ``healpy.mollview`` 

        Parameters
        ----------
        stat: str
            element to be passed to groupby.agg() 
            could be e.g.: 'mean', 'std' etc.
            If stat = 'size', this returns data["fieldid"].value_counts()
            (slightly faster than groupby("fieldid").size()).

        columns: str
            column of the dataframe the stat should be applied to.
            = ignored if stat='size' = 

        title: str
            title of the healpy.mollview plot.
            (healpy.mollview option)
        
        data: pandas.DataFrame, None
            data you want this to be applied to.
            if None, a copy of self.data is used.
            = leave to None if unsure =
            
        Returns
        -------
        None
        
        See also
        --------
        get_fieldstat: get observing statistics for the fields
        """
        if data is None:
            if self.data is None:
                data = np.random.uniform(size=self.nfields)
            else:
                data = self.get_fieldstat(stat=stat, columns=column,
                                              incl_zeros=True, fillna=np.NaN,
                                              data=data)
                
        else:
            if type(data) is dict:
                data = pandas.Series(data)
                
            if type(data) is pandas.Series:
                data = data.reindex(self.fieldids).values

        if vmin is not None:
            data[data<vmin] = vmin
            
        if vmax is not None:
             data[data>vmax] = vmax
             
        return hp.mollview(data, title=title, **kwargs)
        
    # ============== #
    # Static Methods #
    # ============== #        
    @staticmethod
    def _draw_random(nside, size, 
                     bands,  
                     mjd_range, skynoise_range,
                     gain_range=1,
                     zp_range=[27,30],
                     ra_range=None, dec_range=None):
        """Draw observations | internal.

        Parameters
        ----------
        nside : int
            Healpix nside parameter.
        size: int
            Number of observations to draw.
        bands: list of str
            List of bands that should be drawn.
        mjd_range: list or array
            Min and max mjd for the random drawing.
        skynoise_range: list or array
            Min and max skynoise for the random drawing.
        gain_range: list or array, optional
            Min and max gain for the random drawing.
        zp_range: list or array, optional
            Min and max zp for the random drawing.
        ra_range, dec_range: 2d-array, None, optional
            Min and max to define a coordinate range to be considered.
            None means no limit.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the drawn observations.
        """
        # np.resize(1, 2) -> [1,1]
        mjd = np.random.uniform(*np.resize(mjd_range,2), size=size)
        band = np.random.choice(bands, size=size)
        skynoise = np.random.uniform(*np.resize(skynoise_range, 2), size=size)
        gain = np.random.uniform(*np.resize(gain_range, 2), size=size)
        zp = np.random.uniform(*np.resize(zp_range, 2), size=size)
        # = coords
        # no radec limit
        if ra_range is None and dec_range is None:
            npix = hp.nside2npix(nside)
            ipix = np.random.uniform(0, npix, size=size)
        else:
            ipix_ok = get_ipix_in_range(nside, ra_range=ra_range, dec_range=dec_range)
            ipix = np.random.choice(ipix_ok, size=size)
            
        # data sorted by mjd
        data = pandas.DataFrame(zip(mjd, band, skynoise, gain, zp, ipix),
                               columns=["mjd","band","skynoise", "gain", "zp","fieldid"]
                               ).sort_values("mjd"
                               ).reset_index(drop=False) # don't need to know the creation order
        return data
    
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def nside(self):
        """ healpix nside parameter (defines the 'fields' size and number) """
        return self._nside
    
    @property
    def nfields(self):
        """ number of fields (shortcut to npix) """
        return self.npix
    
    @property    
    def npix(self):
        """ number of healpix pixels """
        if not hasattr(self, "_npix") or self._npix is None:
            self._npix = hp.nside2npix(self.nside)
            
        return self._npix

    @property
    def fieldids(self):
        """ id of the individual fields """
        fieldids = np.arange( self.npix )
        # use pandas.index for self consistency with polygon.survey
        return pandas.Index(fieldids, name="fieldid")
   
    def metadata(self):
        """Pandas Series containing meta data information.

        Returns
        -------
        pandas.Series
        """
        meta = super().metadata
        meta["nside"] = self.nside
        return meta
